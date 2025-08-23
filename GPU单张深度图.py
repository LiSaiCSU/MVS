import numpy as np
import cv2
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

# ------------------------------------------------------------------
# 全局设备与 AMP 设置
# ------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = (device.type == "cuda")
AMP_DTYPE = torch.float16  # 可改成 bfloat16(>=Ampere更稳)，但一般 float16 就够
print("Using:", device, "| AMP:", AMP_ENABLED, "| AMP dtype:", AMP_DTYPE if AMP_ENABLED else "N/A")

# ----------------------------
# 读取 temple_par.txt
# ----------------------------
def read_temple_par(par_file_path):
    params = {}
    with open(par_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 22:
                continue
            img = parts[0]
            vals = np.array([float(v) for v in parts[1:]], dtype=np.float64)
            K = vals[0:9].reshape(3,3)
            R = vals[9:18].reshape(3,3)
            t = vals[18:21].reshape(3,1)
            params[img] = {"K":K, "R":R, "t":t, "name":img}
    return params

# ----------------------------
# 参考相机像素射线（保持在 torch / device 上）
# 返回 (3, H*W)
# ----------------------------
def make_ref_rays_torch(K_3x3, H, W):
    K = torch.as_tensor(K_3x3, device=device, dtype=torch.float32)
    K_inv = torch.linalg.inv(K)
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij"
    )
    uv1 = torch.stack([xs.reshape(-1), ys.reshape(-1), torch.ones(H*W, device=device)], dim=1)  # (HW,3)
    dirs = (K_inv @ uv1.T).T  # (HW,3)
    return dirs.T  # (3, HW)

# ----------------------------
# 用 BBOX 自动估计 d_min / d_max（全图/子采样）
# 思路：对下采样像素的射线做：C + t * r 与 AABB 相交，t>0 的最小/最大有效 t
# ----------------------------
def estimate_depth_range_from_bbox(K, R, t, bbox, H, W, sample_stride=8, safety_margin=1.02):
    """
    K: (3,3) np
    R: (3,3) np
    t: (3,1) np
    bbox: {'min': np.array(3,), 'max': np.array(3,)}
    返回: d_min, d_max (float)
    """
    K_t = torch.as_tensor(K, device=device, dtype=torch.float32)
    R_t = torch.as_tensor(R, device=device, dtype=torch.float32)
    t_t = torch.as_tensor(t, device=device, dtype=torch.float32)  # (3,1)

    # 采样像素网格
    ys = torch.arange(0, H, sample_stride, device=device, dtype=torch.float32)
    xs = torch.arange(0, W, sample_stride, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    uv1 = torch.stack([xx.reshape(-1), yy.reshape(-1), torch.ones_like(xx).reshape(-1)], dim=1)  # (N,3)

    K_inv = torch.linalg.inv(K_t)
    rays_cam = (K_inv @ uv1.T).T  # (N,3) 相机系射线方向（未单位化无妨）

    # 相机中心 C = -R^T t  (3,1) -> (3,)
    C = (-R_t.T @ t_t).squeeze(1)  # (3,)
    # 世界系方向 r = R^T * rays_cam^T
    r_world = (R_t.T @ rays_cam.T).T  # (N,3)

    # AABB
    bmin = torch.as_tensor(bbox['min'], device=device, dtype=torch.float32)  # (3,)
    bmax = torch.as_tensor(bbox['max'], device=device, dtype=torch.float32)  # (3,)

    # 与 AABB 相交的 t
    # 对每条射线：t1 = (bmin - C)/r, t2 = (bmax - C)/r
    # 注意除零：用很小的 epsilon
    eps = 1e-8
    r = torch.where(r_world.abs() < eps, torch.full_like(r_world, eps), r_world)
    t1 = (bmin - C) / r  # (N,3)
    t2 = (bmax - C) / r
    t_min = torch.minimum(t1, t2)  # (N,3)
    t_max = torch.maximum(t1, t2)  # (N,3)

    # 每条射线的整体进入/离开
    t_enter, _ = t_min.max(dim=1)  # (N,)
    t_exit,  _ = t_max.min(dim=1)  # (N,)

    valid = (t_exit > t_enter) & (t_exit > 0)
    t_enter = torch.where(valid, t_enter, torch.full_like(t_enter, float('inf')))
    t_exit  = torch.where(valid, t_exit,  torch.full_like(t_exit,  0.0))

    d_min = torch.clamp_min(t_enter.min(), 0.0).item()
    d_max = t_exit.max().item()

    if not np.isfinite(d_min) or not np.isfinite(d_max) or d_max <= 0:
        # 兜底（极少见）
        d_min, d_max = 0.5, 0.8

    # 留一点 margin，避免边缘裁剪
    center = 0.5*(d_min + d_max)
    half   = 0.5*(d_max - d_min)*safety_margin
    d_min2, d_max2 = max(1e-4, center - half), center + half
    return d_min2, d_max2

# ----------------------------
# 生成 plane sweep 重采样后的邻居体 (D, 1, H, W)，单通道灰度
# 支持 AMP：图像/重采样保持 AMP 精度；几何计算用 fp32 更稳
# ----------------------------

@torch.inference_mode()
def neighbor_psv_gray(neighbor_img_gray, d_samples, ref_dirs, ref_R, ref_t, nb_K, nb_R, nb_t, H, W):
    """
    neighbor_img_gray: (1,1,H,W)  (可为 fp16/fp32)
    d_samples: (D,) 或 (D, HW)
    ref_dirs: (3,HW) fp32
    ref_R, ref_t, nb_K, nb_R, nb_t: fp32
    return: (D,1,H,W), valid(D,HW)  （psv 的 dtype 与 neighbor_img_gray 一致）
    """
    D = d_samples.shape[0]
    HW = H*W

    # --- 几何（fp32稳健） ---
    ref_dirs32 = ref_dirs.to(torch.float32)
    d32 = d_samples.to(torch.float32)

    # ==================== BUG 修正 ====================
    # 检查 d_samples 的形状，并选择不同的广播方式
    is_per_pixel_depth = d32.ndim == 2

    if is_per_pixel_depth:
        # 亚像素细化情况: d32.shape is (D, HW)
        # P_cam: (D, 3, HW)
        P_cam = ref_dirs32.unsqueeze(0) * d32.unsqueeze(1)
    else:
        # 粗略搜索情况: d32.shape is (D,)
        # P_cam: (D, 3, HW)
        P_cam = ref_dirs32.unsqueeze(0) * d32.view(D, 1, 1)
    # ================================================

    # 世界坐标 P_w = R^T (P_c - t)
    P_world = (ref_R.T @ (P_cam - ref_t)).transpose(1,2)  # (D,HW,3)

    # 世界 -> 邻居相机
    P_nb = (nb_R @ P_world.transpose(1,2) + nb_t).transpose(1,2)  # (D,HW,3)

    proj = (nb_K @ P_nb.transpose(1,2)).transpose(1,2)  # (D,HW,3)
    z = proj[..., 2:3]                                  # (D,HW,1)
    valid = (z > 1e-6)
    uv = proj[..., :2] / (z + 1e-6)                      # (D,HW,2)

    # 归一化到 [-1,1]
    grid = uv.clone()
    grid[...,0] = grid[...,0] / (W-1) * 2 - 1
    grid[...,1] = grid[...,1] / (H-1) * 2 - 1
    grid = grid.view(D, H, W, 2)

    # --- 采样（AMP 精度） ---
    # 修正 FutureWarning
    with torch.amp.autocast(device_type=device.type, enabled=AMP_ENABLED, dtype=AMP_DTYPE):
        nb_batch = neighbor_img_gray.repeat(D, 1, 1, 1)
        psv = F.grid_sample(
            nb_batch, grid, mode="bilinear", padding_mode="border", align_corners=True
        )

    invalid = (~valid.view(D,H,W,1)).permute(0,3,1,2)
    psv = psv.masked_fill(invalid, 0.0)
    return psv, valid.view(D,HW)

# NCC：参考窗口 vs PSV窗口 -> (D, HW)
# unfold 的统计在 fp32，稳健
# ----------------------------
@torch.inference_mode()
def ncc_volume(psv, ref_img_gray, win):
    """
    psv: (D,1,H,W)  AMP 精度
    ref_img_gray: (1,1,H,W) AMP 精度
    返回 NCC 分数体 (D, HW)  (fp32)
    """
    pad = win//2
    # unfold（保持 AMP，再转 fp32 做统计更稳）
    ref_unf = F.unfold(ref_img_gray, kernel_size=win, padding=pad)  # (1, P, HW)
    psv_unf = F.unfold(psv,          kernel_size=win, padding=pad)  # (D, P, HW)

    ref_unf32 = ref_unf.to(torch.float32)
    psv_unf32 = psv_unf.to(torch.float32)

    ref_zm  = ref_unf32 - ref_unf32.mean(dim=1, keepdim=True)
    ref_nrm = torch.linalg.norm(ref_zm, dim=1, keepdim=True) + 1e-6

    psv_zm  = psv_unf32 - psv_unf32.mean(dim=1, keepdim=True)
    psv_nrm = torch.linalg.norm(psv_zm, dim=1, keepdim=True) + 1e-6

    num = (psv_zm * ref_zm).sum(dim=1)              # (D,HW)
    den = (psv_nrm.squeeze(1) * ref_nrm.squeeze(1)) # (D,HW)
    return (num / den)

# ----------------------------
# 主函数：plane sweep + 亚像素细化（AMP + BBOX 自动范围）
# ----------------------------
@torch.inference_mode()
def compute_depth_map_gpu(ref_img_bgr, neighbor_bgr_dict, ref_cam, neighbor_cams,
                          d_min, d_max, NUM_DEPTH_COARSE=64, WIN=5,
                          DEPTH_CHUNK_SIZE=16):
    # 灰度化到 [0,1]
    ref_gray = cv2.cvtColor(ref_img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    H, W = ref_gray.shape

    # 图像张量（AMP 精度承载）
    with torch.cuda.amp.autocast(enabled=AMP_ENABLED, dtype=AMP_DTYPE):
        ref_ten = torch.from_numpy(ref_gray).to(device).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        neighbors_gray = {
            k: torch.from_numpy(cv2.cvtColor(v, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0).to(device).unsqueeze(0).unsqueeze(0)
            for k, v in neighbor_bgr_dict.items()
        }

    # 相机参数（几何用 fp32）
    K_ref = torch.as_tensor(ref_cam["K"], device=device, dtype=torch.float32)
    R_ref = torch.as_tensor(ref_cam["R"], device=device, dtype=torch.float32)
    t_ref = torch.as_tensor(ref_cam["t"], device=device, dtype=torch.float32)

    # 预计算参考射线（fp32）
    ref_dirs = make_ref_rays_torch(K_ref, H, W)  # (3,HW) fp32

    # 深度采样（fp32）
    d_samples_all = torch.linspace(float(d_min), float(d_max), NUM_DEPTH_COARSE, device=device, dtype=torch.float32)

    best_scores = torch.full((H*W,), -1.0, device=device, dtype=torch.float32)
    best_idx    = torch.zeros((H*W,), dtype=torch.long, device=device)

    # 粗略搜索：分块处理
    for d_chunk in tqdm(torch.split(d_samples_all, DEPTH_CHUNK_SIZE), desc="Depth chunks"):
        D = d_chunk.shape[0]
        chunk_acc = torch.zeros((D, H*W), device=device, dtype=torch.float32)
        chunk_cnt = torch.zeros((D, H*W), device=device, dtype=torch.float32)

        for nb in neighbor_cams:
            nb_img = neighbors_gray[nb["name"]]  # (1,1,H,W) AMP 精度
            K_nb = torch.as_tensor(nb["K"], device=device, dtype=torch.float32)
            R_nb = torch.as_tensor(nb["R"], device=device, dtype=torch.float32)
            t_nb = torch.as_tensor(nb["t"], device=device, dtype=torch.float32)

            psv, valid = neighbor_psv_gray(
                nb_img, d_chunk, ref_dirs, R_ref, t_ref, K_nb, R_nb, t_nb, H, W
            )
            ncc = ncc_volume(psv, ref_ten, WIN)   # (D,HW) fp32
            ncc = torch.where(valid, ncc, torch.zeros_like(ncc))
            chunk_acc += ncc
            chunk_cnt += valid.float()

        # 邻居融合：平均（可改加权/中值）
        chunk_mean = torch.where(chunk_cnt > 0, chunk_acc / (chunk_cnt + 1e-6), torch.full_like(chunk_acc, -1.0))
        chunk_max, chunk_arg = torch.max(chunk_mean, dim=0)

        improve = chunk_max > best_scores
        best_scores[improve] = chunk_max[improve]

        # 定位在全体深度索引中的位置
        base = (d_chunk[0] - d_samples_all[0]) / (d_samples_all[1] - d_samples_all[0])
        base = int(base.item())
        best_idx[improve] = (chunk_arg[improve] + base)

    # ---- 亚像素细化（抛物线）----
    print("Subpixel refinement...")
    idx0 = best_idx.clamp(1, NUM_DEPTH_COARSE - 2)
    idxm1 = idx0 - 1
    idxp1 = idx0 + 1

    depths_triplet = torch.stack([d_samples_all[idxm1], d_samples_all[idx0], d_samples_all[idxp1]], dim=0)  # (3,HW)

    acc3 = torch.zeros((3, H*W), device=device, dtype=torch.float32)
    cnt3 = torch.zeros((3, H*W), device=device, dtype=torch.float32)

    for nb in neighbor_cams:
        nb_img = neighbors_gray[nb["name"]]
        K_nb = torch.as_tensor(nb["K"], device=device, dtype=torch.float32)
        R_nb = torch.as_tensor(nb["R"], device=device, dtype=torch.float32)
        t_nb = torch.as_tensor(nb["t"], device=device, dtype=torch.float32)

        psv, valid = neighbor_psv_gray(
            nb_img, depths_triplet, ref_dirs, R_ref, t_ref, K_nb, R_nb, t_nb, H, W
        )
        ncc = ncc_volume(psv, ref_ten, WIN)
        ncc = torch.where(valid, ncc, torch.zeros_like(ncc))
        acc3 += ncc
        cnt3 += valid.float()

    f_m1, f_0, f_p1 = [
        torch.where(cnt3[i] > 0, acc3[i]/(cnt3[i] + 1e-6), torch.full_like(acc3[i], -1.0))
        for i in range(3)
    ]

    denom = (f_m1 - 2*f_0 + f_p1)
    delta = torch.zeros_like(f_0)
    mask = denom.abs() > 1e-6
    delta[mask] = 0.5*(f_m1[mask] - f_p1[mask]) / denom[mask]
    delta = delta.clamp(-1.0, 1.0)
    step = (d_samples_all[1] - d_samples_all[0]).item()

    depth_refined = d_samples_all[idx0] + delta*step
    depth_map = depth_refined.view(H, W).contiguous()
    depth_map[best_scores.view(H,W) < 0] = 0.0
    return depth_map  # (H,W) fp32

# ----------------------------
# 入口
# ----------------------------
if __name__ == "__main__":
    # 参数
    REF_ID = f"temple{10:04d}.png"
    NEIGHBOR_IDS = [f"temple{i:04d}.png" for i in [7, 8, 9, 11, 12, 13]]
    WIN = 5
    NUM_DEPTH_COARSE = 64
    DEPTH_CHUNK_SIZE = 16

    BBOX = {'min': np.array([-0.054568, 0.001728, -0.042945]),
            'max': np.array([0.047855, 0.161892, 0.032236])}

    print("Loading cameras and images...")
    cams = read_temple_par('temple/temple_par.txt')
    ref_cam = cams[REF_ID]
    neighbors = [cams[k] for k in NEIGHBOR_IDS]

    ref_img = cv2.imread(f"temple/{REF_ID}", cv2.IMREAD_COLOR)
    neighbor_imgs = {k: cv2.imread(f"temple/{k}", cv2.IMREAD_COLOR) for k in NEIGHBOR_IDS}
    H, W, _ = ref_img.shape

    # --- 自动深度范围估计（来自 BBOX） ---
    d_min, d_max = estimate_depth_range_from_bbox(
        ref_cam["K"], ref_cam["R"], ref_cam["t"], BBOX, H, W,
        sample_stride=8, safety_margin=1.02
    )
    # 也可以加个鲁棒修剪（例如限制比值、绝对上下限）
    d_min = max(1e-4, d_min)
    d_max = max(d_min + 1e-4, d_max)

    print(f"Depth sweep in [{d_min:.4f}, {d_max:.4f}] @ {NUM_DEPTH_COARSE} samples")

    depth_map = compute_depth_map_gpu(
        ref_img, neighbor_imgs, ref_cam, neighbors,
        d_min, d_max, NUM_DEPTH_COARSE=NUM_DEPTH_COARSE, WIN=WIN,
        DEPTH_CHUNK_SIZE=DEPTH_CHUNK_SIZE
    ).cpu().numpy()

    np.save(f"depth_map_{REF_ID}.npy", depth_map)

    plt.figure(figsize=(10,8))
    valid = depth_map[depth_map>0]
    if valid.size>0:
        vmin, vmax = np.percentile(valid, 5), np.percentile(valid, 95)
        plt.imshow(depth_map, cmap="magma", vmin=vmin, vmax=vmax)
    else:
        plt.imshow(depth_map, cmap="magma")
    plt.title(f"Depth Map for {ref_cam['name']} (GPU plane sweep + AMP + bbox-range)")
    plt.colorbar(label="Depth (m)")
    plt.savefig(f"depth_map_{REF_ID}.png", dpi=150)
    plt.show()
