import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ------------------------------------------------------------------
# 全局设备与 AMP 设置
# ------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = device.type == "cuda"
AMP_DTYPE = torch.float16  # 可改成 bfloat16(>=Ampere更稳)，但一般 float16 就够
print(
    "Using:",
    device,
    "| AMP:",
    AMP_ENABLED,
    "| AMP dtype:",
    AMP_DTYPE if AMP_ENABLED else "N/A",
)


# ----------------------------
# 数据加载、几何、NCC 函数... (与之前版本相同)
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
            K = vals[:9].reshape(3, 3)
            R = vals[9:18].reshape(3, 3)
            t = vals[18:21].reshape(3, 1)
            params[img] = {"K": K, "R": R, "t": t, "name": img}
    return params


def make_ref_rays_torch(K_3x3, H, W):
    K = torch.as_tensor(K_3x3, device=device, dtype=torch.float32)
    K_inv = torch.linalg.inv(K)
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    uv1 = torch.stack(
        [xs.reshape(-1), ys.reshape(-1), torch.ones(H * W, device=device)], dim=1
    )  # (HW,3)
    dirs = (K_inv @ uv1.T).T  # (HW,3)
    return dirs.T  # (3, HW)


def estimate_depth_range_from_bbox(
    K, R, t, bbox, H, W, sample_stride=8, safety_margin=1.02
):
    K_t = torch.as_tensor(K, device=device, dtype=torch.float32)
    R_t = torch.as_tensor(R, device=device, dtype=torch.float32)
    t_t = torch.as_tensor(t, device=device, dtype=torch.float32)  # (3,1)
    ys = torch.arange(0, H, sample_stride, device=device, dtype=torch.float32)
    xs = torch.arange(0, W, sample_stride, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    uv1 = torch.stack(
        [xx.reshape(-1), yy.reshape(-1), torch.ones_like(xx).reshape(-1)], dim=1
    )  # (N,3)
    K_inv = torch.linalg.inv(K_t)
    rays_cam = (K_inv @ uv1.T).T  # (N,3) 相机系射线方向（未单位化无妨）
    C = (-R_t.T @ t_t).squeeze(1)  # (3,)
    r_world = (R_t.T @ rays_cam.T).T  # (N,3)
    bmin = torch.as_tensor(bbox["min"], device=device, dtype=torch.float32)  # (3,)
    bmax = torch.as_tensor(bbox["max"], device=device, dtype=torch.float32)  # (3,)
    eps = 1e-8
    r = torch.where(r_world.abs() < eps, torch.full_like(r_world, eps), r_world)
    t1 = (bmin - C) / r  # (N,3)
    t2 = (bmax - C) / r
    t_min = torch.minimum(t1, t2)  # (N,3)
    t_max = torch.maximum(t1, t2)  # (N,3)
    t_enter, _ = t_min.max(dim=1)  # (N,)
    t_exit, _ = t_max.min(dim=1)  # (N,)
    valid = (t_exit > t_enter) & (t_exit > 0)
    t_enter = torch.where(valid, t_enter, torch.full_like(t_enter, float("inf")))
    t_exit = torch.where(valid, t_exit, torch.full_like(t_exit, 0.0))
    d_min = torch.clamp_min(t_enter.min(), 0.0).item()
    d_max = t_exit.max().item()
    if not np.isfinite(d_min) or not np.isfinite(d_max) or d_max <= 0:
        d_min, d_max = 0.5, 0.8
    center = 0.5 * (d_min + d_max)
    half = 0.5 * (d_max - d_min) * safety_margin
    d_min2, d_max2 = max(1e-4, center - half), center + half
    return d_min2, d_max2


@torch.inference_mode()
def neighbor_psv_gray(
    neighbor_img_gray, d_samples, ref_dirs, ref_R, ref_t, nb_K, nb_R, nb_t, H, W
):
    D = d_samples.shape[0]
    HW = H * W
    ref_dirs32 = ref_dirs.to(torch.float32)
    d32 = d_samples.to(torch.float32)
    is_per_pixel_depth = d32.ndim == 2
    if is_per_pixel_depth:
        P_cam = ref_dirs32.unsqueeze(0) * d32.unsqueeze(1)
    else:
        P_cam = ref_dirs32.unsqueeze(0) * d32.view(D, 1, 1)
    P_world = (ref_R.T @ (P_cam - ref_t)).transpose(1, 2)
    P_nb = (nb_R @ P_world.transpose(1, 2) + nb_t).transpose(1, 2)
    proj = (nb_K @ P_nb.transpose(1, 2)).transpose(1, 2)
    z = proj[..., 2:3]
    valid = z > 1e-6
    uv = proj[..., :2] / (z + 1e-6)
    grid = uv.clone()
    grid[..., 0] = grid[..., 0] / (W - 1) * 2 - 1
    grid[..., 1] = grid[..., 1] / (H - 1) * 2 - 1
    grid = grid.view(D, H, W, 2)
    with torch.amp.autocast(
        device_type=device.type, enabled=AMP_ENABLED, dtype=AMP_DTYPE
    ):
        nb_batch = neighbor_img_gray.repeat(D, 1, 1, 1)
        psv = F.grid_sample(
            nb_batch, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
    invalid = (~valid.view(D, H, W, 1)).permute(0, 3, 1, 2)
    psv = psv.masked_fill(invalid, 0.0)
    return psv, valid.view(D, HW)


@torch.inference_mode()
def ncc_volume(psv, ref_img_gray, win):
    pad = win // 2
    ref_unf = F.unfold(ref_img_gray, kernel_size=win, padding=pad)
    psv_unf = F.unfold(psv, kernel_size=win, padding=pad)
    ref_unf32 = ref_unf.to(torch.float32)
    psv_unf32 = psv_unf.to(torch.float32)
    ref_zm = ref_unf32 - ref_unf32.mean(dim=1, keepdim=True)
    ref_nrm = torch.linalg.norm(ref_zm, dim=1, keepdim=True) + 1e-6
    psv_zm = psv_unf32 - psv_unf32.mean(dim=1, keepdim=True)
    psv_nrm = torch.linalg.norm(psv_zm, dim=1, keepdim=True) + 1e-6
    num = (psv_zm * ref_zm).sum(dim=1)
    den = psv_nrm.squeeze(1) * ref_nrm.squeeze(1)
    return num / den


# ----------------------------
# 优化: 深度图后处理滤波
# ----------------------------
def filter_depth_map(
    depth_map,
    filter_type="median",
    median_ksize=5,
    bilateral_d=5,
    bilateral_sigma_color=75,
    bilateral_sigma_space=75,
):
    """
    对深度图进行后处理滤波以去除噪声。
    参数:
        depth_map (np.ndarray): 原始深度图，带有0.0的无效区域。
        filter_type (str): 'median' 或 'bilateral'
        median_ksize (int): 中值滤波的核大小 (必须是奇数)。
        bilateral_d (int): 双边滤波的直径。
        bilateral_sigma_color (int): 双边滤波的颜色空间sigma。
        bilateral_sigma_space (int): 双边滤波的空间坐标sigma。
    返回:
        np.ndarray: 滤波后的深度图。
    """
    if filter_type not in ["median", "bilateral", None]:
        print(f"警告: 未知的滤波类型 '{filter_type}'. 跳过滤波.")
        return depth_map

    if filter_type is None:
        print("未启用深度图滤波.")
        return depth_map

    print(f"正在对深度图应用 {filter_type} 滤波...")

    # 临时处理：将无效区域 (0.0) 填充，避免滤波时产生错误深度
    valid_mask = depth_map > 0
    if not np.any(valid_mask):
        return depth_map

    filled_depth = depth_map.copy()

    # 找到有效深度点周围的零值点进行简单填充，这有助于滤波
    # 也可以使用更高级的插值方法，这里为了简洁只做简单的邻域填充
    filled_depth[~valid_mask] = np.mean(filled_depth[valid_mask])  # 均值填充

    filtered_map = filled_depth
    if filter_type == "median":
        if median_ksize % 2 == 0:
            median_ksize += 1
        filtered_map = cv2.medianBlur(
            filled_depth.astype(np.float32), ksize=median_ksize
        )
    elif filter_type == "bilateral":
        # 双边滤波需要深度图为uint8格式，且值范围为0-255，我们先进行归一化
        min_d, max_d = np.percentile(filled_depth[valid_mask], [5, 95])
        normalized_d = (
            (np.clip(filled_depth, min_d, max_d) - min_d)
            / (max_d - min_d + 1e-6)
            * 255.0
        )
        normalized_d = normalized_d.astype(np.uint8)

        filtered_normalized = cv2.bilateralFilter(
            normalized_d,
            d=bilateral_d,
            sigmaColor=bilateral_sigma_color,
            sigmaSpace=bilateral_sigma_space,
        )

        # 将滤波后的结果反归一化回原始深度范围
        filtered_map = (filtered_normalized / 255.0) * (max_d - min_d) + min_d

    # 最后，将滤波后的结果应用回原始的有效区域，并确保无效区域仍然为0
    final_depth = np.zeros_like(depth_map)
    final_depth[valid_mask] = filtered_map[valid_mask]

    return final_depth


# ----------------------------
# 主函数：plane sweep + 亚像素细化 + 滤波
# ----------------------------
@torch.inference_mode()
def compute_depth_map_gpu(
    ref_img_bgr,
    neighbor_bgr_dict,
    ref_cam,
    neighbor_cams,
    d_min,
    d_max,
    NUM_DEPTH_COARSE=64,
    WIN=5,
    DEPTH_CHUNK_SIZE=16,
    NCC_THRESHOLD=-0.5,
):
    ref_gray = cv2.cvtColor(ref_img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    H, W = ref_gray.shape
    with torch.amp.autocast(
        device_type=device.type, enabled=AMP_ENABLED, dtype=AMP_DTYPE
    ):
        ref_ten = torch.from_numpy(ref_gray).to(device).unsqueeze(0).unsqueeze(0)
        neighbors_gray = {
            k: torch.from_numpy(
                cv2.cvtColor(v, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            )
            .to(device)
            .unsqueeze(0)
            .unsqueeze(0)
            for k, v in neighbor_bgr_dict.items()
        }
    K_ref = torch.as_tensor(ref_cam["K"], device=device, dtype=torch.float32)
    R_ref = torch.as_tensor(ref_cam["R"], device=device, dtype=torch.float32)
    t_ref = torch.as_tensor(ref_cam["t"], device=device, dtype=torch.float32)
    ref_dirs = make_ref_rays_torch(K_ref, H, W)
    d_samples_all = torch.linspace(
        float(d_min), float(d_max), NUM_DEPTH_COARSE, device=device, dtype=torch.float32
    )
    best_scores = torch.full((H * W,), -1.0, device=device, dtype=torch.float32)
    best_idx = torch.zeros((H * W,), dtype=torch.long, device=device)
    for d_chunk in tqdm(
        torch.split(d_samples_all, DEPTH_CHUNK_SIZE), desc="Depth chunks"
    ):
        D = d_chunk.shape[0]
        chunk_acc = torch.zeros((D, H * W), device=device, dtype=torch.float32)
        chunk_cnt = torch.zeros((D, H * W), device=device, dtype=torch.float32)
        for nb in neighbor_cams:
            nb_img = neighbors_gray[nb["name"]]
            K_nb = torch.as_tensor(nb["K"], device=device, dtype=torch.float32)
            R_nb = torch.as_tensor(nb["R"], device=device, dtype=torch.float32)
            t_nb = torch.as_tensor(nb["t"], device=device, dtype=torch.float32)
            psv, valid = neighbor_psv_gray(
                nb_img, d_chunk, ref_dirs, R_ref, t_ref, K_nb, R_nb, t_nb, H, W
            )
            ncc = ncc_volume(psv, ref_ten, WIN)
            ncc = torch.where(valid, ncc, torch.zeros_like(ncc))
            chunk_acc += ncc
            chunk_cnt += valid.float()
        chunk_mean = torch.where(
            chunk_cnt > 0,
            chunk_acc / (chunk_cnt + 1e-6),
            torch.full_like(chunk_acc, -1.0),
        )
        chunk_max, chunk_arg = torch.max(chunk_mean, dim=0)
        improve = chunk_max > best_scores
        best_scores[improve] = chunk_max[improve]
        base = (d_chunk[0] - d_samples_all[0]) / (d_samples_all[1] - d_samples_all[0])
        base = int(base.item())
        best_idx[improve] = chunk_arg[improve] + base
    print("Subpixel refinement...")
    idx0 = best_idx.clamp(1, NUM_DEPTH_COARSE - 2)
    idxm1 = idx0 - 1
    idxp1 = idx0 + 1
    depths_triplet = torch.stack(
        [d_samples_all[idxm1], d_samples_all[idx0], d_samples_all[idxp1]], dim=0
    )
    acc3 = torch.zeros((3, H * W), device=device, dtype=torch.float32)
    cnt3 = torch.zeros((3, H * W), device=device, dtype=torch.float32)
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
        torch.where(
            cnt3[i] > 0, acc3[i] / (cnt3[i] + 1e-6), torch.full_like(acc3[i], -1.0)
        )
        for i in range(3)
    ]
    denom = f_m1 - 2 * f_0 + f_p1
    delta = torch.zeros_like(f_0)
    mask = denom.abs() > 1e-6
    delta[mask] = 0.5 * (f_m1[mask] - f_p1[mask]) / denom[mask]
    delta = delta.clamp(-1.0, 1.0)
    step = (d_samples_all[1] - d_samples_all[0]).item()
    depth_refined = d_samples_all[idx0] + delta * step
    depth_map = depth_refined.view(H, W).contiguous()
    # 关键步骤：用 NCC 分数进行过滤
    depth_map[best_scores.view(H, W) < NCC_THRESHOLD] = 0.0
    return depth_map


# ----------------------------
# 入口
# ----------------------------
if __name__ == "__main__":
    # ======================================================================
    # --- 用户可调参数 ---
    # 核心算法参数
    WIN = 3
    NUM_DEPTH_COARSE = 256
    DEPTH_CHUNK_SIZE = 128

    # 深度图质量控制参数
    # NCC阈值: 过滤掉低置信度的深度点 (范围通常在[-1, 1]，0.0到0.2比较保守)
    NCC_THRESHOLD = 0.85

    # 后处理滤波参数
    USE_FILTERING = 0  # 是否启用滤波
    FILTER_TYPE = "median"  # 'median' 或 'bilateral'
    # 中值滤波参数
    MEDIAN_KERNEL_SIZE = 3
    # 双边滤波参数 (d: 邻域直径, sigmaColor: 颜色相似度, sigmaSpace: 空间相似度)
    BILATERAL_D = 5
    BILATERAL_SIGMA_COLOR = 100
    BILATERAL_SIGMA_SPACE = 100

    # 数据集和相机设置
    REF_ID = f"temple{10:04d}.png"
    NEIGHBOR_IDS = [f"temple{i:04d}.png" for i in [7, 8, 9, 11, 12, 13]]
    BBOX = {
        "min": np.array([-0.054568, 0.001728, -0.042945]),
        "max": np.array([0.047855, 0.161892, 0.032236]),
    }
    # ======================================================================

    print("Loading cameras and images...")
    cams = read_temple_par("temple/temple_par.txt")
    ref_cam = cams[REF_ID]
    neighbors = [cams[k] for k in NEIGHBOR_IDS]
    ref_img = cv2.imread(f"temple/{REF_ID}", cv2.IMREAD_COLOR)
    neighbor_imgs = {
        k: cv2.imread(f"temple/{k}", cv2.IMREAD_COLOR) for k in NEIGHBOR_IDS
    }
    H, W, _ = ref_img.shape

    d_min, d_max = estimate_depth_range_from_bbox(
        ref_cam["K"],
        ref_cam["R"],
        ref_cam["t"],
        BBOX,
        H,
        W,
        sample_stride=8,
        safety_margin=1.02,
    )
    d_min = max(1e-4, d_min)
    d_max = max(d_min + 1e-4, d_max)
    print(f"Depth sweep in [{d_min:.4f}, {d_max:.4f}] @ {NUM_DEPTH_COARSE} samples")

    # --- 1. 计算原始深度图 ---
    raw_depth_map = (
        compute_depth_map_gpu(
            ref_img,
            neighbor_imgs,
            ref_cam,
            neighbors,
            d_min,
            d_max,
            NUM_DEPTH_COARSE=NUM_DEPTH_COARSE,
            WIN=WIN,
            DEPTH_CHUNK_SIZE=DEPTH_CHUNK_SIZE,
            NCC_THRESHOLD=NCC_THRESHOLD,  # 传递阈值
        )
        .cpu()
        .numpy()
    )

    # --- 2. 后处理滤波 ---
    filtered_depth_map = filter_depth_map(
        raw_depth_map,
        filter_type=FILTER_TYPE,
        median_ksize=MEDIAN_KERNEL_SIZE,
        bilateral_d=BILATERAL_D,
        bilateral_sigma_color=BILATERAL_SIGMA_COLOR,
        bilateral_sigma_space=BILATERAL_SIGMA_SPACE,
    )

    np.save(f"depth_map_filtered_{REF_ID}.npy", filtered_depth_map)

    # --- 3. 可视化结果 ---
    plt.figure(figsize=(10, 8))
    valid = filtered_depth_map[filtered_depth_map > 0]
    if valid.size > 0:
        vmin, vmax = np.percentile(valid, 5), np.percentile(valid, 95)
        plt.imshow(filtered_depth_map, cmap="magma", vmin=vmin, vmax=vmax)
    else:
        plt.imshow(filtered_depth_map, cmap="magma")
    plt.title(f"Depth Map for {ref_cam['name']} (Filtered)")
    plt.colorbar(label="Depth (m)")
    plt.savefig(f"depth_map_filtered_{REF_ID}.png", dpi=150)
    plt.show()
