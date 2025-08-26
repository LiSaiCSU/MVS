import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ==============================================================================
# 数据加载函数
# ==============================================================================


def read_temple_par(par_file_path):
    """
    读取并解析 temple_par.txt 文件。
    *** 已修正：增加了对行长度的检查，以跳过头部的数字行或任何格式错误的行 ***
    """
    params_dict = {}
    with open(par_file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue  # 跳过空行

            parts = line.split()

            # --- 关键修正 ---
            # 检查这一行是否有足够的参数，如果不够就跳过
            if len(parts) != 22:
                print(f"Skipping malformed line: {line}")
                continue

            img_name = parts[0]
            values = np.array([float(v) for v in parts[1:]])
            k = values[0:9].reshape(3, 3)
            r = values[9:18].reshape(3, 3)
            t = values[18:21].reshape(3, 1)
            params_dict[img_name] = {"K": k, "R": r, "t": t}

    return params_dict


# ==============================================================================
# 核心几何变换函数
# ==============================================================================


def Pic2Cam(uv, depth, K):
    """
    将图像像素坐标(u,v)和深度depth转换为相机坐标系下的三维点。
    """
    K_inv = np.linalg.inv(K)
    uv_h = np.concatenate(
        [np.atleast_2d(uv), np.ones((np.atleast_2d(uv).shape[0], 1))], axis=1
    )  # 齐次坐标 (N,3)
    xyz_cam_norm = (K_inv @ uv_h.T).T  # 归一化相机坐标 (N,3)

    if np.isscalar(depth):
        xyz_cam = xyz_cam_norm[0] * depth
    else:
        xyz_cam = xyz_cam_norm * depth[:, None]
    return xyz_cam


def Cam2World(xyz_cam, R, t):
    """
    *** 已修正 ***
    将相机坐标(x,y,z)转换为世界坐标系下的三维点(X,Y,Z)。
    这是相机外参的逆变换。 P_world = R.T @ (P_cam - t)
    """
    xyz_cam = np.atleast_2d(xyz_cam)
    xyz_world = (np.linalg.inv(R) @ (xyz_cam.T - t)).T
    return xyz_world


def World2Cam(xyz_world, R, t):
    """
    将世界坐标系下的三维点(X,Y,Z)转换为相机坐标系下的三维点(x,y,z)。
    P_cam = R @ P_world + t
    """
    xyz_world = np.atleast_2d(xyz_world)
    xyz_cam = (R @ xyz_world.T + t).T
    return xyz_cam


def Cam2Pic(xyz_cam, K):
    """
    将相机坐标系下的三维点(x,y,z)投影到像素坐标系(u,v)。
    """
    xyz_cam = np.atleast_2d(xyz_cam)
    # 深度必须大于0
    depths = xyz_cam[:, 2]
    if np.any(depths <= 0):
        return None  # 点在相机后面或平面上，无法投影

    proj = (K @ xyz_cam.T).T
    uv = proj[:, :2] / proj[:, 2:3]

    return uv[0] if uv.shape[0] == 1 else uv


# ==============================================================================
# 匹配与评分函数
# ==============================================================================


def extract_window(image, center_uv, size):
    """
    从图像中提取一个窗口，并处理边界情况。
    """
    u, v = np.round(center_uv).astype(int)
    h, w, _ = image.shape
    half_size = size // 2

    if (
        u - half_size < 0
        or u + half_size + 1 > w
        or v - half_size < 0
        or v + half_size + 1 > h
    ):
        return None

    return image[v - half_size : v + half_size + 1, u - half_size : u + half_size + 1]


def calculate_ncc(w1, w2):
    """
    计算两个窗口的归一化互相关(NCC)。
    """
    if w1 is None or w2 is None or w1.shape != w2.shape:
        return -1.0  # 返回-1表示无效

    w1 = w1.astype(np.float64)
    w2 = w2.astype(np.float64)

    # 展平并去均值
    w1_flat = w1.flatten()
    w2_flat = w2.flatten()
    w1_zm = w1_flat - np.mean(w1_flat)
    w2_zm = w2_flat - np.mean(w2_flat)

    numerator = np.dot(w1_zm, w2_zm)
    denominator = np.sqrt(np.sum(w1_zm**2) * np.sum(w2_zm**2))

    return numerator / denominator if denominator > 1e-6 else 0.0


def calculate_correlation_score(
    p_ref,
    depth,
    ref_cam_info,
    neighbor_cams_info,
    ref_img,
    neighbor_images,
    m_size,
    ncc_thresh,
):
    """
    计算一个深度假设的相关性得分。
    """
    ref_K, ref_R, ref_t = ref_cam_info["K"], ref_cam_info["R"], ref_cam_info["t"]

    xyz_cam_ref = Pic2Cam(p_ref, depth, ref_K)
    xyz_world = Cam2World(xyz_cam_ref, ref_R, ref_t)

    W_ref = extract_window(ref_img, p_ref, m_size)
    if W_ref is None:
        return -1.0

    valid_ncc_scores = []
    for neighbor_cam in neighbor_cams_info:
        K, R, t = neighbor_cam["K"], neighbor_cam["R"], neighbor_cam["t"]

        xyz_cam_neighbor = World2Cam(xyz_world, R, t)
        p_neighbor = Cam2Pic(xyz_cam_neighbor, K)

        if p_neighbor is None:
            continue

        W_neighbor = extract_window(
            neighbor_images[neighbor_cam["name"]], p_neighbor, m_size
        )

        ncc = calculate_ncc(W_ref, W_neighbor)

        if ncc > ncc_thresh:
            valid_ncc_scores.append(ncc)

    if len(valid_ncc_scores) < 2:  # 论文要求至少2个视图匹配 [cite: 1]
        return -1.0

    return np.mean(valid_ncc_scores)


def calculate_ray_bbox_intersection(p_ref, cam_info, bbox):
    """
    计算像素射线与边界框的交点以确定深度搜索范围。
    """
    cam_center = -cam_info["R"].T @ cam_info["t"]
    p_cam_norm = np.linalg.inv(cam_info["K"]) @ np.array([p_ref[0], p_ref[1], 1.0])
    ray_direction = cam_info["R"].T @ p_cam_norm
    ray_direction /= np.linalg.norm(ray_direction)

    t_min, t_max = 0.0, np.inf
    for i in range(3):
        if abs(ray_direction[i]) < 1e-6:
            if cam_center[i] < bbox["min"][i] or cam_center[i] > bbox["max"][i]:
                return None, None
        else:
            t1 = (bbox["min"][i] - cam_center[i]) / ray_direction[i]
            t2 = (bbox["max"][i] - cam_center[i]) / ray_direction[i]
            if t1 > t2:
                t1, t2 = t2, t1
            t_min, t_max = max(t_min, t1), min(t_max, t2)

    return (t_min[0], t_max[0]) if t_min < t_max else (None, None)


# ==============================================================================
# 主执行逻辑
# ==============================================================================

if __name__ == "__main__":
    # --- 1. 设置参数 ---
    # 使用f-string进行格式化，更健壮
    REF_ID = f"temple{10:04d}.png"
    NEIGHBOR_IDS = [f"temple{i:04d}.png" for i in [8, 9, 11, 12]]

    # 算法超参数 (来自论文)
    M_WINDOW_SIZE = 5
    NCC_THRESHOLD = 0.6
    DELTA_D_COARSE = 2.5e-3  # 2.5mm

    # Temple数据集的边界框 (来自readme)
    BBOX = {
        "min": np.array([-0.054568, 0.001728, -0.042945]),
        "max": np.array([0.047855, 0.161892, 0.032236]),
    }

    # --- 2. 加载数据 ---
    print("Loading camera parameters...")
    all_cameras = read_temple_par("temple/temple_par.txt")

    # 为每个相机信息添加'name'字段，方便后续引用
    for name, params in all_cameras.items():
        params["name"] = name

    ref_cam_info = all_cameras[REF_ID]
    neighbor_cams_info = [all_cameras[name] for name in NEIGHBOR_IDS]

    ref_img = cv2.imread(f"temple/{REF_ID}")
    if ref_img is None:
        raise FileNotFoundError(
            f"Reference image {REF_ID} not found in 'temple/' directory."
        )

    neighbor_images = {name: cv2.imread(f"temple/{name}") for name in NEIGHBOR_IDS}

    h, w, _ = ref_img.shape
    depth_map = np.zeros((h, w), dtype=np.float64)

    print(f"Generating depth map for {ref_cam_info['name']} ({w}x{h})...")

    # --- 3. 主循环: 遍历每个像素 ---
    for y in tqdm(range(h)):
        for x in range(w):
            d_start, d_end = calculate_ray_bbox_intersection((x, y), ref_cam_info, BBOX)
            if d_start is None:
                continue

            # --- 4a. 粗略搜索 ---
            best_coarse_depth, max_coarse_corr = -1.0, -1.0
            num_steps = int((d_end - d_start) / DELTA_D_COARSE)
            if num_steps <= 1:
                continue

            for d in np.linspace(
                d_start, d_end, num=min(num_steps, 100)
            ):  # 限制最大步数以提高效率
                corr = calculate_correlation_score(
                    (x, y),
                    d,
                    ref_cam_info,
                    neighbor_cams_info,
                    ref_img,
                    neighbor_images,
                    M_WINDOW_SIZE,
                    NCC_THRESHOLD,
                )
                if corr > max_coarse_corr:
                    max_coarse_corr, best_coarse_depth = corr, d

            # --- 4b. 精细搜索 ---
            if best_coarse_depth > 0:
                final_depth, max_fine_corr = -1.0, -1.0
                fine_d_start = best_coarse_depth - DELTA_D_COARSE
                fine_d_end = best_coarse_depth + DELTA_D_COARSE

                for d in np.linspace(fine_d_start, fine_d_end, num=10):
                    corr = calculate_correlation_score(
                        (x, y),
                        d,
                        ref_cam_info,
                        neighbor_cams_info,
                        ref_img,
                        neighbor_images,
                        M_WINDOW_SIZE,
                        NCC_THRESHOLD,
                    )
                    if corr > max_fine_corr:
                        max_fine_corr, final_depth = corr, d

                depth_map[y, x] = final_depth

    # --- 5. 保存和显示结果 ---
    np.save(f"depth_map_{REF_ID}.npy", depth_map)

    plt.figure(figsize=(10, 8))
    valid_depths = depth_map[depth_map > 0]
    if len(valid_depths) > 0:
        vmin = np.percentile(valid_depths, 5)
        vmax = np.percentile(valid_depths, 95)
        plt.imshow(depth_map, cmap="magma", vmin=vmin, vmax=vmax)
    else:
        plt.imshow(depth_map, cmap="magma")

    plt.title(f"Depth Map for {ref_cam_info['name']}")
    plt.colorbar(label="Depth (meters)")
    plt.savefig(f"depth_map_{REF_ID}.png")
    plt.show()
