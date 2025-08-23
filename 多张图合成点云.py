import numpy as np
import open3d as o3d
import cv2
import os
from tqdm import tqdm

# ------------------------------------------------------------------
# 全局设置
# ------------------------------------------------------------------
TEMPLE_DIR = 'temple/'
DEPTH_MAPS_DIR = 'depth_maps1/'
OUTPUT_PLY_FILE = 'reconstructed_point_cloud_filtered2.ply'

# ----------------------------
# 读取 temple_par.txt (与之前版本相同)
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
            K = vals[:9].reshape(3,3)
            R = vals[9:18].reshape(3,3)
            t = vals[18:21].reshape(3,1)
            params[img] = {"K":K, "R":R, "t":t, "name":img}
    return params

def create_4x4_transform_matrix(R, t):
    """
    根据3x3旋转矩阵R和3x1平移向量t创建4x4的齐次变换矩阵。
    """
    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3:4] = t
    return T

# ------------------------------------------------------------------
# 主函数：合成多张深度图
# ------------------------------------------------------------------
def reconstruct_from_depth_maps():
    
    # === 用户可调参数 ===
    COLOR_FILTER_THRESHOLD = 1 # 颜色过滤阈值，0.0-1.0之间。值越大，过滤越严格。
    VOXEL_SIZE = 0.002           # 降采样体素大小
    NB_NEIGHBORS = 200            # 统计离群点移除的邻居数量
    STD_RATIO = 2.0              # 统计离群点移除的标准差倍数
    # ==================
    
    print("Loading camera parameters...")
    all_cams = read_temple_par(os.path.join(TEMPLE_DIR, 'temple_par.txt'))

    merged_point_cloud = o3d.geometry.PointCloud()

    depth_map_files = sorted([f for f in os.listdir(DEPTH_MAPS_DIR) if f.endswith('.npy')])
    
    if not depth_map_files:
        print("错误: 'depth_maps' 文件夹中没有找到任何 .npy 深度图文件。")
        return

    print(f"Found {len(depth_map_files)} depth maps. Starting reconstruction...")
    
    for depth_file in tqdm(depth_map_files, desc="Processing depth maps"):
        base_name = os.path.splitext(depth_file)[0]
        original_img_name = base_name.replace('depth_', '') + '.png'

        if original_img_name not in all_cams:
            continue

        cam_params = all_cams[original_img_name]
        depth_map = np.load(os.path.join(DEPTH_MAPS_DIR, depth_file))
        color_image = cv2.imread(os.path.join(TEMPLE_DIR, original_img_name))
        
        if color_image is None or depth_map.shape[:2] != color_image.shape[:2]:
            continue

        color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        o3d_depth = o3d.geometry.Image(depth_map)
        o3d_color = o3d.geometry.Image(color_image_rgb)
        
        K_o3d = o3d.camera.PinholeCameraIntrinsic()
        K_o3d.set_intrinsics(width=color_image.shape[1], height=color_image.shape[0],
                             fx=cam_params['K'][0,0], fy=cam_params['K'][1,1],
                             cx=cam_params['K'][0,2], cy=cam_params['K'][1,2])

        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_color, o3d_depth, depth_scale=1.0, depth_trunc=2.0, convert_rgb_to_intensity=False
            ),
            K_o3d
        )
        
        R_inv = cam_params['R'].T
        t_world = -R_inv @ cam_params['t']
        T_cam_to_world = create_4x4_transform_matrix(R_inv, t_world)
        point_cloud.transform(T_cam_to_world)

        merged_point_cloud += point_cloud
        
    print("\nPoint cloud merging complete. Starting filtering...")
    print(f"Initial point cloud has {len(merged_point_cloud.points)} points.")
    
    # === 颜色过滤 ===
    # 将颜色转换为numpy数组
    colors = np.asarray(merged_point_cloud.colors)
    # 计算每个点的RGB值之和
    color_sums = np.sum(colors, axis=1)
    # 创建掩码，保留颜色和高于阈值的点
    color_mask = color_sums > COLOR_FILTER_THRESHOLD
    
    # 过滤点云
    merged_point_cloud = merged_point_cloud.select_by_index(np.where(color_mask)[0])
    print(f"After color filtering, point cloud has {len(merged_point_cloud.points)} points.")
    
    # === 点云降采样与去噪 ===
    downsampled_cloud = merged_point_cloud.voxel_down_sample(voxel_size=VOXEL_SIZE)
    
    cleaned_cloud, _ = downsampled_cloud.remove_statistical_outlier(
        nb_neighbors=NB_NEIGHBORS, std_ratio=STD_RATIO
    )
    
    print(f"After statistical filtering, final point cloud has {len(cleaned_cloud.points)} points.")
    
    # 保存点云
    o3d.io.write_point_cloud(OUTPUT_PLY_FILE, cleaned_cloud)
    print(f"Reconstructed point cloud saved to '{OUTPUT_PLY_FILE}'.")
    
    # 可选：可视化结果
    # o3d.visualization.draw_geometries([cleaned_cloud])


if __name__ == "__main__":
    reconstruct_from_depth_maps()