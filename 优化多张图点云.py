import copy
import os

import open3d as o3d


def process_point_cloud(input_file, output_file):
    """
    加载点云文件，并对其进行去噪、平滑和优化，然后保存。

    参数:
        input_file (str): 待处理的 .ply 点云文件路径。
        output_file (str): 优化后的点云保存路径。
    """
    # === 用户可调参数 ===
    # --- 点云优化参数 ---
    # 统计离群点移除：用于去除孤立的噪点
    NB_NEIGHBORS_FILTER = 30  # 邻居点数，用于计算平均距离
    STD_RATIO_FILTER = 2.0  # 标准差倍数，超出此范围的点被移除

    # 体素降采样：用于均匀化点云，使其更平整
    VOXEL_SIZE = 0.002  # 降采样体素大小，值越小，点云越密集

    # --- 可视化选项 ---
    VISUALIZE_BEFORE_AFTER = True  # 是否可视化处理前后的对比
    # ====================

    if not os.path.exists(input_file):
        print(f"错误: 找不到点云文件 '{input_file}'。")
        return

    print(f"Loading point cloud from '{input_file}'...")
    pcd = o3d.io.read_point_cloud(input_file)
    print(f"Loaded point cloud with {len(pcd.points)} points.")

    # 可视化处理前的点云
    if VISUALIZE_BEFORE_AFTER:
        pcd_original = copy.deepcopy(pcd)
        pcd_original.paint_uniform_color([0.5, 0.5, 0.5])  # 原始点云涂成灰色方便对比

    print("\nStarting point cloud filtering and smoothing...")

    # --- 1. 统计离群点移除（第一遍）---
    # 去除初始的、明显的离群噪点
    pcd_cleaned, ind = pcd.remove_statistical_outlier(
        nb_neighbors=NB_NEIGHBORS_FILTER, std_ratio=STD_RATIO_FILTER
    )
    print(
        f"Removed {len(pcd.points) - len(pcd_cleaned.points)} statistical outliers (pass 1)."
    )

    # --- 2. 体素降采样 ---
    # 减少点云密度，均匀化点云，使其更平整
    pcd_downsampled = pcd_cleaned.voxel_down_sample(voxel_size=VOXEL_SIZE)
    print(f"Downsampled to {len(pcd_downsampled.points)} points.")

    # --- 3. 统计离群点移除（第二遍）---
    # 再次移除降采样后可能仍然存在的离群点
    pcd_final, ind = pcd_downsampled.remove_statistical_outlier(
        nb_neighbors=NB_NEIGHBORS_FILTER, std_ratio=STD_RATIO_FILTER
    )
    print(
        f"Removed {len(pcd_downsampled.points) - len(pcd_final.points)} statistical outliers (pass 2)."
    )

    print(f"\nFinal processed point cloud has {len(pcd_final.points)} points.")

    # --- 保存最终点云 ---
    o3d.io.write_point_cloud(output_file, pcd_final)
    print(f"Processed point cloud saved to '{output_file}'.")

    # --- 可视化处理后的点云 ---
    if VISUALIZE_BEFORE_AFTER:
        pcd_final.paint_uniform_color([0.2, 0.8, 0.2])  # 处理后的点云涂成绿色
        print("Displaying processed vs. original point clouds...")
        o3d.visualization.draw_geometries([pcd_original, pcd_final])
    else:
        o3d.visualization.draw_geometries([pcd_final])


if __name__ == "__main__":
    # --- 请在这里设置您的输入和输出文件路径 ---
    INPUT_PLY_FILE = "reconstructed_point_cloud_filtered2.ply"
    OUTPUT_PLY_FILE = "processed_point_cloud2.ply"

    process_point_cloud(INPUT_PLY_FILE, OUTPUT_PLY_FILE)
