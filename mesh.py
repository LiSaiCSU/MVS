import open3d as o3d
import numpy as np
import os

def reconstruct_mesh_with_bpa(input_file, output_file):
    """
    使用球体平铺算法（BPA）重建网格。
    
    参数:
        input_file (str): 待重建的 .ply 点云文件路径。
        output_file (str): 最终重建的网格文件保存路径。
    """
    if not os.path.exists(input_file):
        print(f"错误: 找不到点云文件 '{input_file}'。")
        return

    print(f"Loading point cloud from '{input_file}'...")
    pcd = o3d.io.read_point_cloud(input_file)
    print(f"Loaded point cloud with {len(pcd.points)} points.")
    
    # BPA 的输入需要法线
    print("Estimating normals for BPA...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

    # 根据点云密度估计球体半径列表
    # BPA 对球体半径非常敏感，这里使用一个启发式方法生成半径列表
    pcd_points = np.asarray(pcd.points)
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    
    # 尝试不同倍数的平均距离作为半径
    radii = [avg_dist, avg_dist * 7, avg_dist * 11]
    
    print(f"Using radii: {radii}")
    
    # 使用 Ball Pivoting Algorithm 进行重建
    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )

    # 可选：平滑重建后的网格
    # print("Applying mesh smoothing...")
    # bpa_mesh.filter_smooth_simple(number_of_iterations=10)
    # bpa_mesh.compute_vertex_normals()
    
    # 保存网格模型
    o3d.io.write_triangle_mesh(output_file, bpa_mesh)
    print(f"BPA mesh saved to '{output_file}'.")
    
    o3d.visualization.draw_geometries([bpa_mesh])


if __name__ == "__main__":
    # --- 请在这里设置您的输入和输出文件路径 ---
    # 使用我们之前生成的经过处理的点云作为输入
    INPUT_PLY_FILE = '稀疏点云.ply'
    OUTPUT_PLY_FILE = 'mesh.ply'
    
    reconstruct_mesh_with_bpa(INPUT_PLY_FILE, OUTPUT_PLY_FILE)