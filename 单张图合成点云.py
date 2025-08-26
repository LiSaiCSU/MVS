import os

import cv2
import numpy as np


def create_point_cloud(color_image_path, depth_map_path, output_ply_path, K):
    """
    根据彩色图、深度图和相机内参生成点云文件 (.ply)。

    参数:
    color_image_path (str): 彩色图像的文件路径。
    depth_map_path (str): .npy 格式深度图的文件路径。
    output_ply_path (str): 输出的 .ply 点云文件的保存路径。
    K (np.ndarray): 3x3 的相机内参矩阵。
    """
    # --- 1. 加载数据 ---
    print("正在加载图像和深度图...")

    # 使用 OpenCV 加载彩色图像
    color_image = cv2.imread(color_image_path)
    if color_image is None:
        print(f"错误：无法加载彩色图像于 {color_image_path}")
        return

    # OpenCV 默认以 BGR 格式加载，需要转换为 RGB
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # 加载 .npy 格式的深度图
    depth_map = np.load(depth_map_path)
    if depth_map is None:
        print(f"错误：无法加载深度图于 {depth_map_path}")
        return

    # --- 2. 验证数据尺寸 ---
    h, w, _ = color_image.shape
    if color_image.shape[:2] != depth_map.shape:
        print(f"错误：彩色图 ({h}x{w}) 和深度图 ({depth_map.shape}) 的尺寸不匹配。")
        return

    print(f"图像尺寸: {w}x{h}")

    # --- 3. 从内参矩阵中提取参数 ---
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # --- 4. 逐像素计算三维坐标 ---
    print("正在生成点云...")
    points = []
    colors = []

    for v in range(h):  # 对应图像的行 (y)
        for u in range(w):  # 对应图像的列 (x)

            # 获取深度值
            depth = depth_map[v, u]

            # 关键步骤：过滤掉无效的深度点
            # 深度值为 0 或负数通常表示无效测量
            if depth <= 0:
                continue

            # 从像素坐标 (u, v) 和深度 z 计算相机坐标系下的 (x, y, z)
            # 这是标准的反投影公式
            z_cam = depth
            x_cam = (u - cx) * z_cam / fx
            y_cam = (v - cy) * z_cam / fy

            points.append([x_cam, y_cam, z_cam])

            # 获取对应的颜色
            colors.append(color_image[v, u])

    if not points:
        print("没有有效的深度点来创建点云。")
        return

    # --- 5. 将点云数据写入 .ply 文件 ---
    print(f"正在将 {len(points)} 个点写入到 {output_ply_path}...")

    # 将列表转换为 NumPy 数组以提高效率
    points = np.array(points, dtype=np.float32)
    colors = np.array(colors, dtype=np.uint8)

    # 创建PLY文件头
    ply_header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""

    # 将坐标和颜色数据合并
    # np.hstack 在水平方向上拼接数组
    vertex_data = np.hstack([points, colors])

    # 写入文件
    with open(output_ply_path, "w") as f:
        f.write(ply_header)
        # 使用 numpy.savetxt 高效写入
        np.savetxt(f, vertex_data, fmt="%f %f %f %d %d %d")

    print("点云文件已成功保存！")


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


if __name__ == "__main__":
    # ======================================================================
    # --- 请在这里修改您的文件路径和相机参数 ---

    # 1. 设置文件路径
    # 您的原始彩色图像
    COLOR_IMAGE_PATH = "temple/temple0010.png"
    # 您生成的 .npy 深度图文件
    DEPTH_MAP_PATH = "depth_map_temple0010.png.npy"
    # 您希望保存的点云文件名
    OUTPUT_PLY_PATH = "point_cloud_temple0003.ply"

    # 2. 设置相机内参矩阵 K
    # !!! 警告: 这里的数值是示例，您必须替换成 temple0003.png 对应的真实内参 !!!
    # 您需要从 temple_par.txt 文件中读取 'temple0003.png' 对应的 K 矩阵
    # 格式为 3x3 的 NumPy 数组

    K_matrix = read_temple_par("temple/temple_par.txt")["temple0010.png"]["K"]
    # ======================================================================

    # 检查输入文件是否存在
    if not os.path.exists(COLOR_IMAGE_PATH):
        print(f"错误: 找不到彩色图像 '{COLOR_IMAGE_PATH}'")
    elif not os.path.exists(DEPTH_MAP_PATH):
        print(f"错误: 找不到深度图 '{DEPTH_MAP_PATH}'")
    else:
        # 调用主函数
        create_point_cloud(COLOR_IMAGE_PATH, DEPTH_MAP_PATH, OUTPUT_PLY_PATH, K_matrix)
