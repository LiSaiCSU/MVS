# MVS 3D 重建项目 (MVS 3D Reconstruction Project)

## 项目简介
本项目基于多视图立体匹配（Multi-View Stereo, MVS），实现了从二维图像到三维模型的完整重建流程。通过从不同视角拍摄的图像，逐步生成深度图、点云并最终重建出高精度的三维网格模型。  

项目从 **CPU 实现** 出发，逐步优化到 **多线程** 与 **GPU 加速**，最终形成了一个完整、可复现的 3D 重建 pipeline。

---

## 项目目标
- **算法理解**：深入掌握 MVS 核心原理。  
- **完整流程**：实现从图像深度估计 → 点云合成与优化 → 网格重建的全流程。  
- **性能优化**：探索并应用 PyTorch GPU 加速以提升计算效率。  

---

## 技术栈
- **Python**：主开发语言  
- **PyTorch**：GPU 加速的深度图估计  
- **Open3D**：点云处理与网格重建  
- **OpenCV**：图像读取与预处理  
- **NumPy**：基础科学计算  
- **Matplotlib**：结果可视化  
- **Tqdm**：进度条显示  

---

## 项目目录结构
```
MVS/
├── depth_maps/                  # [输出] 生成的深度图 (npy & png)
├── temple/                      # [输入] 原始图像 & 相机参数
│   ├── temple_par.txt
│   └── temple00xx.png
├── CPU单张深度图.py               # CPU 单张深度图估计
├── CPU多张深度图多线程计算.py      # CPU 多线程优化
├── GPU多张深度图.py               # [核心] GPU 加速版深度图生成
├── 单张图合成点云.py              # [旧版] 单深度图 → 点云
├── 多张图合成点云.py              # [核心] 多深度图 → 点云合成
├── 优化多张点云.py                 # [核心] 点云优化与去噪
├── mesh.py                      # [核心] 网格重建
├── requirements.txt             # 依赖库列表
└── README.md                    # 项目说明文件
```

---

## 核心工作流

### 1. 深度图估计  
- **脚本**：`GPU多张深度图.py`  
- **功能**：基于平面扫描（Plane Sweep）算法，对 `temple/` 中的所有图像生成深度图。  
- **输出**：`depth_maps/depth_temple00xx.npy` 与对应的 `.png` 可视化文件。  

---

### 2. 点云合成与优化  
- **脚本**：`多张图合成点云.py` 与 `优化多张点云.py`  
- **功能**：  
  - 多视图点云合成：将深度图投影到统一的世界坐标系并融合。  
  - 点云优化：颜色过滤、降采样、去除离群点。  
- **输出**：`reconstructed_point_cloud_filtered.ply`  

---

### 3. 网格重建  
- **脚本**：`mesh.py`  
- **功能**：基于泊松重建（Poisson Reconstruction）算法，从优化点云生成光滑、连续的 3D 网格模型。  
- **输出**：`final_reconstructed_mesh.ply`  

---

## 使用方法

### 1. 克隆项目
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```
如无 `requirements.txt`，可手动安装：
```bash
pip install numpy opencv-python tqdm torch open3d matplotlib
```

### 3. 准备数据
确保 `temple/` 文件夹中包含：  
- 多张输入图像（`temple00xx.png`）  
- 相机参数文件（`temple_par.txt`）  

### 4. 运行流程
```bash
# 步骤一：生成深度图
python GPU多张深度图.py

# 步骤二：合成并优化点云
python 多张图合成点云.py
python 优化多张点云.py

# 步骤三：网格重建
python mesh.py
```

### 5. 可视化结果
使用 **MeshLab** 或 **CloudCompare** 打开 `final_reconstructed_mesh.ply` 进行查看。  

---

## 致谢
本项目的实验数据和部分算法参考了经典 MVS 数据集与相关研究成果。  
