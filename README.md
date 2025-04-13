# PixelToVoxelProjector

将2D像素运动投影到3D体素空间的系统，主要用于天文图像处理和运动分析。

## 项目结构

```
.
├── ray_voxel.cpp        # 核心光线投射算法
├── process_image.cpp    # 图像处理模块
├── spacevoxelviewer.py  # 天文可视化工具
├── voxelmotionviewer.py # 3D体素查看器
├── setup.py            # 构建脚本
└── examplebuildvoxelgridfrommotion.bat # 示例运行脚本
```

## 核心组件详解

### C++核心模块

#### ray_voxel.cpp
- 从JSON文件加载相机参数和图像序列元数据
- 实现数字微分分析器(DDA)算法进行高效光线投射
- 检测连续帧之间的像素运动变化
- 构建3D体素网格并保存为二进制格式
- 关键功能：
  - 欧拉角到旋转矩阵的转换
  - 光线与体素网格的相交检测
  - 运动向量计算和聚合

#### process_image.cpp
- 使用pybind11提供Python接口
- 处理FITS格式的天文图像
- 实现背景减除和噪声过滤
- 支持OpenMP并行计算
- 更新体素网格和天球纹理数据

### Python可视化工具

#### spacevoxelviewer.py
- 读取FITS天文图像文件
- 计算地球位置和望远镜指向方向
- 调用C++模块处理图像序列
- 可视化分析结果：
  - 3D体素网格
  - 天球纹理映射
  - 运动轨迹分析

#### voxelmotionviewer.py
- 交互式3D可视化界面
- 支持以下功能：
  - 体素网格旋转和缩放
  - 高亮度体素筛选
  - 视角动画录制
  - 截图自动保存
- 使用PyVista进行高效3D渲染

## 技术细节

### 光线投射算法
- 采用数字微分分析器(DDA)算法
- 支持任意分辨率的体素网格
- 可配置的光线步长和采样率

### 并行处理
- 使用OpenMP实现多线程
- 图像处理并行化
- 光线投射任务分发

### 天文计算
- 精确的坐标系统转换
- 望远镜指向计算
- 时间序列分析

## 安装和使用

### 依赖项
- C++17编译器
- Python 3.8+
- pybind11
- OpenMP
- PyVista
- Astropy

### 构建步骤
1. 安装依赖：`pip install -r requirements.txt`
2. 编译C++模块：`python setup.py build_ext --inplace`
3. 运行示例：`examplebuildvoxelgridfrommotion.bat`

### 使用流程
1. 准备输入数据：
   - 图像序列(FITS格式)
   - 相机参数JSON文件
2. 运行处理：
   ```bash
   ./ray_voxel input.json output.bin
   ```
3. 可视化结果：
   ```python
   python voxelmotionviewer.py output.bin
   ```

## 应用案例
- 天文卫星轨迹分析
- 空间碎片运动追踪
- 望远镜图像中的异常检测
- 科学数据3D可视化

## 开发指南
- 代码规范：遵循Google C++ Style Guide
- 测试：使用catch2进行单元测试
- 性能分析：使用VTune进行优化
