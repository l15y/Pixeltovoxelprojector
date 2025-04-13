"""
pyvista_interactive_view_with_rotation_history.py

需求:
    pip install pyvista numpy

使用方法:
    python pyvista_interactive_view_with_rotation_history.py

功能描述:
    1) 加载C++代码生成的voxel_grid.bin文件
    2) 将3D数组形状解释为(Z, Y, X)
    3) 提取亮度最高的百分位体素
    4) 对整个点云应用用户定义的欧拉旋转
    5) 在PyVista窗口中交互式显示，支持鼠标旋转、缩放和平移
    6) 关闭窗口时，保存1920×1080截图到'screenshots/'文件夹，
       文件名为'voxel_####.png'，用于记录运行历史
"""

import os  # 操作系统接口
import re  # 正则表达式
import math  # 数学函数
import numpy as np  # 数值计算库
import pyvista as pv  # 3D可视化库


def load_voxel_grid(filename):
    """
    从二进制文件读取体素网格数据，文件格式如下:
      1) int32: N (NxNxN网格的大小)
      2) float32: voxel_size (单个体素的大小)
      3) N*N*N float32: 按行优先顺序存储的体素数据
    返回值:
       voxel_grid (N x N x N 的3D数组),
       voxel_size (单个体素大小)
    """
    with open(filename, "rb") as f:
        # 读取N (网格尺寸)
        raw = f.read(4)  # 读取4字节
        N = np.frombuffer(raw, dtype=np.int32)[0]  # 转换为int32

        # 读取voxel_size (体素大小)
        raw = f.read(4)  # 读取4字节
        voxel_size = np.frombuffer(raw, dtype=np.float32)[0]  # 转换为float32

        # 读取体素数据
        count = N*N*N  # 计算总数据量
        raw = f.read(count*4)  # 读取所有体素数据(每个float32占4字节)
        data = np.frombuffer(raw, dtype=np.float32)  # 转换为float32数组
        voxel_grid = data.reshape((N, N, N))  # 重塑为3D数组

    return voxel_grid, voxel_size  # 返回体素网格和体素大小


def extract_top_percentile_z_up(voxel_grid, voxel_size, grid_center,
                                percentile=99.5, use_hard_thresh=False, hard_thresh=700):
    """
    提取亮度最高的'percentile'百分位体素(或高于'hard_thresh'硬阈值)
    我们将数组形状解释为(Z, Y, X)

    索引: voxel[z, y, x]

    返回:
        Nx3数组的点坐标(x, y, z)
        亮度值数组
    """
    N = voxel_grid.shape[0]  # 假设形状为(N,N,N)
    half_side = (N * voxel_size) * 0.5  # 计算网格半边长
    grid_min = grid_center - half_side  # 计算网格最小坐标

    # 展平数组以计算阈值
    flat_vals = voxel_grid.ravel()
    if use_hard_thresh:
        thresh = hard_thresh  # 使用硬阈值
    else:
        thresh = np.percentile(flat_vals, percentile)  # 计算百分位阈值

    # 获取高于阈值的体素坐标
    coords = np.argwhere(voxel_grid > thresh)
    if coords.size == 0:
        print(f"没有体素高于阈值 {thresh}。无内容可显示。")
        return None, None

    # 提取对应坐标的亮度值
    intensities = voxel_grid[coords[:, 0], coords[:, 1], coords[:, 2]]

    # 将数组索引转换为坐标(0 -> z, 1 -> y, 2 -> x)
    z_idx = coords[:, 0] + 0.5  # 添加0.5以获取体素中心坐标
    y_idx = coords[:, 1] + 0.5
    x_idx = coords[:, 2] + 0.5

    # 转换为世界坐标
    x_world = grid_min[0] + x_idx * voxel_size  # x坐标 = 网格最小x + 索引 * 体素大小
    y_world = grid_min[1] + y_idx * voxel_size
    z_world = grid_min[2] + z_idx * voxel_size

    points = np.column_stack((x_world, y_world, z_world))  # 组合成Nx3坐标数组
    return points, intensities  # 返回坐标点和亮度值


def rotation_matrix_xyz(rx_deg, ry_deg, rz_deg):
    """
    构建XYZ欧拉角的3x3旋转矩阵(角度制)
    旋转顺序为X->Y->Z，即:
      R = Rz(rz) * Ry(ry) * Rx(rx)
    先绕X轴旋转rx度，再绕Y轴旋转ry度，最后绕Z轴旋转rz度
    """
    rx = math.radians(rx_deg)  # X轴旋转角度(弧度)
    ry = math.radians(ry_deg)  # Y轴旋转角度(弧度)
    rz = math.radians(rz_deg)  # Z轴旋转角度(弧度)

    cx, sx = math.cos(rx), math.sin(rx)  # X轴旋转的cos/sin值
    cy, sy = math.cos(ry), math.sin(ry)  # Y轴旋转的cos/sin值
    cz, sz = math.cos(rz), math.sin(rz)  # Z轴旋转的cos/sin值

    # X轴旋转矩阵
    Rx = np.array([
        [1,   0,   0],  # X轴不变
        [0,  cx, -sx],  # Y-Z平面旋转
        [0,  sx,  cx]
    ], dtype=np.float32)

    # Y轴旋转矩阵
    Ry = np.array([
        [ cy,  0,  sy],  # X-Z平面旋转
        [  0,  1,   0],  # Y轴不变
        [-sy,  0,  cy]
    ], dtype=np.float32)

    # Z轴旋转矩阵
    Rz = np.array([
        [ cz, -sz,  0],  # X-Y平面旋转
        [ sz,  cz,  0],
        [  0,   0,  1]   # Z轴不变
    ], dtype=np.float32)

    # 组合旋转矩阵: Rz * Ry * Rx
    Rtemp = Rz @ Ry  # 先计算Z和Y的旋转
    Rfinal = Rtemp @ Rx  # 再乘以X旋转
    return Rfinal  # 返回最终旋转矩阵


def get_next_image_index(folder, prefix="voxel_", suffix=".png"):
    """
    扫描文件夹中类似'voxel_XXXX.png'的文件
    找到最大的XXXX数字，返回该数字+1
    如果没有找到匹配文件，返回1
    """
    if not os.path.exists(folder):  # 如果文件夹不存在
        return 1  # 从1开始编号

    pattern = re.compile(rf"^{prefix}(\d+){suffix}$")  # 正则匹配模式
    max_index = 0  # 初始化最大索引
    for fname in os.listdir(folder):  # 遍历文件夹中的文件
        match = pattern.match(fname)  # 尝试匹配文件名
        if match:  # 如果匹配成功
            idx = int(match.group(1))  # 提取数字部分并转为整数
            if idx > max_index:  # 更新最大索引
                max_index = idx
    return max_index + 1  # 返回最大索引+1


def main():
    # 1) 加载体素网格数据
    voxel_grid, vox_size = load_voxel_grid("voxel_grid.bin")
    print("已加载体素网格:", voxel_grid.shape, "体素大小=", vox_size)
    print("最大体素值:", voxel_grid.max())

    # 2) 定义网格中心坐标(x,y,z)
    grid_center = np.array([30, 0, 14000], dtype=np.float32)

    # 3) 提取高亮度体素点(Z轴向上)
    percentile_to_show = 99.9  # 显示前99.9%亮度的体素
    points, intensities = extract_top_percentile_z_up(
        voxel_grid,
        voxel_size=vox_size,
        grid_center=grid_center,
        percentile=percentile_to_show,
        use_hard_thresh=False,
        hard_thresh=700
    )
    if points is None:  # 如果没有提取到体素点
        return  # 直接返回

    # 4) 可选旋转(用于调整方向)
    # 例如: 旋转以修正方向
    rx_deg = 90   # X轴旋转90度
    ry_deg = 270  # Y轴旋转270度
    rz_deg = 0    # Z轴不旋转

    R = rotation_matrix_xyz(rx_deg, ry_deg, rz_deg)  # 生成3x3旋转矩阵
    points_rot = points @ R.T  # 应用旋转矩阵
    # 5) 创建PyVista绘图器(交互式)
    plotter = pv.Plotter(off_screen=True,)  # 离屏渲染模式
    plotter.set_background("white")  # 设置白色背景
    plotter.enable_terrain_style()  # 启用地形样式交互

    # 将点云转换为带标量的PolyData
    cloud = pv.PolyData(points_rot)  # 创建点云对象
    cloud["intensity"] = intensities  # 添加亮度标量

    # 添加点云到绘图器
    plotter.add_points(
        cloud,
        scalars="intensity",  # 使用亮度值着色
        cmap="hot",          # 使用热图颜色映射
        point_size=4.0,      # 点大小
        render_points_as_spheres=True,  # 渲染为球体
        opacity=0.1,         # 透明度
    )

    # 添加颜色条
    plotter.add_scalar_bar(
        title="亮度",        # 颜色条标题
        n_labels=5          # 标签数量
    )

    # 6) 确定下一个截图索引
    screenshot_folder = "screenshots"  # 截图保存目录
    if not os.path.exists(screenshot_folder):
        os.makedirs(screenshot_folder)  # 创建目录
    next_idx = get_next_image_index(screenshot_folder, prefix="voxel_", suffix=".png")
    out_name = f"voxel_{next_idx:04d}.png"  # 生成文件名(4位数字)
    out_path = os.path.join(screenshot_folder, out_name)  # 完整路径

    # 7) 显示交互窗口并保存截图
    #    截图在关闭绘图窗口时生成
    plotter.show(window_size=[3840, 2160], auto_close=False, screenshot=out_path)

    print(f"[信息] 截图已保存到 {out_path}")

    # 创建新的交互式绘图器(非离屏模式)
    plotter = pv.Plotter(off_screen=False, )
    plotter.set_background("white")
    plotter.enable_terrain_style()

    # 再次创建点云对象
    cloud = pv.PolyData(points_rot)
    cloud["intensity"] = intensities

    # 添加点云(使用不同透明度)
    plotter.add_points(
        cloud,
        scalars="intensity",
        cmap="hot",
        point_size=4.0,
        render_points_as_spheres=True,
        opacity=0.05,  # 更低的透明度
    )

    print("[完成]")
    plotter.show(window_size=[1920, 1080], auto_close=False)  # 显示标准分辨率窗口



# Python主程序入口
if __name__ == "__main__":
    main()  # 调用主函数
