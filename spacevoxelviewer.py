import numpy as np  # 数值计算库，用于处理数组和矩阵运算
import matplotlib.pyplot as plt  # 绘图库，用于数据可视化
from astropy.io import fits  # 用于读取FITS文件(天文数据标准格式)
from astropy.time import Time  # 用于处理天文观测时间
from astropy.coordinates import get_body_barycentric, SkyCoord, solar_system_ephemeris  # 天文坐标计算
import astropy.units as u  # 天文单位处理
import os  # 操作系统接口，用于文件路径操作

# 导入编译好的C++模块，用于高性能图像处理
import process_image_cpp

# -------------------------------------------------------------------------------------
# 可配置参数 - 这些参数可以根据具体观测需求进行调整
# -------------------------------------------------------------------------------------

# 体素网格是一个3D数组，用于累积从空间中投射的光线的亮度值
# 根据您的场景调整voxel_grid_size和grid_extent参数
voxel_grid_size = (400, 400, 400)  # (x, y, z)方向上的体素数量，决定了空间分辨率
grid_extent = 3e12  # 体素立方体边长的一半(以米为单位)，决定了空间覆盖范围

# 体素网格的位置和方向基于赤经(RA)和赤纬(Dec)确定
# 我们选择一个RA/Dec坐标，并假设体素网格沿该视线方向居中
# 距离太阳的距离为distance_from_sun
distance_from_sun = 1.496e+11 * 41.714231  # 约1天文单位(AU)，以米为单位
center_ra = 280.50  # 中心赤经(度)，目标区域的中心坐标
center_dec = -20.  # 中心赤纬(度)，目标区域的中心坐标

# 备选参数配置示例(注释掉)
# distance_from_sun = 1.496e+11 * 34  # 约1天文单位(AU)，以米为单位
# center_ra = 287.967022  # 中心赤经(度)
# center_dec = -20.713745  # 中心赤纬(度)

# 用于识别"显著"体素的阈值(如前90%亮度)
brightness_threshold_percentile = .1  # 只显示亮度高于90%分位数的体素

# 光线投射参数：投射到空间中的距离和精细程度
max_distance = distance_from_sun * 100  # 光线投射的最大距离(以米为单位)
num_steps = 20000     # 每条光线的步数，影响光线追踪的精度

# 绘图的可视化参数
marker_size = 5  # 散点图的标记大小，控制3D可视化中点的大小
alpha = 0.5      # 散点的透明度，0为完全透明，1为完全不透明

# 如果在FITS头中找不到时的默认视场(FOV)(以角分为单位)
default_fov_arcminutes = 2.7  # 默认视场大小，约2.7角分

# 包含FITS文件的目录
fits_directory = 'fits'  # 存放观测数据FITS文件的目录路径

# 定义用于构建天球纹理的天空区域参数:
# 我们将把图像方向投影到这个天空区域上
angular_width = 5.0    # 天空区域的角宽度(度)，决定纹理覆盖的天空范围
angular_height = 5.0   # 天空区域的角高度(度)
texture_width = 1024   # 纹理宽度(像素)，决定纹理的分辨率
texture_height = 1024  # 纹理高度(像素)

# -------------------------------------------------------------------------------------
# 辅助函数
# -------------------------------------------------------------------------------------

def get_earth_position_icrs(obs_time):
    """
    计算给定观测时间地球在ICRS坐标系中的日心位置
    ICRS(国际天球参考系)是天文学中使用的标准坐标系

    参数:
    - obs_time: Astropy Time对象，表示观测时间

    返回:
    - earth_pos: Numpy数组[x, y, z]，以米为单位表示地球在ICRS坐标系中的位置
                 数组顺序对应ICRS坐标系的X,Y,Z轴
    """
    with solar_system_ephemeris.set('builtin'):
        earth_barycentric = get_body_barycentric('earth', obs_time)
        earth_icrs = earth_barycentric

    earth_pos = earth_icrs.get_xyz().to(u.meter).value
    earth_pos = np.array(earth_pos).flatten()
    return earth_pos

def get_telescope_pointing(header):
    """
    从FITS头中的RA_TARG和DEC_TARG获取望远镜的指向方向
    FITS(灵活图像传输系统)是天文学中常用的数据格式

    参数:
    - header: 包含RA_TARG和DEC_TARG的FITS头
              RA_TARG: 目标赤经(度)
              DEC_TARG: 目标赤纬(度)

    返回:
    - direction: ICRS坐标系中的单位向量[x, y, z]
                  向量已归一化，长度为1
    """
    ra = header.get('RA_TARG')
    dec = header.get('DEC_TARG')
    if ra is None or dec is None:
        raise ValueError("RA_TARG and DEC_TARG not found in FITS header.")

    coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    cartesian = coord.represent_as('cartesian')
    direction = np.array([cartesian.x.value, cartesian.y.value, cartesian.z.value])
    direction /= np.linalg.norm(direction)
    return direction

def get_observation_time(fits_file):
    """
    从FITS文件中提取观测时间(DATE-OBS, TIME-OBS)并返回为Astropy Time对象
    观测时间对于确定地球位置和天体位置至关重要

    参数:
    - fits_file: FITS文件路径，包含观测数据

    返回:
    - obs_time: 表示观测时间的Astropy Time对象
                使用国际原子时(TAI)时间尺度
                格式为ISO格式(YYYY-MM-DDThh:mm:ss.sss)
    """
    with fits.open(fits_file) as hdulist:
        header = hdulist[0].header
        date_obs = header.get('DATE-OBS')
        time_obs = header.get('TIME-OBS')

        if date_obs is None or time_obs is None:
            raise ValueError(f"DATE-OBS and TIME-OBS not found in FITS header of {fits_file}.")

        obs_time_str = f"{date_obs}T{time_obs}"
        obs_time = Time(obs_time_str, format='isot', scale='utc')
    return obs_time

def process_image(fits_file, voxel_grid, voxel_grid_extent, celestial_sphere_texture):
    """
    处理单个FITS图像的核心函数:
    1. 计算地球位置和望远镜指向
    2. 读取并归一化图像数据
    3. 调用C++函数投射光线，更新体素网格和天球纹理

    参数:
    - fits_file: FITS文件路径，包含观测数据
    - voxel_grid: 用于体素累积的3D numpy数组，形状为voxel_grid_size
    - voxel_grid_extent: 体素网格的空间范围，格式为:
                         ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    - celestial_sphere_texture: 用于天球的2D numpy数组，形状为(texture_height, texture_width)

    返回:
    - earth_position: 观测时间的地球位置，numpy数组[x,y,z] (米)
    - pointing_direction: 望远镜指向方向，单位向量[x,y,z]
    - obs_time: 表示观测时间的Astropy Time对象

    注意:
    - 该函数会修改传入的voxel_grid和celestial_sphere_texture数组
    - 使用C++扩展模块进行高性能光线投射计算
    """
    with fits.open(fits_file) as hdulist:
        print(f"正在处理FITS文件: {fits_file}")
        hdulist.info()

        header = hdulist[0].header
        date_obs = header.get('DATE-OBS')
        time_obs = header.get('TIME-OBS')

        if date_obs is None or time_obs is None:
            raise ValueError("DATE-OBS and TIME-OBS not found in FITS header.")

        obs_time_str = f"{date_obs}T{time_obs}"
        obs_time = Time(obs_time_str, format='isot', scale='utc')

        # 计算地球位置和望远镜指向
        earth_position = get_earth_position_icrs(obs_time)
        pointing_direction = get_telescope_pointing(header)

        # 查找图像数据
        image_data = None
        if hdulist[0].data is not None:
            image_data = hdulist[0].data
            print("在主HDU中找到图像数据。")
        elif 'SCI' in hdulist:
            image_data = hdulist['SCI'].data
            print("在'SCI'扩展中找到图像数据。")
        else:
            for hdu in hdulist:
                if isinstance(hdu, (fits.ImageHDU, fits.CompImageHDU)):
                    if hdu.data is not None:
                        image_data = hdu.data
                        print(f"在扩展'{hdu.name}'中找到图像数据。")
                        break

        if image_data is None:
            raise ValueError("在FITS文件中未找到图像数据。")

        if image_data.ndim != 2:
            raise ValueError("图像数据不是2D的。")

        image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)
        image_min = np.min(image_data)
        image_max = np.max(image_data)
        if image_max - image_min == 0:
            raise ValueError("图像数据动态范围为零。")
        image = (image_data - image_min) / (image_max - image_min)

        # FITS图像的可选可视化
        # plt.figure(figsize=(8, 6))
        # plt.imshow(image, cmap='gray', origin='lower')
        # plt.title(f"FITS图像: {os.path.basename(fits_file)}")
        # plt.xlabel('像素X')
        # plt.ylabel('像素Y')
        # plt.colorbar(label='归一化强度')
        # plt.show()

        height, width = image.shape

        # 确定视场
        fov = header.get('FOV')
        if fov is None:
            cd1_1 = header.get('CD1_1')
            cd1_2 = header.get('CD1_2')
            cd2_1 = header.get('CD2_1')
            cd2_2 = header.get('CD2_2')
            if cd1_1 is not None and cd1_2 is not None and cd2_1 is not None and cd2_2 is not None:
                pixel_scale_x = np.sqrt(cd1_1**2 + cd2_1**2)
                pixel_scale_y = np.sqrt(cd1_2**2 + cd2_2**2)
                fov_x = pixel_scale_x * width
                fov_y = pixel_scale_y * height
                fov = max(fov_x, fov_y)
            else:
                fov = default_fov_arcminutes / 60  # 度
        else:
            fov = float(fov)

        fov_rad = np.deg2rad(fov)

        # 将Python数据转换为C++可用的列表
        earth_position_list = earth_position.tolist()
        pointing_direction_list = pointing_direction.tolist()
        voxel_grid_extent_list = [
            (voxel_grid_extent[0][0], voxel_grid_extent[0][1]),
            (voxel_grid_extent[1][0], voxel_grid_extent[1][1]),
            (voxel_grid_extent[2][0], voxel_grid_extent[2][1])
        ]

        # 以弧度定义天空区域
        c_ra_rad = np.deg2rad(center_ra)
        c_dec_rad = np.deg2rad(center_dec)
        aw_rad = np.deg2rad(angular_width)
        ah_rad = np.deg2rad(angular_height)

        # 调用C++函数处理图像
        process_image_cpp.process_image_cpp(
            image.astype(np.float64),
            earth_position_list,
            pointing_direction_list,
            fov_rad,
            width,
            height,
            voxel_grid,
            voxel_grid_extent_list,
            max_distance,
            num_steps,
            celestial_sphere_texture,
            c_ra_rad,
            c_dec_rad,
            aw_rad,
            ah_rad,
            True,   # update_celestial_sphere: True表示累积天球亮度
            False   # perform_background_subtraction: False表示不进行背景减除
        )

    return earth_position, pointing_direction, obs_time

def main():
    """
    主函数，执行整个体素投影流程:
    1. 设置体素网格和天球纹理
    2. 查找并处理FITS文件(单次遍历)
    3. 分析体素网格，找到最亮点，可视化结果

    工作流程:
    - 初始化体素网格和天球纹理
    - 扫描fits_directory目录下的所有FITS文件
    - 按观测时间排序文件
    - 逐个处理每个FITS文件，累积体素网格
    - 计算平均亮度并识别显著体素
    - 生成3D和2D可视化结果

    注意:
    - 需要预先创建fits_directory目录并放入FITS文件
    - 依赖C++扩展模块process_image_cpp
    - 使用matplotlib进行结果可视化
    """

    # 根据赤经赤纬计算体素网格中心
    center_coord = SkyCoord(ra=center_ra*u.degree, dec=center_dec*u.degree, frame='icrs')
    direction_vector = center_coord.cartesian.xyz.value
    voxel_grid_center = direction_vector * distance_from_sun

    voxel_grid_extent = (
        (voxel_grid_center[0] - grid_extent, voxel_grid_center[0] + grid_extent),
        (voxel_grid_center[1] - grid_extent, voxel_grid_center[1] + grid_extent),
        (voxel_grid_center[2] - grid_extent, voxel_grid_center[2] + grid_extent)
    )

    # 初始化体素网格和天球纹理
    voxel_grid = np.zeros(voxel_grid_size, dtype=np.float64)
    celestial_sphere_texture = np.zeros((texture_height, texture_width), dtype=np.float64)

    # 列出FITS文件
    fits_files = [os.path.join(fits_directory, f) for f in os.listdir(fits_directory) if f.endswith('.fits')]

    if not fits_files:
        print(f"在目录'{fits_directory}'中未找到FITS文件。")
        return

    # 按观测时间排序FITS文件
    fits_files_with_times = []
    for fits_file in fits_files:
        try:
            obs_time = get_observation_time(fits_file)
            fits_files_with_times.append((fits_file, obs_time))
            print(obs_time)
        except ValueError as e:
            print(e)


    fits_files_sorted = sorted(fits_files_with_times, key=lambda x: x[1])
    fits_files_sorted = [(f[0], f[1]) for f in fits_files_sorted]

    earth_positions = []
    pointing_directions = []
    observation_times = []

    # 处理每个FITS文件(单次遍历)
    for fits_file, obs_time in fits_files_sorted:
        earth_pos, p_dir, obs_time = process_image(
            fits_file,
            voxel_grid,
            voxel_grid_extent,
            celestial_sphere_texture
        )
        earth_positions.append(earth_pos)
        pointing_directions.append(p_dir)
        observation_times.append(obs_time)

    # 创建背景模型(可选步骤)
    # 在这个例子中，我们只展示单次遍历后的结果
    background_model = celestial_sphere_texture / len(fits_files_sorted)

    # 分析体素网格
    voxel_grid_avg = voxel_grid / len(fits_files_sorted)

    # 显著体素的阈值计算
    if np.any(voxel_grid_avg > 0):
        threshold = np.percentile(voxel_grid_avg[voxel_grid_avg > 0], brightness_threshold_percentile)
    else:
        threshold = 0

    object_voxels = voxel_grid_avg > threshold
    x_indices, y_indices, z_indices = np.nonzero(object_voxels)

    nx, ny, nz = voxel_grid_avg.shape
    x_min, x_max = voxel_grid_extent[0]
    y_min, y_max = voxel_grid_extent[1]
    z_min, z_max = voxel_grid_extent[2]

    x_coords = x_indices / nx * (x_max - x_min) + x_min
    y_coords = y_indices / ny * (y_max - y_min) + y_min
    z_coords = z_indices / nz * (z_max - z_min) + z_min

    intensities = voxel_grid_avg[object_voxels]

    # 查找最亮点
    if intensities.size > 0:
        brightest_idx = np.argmax(intensities)
        brightest_x = x_coords[brightest_idx]
        brightest_y = y_coords[brightest_idx]
        brightest_z = z_coords[brightest_idx]

        brightest_coord = SkyCoord(
            x=brightest_x * u.meter,
            y=brightest_y * u.meter,
            z=brightest_z * u.meter,
            representation_type='cartesian',
            frame='icrs'
        )

        brightest_sph = brightest_coord.represent_as('spherical')
        brightest_ra = brightest_sph.lon.degree
        brightest_dec = brightest_sph.lat.degree
        distance_to_origin = np.sqrt(brightest_x**2 + brightest_y**2 + brightest_z**2)
        distance_to_origin_au = distance_to_origin / 1.496e+11

        print(f"最亮点坐标:")
        print(f"赤经: {brightest_ra:.6f} 度")
        print(f"赤纬: {brightest_dec:.6f} 度")
        print(f"距原点距离: {distance_to_origin_au:.6f} 天文单位")
    else:
        print("未找到显著体素来识别最亮点。")

    # 体素网格的3D可视化
    if len(x_coords) > 0:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        sc = ax.scatter(x_coords, y_coords, z_coords, c=intensities, cmap='hot', marker='o', s=marker_size, alpha=alpha)
        if intensities.size > 0:
            ax.scatter([brightest_x], [brightest_y], [brightest_z], c='blue', marker='*', s=100, label='最亮点')

        # 绘制地球位置
        earth_x = [pos[0] for pos in earth_positions]
        earth_y = [pos[1] for pos in earth_positions]
        earth_z = [pos[2] for pos in earth_positions]
        ax.scatter(earth_x, earth_y, earth_z, c='green', marker='o', s=200, label='地球位置')

        # 绘制体素网格中心
        ax.scatter([voxel_grid_center[0]], [voxel_grid_center[1]], [voxel_grid_center[2]],
                   c='purple', marker='x', s=100, label='体素网格中心')

        # 绘制随时间变化的相机指向方向箭头
        times = np.array([t.mjd for t in observation_times])
        if len(times) > 1 and times.max() != times.min():
            times_norm = (times - times.min()) / (times.max() - times.min())
        else:
            times_norm = np.zeros_like(times)

        cmap = plt.cm.get_cmap('viridis')

        # 从地球位置绘制指向相机方向的箭头
        arrow_length = grid_extent * 0.5
        for idx, (pos, dir_vec, time_norm) in enumerate(zip(earth_positions, pointing_directions, times_norm)):
            x0, y0, z0 = pos
            dx = dir_vec[0] * arrow_length
            dy = dir_vec[1] * arrow_length
            dz = dir_vec[2] * arrow_length
            color = cmap(time_norm)
            ax.quiver(x0, y0, z0, dx, dy, dz, color=color, length=1.0, normalize=False, arrow_length_ratio=0.1)

        # 添加观测时间的颜色条
        mappable = plt.cm.ScalarMappable(cmap=cmap)
        mappable.set_array(times)
        plt.colorbar(mappable, ax=ax, label='观测时间(MJD)')

        ax.legend()
        plt.colorbar(sc, ax=ax, label='平均亮度')
        ax.set_title('检测到的物体、地球位置和相机方向的3D可视化')
        ax.set_xlabel('X (米)')
        ax.set_ylabel('Y (米)')
        ax.set_zlabel('Z (米)')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        plt.show()
    else:
        print("未找到显著体素进行可视化。")

    # 可视化体素网格的2D切片
    voxel_grid_avg = voxel_grid / len(fits_files_sorted)
    z_slice_index = voxel_grid_avg.shape[2] // 2
    voxel_slice = voxel_grid_avg[:, :, z_slice_index]
    plt.figure(figsize=(8, 6))
    plt.imshow(voxel_slice.T, origin='lower', cmap='hot', extent=(x_min, x_max, y_min, y_max))
    plt.colorbar(label='平均亮度')
    z_slice_pos = z_min + z_slice_index * (z_max - z_min) / voxel_grid_avg.shape[2]
    plt.title(f'体素网格切片 z = {z_slice_pos:.2f} 米处')
    plt.xlabel('X (米)')
    plt.ylabel('Y (米)')
    plt.show()

if __name__ == '__main__':
    main()
