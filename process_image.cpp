// process_image.cpp - 光线投射和体素网格处理核心算法(C++实现)
// 该文件实现了从天文图像到3D体素网格的光线投射算法

#include <pybind11/pybind11.h>  // Python-C++绑定库
#include <pybind11/numpy.h>    // NumPy数组支持
#include <pybind11/stl.h>      // STL容器支持
#include <cmath>               // 数学函数
#include <vector>              // 向量容器
#include <array>               // 固定大小数组
#include <algorithm>           // 算法函数
#include <limits>              // 数值极限

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace py = pybind11;

/**
 * 计算光线与轴对齐包围盒(AABB)的交点
 *
 * 该函数确定光线与由box_min和box_max定义的AABB的进入和退出点
 * 如果光线不与盒子相交，则返回false
 *
 * 算法原理:
 * 1. 对每个坐标轴(x,y,z)分别计算光线与两个平行平面的交点参数t1和t2
 * 2. 取所有轴的交点参数的并集作为最终的交点区间
 * 3. 如果区间有效则返回交点参数，否则返回不相交
 */
bool ray_aabb_intersection(
    const std::array<double, 3>& ray_origin,
    const std::array<double, 3>& ray_direction,
    const std::array<double, 3>& box_min,
    const std::array<double, 3>& box_max,
    double& t_entry,
    double& t_exit)
{
    double tmin = -std::numeric_limits<double>::infinity();
    double tmax = std::numeric_limits<double>::infinity();

    for (int i = 0; i < 3; ++i)
    {
        if (std::abs(ray_direction[i]) > 1e-8)
        {
            double t1 = (box_min[i] - ray_origin[i]) / ray_direction[i];
            double t2 = (box_max[i] - ray_origin[i]) / ray_direction[i];

            if (t1 > t2) std::swap(t1, t2);

            tmin = std::max(tmin, t1);
            tmax = std::min(tmax, t2);

            if (tmin > tmax)
                return false;
        }
        else
        {
            // Ray is parallel to the slab. If the origin is not within the slab, no intersection
            if (ray_origin[i] < box_min[i] || ray_origin[i] > box_max[i])
                return false;
        }
    }

    t_entry = tmin;
    t_exit = tmax;
    return true;
}

/**
 * 将3D点转换为体素网格内的索引
 *
 * 给定空间中的点和体素网格范围，计算它落在哪个体素中
 *
 * 计算步骤:
 * 1. 检查点是否在体素网格范围内
 * 2. 计算点在网格内的归一化坐标(0到1之间)
 * 3. 将归一化坐标转换为体素索引
 * 4. 对索引进行边界裁剪确保有效性
 *
 * 返回:
 * - 如果点在网格内: 返回(x,y,z)体素索引
 * - 如果点在网格外: 返回(-1,-1,-1)
 */
std::tuple<pybind11::ssize_t, pybind11::ssize_t, pybind11::ssize_t> point_to_voxel_indices(
    const std::array<double, 3>& point,
    const std::vector<std::pair<double, double>>& voxel_grid_extent,
    const std::array<pybind11::ssize_t, 3>& voxel_grid_size)
{
    double x_min = voxel_grid_extent[0].first;
    double x_max = voxel_grid_extent[0].second;
    double y_min = voxel_grid_extent[1].first;
    double y_max = voxel_grid_extent[1].second;
    double z_min = voxel_grid_extent[2].first;
    double z_max = voxel_grid_extent[2].second;

    double x = point[0];
    double y = point[1];
    double z = point[2];

    // Check if the point is inside the voxel grid bounds
    if (x_min <= x && x <= x_max && y_min <= y && y <= y_max && z_min <= z && z <= z_max)
    {
        pybind11::ssize_t nx = voxel_grid_size[0];
        pybind11::ssize_t ny = voxel_grid_size[1];
        pybind11::ssize_t nz = voxel_grid_size[2];

        // Compute normalized position within the grid
        double x_norm = (x - x_min) / (x_max - x_min);
        double y_norm = (y - y_min) / (y_max - y_min);
        double z_norm = (z - z_min) / (z_max - z_min);

        // Convert normalized coordinates to voxel indices
        pybind11::ssize_t x_idx = static_cast<pybind11::ssize_t>(x_norm * nx);
        pybind11::ssize_t y_idx = static_cast<pybind11::ssize_t>(y_norm * ny);
        pybind11::ssize_t z_idx = static_cast<pybind11::ssize_t>(z_norm * nz);

        // Clamp the indices to valid range
        x_idx = std::min(std::max(x_idx, pybind11::ssize_t(0)), nx - 1);
        y_idx = std::min(std::max(y_idx, pybind11::ssize_t(0)), ny - 1);
        z_idx = std::min(std::max(z_idx, pybind11::ssize_t(0)), nz - 1);

        return std::make_tuple(x_idx, y_idx, z_idx);
    }
    else
    {
        // The point is outside the voxel grid
        return std::make_tuple(-1, -1, -1);
    }
}

/**
 * 处理图像并更新体素网格和天球纹理
 *
 * 这是核心处理函数，执行以下操作:
 * 1. 计算图像中每个像素的方向向量
 * 2. 将该方向映射到赤经/赤纬(RA/Dec)坐标，找到对应的天球亮度
 * 3. 可选地从图像亮度中减去背景(天球亮度)
 * 4. 如果需要更新天球，则将亮度值累积到celestial_sphere_texture中
 * 5. 如果提供了体素网格，则向网格投射光线并相应更新体素亮度
 *
 * 算法流程:
 * 1. 初始化相机坐标系(基于指向方向)
 * 2. 对每个像素:
 *    a. 计算像素方向在世界坐标系中的向量
 *    b. 转换为RA/Dec坐标
 *    c. 映射到天球纹理坐标
 *    d. 执行背景减除(可选)
 *    e. 更新天球纹理(可选)
 *    f. 向体素网格投射光线(如果启用)
 * 3. 使用OpenMP并行加速像素处理
 *
 * 注意:
 * - 该函数会原地修改voxel_grid和celestial_sphere_texture
 * - 使用原子操作保证多线程安全
 */
void process_image_cpp(
    py::array_t<double> image,
    std::array<double, 3> earth_position,
    std::array<double, 3> pointing_direction,
    double fov,
    pybind11::ssize_t image_width,
    pybind11::ssize_t image_height,
    py::array_t<double> voxel_grid,
    std::vector<std::pair<double, double>> voxel_grid_extent,
    double max_distance,
    int num_steps,
    py::array_t<double> celestial_sphere_texture,
    double center_ra_rad,
    double center_dec_rad,
    double angular_width_rad,
    double angular_height_rad,
    bool update_celestial_sphere,
    bool perform_background_subtraction
)
{
    // Access the image and celestial sphere texture arrays
    auto image_unchecked = image.unchecked<2>();
    auto texture_mutable = celestial_sphere_texture.mutable_unchecked<2>();
    pybind11::ssize_t texture_height = celestial_sphere_texture.shape(0);
    pybind11::ssize_t texture_width = celestial_sphere_texture.shape(1);

    // Check if voxel_grid is provided and non-empty
    bool voxel_grid_provided = voxel_grid && voxel_grid.size() > 0;

    // Variables for voxel grid (only if voxel_grid_provided)
    std::array<pybind11::ssize_t, 3> voxel_grid_size = {0, 0, 0};
    double x_min = 0, x_max = 0;
    double y_min = 0, y_max = 0;
    double z_min = 0, z_max = 0;

    // We only declare voxel_grid_mutable inside the if block if voxel_grid is provided
    // This avoids the need for a default constructor for unchecked_mutable_reference.
    py::detail::unchecked_mutable_reference<double, 3>* voxel_grid_mutable_ptr = nullptr;

    if (voxel_grid_provided)
    {
        // Get a mutable reference to the voxel grid
        auto voxel_grid_mutable = voxel_grid.mutable_unchecked<3>();
        voxel_grid_mutable_ptr = &voxel_grid_mutable;

        // Extract voxel grid dimensions and extents
        voxel_grid_size = {
            voxel_grid.shape(0),
            voxel_grid.shape(1),
            voxel_grid.shape(2)
        };

        x_min = voxel_grid_extent[0].first;
        x_max = voxel_grid_extent[0].second;
        y_min = voxel_grid_extent[1].first;
        y_max = voxel_grid_extent[1].second;
        z_min = voxel_grid_extent[2].first;
        z_max = voxel_grid_extent[2].second;
    }

    // Compute focal length from the field of view
    double focal_length = (image_width / 2.0) / std::tan(fov / 2.0);

    // Principal point (optical center)
    double cx = image_width / 2.0;
    double cy = image_height / 2.0;

    // pointing_direction is the z-axis of the camera frame
    // Normalize pointing_direction to ensure it's a unit vector
    double z_norm = std::sqrt(pointing_direction[0]*pointing_direction[0] +
                              pointing_direction[1]*pointing_direction[1] +
                              pointing_direction[2]*pointing_direction[2]);
    pointing_direction[0] /= z_norm;
    pointing_direction[1] /= z_norm;
    pointing_direction[2] /= z_norm;

    // Define an 'up' vector to avoid singularities
    std::array<double, 3> up = {0.0, 0.0, 1.0};
    if ((std::abs(pointing_direction[0] - up[0]) < 1e-8 &&
         std::abs(pointing_direction[1] - up[1]) < 1e-8 &&
         std::abs(pointing_direction[2] - up[2]) < 1e-8) ||
        (std::abs(pointing_direction[0] + up[0]) < 1e-8 &&
         std::abs(pointing_direction[1] + up[1]) < 1e-8 &&
         std::abs(pointing_direction[2] + up[2]) < 1e-8))
    {
        up = {0.0, 1.0, 0.0};
    }

    // Compute orthonormal basis: x_axis, y_axis, z_axis (z_axis = pointing_direction)
    std::array<double, 3> z_axis = pointing_direction;
    std::array<double, 3> x_axis;
    x_axis[0] = up[1]*z_axis[2] - up[2]*z_axis[1];
    x_axis[1] = up[2]*z_axis[0] - up[0]*z_axis[2];
    x_axis[2] = up[0]*z_axis[1] - up[1]*z_axis[0];

    double x_norm = std::sqrt(x_axis[0]*x_axis[0] + x_axis[1]*x_axis[1] + x_axis[2]*x_axis[2]);
    x_axis[0] /= x_norm;
    x_axis[1] /= x_norm;
    x_axis[2] /= x_norm;

    std::array<double, 3> y_axis;
    y_axis[0] = z_axis[1]*x_axis[2] - z_axis[2]*x_axis[1];
    y_axis[1] = z_axis[2]*x_axis[0] - z_axis[0]*x_axis[2];
    y_axis[2] = z_axis[0]*x_axis[1] - z_axis[1]*x_axis[0];

    // Iterate over each pixel in the image
    #pragma omp parallel for
    for (pybind11::ssize_t i = 0; i < image_height; ++i)
    {
        for (pybind11::ssize_t j = 0; j < image_width; ++j)
        {
            double brightness = image_unchecked(i, j);

            if (brightness > 0)
            {
                // Compute the direction in camera coordinates
                double x_cam = (j - cx);
                double y_cam = (i - cy);
                double z_cam = focal_length;

                double norm = std::sqrt(x_cam*x_cam + y_cam*y_cam + z_cam*z_cam);
                double direction_camera[3] = { x_cam/norm, y_cam/norm, z_cam/norm };

                // Transform direction_camera to world coordinates
                double direction_world[3];
                direction_world[0] = x_axis[0]*direction_camera[0] + y_axis[0]*direction_camera[1] + z_axis[0]*direction_camera[2];
                direction_world[1] = x_axis[1]*direction_camera[0] + y_axis[1]*direction_camera[1] + z_axis[1]*direction_camera[2];
                direction_world[2] = x_axis[2]*direction_camera[0] + y_axis[2]*direction_camera[1] + z_axis[2]*direction_camera[2];

                // Normalize direction_world (should already be unit, but just in case)
                double dir_norm = std::sqrt(direction_world[0]*direction_world[0] +
                                            direction_world[1]*direction_world[1] +
                                            direction_world[2]*direction_world[2]);
                direction_world[0] /= dir_norm;
                direction_world[1] /= dir_norm;
                direction_world[2] /= dir_norm;

                // Compute RA/Dec for direction_world
                double dx = direction_world[0];
                double dy = direction_world[1];
                double dz = direction_world[2];

                double r = std::sqrt(dx*dx + dy*dy + dz*dz);
                double dec = std::asin(dz / r);
                double ra = std::atan2(dy, dx);
                if (ra < 0) ra += 2 * M_PI;

                // Compute offsets from the center of the sky patch
                double ra_offset = ra - center_ra_rad;
                double dec_offset = dec - center_dec_rad;

                // Adjust RA offset for wrapping
                if (ra_offset > M_PI) ra_offset -= 2 * M_PI;
                if (ra_offset < -M_PI) ra_offset += 2 * M_PI;

                // Check if within the defined sky patch
                bool within_sky_patch = (std::abs(ra_offset) <= angular_width_rad / 2) &&
                                        (std::abs(dec_offset) <= angular_height_rad / 2);

                // Map RA/Dec to texture coordinates
                double u = (ra_offset + angular_width_rad / 2) / angular_width_rad * texture_width;
                double v = (dec_offset + angular_height_rad / 2) / angular_height_rad * texture_height;

                pybind11::ssize_t u_idx = static_cast<pybind11::ssize_t>(u);
                pybind11::ssize_t v_idx = static_cast<pybind11::ssize_t>(v);

                // Clamp texture coordinates
                u_idx = std::min(std::max(u_idx, pybind11::ssize_t(0)), texture_width - 1);
                v_idx = std::min(std::max(v_idx, pybind11::ssize_t(0)), texture_height - 1);

                // Background subtraction
                double background_brightness = 0.0;
                if (within_sky_patch)
                {
                    background_brightness = texture_mutable(v_idx, u_idx);
                }

                if (perform_background_subtraction)
                {
                    brightness -= background_brightness;
                    if (brightness <= 0)
                        continue;  // Skip if adjusted brightness is zero or negative
                }

                // Update celestial sphere texture if needed
                if (update_celestial_sphere && within_sky_patch)
                {
                    #pragma omp atomic
                    texture_mutable(v_idx, u_idx) += brightness;
                }

                // If we have a voxel grid, cast rays into it and update voxel brightness
                if (voxel_grid_provided)
                {
                    // Safe to use voxel_grid_mutable_ptr now because voxel_grid_provided is true
                    auto &voxel_grid_mutable = *voxel_grid_mutable_ptr; // Reference to the voxel grid

                    // Ray casting into voxel grid
                    double step_size = max_distance / num_steps;

                    std::array<double, 3> ray_origin = { earth_position[0], earth_position[1], earth_position[2] };
                    std::array<double, 3> ray_direction = { direction_world[0], direction_world[1], direction_world[2] };

                    std::array<double, 3> box_min = { x_min, y_min, z_min };
                    std::array<double, 3> box_max = { x_max, y_max, z_max };
                    double t_entry, t_exit;

                    if (ray_aabb_intersection(ray_origin, ray_direction, box_min, box_max, t_entry, t_exit))
                    {
                        t_entry = std::max(t_entry, 0.0);
                        t_exit = std::min(t_exit, max_distance);

                        if (t_entry <= t_exit)
                        {
                            int s_entry = static_cast<int>(t_entry / step_size);
                            int s_exit = static_cast<int>(t_exit / step_size);

                            for (int s = s_entry; s <= s_exit; ++s)
                            {
                                double d = s * step_size;
                                double px = ray_origin[0] + d * ray_direction[0];
                                double py = ray_origin[1] + d * ray_direction[1];
                                double pz = ray_origin[2] + d * ray_direction[2];

                                auto indices = point_to_voxel_indices({ px, py, pz }, voxel_grid_extent, voxel_grid_size);
                                pybind11::ssize_t x_idx = std::get<0>(indices);
                                pybind11::ssize_t y_idx = std::get<1>(indices);
                                pybind11::ssize_t z_idx = std::get<2>(indices);

                                if (x_idx >= 0)
                                {
                                    #pragma omp atomic
                                    voxel_grid_mutable(x_idx, y_idx, z_idx) += brightness;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// 将函数暴露给Python
PYBIND11_MODULE(process_image_cpp, m) {
    m.doc() = "process_image函数的C++实现";
    m.def("process_image_cpp", &process_image_cpp, "用C++处理图像并更新体素网格",
          py::arg("image"),
          py::arg("earth_position"),
          py::arg("pointing_direction"),
          py::arg("fov"),
          py::arg("image_width"),
          py::arg("image_height"),
          py::arg("voxel_grid"),
          py::arg("voxel_grid_extent"),
          py::arg("max_distance"),
          py::arg("num_steps"),
          py::arg("celestial_sphere_texture"),
          py::arg("center_ra_rad"),
          py::arg("center_dec_rad"),
          py::arg("angular_width_rad"),
          py::arg("angular_height_rad"),
          py::arg("update_celestial_sphere"),
          py::arg("perform_background_subtraction")
    );
}
