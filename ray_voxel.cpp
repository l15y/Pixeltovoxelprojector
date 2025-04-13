/***************************************************
 * ray_voxel.cpp - 基于多视角图像的运动体素重建
 *
 * 完整C++实现流程:
 *   1) 使用nlohmann::json解析metadata.json
 *   2) 加载灰度图像(stb_image)
 *   3) 对每个相机的连续帧进行运动检测
 *   4) 为变化像素投射光线(体素DDA算法)
 *   5) 在共享3D体素网格中累积
 *   6) 将体素网格保存为.bin文件
 *
 * 核心算法:
 * - 数字微分分析器(DDA)用于高效光线遍历
 * - 基于帧间差异的运动检测
 * - 多相机数据融合
 ***************************************************/




#include <iostream>
#include <fstream>
#include <cmath>
#include <limits>
#include <vector>
#include <map>
#include <string>
#include <algorithm>

// External libraries for JSON & image loading
#include "nlohmann/json.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"



// For convenience
using json = nlohmann::json;

//----------------------------------------------
// 1) Data Structures
//----------------------------------------------
// 3D向量结构
struct Vec3 {
    float x, y, z;  // 三维坐标
};

// 3x3矩阵结构
struct Mat3 {
    float m[9];  // 按行优先存储的矩阵元素
};

// 帧信息结构体
struct FrameInfo {
    int camera_index;     // 相机索引
    int frame_index;      // 帧序号
    Vec3 camera_position; // 相机位置(x,y,z)
    float yaw, pitch, roll; // 欧拉角(偏航、俯仰、滚转)
    float fov_degrees;    // 视场角(度)
    std::string image_file; // 图像文件名
    // 可选存储对象名称和位置(如果需要)
};

//----------------------------------------------
// 2) Basic Math Helpers
//----------------------------------------------
// 角度转弧度
static inline float deg2rad(float deg) {
    return deg * 3.14159265358979323846f / 180.0f;
}

// 向量归一化
static inline Vec3 normalize(const Vec3 &v) {
    float len = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    if(len < 1e-12f) {
        return {0.f, 0.f, 0.f};  // 零向量处理
    }
    return { v.x/len, v.y/len, v.z/len };  // 归一化向量
}

// 3x3矩阵乘以3D向量
static inline Vec3 mat3_mul_vec3(const Mat3 &M, const Vec3 &v) {
    Vec3 r;
    r.x = M.m[0]*v.x + M.m[1]*v.y + M.m[2]*v.z;  // 第一行点积
    r.y = M.m[3]*v.x + M.m[4]*v.y + M.m[5]*v.z;  // 第二行点积
    r.z = M.m[6]*v.x + M.m[7]*v.y + M.m[8]*v.z;  // 第三行点积
    return r;
}

//----------------------------------------------
// 3) Euler -> Rotation Matrix
//----------------------------------------------
// 根据欧拉角(yaw/pitch/roll)构建旋转矩阵
Mat3 rotation_matrix_yaw_pitch_roll(float yaw_deg, float pitch_deg, float roll_deg) {
    float y = deg2rad(yaw_deg);    // 偏航角(绕Z轴)
    float p = deg2rad(pitch_deg);  // 俯仰角(绕Y轴)
    float r = deg2rad(roll_deg);   // 滚转角(绕X轴)

    // 构建各轴旋转矩阵
    // Rz(yaw) - 绕Z轴旋转
    float cy = std::cos(y), sy = std::sin(y);
    float Rz[9] = {
        cy, -sy, 0.f,  // 第一行
        sy,  cy, 0.f,  // 第二行
        0.f, 0.f, 1.f  // 第三行
    };

    // Ry(roll) - 绕Y轴旋转
    float cr = std::cos(r), sr = std::sin(r);
    float Ry[9] = {
        cr,  0.f, sr,   // 第一行
        0.f, 1.f, 0.f,  // 第二行
        -sr, 0.f, cr    // 第三行
    };

    // Rx(pitch) - 绕X轴旋转
    float cp = std::cos(p), sp = std::sin(p);
    float Rx[9] = {
        1.f,  0.f,  0.f,  // 第一行
        0.f,  cp,  -sp,   // 第二行
        0.f,  sp,   cp    // 第三行
    };

    // 3x3矩阵乘法辅助函数
    auto matmul3x3 = [&](const float A[9], const float B[9], float C[9]){
        for(int row=0; row<3; ++row) {
            for(int col=0; col<3; ++col) {
                C[row*3+col] =
                    A[row*3+0]*B[0*3+col] +  // 行*列
                    A[row*3+1]*B[1*3+col] +
                    A[row*3+2]*B[2*3+col];
            }
        }
    };

    // 组合旋转矩阵: R = Rz * Ry * Rx
    float Rtemp[9], Rfinal[9];
    matmul3x3(Rz, Ry, Rtemp);     // 先计算Rz*Ry
    matmul3x3(Rtemp, Rx, Rfinal); // 再乘以Rx

    Mat3 out;
    for(int i=0; i<9; i++){
        out.m[i] = Rfinal[i];  // 复制结果
    }
    return out;
}

//----------------------------------------------
// 4) Load JSON Metadata
//----------------------------------------------
std::vector<FrameInfo> load_metadata(const std::string &json_path) {
    std::vector<FrameInfo> frames;

    std::ifstream ifs(json_path);
    if(!ifs.is_open()){
        std::cerr << "ERROR: Cannot open " << json_path << std::endl;
        return frames;
    }
    json j;
    ifs >> j;
    if(!j.is_array()){
        std::cerr << "ERROR: JSON top level is not an array.\n";
        return frames;
    }

    for(const auto &entry : j) {
        FrameInfo fi;
        fi.camera_index   = entry.value("camera_index", 0);
        fi.frame_index    = entry.value("frame_index", 0);
        fi.yaw            = entry.value("yaw", 0.f);
        fi.pitch          = entry.value("pitch", 0.f);
        fi.roll           = entry.value("roll", 0.f);
        fi.fov_degrees    = entry.value("fov_degrees", 60.f);
        fi.image_file     = entry.value("image_file", "");

        // camera_position array
        if(entry.contains("camera_position") && entry["camera_position"].is_array()){
            auto arr = entry["camera_position"];
            if(arr.size()>=3){
                fi.camera_position.x = arr[0].get<float>();
                fi.camera_position.y = arr[1].get<float>();
                fi.camera_position.z = arr[2].get<float>();
            }
        }
        frames.push_back(fi);
    }

    return frames;
}

//----------------------------------------------
// 5) Image Loading (Gray) & Motion Detection
//----------------------------------------------
struct ImageGray {
    int width;
    int height;
    std::vector<float> pixels;  // grayscale float
};

#include <random>  // for std::mt19937, std::uniform_real_distribution

// Load image in grayscale (0-255 float) and add uniform noise.
bool load_image_gray(const std::string &img_path, ImageGray &out) {
    int w, h, channels;
    // stbi_load returns 8-bit data by default
    unsigned char* data = stbi_load(img_path.c_str(), &w, &h, &channels, 1);
    if (!data) {
        std::cerr << "Failed to load image: " << img_path << std::endl;
        return false;
    }

    out.width = w;
    out.height = h;
    out.pixels.resize(w * h);

    // Prepare random noise generator
    static std::random_device rd;
    static std::mt19937 gen(rd());
    // Noise in [-3, +3]
    std::uniform_real_distribution<float> noise_dist(-1.0f, 1.0f);

    // Copy pixels and add noise
    for (int i = 0; i < w * h; i++) {
        float val = static_cast<float>(data[i]);  // 0..255
        // Add uniform noise
        val += noise_dist(gen);
        // Clamp to [0, 255]
        if (val < 0.0f) val = 0.0f;
        if (val > 255.0f) val = 255.0f;
        // Store in out.pixels
        out.pixels[i] = val;
    }

    stbi_image_free(data);
    return true;
}

// Detect motion by absolute difference
// Returns a boolean mask + the difference for each pixel
struct MotionMask {
    int width;
    int height;
    std::vector<bool> changed;
    std::vector<float> diff; // absolute difference
};

MotionMask detect_motion(const ImageGray &prev, const ImageGray &next, float threshold) {
    MotionMask mm;
    if(prev.width != next.width || prev.height != next.height) {
        std::cerr << "Images differ in size. Can't do motion detection!\n";
        mm.width = 0;
        mm.height = 0;
        return mm;
    }
    mm.width = prev.width;
    mm.height = prev.height;
    mm.changed.resize(mm.width * mm.height, false);
    mm.diff.resize(mm.width * mm.height, 0.f);

    for(int i=0; i < mm.width*mm.height; i++){
        float d = std::fabs(prev.pixels[i] - next.pixels[i]);
        mm.diff[i] = d;
        mm.changed[i] = (d > threshold);
    }
    return mm;
}

//----------------------------------------------
// 6) Voxel DDA
//----------------------------------------------
// 光线步进信息
struct RayStep {
    int ix, iy, iz;      // 当前体素索引
    int step_count;       // 步进计数
    float distance;       // 当前距离
};

// 安全除法(避免除以零)
static inline float safe_div(float num, float den) {
    float eps = 1e-12f;
    if(std::fabs(den) < eps) {
        return std::numeric_limits<float>::infinity();  // 返回无穷大
    }
    return num / den;
}

// 使用DDA算法将光线投射到体素网格中
std::vector<RayStep> cast_ray_into_grid(
    const Vec3 &camera_pos,     // 相机位置
    const Vec3 &dir_normalized, // 归一化光线方向
    int N,                      // 网格分辨率(NxNxN)
    float voxel_size,           // 体素大小(单位:米)
    const Vec3 &grid_center)    // 网格中心位置
{
    std::vector<RayStep> steps;
    steps.reserve(64);  // 预分配空间

    // 计算网格边界
    float half_size = 0.5f * (N * voxel_size);
    Vec3 grid_min = { grid_center.x - half_size,
                      grid_center.y - half_size,
                      grid_center.z - half_size };
    Vec3 grid_max = { grid_center.x + half_size,
                      grid_center.y + half_size,
                      grid_center.z + half_size };

    float t_min = 0.f;  // 进入距离
    float t_max = std::numeric_limits<float>::infinity();  // 退出距离

    // 1) 光线与包围盒相交检测
    for(int i=0; i<3; i++){
        float origin = (i==0)? camera_pos.x : ((i==1)? camera_pos.y : camera_pos.z);
        float d      = (i==0)? dir_normalized.x : ((i==1)? dir_normalized.y : dir_normalized.z);
        float mn     = (i==0)? grid_min.x : ((i==1)? grid_min.y : grid_min.z);
        float mx     = (i==0)? grid_max.x : ((i==1)? grid_max.y : grid_max.z);

        if(std::fabs(d) < 1e-12f){  // 光线平行于轴
            if(origin < mn || origin > mx){
                return steps; // 不相交
            }
        } else {
            float t1 = (mn - origin)/d;  // 与最小平面相交
            float t2 = (mx - origin)/d;  // 与最大平面相交
            float t_near = std::fmin(t1, t2);
            float t_far  = std::fmax(t1, t2);
            if(t_near > t_min) t_min = t_near;  // 更新最近交点
            if(t_far  < t_max) t_max = t_far;   // 更新最远交点
            if(t_min > t_max){  // 无有效交点
                return steps;
            }
        }
    }

    if(t_min < 0.f) t_min = 0.f;  // 确保在光线前方

    // 2) 计算起始体素
    Vec3 start_world = { camera_pos.x + t_min*dir_normalized.x,
                         camera_pos.y + t_min*dir_normalized.y,
                         camera_pos.z + t_min*dir_normalized.z };
    // 转换为网格坐标
    float fx = (start_world.x - grid_min.x)/voxel_size;
    float fy = (start_world.y - grid_min.y)/voxel_size;
    float fz = (start_world.z - grid_min.z)/voxel_size;

    int ix = int(fx);  // x索引
    int iy = int(fy);  // y索引
    int iz = int(fz);  // z索引
    if(ix<0 || ix>=N || iy<0 || iy>=N || iz<0 || iz>=N) {
        return steps;  // 起始点在网格外
    }

    // 3) 确定步进方向
    int step_x = (dir_normalized.x >= 0.f)? 1 : -1;  // x步进方向
    int step_y = (dir_normalized.y >= 0.f)? 1 : -1;  // y步进方向
    int step_z = (dir_normalized.z >= 0.f)? 1 : -1;  // z步进方向

    // 边界计算辅助函数
    auto boundary_in_world_x = [&](int i_x){ return grid_min.x + i_x*voxel_size; };
    auto boundary_in_world_y = [&](int i_y){ return grid_min.y + i_y*voxel_size; };
    auto boundary_in_world_z = [&](int i_z){ return grid_min.z + i_z*voxel_size; };

    // 计算下一个边界
    int nx_x = ix + (step_x>0?1:0);
    int nx_y = iy + (step_y>0?1:0);
    int nx_z = iz + (step_z>0?1:0);

    float next_bx = boundary_in_world_x(nx_x);  // 下一个x边界
    float next_by = boundary_in_world_y(nx_y);  // 下一个y边界
    float next_bz = boundary_in_world_z(nx_z);  // 下一个z边界

    // 计算到各轴边界的距离
    float t_max_x = safe_div(next_bx - camera_pos.x, dir_normalized.x);
    float t_max_y = safe_div(next_by - camera_pos.y, dir_normalized.y);
    float t_max_z = safe_div(next_bz - camera_pos.z, dir_normalized.z);

    // 计算步长增量
    float t_delta_x = safe_div(voxel_size, std::fabs(dir_normalized.x));
    float t_delta_y = safe_div(voxel_size, std::fabs(dir_normalized.y));
    float t_delta_z = safe_div(voxel_size, std::fabs(dir_normalized.z));

    float t_current = t_min;  // 当前距离
    int step_count = 0;       // 步进计数器

    // 4) DDA步进循环
    while(t_current <= t_max){
        RayStep rs;
        rs.ix = ix;  // 当前x索引
        rs.iy = iy;  // 当前y索引
        rs.iz = iz;  // 当前z索引
        rs.step_count = step_count;
        rs.distance = t_current;

        steps.push_back(rs);  // 记录步进信息

        // 选择下一个最近的边界
        if(t_max_x < t_max_y && t_max_x < t_max_z){
            ix += step_x;  // x方向步进
            t_current = t_max_x;
            t_max_x += t_delta_x;
        } else if(t_max_y < t_max_z){
            iy += step_y;  // y方向步进
            t_current = t_max_y;
            t_max_y += t_delta_y;
        } else {
            iz += step_z;  // z方向步进
            t_current = t_max_z;
            t_max_z += t_delta_z;
        }
        step_count++;
        if(ix<0 || ix>=N || iy<0 || iy>=N || iz<0 || iz>=N){
            break;  // 超出网格边界
        }
    }

    return steps;
}

//----------------------------------------------
// 7) Main Pipeline
//----------------------------------------------
int main(int argc, char** argv) {
    if(argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <metadata.json> <image_folder> <output_voxel_bin>\n";
        return 1;
    }
    std::string metadata_path = argv[1];
    std::string images_folder = argv[2];
    std::string output_bin    = argv[3];

    //------------------------------------------
    // 7.1) Load metadata
    //------------------------------------------
    std::vector<FrameInfo> frames = load_metadata(metadata_path);
    if(frames.empty()) {
        std::cerr << "No frames loaded.\n";
        return 1;
    }
    // Group by camera_index
    // map< camera_index, vector<FrameInfo> >
    std::map<int, std::vector<FrameInfo>> frames_by_cam;
    for(const auto &f : frames) {
        frames_by_cam[f.camera_index].push_back(f);
    }
    // Sort each by frame_index
    for(auto &kv : frames_by_cam) {
        auto &v = kv.second;
        std::sort(v.begin(), v.end(), [](auto &a, auto &b){
            return a.frame_index < b.frame_index;
        });
    }

    //------------------------------------------
    // 7.2) Create a 3D voxel grid
    //------------------------------------------
    const int N = 500;
    const float voxel_size = 6.f;
    // Hard-coded center (like your Python example):
    Vec3 grid_center = {-0.f, 0.f, 500.f};
    //    Vec3 grid_center = {-0.f, 0.f, 200.f}; // For birds

    std::vector<float> voxel_grid(N*N*N, 0.f);

    //------------------------------------------
    // 7.3) For each camera, load consecutive frames, detect motion,
    //      and cast rays for changed pixels
    //------------------------------------------
    // Basic parameters
    float motion_threshold = 2.0f;  // difference threshold
    float alpha = 0.1f;            // distance-based attenuation factor

    for(auto &kv : frames_by_cam) {
        int cam_id = kv.first;
        auto &cam_frames = kv.second;

        if(cam_frames.size() < 2) {
            // Need at least two frames to see motion
            continue;
        }

        // We'll keep the previous image to compare
        ImageGray prev_img;
        bool prev_valid = false;
        FrameInfo prev_info;

        for(size_t i=0; i<cam_frames.size(); i++){
            // Load current frame
            FrameInfo curr_info = cam_frames[i];
            std::string img_path = images_folder + "/" + curr_info.image_file;

            ImageGray curr_img;
            if(!load_image_gray(img_path, curr_img)) {
                std::cerr << "Skipping frame due to load error.\n";
                continue;
            }

            if(!prev_valid) {
                // Just store it, and wait for next
                prev_img = curr_img;
                prev_info = curr_info;
                prev_valid = true;
                continue;
            }

            // Now we have prev + curr => detect motion
            MotionMask mm = detect_motion(prev_img, curr_img, motion_threshold);

            // Use the "current" frame's camera info for ray-casting
            // (adjust if you prefer the previous frame's camera)
            Vec3 cam_pos    = curr_info.camera_position;
            Mat3 cam_rot    = rotation_matrix_yaw_pitch_roll(curr_info.yaw, curr_info.pitch, curr_info.roll);
            float fov_rad   = deg2rad(curr_info.fov_degrees);
            float focal_len = (mm.width*0.5f) / std::tan(fov_rad*0.5f);

            // For each changed pixel, accumulate into the voxel grid
            for(int v = 0; v < mm.height; v++){
                for(int u = 0; u < mm.width; u++){
                    if(!mm.changed[v*mm.width + u]){
                        continue; // skip if no motion
                    }
                    // Pixel brightness from current or use mm.diff
                    float pix_val = mm.diff[v*mm.width + u];
                    if(pix_val < 1e-3f) {
                        continue;
                    }

                    // Build local camera direction
                    float x = (float(u) - 0.5f*mm.width);
                    float y = - (float(v) - 0.5f*mm.height);
                    float z = -focal_len;

                    Vec3 ray_cam = {x,y,z};
                    ray_cam = normalize(ray_cam);

                    // transform to world
                    Vec3 ray_world = mat3_mul_vec3(cam_rot, ray_cam);
                    ray_world = normalize(ray_world);

                    // DDA
                    std::vector<RayStep> steps = cast_ray_into_grid(
                        cam_pos, ray_world, N, voxel_size, grid_center
                    );

                    // Accumulate
                    for(const auto &rs : steps) {
                        float dist = rs.distance;
                        float attenuation = 1.f/(1.f + alpha*dist);
                        float val = pix_val * 1.f; //attenuation, need to fix this to work better so that it scales with the size of the image as it would appear at that distance but for now this works; 
                        int idx = rs.ix*N*N + rs.iy*N + rs.iz;
                        voxel_grid[idx] += val;
                    }
                }
            }

            // Move current -> previous
            prev_img = curr_img;
            prev_info = curr_info;
        }
    }

    //------------------------------------------
    // 7.4) Save the voxel grid to .bin
    //------------------------------------------
    {
        std::ofstream ofs(output_bin, std::ios::binary);
        if(!ofs) {
            std::cerr << "Cannot open output file: " << output_bin << "\n";
            return 1;
        }
        // Write metadata (N, voxel_size)
        ofs.write(reinterpret_cast<const char*>(&N), sizeof(int));
        ofs.write(reinterpret_cast<const char*>(&voxel_size), sizeof(float));
        // Write the data
        ofs.write(reinterpret_cast<const char*>(voxel_grid.data()), voxel_grid.size()*sizeof(float));
        ofs.close();
        std::cout << "Saved voxel grid to " << output_bin << "\n";
    }

    return 0;
}
