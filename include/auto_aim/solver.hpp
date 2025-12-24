#pragma once

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include "config.hpp"
#include "detector_state.hpp"
#include "types.hpp"

namespace auto_aim {

// 装甲板物理尺寸
constexpr double LIGHTBAR_LENGTH   = 56e-3;  // m
constexpr double BIG_ARMOR_WIDTH   = 230e-3; // m
constexpr double SMALL_ARMOR_WIDTH = 135e-3; // m

/**
 * @brief PnP解算器
 *
 * 移植自 sp_vision_25/tasks/auto_aim/solver
 * 功能：
 * 1. 对检测到的装甲板执行PnP解算，获取3D位姿
 * 2. 坐标系变换：camera_optical->ROS相机->云台->世界
 * 3. Yaw角优化（通过重投影误差最小化）
 *
 * 坐标系定义（ROS标准REP0103）：
 * - camera_optical坐标系（OpenCV/PnP输出）：x右 y下 z前（沿光轴）
 * - ROS相机坐标系（标准）：x前 y左 z上
 * - 云台/世界坐标系：x前 y左 z上（当相机正向安装时）
 *
 * 完整转换链：
 * camera_optical → [R_cameraOptical_to_ros] → ROS相机
 *                → [R_camera2gimbal_] → 云台
 *                → [R_gimbal2world_] → 世界
 */
class Solver {
public:
    explicit Solver(const SpVisionSolverConfig& config);

    /**
     * @brief 设置云台到世界的旋转矩阵
     * @param q IMU四元数 (从IMU body到IMU abs)
     */
    void set_R_gimbal2world(const Eigen::Quaterniond& q);

    /**
     * @brief 获取当前云台到世界的旋转矩阵
     */
    [[nodiscard]] Eigen::Matrix3d R_gimbal2world() const { return R_gimbal2world_; }

    /**
     * @brief 对单个装甲板执行PnP解算
     * @param armor 检测到的装甲板数据
     * @param pose3d 输出的3D位姿
     * @param optimize_yaw 是否执行yaw优化 (平衡步兵不做yaw优化)
     * @return 是否解算成功
     */
    bool solve(const talos::Armor& armor, ArmorPose3D& pose3d, bool optimize_yaw = true) const;

    /**
     * @brief 对ArmorSlots中的所有装甲板执行PnP解算
     * @param slots 检测结果，poses数组会被填充
     * @param poses3d 输出的详细3D位姿数组
     */
    void solve_all(const talos::ArmorSlots& slots, std::array<ArmorPose3D, 10>& poses3d) const;

    /**
     * @brief 重投影装甲板角点到图像平面
     * @param xyz_in_world 世界坐标系下的位置
     * @param yaw 装甲板yaw角
     * @param type 装甲板类型
     * @param name 装甲板名称
     * @return 重投影的四个角点
     */
    [[nodiscard]] std::vector<cv::Point2f> reproject_armor(
        const Eigen::Vector3d& xyz_in_world, double yaw, talos::ArmorType type,
        talos::ArmorName name) const;

    /**
     * @brief 世界坐标到像素坐标的转换
     * @param world_points 世界坐标点列表
     * @return 像素坐标点列表
     */
    [[nodiscard]] std::vector<cv::Point2f>
        world2pixel(const std::vector<cv::Point3f>& world_points) const;

private:
    // 相机参数
    cv::Mat camera_matrix_;
    cv::Mat distort_coeffs_;

    // 外参矩阵和坐标系转换
    Eigen::Matrix3d R_gimbal2imubody_;
    Eigen::Matrix3d R_camera2gimbal_;
    Eigen::Vector3d t_camera2gimbal_;
    Eigen::Matrix3d R_gimbal2world_;

    // camera_optical到ROS标准相机坐标系的转换矩阵
    // OpenCV/PnP输出 camera_optical (x右, y下, z前)
    // → ROS标准 (x前, y左, z上)
    Eigen::Matrix3d R_cameraOptical_to_ros_;

    /**
     * @brief 执行Yaw角优化
     * @param armor 装甲板数据
     * @param pose3d 3D位姿 (ypr_in_world[0]会被更新)
     */
    void optimize_yaw(const talos::Armor& armor, ArmorPose3D& pose3d) const;

    /**
     * @brief 计算装甲板重投影误差
     * @param armor 装甲板数据
     * @param xyz_in_world 世界坐标系位置
     * @param yaw 假设的yaw角
     * @param type 装甲板类型
     * @param name 装甲板名称
     * @return 重投影误差
     */
    [[nodiscard]] double armor_reprojection_error(
        const talos::Armor& armor, const Eigen::Vector3d& xyz_in_world, double yaw,
        talos::ArmorType type, talos::ArmorName name) const;
};

// ============ 数学工具函数 ============
namespace math {

/**
 * @brief 将弧度值限制在(-pi, pi]
 */
double limit_rad(double angle);

/**
 * @brief 旋转矩阵转欧拉角 (ZYX顺序)
 * @param R 旋转矩阵
 * @param axis0 , axis1 , axis2 轴顺序 (0=x, 1=y, 2=z)
 * @param extrinsic 是否为外旋
 * @return 欧拉角向量
 */
Eigen::Vector3d
    eulers(const Eigen::Matrix3d& R, int axis0, int axis1, int axis2, bool extrinsic = false);

/**
 * @brief 四元数转欧拉角
 */
Eigen::Vector3d
    eulers(const Eigen::Quaterniond& q, int axis0, int axis1, int axis2, bool extrinsic = false);

/**
 * @brief 直角坐标系转球坐标系 (yaw, pitch, distance)
 */
Eigen::Vector3d xyz2ypd(const Eigen::Vector3d& xyz);

/**
 * @brief 直角坐标系转球坐标系的雅可比矩阵
 */
Eigen::MatrixXd xyz2ypd_jacobian(const Eigen::Vector3d& xyz);

/**
 * @brief 球坐标系转直角坐标系
 */
Eigen::Vector3d ypd2xyz(const Eigen::Vector3d& ypd);

} // namespace math

} // namespace auto_aim
