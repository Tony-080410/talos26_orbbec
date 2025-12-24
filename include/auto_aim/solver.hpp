#pragma once

#include <Eigen/Geometry>
#include <cstdint>
#include <opencv2/opencv.hpp>

#include "config.hpp"
#include "detector_state.hpp"
#include "pipeline_types.hpp"
#include "types.hpp"

namespace auto_aim {

// 装甲板物理尺寸
constexpr double LIGHTBAR_LENGTH   = 56e-3;  // m
constexpr double BIG_ARMOR_WIDTH   = 230e-3; // m
constexpr double SMALL_ARMOR_WIDTH = 135e-3; // m

/**
 * @brief PnP解算器
 *
 * 功能：对检测到的装甲板执行PnP解算，获取3D位姿
 *
 * 坐标系输出：
 * - camera_optical_frame (OpenCV/PnP输出)：x右 y下 z前（沿光轴）
 * - 其他坐标系的转换由使用者通过 fast_tf 完成
 *
 * 设计说明：
 * - Solver 只负责 PnP 解算，输出 camera_optical_frame 坐标系数据
 * - 解算结果带有帧时间戳 (timestamp_ns)，用于 fast_tf 时间插值查询
 * - tf 发布和坐标转换由 Tracker/Target 负责
 */
class Solver {
public:
    /**
     * @brief 构造解算器
     * @param config 配置（相机内参）
     */
    explicit Solver(const SpVisionSolverConfig& config);

    // ============ SolverConcept 方法 ============

    /**
     * @brief Concept 兼容的解算方法
     * @param det 检测结果 (variant)
     * @param out 解算结果 (variant)
     * @param debug 调试数据
     */
    void solve(const DetectionResult& det, SolveResult& out, DebugData& debug) noexcept;

    // ============ 核心方法 ============

    /**
     * @brief 对单个装甲板执行PnP解算
     * @param armor 检测到的装甲板数据
     * @param pose3d 输出的3D位姿 (camera_optical_frame坐标系)
     * @param timestamp_ns 帧时间戳 (纳秒)
     * @return 是否解算成功
     */
    bool solve(const talos::Armor& armor, ArmorPose3D& pose3d, uint64_t timestamp_ns) const;

    /**
     * @brief 对ArmorSlots中的所有装甲板执行PnP解算
     * @param slots 检测结果
     * @param poses3d 输出的3D位姿数组 (camera_optical_frame坐标系)
     * @param timestamp_ns 帧时间戳 (纳秒)
     */
    void solve_all(
        const talos::ArmorSlots& slots, std::array<ArmorPose3D, 10>& poses3d,
        uint64_t timestamp_ns) const;

    /**
     * @brief 重投影装甲板角点到图像平面
     * @param xyz_in_camera 相机光学坐标系下的位置
     * @param R_in_camera 相机光学坐标系下的旋转矩阵
     * @param type 装甲板类型
     * @return 重投影的四个角点
     */
    [[nodiscard]] std::vector<cv::Point2f> reproject_armor(
        const Eigen::Vector3d& xyz_in_camera, const Eigen::Matrix3d& R_in_camera,
        talos::ArmorType type) const;

private:
    // 相机内参
    cv::Mat camera_matrix_;
    cv::Mat distort_coeffs_;
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

/**
 * @brief BA 解算器占位符
 *
 * 未来扩展，目前只提供空实现以满足 variant 定义。
 */
class BASolver {
public:
    void solve(const DetectionResult& /*det*/, SolveResult& out, DebugData& /*debug*/) noexcept {
        out = std::monostate{};
    }
};

} // namespace auto_aim
