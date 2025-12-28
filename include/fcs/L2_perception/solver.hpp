#pragma once

#include "../core/types.hpp"
#include "detector.hpp"
#include "math/fast_tf/frame.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstdint>
#include <opencv2/core.hpp>
#include <optional>
#include <vector>

namespace fcs::L2 {

// ============================================================================
// Measurement - PnP解算结果
// ============================================================================

struct Measurement {
    // 位姿 (camera_optical 坐标系)
    fast_tf::FuckTransform<fast_tf::camera_optical_fuxk_frame> transform;

    // 球坐标 (camera_optical 坐标系)
    fast_tf::Fuck<fast_tf::camera_optical_fuxk_frame, std::array<double, 4>> xyza;

    // 元数据
    uint64_t timestamp_ns;
    ArmorType type;
    ArmorName name;

    // 原始检测数据 (用于调试/可视化)
    std::array<cv::Point2f, 4> corners;
};

// ============================================================================
// SolverConfig - 解算器配置
// ============================================================================

struct SolverConfig {
    // 相机内参
    double fx{1000.0};
    double fy{1000.0};
    double cx{640.0};
    double cy{360.0};

    // 畸变系数 [k1, k2, p1, p2, k3]
    std::array<double, 5> dist_coeffs{0, 0, 0, 0, 0};

    // 装甲板物理尺寸 (米)
    double lightbar_length{0.056};   // 灯条长度
    double big_armor_width{0.230};   // 大装甲板宽度
    double small_armor_width{0.135}; // 小装甲板宽度
};

// ============================================================================
// PnPSolver - PnP解算器
// ============================================================================

class PnPSolver {
public:
    explicit PnPSolver(const SolverConfig& config);

    /**
     * @brief 解算单个检测
     * @param det 检测结果
     * @return 解算结果 (失败返回 nullopt)
     */
    [[nodiscard]] std::optional<Measurement> solve(const Detection& det) const noexcept;

    /**
     * @brief 批量解算
     * @param detections 检测结果列表
     * @return 解算结果列表 (与输入一一对应，失败的为 nullopt)
     */
    [[nodiscard]] std::vector<std::optional<Measurement>>
        solve_all(std::span<const Detection> detections) const noexcept;

    /**
     * @brief 重投影装甲板角点到图像
     * @param meas 解算结果
     * @return 重投影的四角点
     */
    [[nodiscard]] std::array<cv::Point2f, 4> reproject(const Measurement& meas) const noexcept;

private:
    SolverConfig config_;
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;

    // 物点模板 (BIG/SMALL)
    std::vector<cv::Point3f> big_armor_points_;
    std::vector<cv::Point3f> small_armor_points_;

    [[nodiscard]] const std::vector<cv::Point3f>& get_object_points(ArmorType type) const noexcept;
};

// ============================================================================
// 数学工具
// ============================================================================

namespace math {

/**
 * @brief 角度归一化到 (-pi, pi]
 */
[[nodiscard]] double normalize_angle(double angle) noexcept;

/**
 * @brief 直角坐标 → 球坐标 (yaw, pitch, distance)
 */
[[nodiscard]] Eigen::Vector3d xyz_to_ypd(const Eigen::Vector3d& xyz) noexcept;

/**
 * @brief 球坐标 → 直角坐标
 */
[[nodiscard]] Eigen::Vector3d ypd_to_xyz(const Eigen::Vector3d& ypd) noexcept;

/**
 * @brief xyz_to_ypd 的雅可比矩阵
 */
[[nodiscard]] Eigen::Matrix3d xyz_to_ypd_jacobian(const Eigen::Vector3d& xyz) noexcept;

} // namespace math

} // namespace fcs::L2
