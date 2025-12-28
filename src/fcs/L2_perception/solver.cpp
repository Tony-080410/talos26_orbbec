#include "fcs/L2_perception/solver.hpp"

#include <cmath>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <spdlog/spdlog.h>

namespace fcs::L2 {

// ============================================================================
// 数学工具实现
// ============================================================================

namespace math {

double normalize_angle(double angle) noexcept {
    while (angle > M_PI)
        angle -= 2 * M_PI;
    while (angle <= -M_PI)
        angle += 2 * M_PI;
    return angle;
}

Eigen::Vector3d xyz_to_ypd(const Eigen::Vector3d& xyz) noexcept {
    const auto x = xyz[0], y = xyz[1], z = xyz[2];
    const auto yaw      = std::atan2(y, x);
    const auto pitch    = std::atan2(z, std::sqrt(x * x + y * y));
    const auto distance = std::sqrt(x * x + y * y + z * z);
    return {yaw, pitch, distance};
}

Eigen::Vector3d ypd_to_xyz(const Eigen::Vector3d& ypd) noexcept {
    const auto yaw = ypd[0], pitch = ypd[1], distance = ypd[2];
    const auto x = distance * std::cos(pitch) * std::cos(yaw);
    const auto y = distance * std::cos(pitch) * std::sin(yaw);
    const auto z = distance * std::sin(pitch);
    return {x, y, z};
}

Eigen::Matrix3d xyz_to_ypd_jacobian(const Eigen::Vector3d& xyz) noexcept {
    const auto x = xyz[0], y = xyz[1], z = xyz[2];
    const auto xy_sq    = x * x + y * y;
    const auto xy_norm  = std::sqrt(xy_sq);
    const auto xyz_sq   = xy_sq + z * z;
    const auto xyz_norm = std::sqrt(xyz_sq);

    // 避免除零
    if (xy_sq < 1e-10 || xyz_sq < 1e-10) {
        return Eigen::Matrix3d::Identity();
    }

    Eigen::Matrix3d J;
    // d(yaw)/d(x,y,z)
    J(0, 0) = -y / xy_sq;
    J(0, 1) = x / xy_sq;
    J(0, 2) = 0.0;

    // d(pitch)/d(x,y,z)
    const auto denom = xyz_sq * xy_norm;
    J(1, 0)          = -x * z / denom;
    J(1, 1)          = -y * z / denom;
    J(1, 2)          = xy_norm / xyz_sq;

    // d(distance)/d(x,y,z)
    J(2, 0) = x / xyz_norm;
    J(2, 1) = y / xyz_norm;
    J(2, 2) = z / xyz_norm;

    return J;
}

} // namespace math

// ============================================================================
// PnPSolver 实现
// ============================================================================

PnPSolver::PnPSolver(const SolverConfig& config)
    : config_(config) {
    // 构造相机内参矩阵
    camera_matrix_ =
        (cv::Mat_<double>(3, 3) << config_.fx, 0, config_.cx, 0, config_.fy, config_.cy, 0, 0, 1);

    // 构造畸变系数
    dist_coeffs_ =
        (cv::Mat_<double>(5, 1) << config_.dist_coeffs[0], config_.dist_coeffs[1],
         config_.dist_coeffs[2], config_.dist_coeffs[3], config_.dist_coeffs[4]);

    // 构造物点模板 (装甲板坐标系: x前, y左, z上)
    // 角点顺序: tl, tr, br, bl (与 Detection::corners 对应)
    const auto half_lb    = config_.lightbar_length / 2.0;
    const auto half_big   = config_.big_armor_width / 2.0;
    const auto half_small = config_.small_armor_width / 2.0;

    big_armor_points_ = {
        cv::Point3f(0, static_cast<float>(half_big), static_cast<float>(half_lb)),     // tl
        cv::Point3f(0, static_cast<float>(-half_big), static_cast<float>(half_lb)),    // tr
        cv::Point3f(0, static_cast<float>(-half_big), static_cast<float>(-half_lb)),   // br
        cv::Point3f(0, static_cast<float>(half_big), static_cast<float>(-half_lb))     // bl
    };

    small_armor_points_ = {
        cv::Point3f(0, static_cast<float>(half_small), static_cast<float>(half_lb)),   // tl
        cv::Point3f(0, static_cast<float>(-half_small), static_cast<float>(half_lb)),  // tr
        cv::Point3f(0, static_cast<float>(-half_small), static_cast<float>(-half_lb)), // br
        cv::Point3f(0, static_cast<float>(half_small), static_cast<float>(-half_lb))   // bl
    };

    spdlog::debug(
        "[PnPSolver] initialized with fx={:.1f}, fy={:.1f}, cx={:.1f}, cy={:.1f}", config_.fx,
        config_.fy, config_.cx, config_.cy);
}

std::optional<Measurement> PnPSolver::solve(const Detection& det) const noexcept {
    // 提取图像点
    std::vector<cv::Point2f> image_points(det.corners.begin(), det.corners.end());

    // 选择物点模板
    const auto& object_points = get_object_points(det.type);

    // PnP 解算
    cv::Vec3d rvec, tvec;
    const bool success = cv::solvePnP(
        object_points, image_points, camera_matrix_, dist_coeffs_, rvec, tvec, false,
        cv::SOLVEPNP_IPPE);

    if (!success) {
        spdlog::debug("[PnPSolver] solvePnP failed");
        return std::nullopt;
    }

    // 深度检查
    if (tvec[2] <= 0) {
        spdlog::debug("[PnPSolver] negative depth: {:.3f}", tvec[2]);
        return std::nullopt;
    }

    // 距离检查
    const Eigen::Vector3d position(tvec[0], tvec[1], tvec[2]);
    const double distance = position.norm();
    if (distance < 0.3 || distance > 15.0) {
        spdlog::debug("[PnPSolver] distance out of range: {:.2f}m", distance);
        return std::nullopt;
    }

    // 转换旋转向量到旋转矩阵
    cv::Mat R_cv;
    cv::Rodrigues(rvec, R_cv);
    Eigen::Matrix3d rotation;
    cv::cv2eigen(R_cv, rotation);

    // 提取欧拉角
    // 使用简化的 ZYX 欧拉角提取
    const double sin_pitch = std::clamp(-rotation(2, 0), -1.0, 1.0);
    double roll, pitch, yaw;
    if (std::abs(sin_pitch) < 0.9999999) {
        roll  = std::atan2(rotation(2, 1), rotation(2, 2));
        pitch = std::asin(sin_pitch);
        yaw   = std::atan2(rotation(1, 0), rotation(0, 0));
    } else {
        roll  = 0.0;
        pitch = sin_pitch > 0 ? M_PI_2 : -M_PI_2;
        yaw   = std::atan2(-rotation(0, 1), rotation(1, 1));
    }

    // 计算球坐标
    const auto ypd = math::xyz_to_ypd(position);

    // 计算位置协方差 (简化模型: 与距离平方成正比)
    const double sigma_base      = 0.01; // 1cm 基准误差
    const double sigma           = sigma_base * distance;
    Eigen::Matrix3d position_cov = Eigen::Matrix3d::Identity() * sigma * sigma;

    // 构造结果
    Measurement meas;
    meas.position     = position;
    meas.rotation     = rotation;
    meas.euler        = Eigen::Vector3d(roll, pitch, yaw);
    meas.yaw          = ypd[0];
    meas.pitch        = ypd[1];
    meas.distance     = ypd[2];
    meas.position_cov = position_cov;
    meas.timestamp_ns = det.timestamp_ns;
    meas.type         = det.type;
    meas.name         = det.name;
    meas.corners      = det.corners;

    return meas;
}

std::vector<std::optional<Measurement>>
    PnPSolver::solve_all(std::span<const Detection> detections) const noexcept {
    std::vector<std::optional<Measurement>> results;
    results.reserve(detections.size());

    for (const auto& det : detections) {
        results.push_back(solve(det));
    }

    return results;
}

std::array<cv::Point2f, 4> PnPSolver::reproject(const Measurement& meas) const noexcept {
    // 转换为 OpenCV 格式
    cv::Mat R_cv;
    cv::eigen2cv(meas.rotation, R_cv);
    cv::Vec3d rvec;
    cv::Rodrigues(R_cv, rvec);
    const cv::Vec3d tvec(meas.position[0], meas.position[1], meas.position[2]);

    // 重投影
    std::vector<cv::Point2f> image_points;
    const auto& object_points = get_object_points(meas.type);
    cv::projectPoints(object_points, rvec, tvec, camera_matrix_, dist_coeffs_, image_points);

    std::array<cv::Point2f, 4> result;
    std::copy(image_points.begin(), image_points.end(), result.begin());
    return result;
}

const std::vector<cv::Point3f>& PnPSolver::get_object_points(ArmorType type) const noexcept {
    return type == ArmorType::Big ? big_armor_points_ : small_armor_points_;
}

} // namespace fcs::L2
