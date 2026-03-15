#include "at_legacy_pnp_sort.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <opencv2/calib3d.hpp>
#include <spdlog/spdlog.h>

namespace fcs::L2 {

// ============================================================================
// PnPSolutionSorter Implementation (移植自 at_vision armor_pose_estimator.cpp)
// ============================================================================

bool PnPSolutionSorter::sortSolutions(
    std::vector<cv::Mat>& rvecs, std::vector<cv::Mat>& tvecs, const Eigen::Matrix3d& R_imu_cam,
    const AtLegacyArmorDetection& detection, const Eigen::Matrix3d& K,
    const std::vector<cv::Point3f>& obj_points) {

    // 从 at_vision armor_pose_estimator.cpp:102-154
    // 完全移植，不添加任何额外语义

    // 获取这两个解
    cv::Mat& rvec1 = rvecs.at(0);
    cv::Mat& tvec1 = tvecs.at(0);
    cv::Mat& rvec2 = rvecs.at(1);
    cv::Mat& tvec2 = tvecs.at(1);

    // 将旋转向量转换为旋转矩阵
    cv::Mat R1_cv, R2_cv;
    cv::Rodrigues(rvec1, R1_cv);
    cv::Rodrigues(rvec2, R2_cv);

    // 转换为Eigen矩阵（使用cv::cv2eigen进行零拷贝转换）
    Eigen::Matrix3d R1, R2;
    cv::cv2eigen(R1_cv, R1);
    cv::cv2eigen(R2_cv, R2);

    // 计算云台系下装甲板的RPY角
    auto rpy1 = rotationMatrixToRPY(R_imu_cam * R1);
    auto rpy2 = rotationMatrixToRPY(R_imu_cam * R2);

    // 计算重投影误差
    std::vector<cv::Point2f> img_points = {
        detection.corners[0], detection.corners[1], detection.corners[2], detection.corners[3]};

    double error1 = calculateReprojectionError(rvec1, tvec1, obj_points, img_points, K);
    double error2 = calculateReprojectionError(rvec2, tvec2, obj_points, img_points, K);

    // 提前终止条件 (from at_vision line 132-135)
    if ((error2 / error1 > REPROJECTION_ERROR_RATIO_THRESHOLD)
        || (std::abs(rpy1[0]) > ROLL_ANGLE_THRESHOLD_RAD)
        || (std::abs(rpy2[0]) > ROLL_ANGLE_THRESHOLD_RAD)) {
        return false;
    }

    // 计算灯条在图像中的倾斜角度 (from at_vision line 138-144)
    cv::Point2f left_axis  = detection.corners[3] - detection.corners[0]; // BL - TL
    cv::Point2f right_axis = detection.corners[2] - detection.corners[1]; // BR - TR

    // Normalize axes
    float left_norm  = cv::norm(left_axis);
    float right_norm = cv::norm(right_axis);
    if (left_norm > 1e-6f) {
        left_axis = left_axis / left_norm;
    }
    if (right_norm > 1e-6f) {
        right_axis = right_axis / right_norm;
    }

    double l_angle = std::atan2(static_cast<double>(left_axis.y), static_cast<double>(left_axis.x))
                   * 180.0 / std::numbers::pi;
    double r_angle =
        std::atan2(static_cast<double>(right_axis.y), static_cast<double>(right_axis.x)) * 180.0
        / std::numbers::pi;
    double angle = (l_angle + r_angle) / 2.0;
    angle += 90.0;

    // Outpost 特殊处理 (from at_vision line 143-144)
    if (detection.name == ArmorName::Outpost) {
        angle = -angle;
    }

    // 根据倾斜角度选择解 (from at_vision line 146-153)
    // 如果装甲板左倾（angle > 0），选择Yaw为负的解
    // 如果装甲板右倾（angle < 0），选择Yaw为正的解
    if ((angle > 0 && rpy1[2] > 0 && rpy2[2] < 0) || (angle < 0 && rpy1[2] < 0 && rpy2[2] > 0)) {
        std::swap(rvec1, rvec2);
        std::swap(tvec1, tvec2);
        SPDLOG_DEBUG(
            "PnP Solution 2 Selected (angle={:.3f}, yaw1={:.3f}, yaw2={:.3f})", angle, rpy1[2],
            rpy2[2]);
        return true;
    }

    return false;
}

double PnPSolutionSorter::calculateReprojectionError(
    const cv::Mat& rvec, const cv::Mat& tvec, const std::vector<cv::Point3f>& obj_points,
    const std::vector<cv::Point2f>& img_points, const Eigen::Matrix3d& K) {

    // Project 3D points to 2D
    std::vector<cv::Point2f> projected_points;
    cv::Mat K_cv;
    cv::eigen2cv(K, K_cv);

    cv::projectPoints(obj_points, rvec, tvec, K_cv, cv::noArray(), projected_points);

    // Calculate RMS error
    double total_error = 0.0;
    for (size_t i = 0; i < img_points.size(); i++) {
        double dx = static_cast<double>(projected_points[i].x - img_points[i].x);
        double dy = static_cast<double>(projected_points[i].y - img_points[i].y);
        total_error += dx * dx + dy * dy;
    }

    return std::sqrt(total_error / static_cast<double>(img_points.size()));
}

std::array<double, 3> PnPSolutionSorter::rotationMatrixToRPY(const Eigen::Matrix3d& R) noexcept {
    // Convert rotation matrix to roll-pitch-yaw angles
    // Using the ZYX Euler angle convention (yaw-pitch-roll)
    double roll  = std::atan2(R(2, 1), R(2, 2));
    double pitch = std::atan2(-R(2, 0), std::sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0)));
    double yaw   = std::atan2(R(1, 0), R(0, 0));

    return {roll, pitch, yaw};
}

} // namespace fcs::L2
