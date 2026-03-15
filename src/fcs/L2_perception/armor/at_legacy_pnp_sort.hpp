#pragma once

#include "core/types.hpp"

#include <Eigen/Core>
#include <array>
#include <numbers>
#include <opencv2/core.hpp>

namespace fcs::L2 {

// ============================================================================
// PnP Solution Sorting (Geometric Heuristic for IPPE Dual Solutions)
// 完全移植自 at_vision (armor_detector/armor_pose_estimator.cpp:sortPnPResult)
// ============================================================================

/// PnP solution disambiguator for IPPE dual solutions
///
/// 移植自 at_vision armor_pose_estimator.cpp:102-154
/// 核心逻辑：
/// 1. 计算重投影误差，检查提前终止条件
/// 2. 计算图像倾斜角 (from light.axis)
/// 3. 几何启发式: (angle > 0 && yaw1 > 0 && yaw2 < 0) || (angle < 0 && yaw1 < 0 && yaw2 > 0)
class PnPSolutionSorter {
public:
    /// Sort IPPE dual solutions using geometric heuristic
    /// @param rvecs Rotation vectors from solvePnP (vector with 2 solutions)
    /// @param tvecs Translation vectors from solvePnP (vector with 2 solutions)
    /// @param R_imu_cam Rotation from IMU to camera frame
    /// @param detection Armor detection (for extracting light axes)
    /// @param K Camera intrinsic matrix
    /// @param obj_points 3D model points (small/large armor)
    /// @return true if swapped to second solution, false otherwise
    [[nodiscard]] static bool sortSolutions(
        std::vector<cv::Mat>& rvecs, std::vector<cv::Mat>& tvecs, const Eigen::Matrix3d& R_imu_cam,
        const AtLegacyArmorDetection& detection, const Eigen::Matrix3d& K,
        const std::vector<cv::Point3f>& obj_points);

private:
    /// Calculate reprojection error for a PnP solution
    /// @param rvec Rotation vector
    /// @param tvec Translation vector
    /// @param obj_points 3D model points
    /// @param img_points 2D image points
    /// @param K Camera intrinsic matrix
    /// @return Reprojection error (RMS)
    [[nodiscard]] static double calculateReprojectionError(
        const cv::Mat& rvec, const cv::Mat& tvec, const std::vector<cv::Point3f>& obj_points,
        const std::vector<cv::Point2f>& img_points, const Eigen::Matrix3d& K);

    /// Convert rotation matrix to RPY angles
    /// @param R Rotation matrix
    /// @return {roll, pitch, yaw} in radians
    [[nodiscard]] static std::array<double, 3>
        rotationMatrixToRPY(const Eigen::Matrix3d& R) noexcept;

    // Constants from at_vision (armor_pose_estimator.cpp:104, 131)
    static constexpr double REPROJECTION_ERROR_RATIO_THRESHOLD =
        3.0;                                                 ///< Reprojection error ratio threshold
    static constexpr double ROLL_ANGLE_THRESHOLD_DEG = 10.0; ///< Roll angle threshold (degrees)
    static constexpr double ROLL_ANGLE_THRESHOLD_RAD = ROLL_ANGLE_THRESHOLD_DEG
                                                     * std::numbers::pi_v<double>
                                                     / 180.0; ///< Roll angle threshold (radians)
};

} // namespace fcs::L2
