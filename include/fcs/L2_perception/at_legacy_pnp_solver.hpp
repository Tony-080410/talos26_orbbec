#pragma once

#include "core/types.hpp"

#include <cmath>
#include <expected>
#include <numbers>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

#include "fcs/camera_config.hpp"

namespace fcs {

// ============================================================================
// PnP Solver Error Types
// ============================================================================

enum class PnPError {
    NotInitialized,
    InvalidDetection,
    SolveFailed,
};

[[nodiscard]] constexpr std::string_view pnp_error_str(PnPError e) noexcept {
    switch (e) {
    case PnPError::NotInitialized: return "Not initialized";
    case PnPError::InvalidDetection: return "Invalid detection";
    case PnPError::SolveFailed: return "Solve failed";
    }
    return "Unknown error";
}

// ============================================================================
// Yaw-Only Reprojection Error (4DOF Optimization)
// ============================================================================

/// Ceres cost function for yaw-only bundle adjustment
/// Optimizes only yaw angle in IMU/gimbal frame while keeping pitch fixed
/// Based on FYT's implementation
struct YawOnlyReprojError {
    YawOnlyReprojError(
        const Eigen::Vector3d& Pw, const Eigen::Vector2d& uv, const Eigen::Matrix3d& R_cam_imu,
        const Eigen::Matrix3d& R_pitch, const Eigen::Vector3d& t_cam_armor,
        const Eigen::Matrix3d& K)
        : Pw_(Pw)
        , uv_(uv)
        , R_cam_imu_(R_cam_imu)
        , R_pitch_(R_pitch)
        , t_(t_cam_armor)
        , K_(K) {}

    template <typename T>
    bool operator()(const T* const yaw, T* residuals) const {
        // R_yaw: rotation around Z axis in IMU frame
        const T cy = ceres::cos(yaw[0]);
        const T sy = ceres::sin(yaw[0]);
        Eigen::Matrix<T, 3, 3> R_yaw;
        R_yaw << cy, -sy, T(0), sy, cy, T(0), T(0), T(0), T(1);

        // R = R_cam_imu * R_yaw * R_pitch
        Eigen::Matrix<T, 3, 3> R_cam_imu = R_cam_imu_.template cast<T>();
        Eigen::Matrix<T, 3, 3> R_pitch   = R_pitch_.template cast<T>();
        Eigen::Matrix<T, 3, 3> R         = R_cam_imu * R_yaw * R_pitch;

        // 3D model point (armor frame)
        Eigen::Matrix<T, 3, 1> Pw;
        Pw << T(Pw_.x()), T(Pw_.y()), T(Pw_.z());

        // Translation (fixed, from initial PnP)
        Eigen::Matrix<T, 3, 1> t;
        t << T(t_.x()), T(t_.y()), T(t_.z());

        // Transform to camera frame
        Eigen::Matrix<T, 3, 1> Pc = R * Pw + t;
        const T& Xc               = Pc(0);
        const T& Yc               = Pc(1);
        const T& Zc               = Pc(2);

        if (Zc <= T(0)) {
            // residuals[0] = T(0);
            // residuals[1] = T(0);
            return false;
        }

        // Normalized image plane coordinates
        Eigen::Matrix<T, 3, 1> p_norm;
        p_norm << Xc / Zc, Yc / Zc, T(1);

        // Pixel coordinates (no distortion, consistent with FYT BA)
        Eigen::Matrix<T, 3, 3> K     = K_.template cast<T>();
        Eigen::Matrix<T, 3, 1> p_img = K * p_norm;

        residuals[0] = p_img(0) - T(uv_.x());
        residuals[1] = p_img(1) - T(uv_.y());
        return true;
    }

    static ceres::CostFunction* Create(
        const Eigen::Vector3d& Pw, const Eigen::Vector2d& uv, const Eigen::Matrix3d& R_cam_imu,
        const Eigen::Matrix3d& R_pitch, const Eigen::Vector3d& t_cam_armor,
        const Eigen::Matrix3d& K) {
        return new ceres::AutoDiffCostFunction<YawOnlyReprojError, 2, 1>(
            new YawOnlyReprojError(Pw, uv, R_cam_imu, R_pitch, t_cam_armor, K));
    }

private:
    Eigen::Vector3d Pw_;
    Eigen::Vector2d uv_;
    Eigen::Matrix3d R_cam_imu_;
    Eigen::Matrix3d R_pitch_;
    Eigen::Vector3d t_;
    Eigen::Matrix3d K_;
};

// ============================================================================
// AT Legacy PnP Solver
// ============================================================================

/// PnP solver for armor pose estimation
class AtLegacyPnPSolver {
public:
    using PoseResult = std::expected<AtLegacyArmorMeasurement, PnPError>;

    AtLegacyPnPSolver() = default;

    /// Initialize camera parameters
    /// @param config camera config
    bool init(const CameraConfig& config) {
        cv::eigen2cv(config.camera_matrix, camera_matrix_);
        cv::eigen2cv(config.distort_coefficient, dist_coeffs_);

        // Pre-compute image center for distance calculation
        image_center_ = cv::Point2f(
            config.width / 2.0, // cx
            config.height / 2.0 // cy
        );

        build_model_points();
        initialized_ = true;
        return true;
    }

    /// Solve PnP with yaw-only BA refinement (4DOF optimization)
    /// First solves standard PnP, then refines yaw using Ceres BA
    /// @param detection Armor detection with 4 corners
    /// @param R_imu_cam Rotation from IMU/gimbal to camera frame
    /// @param timestamp_ns Timestamp for the measurement
    /// @return Measurement with refined 3D pose, or error
    [[nodiscard]] PoseResult solve_with_ba(
        const AtLegacyArmorDetection& detection, const Eigen::Matrix3d& R_imu_cam,
        uint64_t timestamp_ns) const {
        auto input = make_pnp_input(detection);
        if (!input) {
            return std::unexpected(input.error());
        }

        // Step 1: Initial 6DoF PnP using IPPE
        cv::Mat rvec, tvec;
        bool success = cv::solvePnP(
            input->obj_points, input->img_points, camera_matrix_, dist_coeffs_, rvec, tvec, false,
            cv::SOLVEPNP_IPPE);

        if (!success) {
            return std::unexpected(PnPError::SolveFailed);
        }

        // Step 2: Convert rvec/tvec to Eigen
        cv::Mat R_cv;
        cv::Rodrigues(rvec, R_cv);

        Eigen::Matrix3d R_cam_armor;
        cv::cv2eigen(R_cv, R_cam_armor);

        Eigen::Vector3d t_cam_armor(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

        // R_cam_imu = R_imu_cam.transpose()
        Eigen::Matrix3d R_cam_imu = R_imu_cam.transpose();

        // R_imu_armor = R_imu_cam * R_cam_armor
        Eigen::Matrix3d R_imu_armor = R_imu_cam * R_cam_armor;

        // Step 3: Extract initial yaw from R_imu_armor
        // Yaw is rotation around Z axis in IMU frame
        double theta_by_sin = std::asin(-R_imu_armor(0, 1));
        double theta_by_cos = std::acos(R_imu_armor(1, 1));
        double yaw_init     = 0.0;

        if (std::abs(theta_by_sin) > 1e-5) {
            yaw_init = (theta_by_sin > 0.0) ? theta_by_cos : -theta_by_cos;
        } else {
            yaw_init = (R_imu_armor(1, 1) > 0.0) ? 0.0 : std::numbers::pi;
        }

        // Step 4: Armor pitch constraint (15 deg, negative for outpost)
        constexpr double FIFTEEN_DEG_RAD = 15.0 * std::numbers::pi / 180.0;
        double armor_pitch_rad           = FIFTEEN_DEG_RAD;
        if (detection.name == ArmorName::Outpost) {
            armor_pitch_rad = -FIFTEEN_DEG_RAD;
        }

        Eigen::Matrix3d R_pitch;
        {
            double cp = std::cos(armor_pitch_rad);
            double sp = std::sin(armor_pitch_rad);
            R_pitch << cp, 0.0, sp, 0.0, 1.0, 0.0, -sp, 0.0, cp;
        }

        // Step 5: Camera intrinsic matrix K (without distortion for BA)
        Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
        K(0, 0)           = camera_matrix_.at<double>(0, 0);
        K(1, 1)           = camera_matrix_.at<double>(1, 1);
        K(0, 2)           = camera_matrix_.at<double>(0, 2);
        K(1, 2)           = camera_matrix_.at<double>(1, 2);

        // Step 6: Prepare 3D model points and 2D observations
        std::vector<Eigen::Vector3d> obj_pts_eigen;
        obj_pts_eigen.reserve(input->obj_points.size());
        for (const auto& p : input->obj_points) {
            obj_pts_eigen.emplace_back(p.x, p.y, p.z);
        }

        std::vector<Eigen::Vector2d> img_pts_eigen;
        img_pts_eigen.reserve(input->img_points.size());
        for (const auto& p : input->img_points) {
            img_pts_eigen.emplace_back(p.x, p.y);
        }

        // Step 7: Build Ceres problem - optimize yaw only
        double yaw = yaw_init;

        ceres::Problem problem;
        // Huber loss for robustness
        auto* loss_function = new ceres::HuberLoss(1.0);

        for (size_t i = 0; i < obj_pts_eigen.size(); ++i) {
            ceres::CostFunction* cost = YawOnlyReprojError::Create(
                obj_pts_eigen[i], img_pts_eigen[i], R_cam_imu, R_pitch, t_cam_armor, K);
            problem.AddResidualBlock(cost, loss_function, &yaw);
        }

        // Constrain yaw to [-pi, pi]
        problem.SetParameterLowerBound(&yaw, 0, -std::numbers::pi);
        problem.SetParameterUpperBound(&yaw, 0, std::numbers::pi);

        ceres::Solver::Options options;
        options.linear_solver_type           = ceres::DENSE_QR;
        options.max_num_iterations           = 20;
        options.minimizer_progress_to_stdout = false;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // Step 8: Reconstruct refined rotation if optimization succeeded
        if (std::isfinite(yaw)) {
            double cy = std::cos(yaw);
            double sy = std::sin(yaw);
            Eigen::Matrix3d R_yaw;
            R_yaw << cy, -sy, 0.0, sy, cy, 0.0, 0.0, 0.0, 1.0;

            Eigen::Matrix3d R_cam_armor_refined = R_cam_imu * R_yaw * R_pitch;

            // Write back to rvec (tvec unchanged)
            cv::Mat R_refined_cv;
            cv::eigen2cv(R_cam_armor_refined, R_refined_cv);
            cv::Rodrigues(R_refined_cv, rvec);
        }
        // If optimization failed, use original PnP result

        return AtLegacyArmorMeasurement(detection, rvec, tvec, timestamp_ns, input->dist_to_center);
    }

    /// Solve PnP with BA for multiple detections
    [[nodiscard]] std::vector<AtLegacyArmorMeasurement> solve_batch_with_ba(
        const std::vector<AtLegacyArmorDetection>& detections, const Eigen::Matrix3d& R_imu_cam,
        uint64_t timestamp_ns) const {
        std::vector<AtLegacyArmorMeasurement> measurements;
        measurements.reserve(detections.size());

        for (const auto& det : detections) {
            auto result = solve_with_ba(det, R_imu_cam, timestamp_ns);
            if (result) {
                measurements.push_back(std::move(*result));
            }
        }

        return measurements;
    }

private:
    struct PnPInput {
        const std::vector<cv::Point3f>& obj_points;
        std::vector<cv::Point2f> img_points;
        float dist_to_center;
    };

    [[nodiscard]] std::expected<PnPInput, PnPError>
        make_pnp_input(const AtLegacyArmorDetection& detection) const {
        if (!initialized_) {
            return std::unexpected(PnPError::NotInitialized);
        }

        auto img_points = detection.image_points();
        if (img_points.size() != 4) {
            return std::unexpected(PnPError::InvalidDetection);
        }

        const auto& obj_points =
            (detection.type == ArmorType::Large) ? large_armor_points_ : small_armor_points_;

        return PnPInput{
            .obj_points     = obj_points,
            .img_points     = std::move(img_points),
            .dist_to_center = static_cast<float>(cv::norm(detection.center() - image_center_)),
        };
    }

    void build_model_points() {
        // Small armor: 135mm x 55mm
        const float sw      = static_cast<float>(AT_LEGACY_SMALL_ARMOR_WIDTH / 2.0);
        const float sh      = static_cast<float>(AT_LEGACY_SMALL_ARMOR_HEIGHT / 2.0);
        small_armor_points_ = {
            cv::Point3f(0, sw, sh),   // Top-left
            cv::Point3f(0, -sw, sh),  // Top-right
            cv::Point3f(0, -sw, -sh), // Bottom-right
            cv::Point3f(0, sw, -sh)   // Bottom-left
        };

        // Large armor: 230mm x 55mm
        const float lw      = static_cast<float>(AT_LEGACY_LARGE_ARMOR_WIDTH / 2.0);
        const float lh      = static_cast<float>(AT_LEGACY_LARGE_ARMOR_HEIGHT / 2.0);
        large_armor_points_ = {
            cv::Point3f(0, lw, lh),   // Top-left
            cv::Point3f(0, -lw, lh),  // Top-right
            cv::Point3f(0, -lw, -lh), // Bottom-right
            cv::Point3f(0, lw, -lh)   // Bottom-left
        };
    }

private:
    bool initialized_ = false;
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    cv::Point2f image_center_;

    std::vector<cv::Point3f> small_armor_points_;
    std::vector<cv::Point3f> large_armor_points_;
};

} // namespace fcs
