#pragma once

#include "core/types.hpp"
#include "core/types_pnp.hpp"

#include <algorithm>
#include <cmath>
#include <expected>
#include <limits>
#include <numbers>
#include <optional>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

#include "camera_config.hpp"

namespace fcs::L2 {

// ============================================================================
// PnP Solver Error Types
// ============================================================================

enum class PnPError {
    InvalidDetection,
    SolveFailed,
};

[[nodiscard]] constexpr std::string_view pnp_error_str(PnPError e) noexcept {
    switch (e) {
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

        // Reject points behind camera or too close (avoids division by ~0 → NaN)
        constexpr double kMinDepth = 1e-3; // 1mm minimum depth
        if (Zc < T(kMinDepth)) {
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

    // ====================================================================
    // CERES SOLVER OWNERSHIP TRANSFER (RAII-compliant)
    // ====================================================================
    //
    // Ceres Problem takes ownership of CostFunction and LossFunction pointers
    // via Problem::AddResidualBlock(). The Problem destructor deletes them.
    //
    // Source: ceres-solver/include/ceres/problem.h
    //   "If set to TAKE_OWNERSHIP (default), the problem object will delete
    //    the corresponding object on destruction."
    //
    // Therefore:
    //   - We allocate with new (required by Ceres API)
    //   - We transfer ownership via AddResidualBlock()
    //   - We DO NOT delete (Ceres Problem destructor handles it)
    //   - NO LEAKS, NO DOUBLE-FREE
    //
    // This is NOT like tinympc - Ceres is RAII-compliant~ ♪
    // ====================================================================
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

[[nodiscard]] inline double armor_pitch_rad_for(ArmorName name) noexcept {
    constexpr double kTiltRad = 15.0 * std::numbers::pi / 180.0;
    return (name == ArmorName::Outpost) ? -kTiltRad : kTiltRad;
}

[[nodiscard]] inline Eigen::Matrix3d armor_pitch_rotation(double armor_pitch_rad) noexcept {
    const double cp = std::cos(armor_pitch_rad);
    const double sp = std::sin(armor_pitch_rad);
    Eigen::Matrix3d R_pitch;
    R_pitch << cp, 0.0, sp, 0.0, 1.0, 0.0, -sp, 0.0, cp;
    return R_pitch;
}

[[nodiscard]] inline Eigen::Matrix3d yaw_rotation(double yaw) noexcept {
    const double cy = std::cos(yaw);
    const double sy = std::sin(yaw);
    Eigen::Matrix3d R_yaw;
    R_yaw << cy, -sy, 0.0, sy, cy, 0.0, 0.0, 0.0, 1.0;
    return R_yaw;
}

[[nodiscard]] inline double extract_yaw_from_rotation(
    const Eigen::Matrix3d& R_cam_armor, const Eigen::Matrix3d& R_imu_cam) noexcept {
    const Eigen::Matrix3d R_imu_armor = R_imu_cam * R_cam_armor;

    // Clamp to [-1, 1] — noisy PnP can produce slightly non-orthogonal R.
    const double r01   = std::clamp(-R_imu_armor(0, 1), -1.0, 1.0);
    const double r11   = std::clamp(R_imu_armor(1, 1), -1.0, 1.0);
    const double yaw_s = std::asin(r01);
    const double yaw_c = std::acos(r11);

    if (std::abs(yaw_s) > 1e-5) {
        return (yaw_s > 0.0) ? yaw_c : -yaw_c;
    }
    return (r11 > 0.0) ? 0.0 : std::numbers::pi;
}

// ============================================================================
// AT Legacy PnP Solver
// ============================================================================

/// PnP solver for armor pose estimation
class PnPSolver {
public:
    using PoseResult = std::expected<CameraArmorMeasurement, PnPError>;

    struct PosePrior {
        cv::Vec3d rvec{};
        cv::Vec3d tvec{};
        double hint_cost{0.0};
        int armor_id{-1};
    };

    /// Construct with camera parameters — fully initialized, no separate init() needed.
    explicit PnPSolver(const CameraConfig& config) {
        cv::eigen2cv(config.camera_matrix, camera_matrix_);
        cv::eigen2cv(config.distort_coefficient, dist_coeffs_);

        // Pre-compute image center for distance calculation
        image_center_ = cv::Point2f(
            config.width / 2.0, // cx
            config.height / 2.0 // cy
        );

        build_model_points();
    }

    /// Solve PnP with yaw-only BA refinement (4DOF optimization)
    /// First solves standard PnP, then refines yaw using Ceres BA
    /// @param detection Armor detection with 4 corners
    /// @param R_imu_cam Rotation from IMU/gimbal to camera frame
    /// @param timestamp_ns Timestamp for the measurement
    /// @return Measurement with refined 3D pose, or error
    [[nodiscard]] PoseResult solve_with_ba(
        const ArmorDetection& detection, const Eigen::Matrix3d& R_imu_cam, uint64_t timestamp_ns,
        const std::vector<PosePrior>& pose_priors = {}) const {
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

        // Bail out on degenerate PnP (NaN/Inf from IPPE with bad geometry)
        {
            const double* rd = rvec.ptr<double>();
            const double* td = tvec.ptr<double>();
            for (int i = 0; i < 3; ++i) {
                if (!std::isfinite(rd[i]) || !std::isfinite(td[i])) {
                    return std::unexpected(PnPError::SolveFailed);
                }
            }
        }

        // R_cam_imu = R_imu_cam.transpose()
        Eigen::Matrix3d R_cam_imu = R_imu_cam.transpose();

        // R_imu_armor = R_imu_cam * R_cam_armor
        Eigen::Matrix3d R_imu_armor = R_imu_cam * R_cam_armor;

        // Step 3: Extract initial yaw from R_imu_armor
        // Yaw is rotation around Z axis in IMU frame
        // Clamp to [-1, 1] — noisy PnP can produce slightly non-orthogonal R
        const double r01    = std::clamp(-R_imu_armor(0, 1), -1.0, 1.0);
        const double r11    = std::clamp(R_imu_armor(1, 1), -1.0, 1.0);
        double theta_by_sin = std::asin(r01);
        double theta_by_cos = std::acos(r11);
        double yaw_init     = 0.0;

        if (std::abs(theta_by_sin) > 1e-5) {
            yaw_init = (theta_by_sin > 0.0) ? theta_by_cos : -theta_by_cos;
        } else {
            yaw_init = (r11 > 0.0) ? 0.0 : std::numbers::pi;
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

        // ====================================================================
        // CERES SOLVER OWNERSHIP TRANSFER (RAII-compliant)
        // ====================================================================
        //
        // Ceres Problem takes ownership of CostFunction and LossFunction pointers.
        // See YawOnlyReprojError::Create() above for full provenance.
        //
        // We allocate with new (required by Ceres API), transfer ownership
        // via AddResidualBlock(), and let Ceres Problem destructor handle cleanup.
        //
        // NO manual delete needed - Ceres is RAII-compliant~ ♪
        // ====================================================================
        ceres::Problem problem;
        // Huber loss for robustness (ownership transferred to problem)
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

        return make_camera_measurement(detection, rvec, tvec, timestamp_ns, input->dist_to_center);
    }

    /// Solve PnP with BA for multiple detections
    [[nodiscard]] std::vector<CameraArmorMeasurement> solve_batch_with_ba(
        const std::vector<ArmorDetection>& detections, const Eigen::Matrix3d& R_imu_cam,
        uint64_t timestamp_ns) const {
        std::vector<CameraArmorMeasurement> measurements;
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
        make_pnp_input(const ArmorDetection& detection) const {
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
        constexpr double SMALL_ARMOR_WIDTH  = 135.0 / 1000.0;
        constexpr double SMALL_ARMOR_HEIGHT = 55.0 / 1000.0;
        constexpr double LARGE_ARMOR_WIDTH  = 230.0 / 1000.0;
        constexpr double LARGE_ARMOR_HEIGHT = 55.0 / 1000.0;

        // Small armor: 135mm x 55mm
        const float sw      = static_cast<float>(SMALL_ARMOR_WIDTH / 2.0);
        const float sh      = static_cast<float>(SMALL_ARMOR_HEIGHT / 2.0);
        small_armor_points_ = {
            cv::Point3f(0, sw, sh),   // Top-left
            cv::Point3f(0, -sw, sh),  // Top-right
            cv::Point3f(0, -sw, -sh), // Bottom-right
            cv::Point3f(0, sw, -sh)   // Bottom-left
        };

        // Large armor: 230mm x 55mm
        const float lw      = static_cast<float>(LARGE_ARMOR_WIDTH / 2.0);
        const float lh      = static_cast<float>(LARGE_ARMOR_HEIGHT / 2.0);
        large_armor_points_ = {
            cv::Point3f(0, lw, lh),   // Top-left
            cv::Point3f(0, -lw, lh),  // Top-right
            cv::Point3f(0, -lw, -lh), // Bottom-right
            cv::Point3f(0, lw, -lh)   // Bottom-left
        };
    }

private:
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    cv::Point2f image_center_;

    std::vector<cv::Point3f> small_armor_points_;
    std::vector<cv::Point3f> large_armor_points_;
};

} // namespace fcs::L2
