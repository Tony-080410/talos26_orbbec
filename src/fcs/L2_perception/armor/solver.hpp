#pragma once

#include "core/types.hpp"
#include "core/types_pnp.hpp"

#include <algorithm>
#include <cmath>
#include <expected>
#include <limits>
#include <numbers>
#include <opencv2/core/matx.hpp>
#include <optional>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

#include "camera_config.hpp"
#include "euler.hpp"

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
// Constrained Reprojection Error (4DOF Optimization)
// ============================================================================

/// Ceres cost function for constrained reprojection refinement.
/// Optimizes armor yaw in the odom/imu frame plus camera-frame translation
/// parameterized as optical-frame yaw/pitch/distance. Armor pitch stays fixed.
struct ConstrainedReprojError {
    ConstrainedReprojError(
        const Eigen::Vector3d& Pw, const Eigen::Vector2d& uv, const Eigen::Matrix3d& R_cam_imu,
        const Eigen::Matrix3d& R_pitch)
        : Pw_(Pw)
        , uv_(uv)
        , R_cam_imu_(R_cam_imu)
        , R_pitch_(R_pitch) {}

    template <typename T>
    bool operator()(const T* const yaw, const T* const translation_ypd, T* residuals) const {
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

        // Translation in camera optical frame, parameterized as:
        // yaw = atan2(x, z), pitch = atan2(y, hypot(x, z)), distance = ||t||.
        const T t_yaw      = translation_ypd[0];
        const T t_pitch    = translation_ypd[1];
        const T t_distance = translation_ypd[2];
        const T cp         = ceres::cos(t_pitch);
        Eigen::Matrix<T, 3, 1> t;
        t << t_distance * cp * ceres::sin(t_yaw), t_distance * ceres::sin(t_pitch),
            t_distance * cp * ceres::cos(t_yaw);

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
        residuals[0] = Xc / Zc - T(uv_.x());
        residuals[1] = Yc / Zc - T(uv_.y());
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
        const Eigen::Matrix3d& R_pitch) {
        return new ceres::AutoDiffCostFunction<ConstrainedReprojError, 2, 1, 3>(
            new ConstrainedReprojError(Pw, uv, R_cam_imu, R_pitch));
    }

private:
    Eigen::Vector3d Pw_;
    Eigen::Vector2d uv_;
    Eigen::Matrix3d R_cam_imu_;
    Eigen::Matrix3d R_pitch_;
};

[[nodiscard]] inline Eigen::Vector3d
    camera_translation_to_ypd(const Eigen::Vector3d& translation) noexcept {
    const double horizontal = std::hypot(translation.x(), translation.z());
    return {
        std::atan2(translation.x(), translation.z()),
        std::atan2(translation.y(), horizontal),
        translation.norm(),
    };
}

[[nodiscard]] inline Eigen::Vector3d
    camera_ypd_to_translation(const Eigen::Vector3d& translation_ypd) noexcept {
    const double yaw      = translation_ypd.x();
    const double pitch    = translation_ypd.y();
    const double distance = translation_ypd.z();
    const double cp       = std::cos(pitch);

    return {
        distance * cp * std::sin(yaw),
        distance * std::sin(pitch),
        distance * cp * std::cos(yaw),
    };
}

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

    /// Solve armor pose with prior-seeded constrained reprojection refinement.
    /// OpenCV PnP is used only to enter a plausible local basin; the final solve
    /// happens on the physical 4DoF manifold (yaw + camera-frame translation).
    /// @param detection Armor detection with 4 corners
    /// @param R_imu_cam Rotation from camera frame into the yaw frame (imu/odom)
    /// @param timestamp_ns Timestamp for the measurement
    /// @return Measurement with refined 3D pose, or error
    [[nodiscard]] PoseResult solve_with_ba(
        const ArmorDetection& detection, const Eigen::Matrix3d& R_imu_cam, uint64_t timestamp_ns,
        const std::vector<PosePrior>& pose_priors = {}) const {
        auto input = make_pnp_input(detection);
        if (!input) {
            return std::unexpected(input.error());
        }

        // Step 1: Initial 6DoF pose estimate in the undistorted normalized plane
        const auto initial_pose = solve_initial_pose(*input, pose_priors);
        if (!initial_pose.has_value()) {
            return std::unexpected(PnPError::SolveFailed);
        }
        cv::Mat rvec = initial_pose->rvec.clone();
        cv::Mat tvec = initial_pose->tvec.clone();

        // Step 2: Convert rvec/tvec to Eigen.
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

        // Step 3: Prepare the constrained pose model.
        const Eigen::Matrix3d R_cam_imu = R_imu_cam.transpose();
        const double armor_pitch_rad    = armor_pitch_rad_for(detection.name);
        const Eigen::Matrix3d R_pitch   = armor_pitch_rotation(armor_pitch_rad);
        double yaw                      = extract_yaw_from_rotation(R_cam_armor, R_imu_cam);
        const Eigen::Vector3d initial_translation_ypd = camera_translation_to_ypd(t_cam_armor);
        double translation_ypd[3]                     = {
            initial_translation_ypd.x(),
            initial_translation_ypd.y(),
            initial_translation_ypd.z(),
        };

        // Step 4: Prepare 3D model points and normalized 2D observations.
        std::vector<Eigen::Vector3d> obj_pts_eigen;
        obj_pts_eigen.reserve(input->obj_points.size());
        for (const auto& p : input->obj_points) {
            obj_pts_eigen.emplace_back(p.x, p.y, p.z);
        }

        std::vector<Eigen::Vector2d> img_pts_eigen;
        img_pts_eigen.reserve(input->norm_points.size());
        for (const auto& p : input->norm_points) {
            img_pts_eigen.emplace_back(p.x, p.y);
        }

        // Step 5: Build Ceres problem - optimize constrained pose on the normalized plane.

        // ====================================================================
        // CERES SOLVER OWNERSHIP TRANSFER (RAII-compliant)
        // ====================================================================
        //
        // Ceres Problem takes ownership of CostFunction and LossFunction pointers.
        // See ConstrainedReprojError::Create() above for full provenance.
        //
        // We allocate with new (required by Ceres API), transfer ownership
        // via AddResidualBlock(), and let Ceres Problem destructor handle cleanup.
        //
        // NO manual delete needed - Ceres is RAII-compliant~ ♪
        // ====================================================================
        ceres::Problem problem;
        // Huber loss for robustness (ownership transferred to problem)
        auto* loss_function = new ceres::HuberLoss(0.5);

        for (size_t i = 0; i < obj_pts_eigen.size(); ++i) {
            ceres::CostFunction* cost = ConstrainedReprojError::Create(
                obj_pts_eigen[i], img_pts_eigen[i], R_cam_imu, R_pitch);
            problem.AddResidualBlock(cost, nullptr, &yaw, translation_ypd);
        }

        // Constrain armor yaw to [-pi, pi] and optical bearing to the front camera hemisphere.
        problem.SetParameterLowerBound(&yaw, 0, -std::numbers::pi);
        problem.SetParameterUpperBound(&yaw, 0, std::numbers::pi);
        constexpr double kBearingLimit = std::numbers::pi / 2.0 - 1e-6;
        problem.SetParameterLowerBound(translation_ypd, 0, -kBearingLimit);
        problem.SetParameterUpperBound(translation_ypd, 0, kBearingLimit);
        problem.SetParameterLowerBound(translation_ypd, 1, -kBearingLimit);
        problem.SetParameterUpperBound(translation_ypd, 1, kBearingLimit);
        if (detection.name == ArmorName::Outpost) {
            if (initial_pose.has_value() && initial_pose->prior_armor_yaw.has_value()) {
                auto pr = initial_pose->prior_armor_yaw.value();
                problem.SetParameterLowerBound(&yaw, 0, std::max(-std::numbers::pi, pr - 0.1256));
                problem.SetParameterUpperBound(&yaw, 0, std::min(pr + 0.1256, std::numbers::pi));
            }
            auto original_norm = initial_pose.value().prior_distance_m
                                   ? initial_pose.value().prior_distance_m.value()
                                   : cv::norm(tvec);
            problem.SetParameterLowerBound(translation_ypd, 2, std::max(0.0, original_norm - 0.2));
            problem.SetParameterUpperBound(translation_ypd, 2, original_norm + 0.2);
        }

        ceres::Solver::Options options;
        options.linear_solver_type           = ceres::DENSE_QR;
        options.max_num_iterations           = 20;
        options.minimizer_progress_to_stdout = false;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // Step 6: Reconstruct refined constrained pose if optimization succeeded.
        const Eigen::Vector3d refined_translation = camera_ypd_to_translation(
            Eigen::Vector3d{translation_ypd[0], translation_ypd[1], translation_ypd[2]});
        const bool translation_finite =
            std::isfinite(translation_ypd[0]) && std::isfinite(translation_ypd[1])
            && std::isfinite(translation_ypd[2]) && refined_translation.allFinite()
            && refined_translation.z() > 1e-3;
        if (summary.IsSolutionUsable() && std::isfinite(yaw) && translation_finite) {
            const Eigen::Matrix3d R_cam_armor_refined = R_cam_imu * yaw_rotation(yaw) * R_pitch;

            // Write back the constrained pose.
            cv::Mat R_refined_cv;
            cv::eigen2cv(R_cam_armor_refined, R_refined_cv);
            cv::Rodrigues(R_refined_cv, rvec);
            tvec =
                (cv::Mat_<double>(3, 1) << refined_translation.x(), refined_translation.y(),
                 refined_translation.z());
        }
        // If optimization failed, keep the initial basin pose.

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
    struct SolvePose {
        cv::Mat rvec;
        cv::Mat tvec;
        double reprojection_rmse{std::numeric_limits<double>::infinity()};
        std::optional<double> prior_armor_yaw{};
        std::optional<double> prior_distance_m{};
    };

    struct PnPInput {
        const std::vector<cv::Point3f>& obj_points;
        std::vector<cv::Point2f> img_points;
        std::vector<cv::Point2f> norm_points;
        float dist_to_center;
    };

    [[nodiscard]] static cv::Mat identity_camera_matrix() { return cv::Mat::eye(3, 3, CV_64F); }

    [[nodiscard]] static bool pose_is_finite(const cv::Mat& rvec, const cv::Mat& tvec) {
        const double* rd = rvec.ptr<double>();
        const double* td = tvec.ptr<double>();
        for (int i = 0; i < 3; ++i) {
            if (!std::isfinite(rd[i]) || !std::isfinite(td[i])) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] static double normalized_reprojection_rmse(
        const PnPInput& input, const cv::Mat& rvec, const cv::Mat& tvec) {
        std::vector<cv::Point2f> projected;
        cv::projectPoints(
            input.obj_points, rvec, tvec, identity_camera_matrix(), cv::Mat{}, projected);
        if (projected.size() != input.norm_points.size()) {
            return std::numeric_limits<double>::infinity();
        }

        double sum_sq = 0.0;
        for (size_t i = 0; i < projected.size(); ++i) {
            const cv::Point2f d = projected[i] - input.norm_points[i];
            sum_sq += static_cast<double>(d.x) * static_cast<double>(d.x)
                    + static_cast<double>(d.y) * static_cast<double>(d.y);
        }
        return std::sqrt(sum_sq / static_cast<double>(projected.size()));
    }

    [[nodiscard]] std::optional<SolvePose> solve_with_ippe(const PnPInput& input) const {
        cv::Mat rvec, tvec;
        const bool success = cv::solvePnP(
            input.obj_points, input.norm_points, identity_camera_matrix(), cv::Mat{}, rvec, tvec,
            false, cv::SOLVEPNP_IPPE);
        if (!success || !pose_is_finite(rvec, tvec)) {
            return std::nullopt;
        }

        const double reprojection_rmse = normalized_reprojection_rmse(input, rvec, tvec);

        return SolvePose{
            .rvec              = std::move(rvec),
            .tvec              = std::move(tvec),
            .reprojection_rmse = reprojection_rmse,
        };
    }

    [[nodiscard]] std::optional<SolvePose>
        solve_with_priors(const PnPInput& input, const std::vector<PosePrior>& pose_priors) const {
        if (pose_priors.empty()) {
            return std::nullopt;
        }
        auto ippe = solve_with_ippe(input);
        if (!ippe) {
            return std::nullopt;
        }
        auto norm = cv::norm(ippe->tvec);

        std::vector<PosePrior> sorted_priors = pose_priors;
        std::sort(
            sorted_priors.begin(), sorted_priors.end(),
            [norm](const PosePrior& a, const PosePrior& b) {
                return std::abs(norm - cv::norm(a.tvec)) < std::abs(norm - cv::norm(b.tvec));
            });

        std::optional<SolvePose> best;
        for (const auto& prior : sorted_priors) {
            const double prior_distance = std::sqrt(
                prior.tvec[0] * prior.tvec[0] + prior.tvec[1] * prior.tvec[1]
                + prior.tvec[2] * prior.tvec[2]);
            if (!std::isfinite(prior_distance) || prior_distance <= 1e-3) {
                continue;
            }

            cv::Mat rvec = (cv::Mat_<double>(3, 1) << prior.rvec[0], prior.rvec[1], prior.rvec[2]);
            cv::Mat tvec = (cv::Mat_<double>(3, 1) << prior.tvec[0], prior.tvec[1], prior.tvec[2]);
            const bool success = cv::solvePnP(
                input.obj_points, input.norm_points, identity_camera_matrix(), cv::Mat{}, rvec,
                tvec, true, cv::SOLVEPNP_ITERATIVE);
            if (!success || !pose_is_finite(rvec, tvec)) {
                continue;
            }

            const double reprojection_rmse = normalized_reprojection_rmse(input, rvec, tvec);
            if (!std::isfinite(reprojection_rmse)) {
                continue;
            }
            Eigen::Matrix3d x;
            cv::Mat xx;
            cv::Rodrigues(prior.rvec, xx);
            cv::cv2eigen(xx, x);
            if (!best.has_value() || reprojection_rmse < best->reprojection_rmse) {
                best = SolvePose{
                    .rvec              = std::move(rvec),
                    .tvec              = std::move(tvec),
                    .reprojection_rmse = reprojection_rmse,
                    .prior_armor_yaw   = (math_fuxk::rpy(x).yaw / 180.0) * std::numbers::pi,
                    .prior_distance_m  = prior_distance,
                };
            }
        }

        return best;
    }

    [[nodiscard]] std::optional<SolvePose>
        solve_initial_pose(const PnPInput& input, const std::vector<PosePrior>& pose_priors) const {
        const auto prior_pose = solve_with_priors(input, pose_priors);
        if (prior_pose.has_value()) {
            return prior_pose;
        }
        return solve_with_ippe(input);
    }

    [[nodiscard]] std::expected<PnPInput, PnPError>
        make_pnp_input(const ArmorDetection& detection) const {
        auto img_points = detection.image_points();
        if (img_points.size() != 4) {
            return std::unexpected(PnPError::InvalidDetection);
        }

        std::vector<cv::Point2f> norm_points;
        cv::undistortPoints(img_points, norm_points, camera_matrix_, dist_coeffs_);
        if (norm_points.size() != img_points.size()) {
            return std::unexpected(PnPError::InvalidDetection);
        }

        const auto& obj_points =
            (detection.type == ArmorType::Large) ? large_armor_points_ : small_armor_points_;

        return PnPInput{
            .obj_points     = obj_points,
            .img_points     = std::move(img_points),
            .norm_points    = std::move(norm_points),
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
