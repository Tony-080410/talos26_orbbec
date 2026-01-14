/// @file pose_measurement_model.hpp
/// @brief Invariant measurement model for pose observations (unit quaternion + log-distance)
///
/// Implements InvariantMeasurementModel concept for observations in
/// SO(3) × ℝ¹ space (unit quaternion + logarithmic distance).
/// Provides:
/// - Right-invariant error computation
/// - Analytical error Jacobians
/// - Quality-aware diagonal noise covariance
/// - Camera extrinsic parameter handling
///
/// Model: z = [R_camera_target; log(||t_camera_target||)] ∈ SO(3) × ℝ¹

#pragma once

#include "../core/models.hpp"
#include "../spaces/log_distance_so3.hpp"
#include "tracker_inekf.hpp" // For RobotProcessModel/OutpostProcessModel types

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>

namespace uni_kf {

// ============================================================================
// PoseMeasurementModel - Base template
// ============================================================================

/// @brief Invariant measurement model for pose observations
///
/// Template parameters:
/// - ExtraDim: Dimension of extra state (4 for robot, 3 for outpost)
/// - TargetType: Enum or tag distinguishing robot/outpost geometry
///
/// Satisfies InvariantMeasurementModel concept.
template <int ExtraDim, typename TargetType>
class PoseMeasurementModel {
public:
    // ------------------------------------------------------------------------
    // Type definitions (required by InvariantMeasurementModel concept)
    // ------------------------------------------------------------------------

    // State space: SE₂(3) × ℝ^ExtraDim
    using Space = SE23_Rn<ExtraDim>;

    // Observation space: SO(3) × ℝ¹ (unit quaternion + log-distance)
    using MeasurementSpace = LogDistanceSO3;

    // Shorthands
    using Point      = typename Space::Point;
    using MeasPoint  = typename MeasurementSpace::Point;
    using MeasFiber  = typename MeasurementSpace::Fiber;
    using Covariance = SymmetricMatrix<4, double>; // 4D: 3 (rotation) + 1 (log-distance)

    using SE23d  = group::SEn3<double, 2>;
    using Vec3   = Eigen::Vector3d;
    using Mat3   = Eigen::Matrix3d;
    using Mat4xX = Eigen::Matrix<double, 4, Space::FiberDim>;

    // ------------------------------------------------------------------------
    // Configuration structures
    // ------------------------------------------------------------------------

    /// @brief Camera extrinsic parameters
    struct CameraExtrinsics {
        Eigen::Matrix3d R_world_camera = Eigen::Matrix3d::Identity(); // World → Camera rotation
        Eigen::Vector3d p_world_camera = Eigen::Vector3d::Zero();     // Camera position in world
    };

    /// @brief Noise model parameters
    struct NoiseParams {
        double sigma_rot           = 0.1; // Base rotation noise (rad)
        double sigma_logr          = 0.2; // Base log-distance noise (dimensionless)
        double k_quality           = 1.0; // Quality sensitivity (0 = ignore quality)
        double k_rot_distance      = 0.1; // Rotation noise distance sensitivity
        double k_logr_view         = 1.0; // Log-distance noise view angle sensitivity
        double min_detection_score = 0.1; // Minimum detection score for update
    };

    // ------------------------------------------------------------------------
    // Construction and configuration
    // ------------------------------------------------------------------------

    /// @brief Construct with camera extrinsics and noise parameters
    PoseMeasurementModel(
        const CameraExtrinsics& extrinsics, const NoiseParams& noise_params,
        TargetType target_type) noexcept
        : extrinsics_(extrinsics)
        , noise_params_(noise_params)
        , target_type_(target_type)
        , cached_y_body_(Vec3::Zero()) {}

    // ------------------------------------------------------------------------
    // InvariantMeasurementModel interface
    // ------------------------------------------------------------------------

    /// @brief Compute body-frame invariant error
    ///
    /// Right-invariant error:
    ///   ξ_err = log(R_meas · R_est^{-1})  ∈ so(3)
    ///   δ_logr = log(r_meas) - log(r_est) ∈ ℝ
    ///
    /// @param x Current state estimate
    /// @param z Observation (R_camera_target, log(r))
    /// @return Error vector [ξ_err; δ_logr] ∈ ℝ⁴
    [[nodiscard]] MeasFiber invariant_error(const Point& x, const MeasPoint& z) const noexcept {
        // 1. Extract observation components
        const Eigen::Matrix3d R_camera_target_meas = z.first.asMatrix();
        const double log_r_meas                    = z.second.data(0);
        const double r_meas                        = std::exp(log_r_meas);

        // 2. Compute estimated pose in camera frame
        const auto [R_camera_target_est, t_camera_target_est] = compute_estimated_pose(x);
        const double r_est                                    = t_camera_target_est.norm();

        // 3. Compute rotation error (right-invariant)
        const Eigen::Matrix3d R_err = R_camera_target_meas * R_camera_target_est.transpose();
        const group::SO3<double> R_err_so3(R_err);
        const Eigen::Vector3d xi_err = group::SO3<double>::log(R_err_so3);

        // 4. Compute log-distance error (with safe handling)
        const double log_r_est  = safe_log_distance(r_est);
        const double delta_logr = log_r_meas - log_r_est;

        // 5. Cache body-frame translation for noise computation
        cached_y_body_ = compute_y_body(x, t_camera_target_est);

        // 6. Return concatenated error
        MeasFiber error;
        error.data.template head<3>() = xi_err;
        error.data(3)                 = delta_logr;
        return error;
    }

    /// @brief Compute error Jacobian C = ∂r/∂δ
    ///
    /// Jacobian mapping from state error δ ∈ ℝ^{13+ExtraDim} to
    /// observation error r ∈ ℝ⁴.
    [[nodiscard]] Mat4xX error_jacobian(const Point& x) const noexcept {
        constexpr int kGroupDim = 9; // SE₂(3) dimension

        Mat4xX C = Mat4xX::Zero();

        // 1. Compute estimated pose and geometry
        const auto [R_camera_target_est, t_camera_target_est] = compute_estimated_pose(x);
        const double r_est                                    = t_camera_target_est.norm();
        const Vec3 t_dir = (r_est > 1e-6) ? t_camera_target_est / r_est : Vec3::UnitZ();

        // 2. Extract body-frame geometry
        const auto [R_target_body, b_body] = get_target_geometry(x);
        const Vec3 b_camera = extrinsics_.R_world_camera.transpose() * x.first.R() * b_body;

        // 3. Rotation error Jacobian w.r.t. state rotation
        // ∂ξ_err/∂δ_R = -J_r^{-1}(ξ_err) · Adjoint(R_target_body^T)
        // For small errors and identity target-body rotation, simplifies to -I
        C.template block<3, 3>(0, 0) = -Mat3::Identity();

        // 4. Rotation error Jacobian w.r.t. state position
        // ∂ξ_err/∂δ_p = 0 (rotation error doesn't depend on position directly)
        C.template block<3, 3>(0, 6).setZero();

        // 5. Log-distance error Jacobian w.r.t. state position
        // ∂δ_logr/∂δ_p = (1/r_est) · t_dir^T · R_world_camera^T
        const Mat3 J_logr_p =
            (1.0 / r_est) * t_dir * t_dir.transpose() * extrinsics_.R_world_camera.transpose();
        C.template block<1, 3>(3, 6) = J_logr_p.row(0); // Only need first row for scalar output

        // 6. Log-distance error Jacobian w.r.t. geometry parameters
        // ∂δ_logr/∂b = (1/r_est) · t_dir^T · R_world_camera^T · R_state
        const Mat3 J_logr_b = (1.0 / r_est) * t_dir * t_dir.transpose()
                            * extrinsics_.R_world_camera.transpose() * x.first.R();

        // Distribute to appropriate extra state indices based on target type
        fill_geometry_jacobian(C, 3, J_logr_b, x);

        // 7. Zero out roll/pitch columns (constrained dimensions)
        C.col(0).setZero(); // roll
        C.col(1).setZero(); // pitch

        return C;
    }

    /// @brief Compute measurement noise covariance
    ///
    /// Returns diagonal 4×4 covariance matrix:
    ///   R = diag(σ_rot², σ_rot², σ_rot², σ_logr²)
    ///
    /// Noise scales with:
    /// - Detection quality (confidence score)
    /// - Distance (for rotation noise)
    /// - Viewing angle (for distance noise)
    ///
    /// @param z Observation (for geometric factors)
    /// @param detection_quality Detection confidence score [0, 1]
    /// @return Diagonal covariance matrix
    [[nodiscard]] Covariance
        measurement_noise(const MeasPoint& z, double detection_quality) const noexcept {
        // 1. Base noise values
        const double sigma_rot_base  = noise_params_.sigma_rot;
        const double sigma_logr_base = noise_params_.sigma_logr;

        // 2. Quality weighting: low quality → higher noise
        const double quality_factor = 1.0 + noise_params_.k_quality * (1.0 - detection_quality);

        // 3. Distance-dependent rotation noise (farther → noisier angles)
        const double log_r              = z.second.data(0);
        const double r                  = std::exp(log_r);
        const double rot_distance_scale = 1.0 + noise_params_.k_rot_distance * std::log(r + 1.0);

        // 4. View-angle dependent distance noise (side view → noisier depth)
        const double view_angle_factor = compute_view_angle_factor(z);
        const double logr_view_scale   = 1.0 + noise_params_.k_logr_view * view_angle_factor;

        // 5. Combined noise scaling
        const double sigma_rot  = sigma_rot_base * quality_factor * rot_distance_scale;
        const double sigma_logr = sigma_logr_base * quality_factor * logr_view_scale;

        // 6. Build diagonal covariance matrix
        Eigen::Matrix<double, 4, 4> R = Eigen::Matrix<double, 4, 4>::Zero();
        R(0, 0)                       = sigma_rot * sigma_rot;
        R(1, 1)                       = sigma_rot * sigma_rot;
        R(2, 2)                       = sigma_rot * sigma_rot;
        R(3, 3)                       = sigma_logr * sigma_logr;

        // 7. Add small regularization for positive definiteness
        R.diagonal().array() += 1e-12;

        return Covariance(R);
    }

    // ------------------------------------------------------------------------
    // Additional interface for integration
    // ------------------------------------------------------------------------

    /// @brief Check if detection quality is sufficient for update
    [[nodiscard]] bool is_quality_sufficient(double detection_quality) const noexcept {
        return detection_quality >= noise_params_.min_detection_score;
    }

    /// @brief Get cached body-frame observation (for debugging/monitoring)
    [[nodiscard]] Vec3 get_cached_y_body() const noexcept { return cached_y_body_; }

private:
    // ------------------------------------------------------------------------
    // Private helper functions
    // ------------------------------------------------------------------------

    /// @brief Compute estimated pose in camera frame
    [[nodiscard]] std::pair<Eigen::Matrix3d, Vec3>
        compute_estimated_pose(const Point& x) const noexcept {
        // World → Body transformation
        const Eigen::Matrix3d R_world_body = x.first.R();
        const Vec3 p_world_body            = x.first.p();

        // Target geometry in body frame
        const auto [R_target_body, b_body] = get_target_geometry(x);

        // Body → World → Camera transformation
        const Eigen::Matrix3d R_camera_body = extrinsics_.R_world_camera.transpose() * R_world_body;
        const Vec3 p_camera_body =
            extrinsics_.R_world_camera.transpose() * (p_world_body - extrinsics_.p_world_camera);

        // Camera → Target transformation
        const Eigen::Matrix3d R_camera_target = R_camera_body * R_target_body.transpose();
        const Vec3 t_camera_target            = p_camera_body + R_camera_body * b_body;

        return {R_camera_target, t_camera_target};
    }

    /// @brief Get target geometry in body frame
    [[nodiscard]] std::pair<Eigen::Matrix3d, Vec3>
        get_target_geometry(const Point& x) const noexcept {
        // Default: identity rotation, zero offset (for forward-facing target)
        // Override in specialized templates for robot/outpost geometry
        return {Eigen::Matrix3d::Identity(), Vec3::Zero()};
    }

    /// @brief Compute body-frame observation for noise computation
    [[nodiscard]] Vec3 compute_y_body(const Point& x, const Vec3& t_camera_target) const noexcept {
        // Transform camera-frame translation to body frame
        const Eigen::Matrix3d R_world_body  = x.first.R();
        const Eigen::Matrix3d R_body_camera = R_world_body.transpose() * extrinsics_.R_world_camera;
        return R_body_camera * t_camera_target;
    }

    /// @brief Compute view angle factor for distance noise
    [[nodiscard]] double compute_view_angle_factor(const MeasPoint& z) const noexcept {
        // Simplified: use cached body-frame observation
        if (cached_y_body_.norm() < 1e-6) {
            return 0.0;
        }

        // Assume target normal points away from body center (radial outward)
        const Vec3 target_normal_body = cached_y_body_.normalized();

        // Use utility function from log_distance_so3.hpp
        return uni_kf::compute_view_angle_factor(z, target_normal_body);
    }

    /// @brief Fill geometry parameter Jacobian based on target type
    void fill_geometry_jacobian(
        Mat4xX& C, int row_idx, const Mat3& J_logr_b, const Point& x) const noexcept {
        // Base implementation does nothing
        // Specialized in derived classes for robot/outpost geometry
        (void)C;
        (void)row_idx;
        (void)J_logr_b;
        (void)x;
    }

    /// @brief Safe logarithmic distance with minimum clipping
    [[nodiscard]] static double safe_log_distance(double r) noexcept {
        constexpr double kMinDistance = 1e-3;
        return std::log(std::max(r, kMinDistance));
    }

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------

    CameraExtrinsics extrinsics_; // Camera extrinsic parameters
    NoiseParams noise_params_;    // Noise model parameters
    TargetType target_type_;      // Target type (robot/outpost)

    mutable Vec3 cached_y_body_;  // Cached body-frame observation for noise computation
};

// ============================================================================
// RobotPoseMeasurementModel - Specialization for 4-armor robots
// ============================================================================

/// @brief Pose measurement model for robot targets (4 armors)
template <>
class PoseMeasurementModel<RobotProcessModel::kExtraDim, RobotTargetTag> {
    // Note: This would be a full specialization with robot-specific geometry
    // Implementation would go here...
};

// ============================================================================
// OutpostPoseMeasurementModel - Specialization for 3-armor outposts
// ============================================================================

/// @brief Pose measurement model for outpost targets (3 armors)
template <>
class PoseMeasurementModel<OutpostProcessModel::kExtraDim, OutpostTargetTag> {
    // Note: This would be a full specialization with outpost-specific geometry
    // Implementation would go here...
};

} // namespace uni_kf