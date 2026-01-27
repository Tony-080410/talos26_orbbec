/// @file log_distance_so3.hpp
/// @brief SO(3) × log-distance observation space utilities for redact
///
/// Provides type aliases and utility functions for working with
/// unit-quaternion + log-distance observation spaces.
/// Observation: z = [R; log(r)] ∈ SO(3) × ℝ¹
///
/// Key features:
/// - Safe logarithmic distance handling with minimum distance clipping
/// - Type-safe observation construction from rotation matrix + distance
/// - Utility functions for converting between representations
/// - Numerical stability guarantees for small distances

#pragma once

#include "euclidean.hpp"
#include "lie_group.hpp"
#include "product_space.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <utility>

namespace uni_kf::redact::spaces {

// ============================================================================
// Type Aliases
// ============================================================================

/// @brief SO(3) space with right-invariant error convention
using SO3d = SO3<LieSide::Right>;

/// @brief 1D Euclidean space for log-distance
using R1 = Euclidean<1>;

/// @brief Product space: SO(3) × ℝ¹ for unit-quaternion + log-distance observations
using LogDistanceSO3 = ProductSpace<SO3d, R1>;

// ============================================================================
// Safe Logarithmic Distance Utilities
// ============================================================================

/// @brief Minimum allowed distance for log-transformation (1mm)
inline constexpr double kMinLogDistance = 1e-3;

/// @brief Compute safe log-distance with lower bound
/// @param r Distance (must be positive)
/// @return log(max(r, kMinLogDistance))
[[nodiscard]] inline double safe_log_distance(double r) noexcept {
    return std::log(std::max(r, kMinLogDistance));
}

/// @brief Exponentiate log-distance with numerical safety
/// @param log_r Logarithm of distance
/// @return exp(log_r), guaranteed ≥ kMinLogDistance
[[nodiscard]] inline double safe_exp_distance(double log_r) noexcept {
    const double r = std::exp(log_r);
    return std::max(r, kMinLogDistance);
}

/// @brief Compute relative log-distance error with safe handling
/// @param log_r_meas Measured log-distance
/// @param log_r_est Estimated log-distance
/// @return log_r_meas - log_r_est (standard subtraction in log-space)
[[nodiscard]] inline double log_distance_error(double log_r_meas, double log_r_est) noexcept {
    return log_r_meas - log_r_est;
}

/// @brief Convert rotation matrix and distance to observation point
/// @param R Rotation matrix ∈ SO(3)
/// @param r Distance (linear scale)
/// @return Observation point in LogDistanceSO3 space
[[nodiscard]] inline LogDistanceSO3::Flux
    make_observation(const Eigen::Matrix3d& R, double r) noexcept {
    // SO3 point from rotation matrix
    const group::SO3<double> R_so3(R);

    // Log-distance with safety
    const double log_r = safe_log_distance(r);
    const R1::Flux log_r_point{Eigen::Matrix<double, 1, 1>{log_r}};

    return {R_so3, log_r_point};
}

/// @brief Convert observation point back to rotation matrix and distance
/// @param obs Observation point in LogDistanceSO3 space
/// @return Pair of (rotation matrix, distance)
[[nodiscard]] inline std::pair<Eigen::Matrix3d, double>
    from_observation(const LogDistanceSO3::Flux& obs) noexcept {
    // Extract rotation matrix
    const Eigen::Matrix3d R = obs.first.asMatrix();

    // Extract and exponentiate log-distance
    const double log_r = obs.second.data(0);
    const double r     = safe_exp_distance(log_r);

    return {R, r};
}

/// @brief Compute observation error in fiber space
/// @param z_meas Measured observation
/// @param z_est Estimated observation
/// @return Error vector [ξ; δ_logr] where ξ ∈ so(3), δ_logr ∈ ℝ
[[nodiscard]] inline LogDistanceSO3::Drift observation_error(
    const LogDistanceSO3::Flux& z_meas, const LogDistanceSO3::Flux& z_est) noexcept {
    return LogDistanceSO3::diff(z_est, z_meas);
}

/// @brief Check if observation point is valid (finite values, valid rotation)
/// @param obs Observation point to check
/// @return true if observation is valid
[[nodiscard]] inline bool is_valid_observation(const LogDistanceSO3::Flux& obs) noexcept {
    // Check rotation matrix (SO3 validity)
    const Eigen::Matrix3d R = obs.first.asMatrix();
    const double det        = R.determinant();
    if (std::abs(det - 1.0) > 1e-6) {
        return false;
    }

    // Check orthogonality
    const Eigen::Matrix3d I = R * R.transpose();
    if (!(I - Eigen::Matrix3d::Identity()).isZero(1e-6)) {
        return false;
    }

    // Check log-distance is finite
    const double log_r = obs.second.data(0);
    if (!std::isfinite(log_r)) {
        return false;
    }

    return true;
}

/// @brief Normalize observation (ensure rotation matrix is proper SO(3))
/// @param obs Observation point (modified in-place if needed)
/// @return true if normalization was successful
inline bool normalize_observation(LogDistanceSO3::Flux& obs) noexcept {
    // Normalize rotation matrix to proper SO(3)
    Eigen::Matrix3d R = obs.first.asMatrix();

    // SVD-based orthogonalization
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    // Ensure proper rotation (det = +1)
    if (U.determinant() * V.determinant() < 0) {
        V.col(2) *= -1;
    }

    R         = U * V.transpose();
    obs.first = group::SO3<double>(R);

    // Ensure log-distance is finite
    if (!std::isfinite(obs.second.data(0))) {
        obs.second.data(0) = safe_log_distance(kMinLogDistance);
    }

    return is_valid_observation(obs);
}

// ============================================================================
// Geometric Utility Functions
// ============================================================================

/// @brief Compute viewing angle factor for noise scaling
/// @param obs Observation point
/// @param target_normal_body Target surface normal in body frame (unit vector)
/// @return View angle factor ∈ [0, ∞), larger for side views
[[nodiscard]] inline double compute_view_angle_factor(
    const LogDistanceSO3::Flux& obs, const Eigen::Vector3d& target_normal_body) noexcept {

    // Extract rotation matrix (camera -> target orientation)
    const Eigen::Matrix3d R_camera_target = obs.first.asMatrix();

    // Camera line-of-sight direction in target frame (assuming looking at target)
    const Eigen::Vector3d line_of_sight_camera = Eigen::Vector3d::UnitZ();
    const Eigen::Vector3d line_of_sight_target = R_camera_target.transpose() * line_of_sight_camera;

    // Angle between line-of-sight and target normal
    const double cos_angle = std::abs(line_of_sight_target.dot(target_normal_body));
    const double sin_angle = std::sqrt(1.0 - cos_angle * cos_angle);

    // Return factor: 0 for head-on, larger for side views
    return std::log(sin_angle + 1.0);
}

/// @brief Convert from Cartesian position to observation
/// @param t_camera_target Translation vector from camera to target (camera frame)
/// @return Observation point [R; log(r)] where R is identity and r = ||t||
[[nodiscard]] inline LogDistanceSO3::Flux
    from_cartesian_position(const Eigen::Vector3d& t_camera_target) noexcept {
    const double r          = t_camera_target.norm();
    const Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    return make_observation(R, r);
}

/// @brief Convert observation to Cartesian position (camera frame)
/// @param obs Observation point
/// @return Translation vector assuming identity rotation
[[nodiscard]] inline Eigen::Vector3d
    to_cartesian_position(const LogDistanceSO3::Flux& obs) noexcept {
    auto [R, r] = from_observation(obs);
    // Assume target is along camera's z-axis at distance r
    return {0.0, 0.0, r};
}

// ============================================================================
// Concept Verification
// ============================================================================

static_assert(Space<LogDistanceSO3>, "LogDistanceSO3 must satisfy Space concept");

} // namespace uni_kf::redact::spaces
