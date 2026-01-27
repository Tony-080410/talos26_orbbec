#pragma once
#include "covariance.hpp"
#include "filter.hpp"

namespace uni_kf::redact {

// ============================================================================
// Strategy-Only Aliases (use default Direct storage)
// ============================================================================

/// @brief Standard Kalman Filter (KF) with Direct storage
///
/// Requirements:
/// - Belief must satisfy LinearDynamics (constant F and H matrices)
/// - Belief::state_transition_matrix() returns F
/// - Belief::measurement_matrix() returns H
///
/// @tparam Belief The state space type
/// @tparam Perception The measurement model type
/// @tparam Assumption The dynamics model type
template <typename Belief, typename Perception, typename Assumption>
requires LinearDynamics<Belief> using KF = KalmanFilter<
    Covariance<Belief, Representation::Direct, LinearStrategy>, Perception, Assumption>;

/// @brief Extended Kalman Filter (EKF) with Direct storage
///
/// Requirements:
/// - Belief must satisfy Linearizable (has Jacobians)
/// - Belief::dynamics_jacobian(state, dt) returns J_f = df/dx
/// - Belief::measurement_jacobian(state) returns J_h = dh/dx
///
/// @tparam Belief The state space type
/// @tparam Perception The measurement model type
/// @tparam Assumption The dynamics model type
template <typename Belief, typename Perception, typename Assumption>
requires Linearizable<Belief> using EKF = KalmanFilter<
    Covariance<Belief, Representation::Direct, LinearizedStrategy>, Perception, Assumption>;

/// @brief Unscented Kalman Filter (UKF) with Direct storage
///
/// Requirements:
/// - Belief must satisfy Sampleable (has sigma-point methods)
/// - Belief::sigma_points(state, P) returns weighted sigma points
/// - Belief::mean(points) returns mean from weighted points
/// - Belief::covariance(points, mean) returns covariance from weighted points
///
/// @tparam Belief The state space type
/// @tparam Perception The measurement model type
/// @tparam Assumption The dynamics model type
template <typename Belief, typename Perception, typename Assumption>
requires Sampleable<Belief> using UKF = KalmanFilter<
    Covariance<Belief, Representation::Direct, UnscentedStrategy>, Perception, Assumption>;

/// @brief Cubature Kalman Filter (CKF) with Direct storage
///
/// Requirements:
/// - Belief must satisfy Sampleable (has sigma-point methods)
/// - Uses spherical-radial cubature rule for sigma points
///
/// @tparam Belief The state space type
/// @tparam Perception The measurement model type
/// @tparam Assumption The dynamics model type
template <typename Belief, typename Perception, typename Assumption>
requires Sampleable<Belief> using CKF = KalmanFilter<
    Covariance<Belief, Representation::Direct, CubatureStrategy>, Perception, Assumption>;

// ============================================================================
// Storage Variant Aliases
// ============================================================================

/// @brief KF with Square-Root (SR) storage
template <typename Belief, typename Perception, typename Assumption>
requires LinearDynamics<Belief> using KF_SR =
    KalmanFilter<Covariance<Belief, Representation::Sr, LinearStrategy>, Perception, Assumption>;

/// @brief EKF with Square-Root (SR) storage
template <typename Belief, typename Perception, typename Assumption>
requires Linearizable<Belief> using EKF_SR = KalmanFilter<
    Covariance<Belief, Representation::Sr, LinearizedStrategy>, Perception, Assumption>;

/// @brief UKF with Square-Root (SR) storage
template <typename Belief, typename Perception, typename Assumption>
requires Sampleable<Belief> using UKF_SR =
    KalmanFilter<Covariance<Belief, Representation::Sr, UnscentedStrategy>, Perception, Assumption>;

/// @brief CKF with Square-Root (SR) storage
template <typename Belief, typename Perception, typename Assumption>
requires Sampleable<Belief> using CKF_SR =
    KalmanFilter<Covariance<Belief, Representation::Sr, CubatureStrategy>, Perception, Assumption>;

// ============================================================================
// UDU Storage Variant Aliases
// ============================================================================

/// @brief KF with UDU storage (Bierman-Thornton UD filter)
template <typename Belief, typename Perception, typename Assumption>
requires LinearDynamics<Belief> using KF_UDU =
    KalmanFilter<Covariance<Belief, Representation::Udu, LinearStrategy>, Perception, Assumption>;

/// @brief EKF with UDU storage
template <typename Belief, typename Perception, typename Assumption>
requires Linearizable<Belief> using EKF_UDU = KalmanFilter<
    Covariance<Belief, Representation::Udu, LinearizedStrategy>, Perception, Assumption>;

// ============================================================================
// Joseph Update Aliases (Direct storage + Joseph updater)
// ============================================================================

/// @brief KF with Joseph update
template <typename Belief, typename Perception, typename Assumption>
requires LinearDynamics<Belief> using KF_Joseph = KalmanFilter<
    Covariance<Belief, Representation::Direct, LinearStrategy, DirectJosephUpdater>, Perception,
    Assumption>;

/// @brief EKF with Joseph update
template <typename Belief, typename Perception, typename Assumption>
requires Linearizable<Belief> using EKF_Joseph = KalmanFilter<
    Covariance<Belief, Representation::Direct, LinearizedStrategy, DirectJosephUpdater>, Perception,
    Assumption>;

// ============================================================================
// Invariant Kalman Filter Aliases (for Lie groups)
// ============================================================================

/// @brief Invariant Kalman Filter (InKF) with Direct storage
///
/// Requirements:
/// - Belief must be a LieGroup with adjoint_matrix()
/// - Uses Adjoint operator for covariance propagation on tangent space
/// - Suitable for SO(3), SE(3), etc.
///
/// @tparam Belief The Lie group state space type
/// @tparam Perception The measurement model type
/// @tparam Assumption The dynamics model type
template <typename Belief, typename Perception, typename Assumption>
requires InvariantGroup<Belief> using InKF = KalmanFilter<
    Covariance<Belief, Representation::Direct, InvariantLinearStrategy>, Perception, Assumption>;

/// @brief Invariant Extended Kalman Filter (InEKF) with Direct storage
///
/// Requirements:
/// - Belief must be a LieGroup with adjoint_matrix()
/// - Uses Adjoint operator with Jacobians on tangent space
/// - Suitable for nonlinear dynamics on manifolds
///
/// @tparam Belief The Lie group state space type
/// @tparam Perception The measurement model type
/// @tparam Assumption The dynamics model type
template <typename Belief, typename Perception, typename Assumption>
requires InvariantLinearizableGroup<Belief> using InEKF = KalmanFilter<
    Covariance<Belief, Representation::Direct, InvariantLinearizedStrategy>, Perception,
    Assumption>;

// ============================================================================
// Invariant Filter Storage Variants
// ============================================================================

/// @brief InKF with Square-Root (SR) storage
template <typename Belief, typename Perception, typename Assumption>
using InKF_SR = KalmanFilter<
    Covariance<Belief, Representation::Sr, InvariantLinearStrategy>, Perception, Assumption>;

/// @brief InEKF with Square-Root (SR) storage
template <typename Belief, typename Perception, typename Assumption>
using InEKF_SR = KalmanFilter<
    Covariance<Belief, Representation::Sr, InvariantLinearizedStrategy>, Perception, Assumption>;

/// @brief InKF with UDU storage
template <typename Belief, typename Perception, typename Assumption>
using InKF_UDU = KalmanFilter<
    Covariance<Belief, Representation::Udu, InvariantLinearStrategy>, Perception, Assumption>;

/// @brief InEKF with UDU storage
template <typename Belief, typename Perception, typename Assumption>
using InEKF_UDU = KalmanFilter<
    Covariance<Belief, Representation::Udu, InvariantLinearizedStrategy>, Perception, Assumption>;

// NOTE: Joseph update is defined on Direct P; invariant SR/UDU updates have
// dedicated numerically-stable factorized algorithms.

} // namespace uni_kf::redact
