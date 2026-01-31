#pragma once
#include "covariance.hpp"
#include "filter.hpp"

namespace uni_kf::redact {

namespace detail {
template <typename Assumption, typename Flux>
concept HasTransitionJacobian =
    HasMemberDiscreteDynamicsJacobian<Assumption, Flux>
    || HasStaticDiscreteDynamicsJacobian<Assumption, Flux>
    || HasMemberContinuousDynamicsJacobian<Assumption, Flux>
    || HasStaticContinuousDynamicsJacobian<Assumption, Flux>
    || HasMemberLegacyDynamicsJacobian<Assumption, Flux>
    || HasStaticLegacyDynamicsJacobian<Assumption, Flux> || HasMemberJacobian<Assumption, Flux>
    || HasStaticJacobian<Assumption, Flux>;
} // namespace detail

// ============================================================================
// Strategy-Only Aliases (use default Direct storage)
// ============================================================================

/// @brief Extended Kalman Filter (EKF) with Direct storage
///
/// Requirements:
/// - Assumption must provide a transition Jacobian Φ via one of:
///   - `assumption.dynamics_jacobian_discrete(x, dt)` returning Φ
///   - `assumption.dynamics_jacobian_continuous(x)` returning A (used as Φ = expm(A·dt))
///   - legacy `assumption.dynamics_jacobian(x, dt)` (interpreted as Φ)
///   - `assumption.jacobian(x, dt)` (synonym for Φ)
///
/// @tparam LieGroup The lieplusplus Lie group type
/// @tparam Perception The measurement model type
/// @tparam Assumption The dynamics model type
template <typename LieGroup, typename Perception, typename Assumption>
requires(detail::HasTransitionJacobian<Assumption, LieGroup>) using EKF = KalmanFilter<
    Covariance<LieGroup, Representation::Direct, LinearizedStrategy>, Perception, Assumption>;

/// @brief Unscented Kalman Filter (UKF) with Direct storage
///
/// Requirements:
/// - LieGroup must support Lie group operations (*, inv, log)
/// - LieGroup::exp(ξ) for exponential map
///
/// @tparam LieGroup The lieplusplus Lie group type
/// @tparam Perception The measurement model type
/// @tparam Assumption The dynamics model type
template <typename LieGroup, typename Perception, typename Assumption>
requires requires(const LieGroup& x, const LieGroup& y) {
    { x * y } -> std::convertible_to<LieGroup>;
    { x.inv() } -> std::convertible_to<LieGroup>;
    { LieGroup::log(x) } -> std::convertible_to<typename LieGroup::VectorType>;
    { LieGroup::exp(typename LieGroup::VectorType::Zero()) } -> std::convertible_to<LieGroup>;
}
using UKF = KalmanFilter<
    Covariance<LieGroup, Representation::Direct, UnscentedStrategy>, Perception, Assumption>;

/// @brief Cubature Kalman Filter (CKF) with Direct storage
///
/// Requirements:
/// - LieGroup must support Lie group operations (*, inv, log)
/// - LieGroup::exp(ξ) for exponential map
/// - Uses spherical-radial cubature rule for sigma points
///
/// @tparam LieGroup The lieplusplus Lie group type
/// @tparam Perception The measurement model type
/// @tparam Assumption The dynamics model type
template <typename LieGroup, typename Perception, typename Assumption>
requires requires(const LieGroup& x, const LieGroup& y) {
    { x * y } -> std::convertible_to<LieGroup>;
    { x.inv() } -> std::convertible_to<LieGroup>;
    { LieGroup::log(x) } -> std::convertible_to<typename LieGroup::VectorType>;
    { LieGroup::exp(typename LieGroup::VectorType::Zero()) } -> std::convertible_to<LieGroup>;
}
using CKF = KalmanFilter<
    Covariance<LieGroup, Representation::Direct, CubatureStrategy>, Perception, Assumption>;

// ============================================================================
// Storage Variant Aliases
// ============================================================================

/// @brief EKF with Square-Root (SR) storage
template <typename LieGroup, typename Perception, typename Assumption>
requires(detail::HasTransitionJacobian<Assumption, LieGroup>) using EKF_SR = KalmanFilter<
    Covariance<LieGroup, Representation::Sr, LinearizedStrategy>, Perception, Assumption>;

/// @brief UKF with Square-Root (SR) storage
template <typename LieGroup, typename Perception, typename Assumption>
requires requires(const LieGroup& x, const LieGroup& y) {
    { x * y } -> std::convertible_to<LieGroup>;
    { x.inv() } -> std::convertible_to<LieGroup>;
    { LieGroup::log(x) } -> std::convertible_to<typename LieGroup::VectorType>;
    { LieGroup::exp(typename LieGroup::VectorType::Zero()) } -> std::convertible_to<LieGroup>;
}
using UKF_SR = KalmanFilter<
    Covariance<LieGroup, Representation::Sr, UnscentedStrategy>, Perception, Assumption>;

/// @brief CKF with Square-Root (SR) storage
template <typename LieGroup, typename Perception, typename Assumption>
requires requires(const LieGroup& x, const LieGroup& y) {
    { x * y } -> std::convertible_to<LieGroup>;
    { x.inv() } -> std::convertible_to<LieGroup>;
    { LieGroup::log(x) } -> std::convertible_to<typename LieGroup::VectorType>;
    { LieGroup::exp(typename LieGroup::VectorType::Zero()) } -> std::convertible_to<LieGroup>;
}
using CKF_SR = KalmanFilter<
    Covariance<LieGroup, Representation::Sr, CubatureStrategy>, Perception, Assumption>;

// ============================================================================
// UDU Storage Variant Aliases
// ============================================================================

/// @brief EKF with UDU storage
template <typename LieGroup, typename Perception, typename Assumption>
requires(detail::HasTransitionJacobian<Assumption, LieGroup>) using EKF_UDU = KalmanFilter<
    Covariance<LieGroup, Representation::Udu, LinearizedStrategy>, Perception, Assumption>;

// ============================================================================
// Joseph Update Aliases (Direct storage + Joseph updater)
// ============================================================================

/// @brief EKF with Joseph update
template <typename LieGroup, typename Perception, typename Assumption>
requires(detail::HasTransitionJacobian<Assumption, LieGroup>) using EKF_Joseph = KalmanFilter<
    Covariance<LieGroup, Representation::Direct, LinearizedStrategy, DirectJosephUpdater>,
    Perception, Assumption>;

// ============================================================================
// Invariant Kalman Filter Aliases (for Lie groups)
// ============================================================================

/// @brief Invariant Kalman Filter (InKF) with Direct storage
///
/// Requirements:
/// - LieGroup must support inv() and log() (lieplusplus native methods)
/// - Suitable for SO(3), SE(3), etc.
///
/// @tparam LieGroup The lieplusplus Lie group type
/// @tparam Perception The measurement model type
/// @tparam Assumption The dynamics model type
template <typename LieGroup, typename Perception, typename Assumption>
requires requires(const LieGroup& x) {
    { x.inv() } -> std::convertible_to<LieGroup>;
    { LieGroup::log(x) } -> std::convertible_to<typename LieGroup::VectorType>;
}
using InKF = KalmanFilter<
    Covariance<LieGroup, Representation::Direct, InvariantLinearStrategy>, Perception, Assumption>;

/// @brief Invariant Extended Kalman Filter (InEKF) with Direct storage
///
/// Requirements:
/// - LieGroup must support inv(), log() (used by the filter layer for innovations)
/// - Assumption must provide a transition Jacobian Φ (see EKF requirements above)
/// - Suitable for nonlinear dynamics on manifolds
///
/// @tparam LieGroup The lieplusplus Lie group type
/// @tparam Perception The measurement model type
/// @tparam Assumption The dynamics model type
template <typename LieGroup, typename Perception, typename Assumption>
requires(detail::HasTransitionJacobian<Assumption, LieGroup>) using InEKF = KalmanFilter<
    Covariance<LieGroup, Representation::Direct, InvariantLinearizedStrategy>, Perception,
    Assumption>;

// ============================================================================
// Invariant Filter Storage Variants
// ============================================================================

/// @brief InKF with Square-Root (SR) storage
template <typename LieGroup, typename Perception, typename Assumption>
requires requires(const LieGroup& x) {
    { x.inv() } -> std::convertible_to<LieGroup>;
    { LieGroup::log(x) } -> std::convertible_to<typename LieGroup::VectorType>;
}
using InKF_SR = KalmanFilter<
    Covariance<LieGroup, Representation::Sr, InvariantLinearStrategy>, Perception, Assumption>;

/// @brief InEKF with Square-Root (SR) storage
template <typename LieGroup, typename Perception, typename Assumption>
requires(detail::HasTransitionJacobian<Assumption, LieGroup>) using InEKF_SR = KalmanFilter<
    Covariance<LieGroup, Representation::Sr, InvariantLinearizedStrategy>, Perception, Assumption>;

/// @brief InKF with UDU storage
template <typename LieGroup, typename Perception, typename Assumption>
requires requires(const LieGroup& x) {
    { x.inv() } -> std::convertible_to<LieGroup>;
    { LieGroup::log(x) } -> std::convertible_to<typename LieGroup::VectorType>;
}
using InKF_UDU = KalmanFilter<
    Covariance<LieGroup, Representation::Udu, InvariantLinearStrategy>, Perception, Assumption>;

/// @brief InEKF with UDU storage
template <typename LieGroup, typename Perception, typename Assumption>
requires(detail::HasTransitionJacobian<Assumption, LieGroup>) using InEKF_UDU = KalmanFilter<
    Covariance<LieGroup, Representation::Udu, InvariantLinearizedStrategy>, Perception, Assumption>;

// NOTE: Joseph update is defined on Direct P; invariant SR/UDU updates have
// dedicated numerically-stable factorized algorithms.

} // namespace uni_kf::redact
