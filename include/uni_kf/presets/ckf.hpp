/// @file ckf.hpp
/// @brief Cubature Kalman Filter (CKF) presets and type aliases
///
/// CKF uses spherical-radial cubature rule for state and covariance propagation.
/// Advantages over EKF:
/// - 3rd order accuracy (vs 1st order for EKF)
/// - Better handling of strong nonlinearities
/// - No need for Jacobian computation
///
/// Advantages over UKF:
/// - No tuning parameters (α, β, κ)
/// - Simpler implementation
/// - Deterministic cubature points
///
/// Use when:
/// - Nonlinearities are significant
/// - Jacobians are difficult or expensive to compute
/// - You want better accuracy than EKF without UKF's complexity

#pragma once

#include "../core/filter.hpp"
#include "../spaces/euclidean.hpp"
#include "../spaces/lie_manifold.hpp"

namespace uni_kf {

// ============================================================================
// Euclidean CKF (standard CKF on ℝⁿ)
// ============================================================================

/// @brief Cubature Kalman Filter on Euclidean space ℝⁿ
///
/// State space: ℝⁿ (vector space)
/// Propagation: Cubature rule with 2n points
/// Use for: Standard nonlinear filtering on vector spaces
///
/// Example:
/// ```cpp
/// uni_kf::CKF<6> filter;  // 6D state (position + velocity)
/// filter.init(initial_state, initial_covariance);
///
/// // Predict
/// auto dynamics = [](const auto& x, double dt) { /* ... */ };
/// filter.predict(dynamics, Q, dt);
///
/// // Update
/// auto measurement_fn = [](const auto& x) { /* ... */ };
/// filter.update<uni_kf::Euclidean<3>>(z, measurement_fn, R);
/// ```
template <int N, typename ReprTag = CovarianceTag, typename Scalar = double>
using CKF = UnifiedKalmanFilter<Euclidean<N, Scalar>, CubaturePointsTag, ReprTag, Scalar>;

// Convenient aliases for common dimensions
using CKF_1D = CKF<1>; ///< 1D Cubature Kalman Filter
using CKF_2D = CKF<2>; ///< 2D Cubature Kalman Filter
using CKF_3D = CKF<3>; ///< 3D Cubature Kalman Filter
using CKF_6D = CKF<6>; ///< 6D Cubature Kalman Filter (common for position + velocity)

// ============================================================================
// Lie Group CKF (CKF on manifolds)
// ============================================================================

/// @brief Cubature Kalman Filter on Lie groups
///
/// State space: Lie group G (SO(3), SE(3), etc.)
/// Propagation: Cubature rule in tangent space
/// Use for: Nonlinear filtering on rotation/pose manifolds
///
/// The cubature points are generated in the tangent space (Lie algebra),
/// mapped to the manifold, propagated, and mapped back for covariance computation.
///
/// Example (attitude estimation):
/// ```cpp
/// using SO3 = group::SO3<double>;
/// uni_kf::LieCKF<SO3> filter;
/// filter.init(initial_rotation, initial_covariance);
///
/// // Predict with angular velocity
/// auto dynamics = [omega](const SO3& R, double dt) {
///     return R * SO3::exp(omega * dt);
/// };
/// filter.predict(dynamics, Q, dt);
///
/// // Update with gravity measurement
/// filter.update<uni_kf::Euclidean<3>>(g_meas, measurement_fn, R);
/// ```
template <
    typename G, LieSide Side = LieSide::Right, typename ReprTag = CovarianceTag,
    typename Scalar = double>
using LieCKF = UnifiedKalmanFilter<LieManifold<G, Side>, CubaturePointsTag, ReprTag, Scalar>;

// ============================================================================
// Special case: CKF on SE(3) and SO(3)
// ============================================================================

/// @brief Cubature Kalman Filter on SO(3) (3D rotations)
///
/// State: 3D rotation matrix
/// Tangent space: ℝ³ (axis-angle representation)
/// Cubature points: 6 points (±√3 * e_i for i=1,2,3)
///
/// Use for: Attitude estimation, IMU orientation tracking
template <typename ReprTag = CovarianceTag, typename Scalar = double>
using CKF_SO3 = LieCKF<group::SO3<Scalar>, LieSide::Right, ReprTag, Scalar>;

/// @brief Cubature Kalman Filter on SE₂(3) (extended pose)
///
/// State: SE₂(3) = SO(3) ⋉ (ℝ³ × ℝ³) (rotation + velocity + position)
/// Tangent space: ℝ⁹
/// Cubature points: 18 points
///
/// Use for: Navigation with velocity states (SLAM, VIO)
template <typename ReprTag = CovarianceTag, typename Scalar = double>
using CKF_SE23 = LieCKF<group::SEn3<Scalar, 2>, LieSide::Right, ReprTag, Scalar>;

// ============================================================================
// Square Root CKF (SRCKF)
// ============================================================================

/// @brief Square Root Cubature Kalman Filter
/// More numerically stable than standard CKF
template <int N, typename Scalar = double>
using SRCKF = UnifiedKalmanFilter<Euclidean<N, Scalar>, CubaturePointsTag, SquareRootTag, Scalar>;

template <typename G, LieSide Side = LieSide::Right, typename Scalar = double>
using LieSRCKF =
    UnifiedKalmanFilter<LieManifold<G, Side>, CubaturePointsTag, SquareRootTag, Scalar>;

// Common dimensions
using SRCKF_1D = SRCKF<1>;
using SRCKF_2D = SRCKF<2>;
using SRCKF_3D = SRCKF<3>;
using SRCKF_6D = SRCKF<6>;

// ============================================================================
// Factory functions
// ============================================================================

/// @brief Create a CKF on Euclidean space with initial state
///
/// @param initial_state Initial mean state
/// @param initial_cov Initial covariance
/// @return Initialized CKF filter
template <int N, typename Scalar = double>
[[nodiscard]] inline auto make_ckf(
    const Eigen::Matrix<Scalar, N, 1>& initial_state,
    const SymmetricMatrix<N, Scalar>& initial_cov) noexcept -> CKF<N, Scalar> {
    CKF<N, Scalar> filter;
    filter.init(typename Euclidean<N, Scalar>::Point{initial_state}, initial_cov);
    return filter;
}

/// @brief Create a CKF on Lie group with initial state
///
/// @param initial_element Initial group element
/// @param initial_cov Initial covariance in tangent space
/// @return Initialized LieCKF filter
template <typename G, LieSide Side = LieSide::Right>
[[nodiscard]] inline auto make_lie_ckf(
    const G& initial_element,
    const SymmetricMatrix<G::VectorType::RowsAtCompileTime, typename G::Scalar>&
        initial_cov) noexcept -> LieCKF<G, Side, typename G::Scalar> {
    LieCKF<G, Side, typename G::Scalar> filter;
    filter.init(initial_element, initial_cov);
    return filter;
}

// ============================================================================
// Usage Notes
// ============================================================================

/// CKF is recommended when:
/// 1. Nonlinearities are moderate to strong (EKF may diverge)
/// 2. You want better accuracy than EKF without the complexity of UKF
/// 3. State dimension is not too large (cubature points = 2n)
/// 4. You prefer no tuning parameters (unlike UKF's α, β, κ)
///
/// Implementation details:
/// - Uses 2n cubature points for n-dimensional state
/// - Equal weights: w_i = 1/(2n)
/// - Points: ξ_i = √n * [I | -I] (columns of scaled identity matrix)
/// - 3rd order accuracy for polynomial nonlinearities
/// - Manifold-aware: works on both Euclidean spaces and Lie groups
///
/// Performance considerations:
/// - Computational cost: O(n³) (same as EKF, less than UKF)
/// - Memory: O(n²)
/// - Accuracy: Better than EKF, similar to UKF
/// - No matrix square root needed (unlike square-root UKF)
///
/// Limitations:
/// - Not suitable for very high-dimensional states (n > 100)
/// - 3rd order accuracy (vs arbitrary order for particle filters)
/// - Assumes Gaussian-like distributions

} // namespace uni_kf
