/// @file ukf.hpp
/// @brief Unscented Kalman Filter presets
///
/// Provides convenient type aliases for UKF configurations using sigma points.

#pragma once

#include "../core/filter.hpp"
#include "../strategies/sigma_points.hpp"

namespace uni_kf {

// ============================================================================
// UKF Presets
// ============================================================================

/// @brief Unscented Kalman Filter on Euclidean space
/// @tparam N State dimension
/// @tparam ReprTag Uncertainty representation (default: Covariance)
/// @tparam Scalar Floating point type
template <int N, typename ReprTag = CovarianceTag, typename Scalar = double>
using UKF = UnifiedKalmanFilter<Euclidean<N, Scalar>, SigmaPointsTag, ReprTag, Scalar>;

/// @brief Unscented Kalman Filter on a Lie group
template <
    typename G, LieSide Side = LieSide::Right, typename ReprTag = CovarianceTag,
    typename Scalar = double>
using LieUKF = UnifiedKalmanFilter<LieManifold<G, Side>, SigmaPointsTag, ReprTag, Scalar>;

// Common dimensions
using UKF_1D       = UKF<2>;
using UKF_2D       = UKF<4>;
using UKF_3D       = UKF<6>;
using UKF_3D_Accel = UKF<9>;

// ============================================================================
// Square Root UKF (SRUKF)
// ============================================================================

/// @brief Square Root Unscented Kalman Filter
/// More numerically stable than standard UKF
template <int N, typename Scalar = double>
using SRUKF = UnifiedKalmanFilter<Euclidean<N, Scalar>, SigmaPoints<>, SquareRootTag, Scalar>;

template <typename G, LieSide Side = LieSide::Right, typename Scalar = double>
using LieSRUKF = UnifiedKalmanFilter<LieManifold<G, Side>, SigmaPointsTag, SquareRootTag, Scalar>;

// Common dimensions
using SRUKF_1D       = SRUKF<2>;
using SRUKF_2D       = SRUKF<4>;
using SRUKF_3D       = SRUKF<6>;
using SRUKF_3D_Accel = SRUKF<9>;

// ============================================================================
// UKF Parameters
// ============================================================================

/// @brief Standard UKF parameters (α=1e-3, β=2, κ=0)
inline constexpr SigmaPointParams<double> StandardUKFParams{1e-3, 2.0, 0.0};

/// @brief More spread sigma points (α=1, β=2, κ=0) - better for highly nonlinear
inline constexpr SigmaPointParams<double> SpreadUKFParams{1.0, 2.0, 0.0};

/// @brief Minimal spread (α=1e-4, β=2, κ=0) - more linear-like behavior
inline constexpr SigmaPointParams<double> MinimalUKFParams{1e-4, 2.0, 0.0};

} // namespace uni_kf
