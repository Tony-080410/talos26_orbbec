/// @file sigma_points.hpp
/// @brief Sigma point (unscented) propagation strategy for UKF
///
/// Propagates mean and covariance through nonlinear functions using
/// deterministic sigma point sampling. Works on any DifferentialSpace.

#pragma once

#include "../concepts/differential_space.hpp"
#include "../concepts/quadratic_form.hpp"

#include <array>
#include <cmath>

namespace uni_kf {

// ============================================================================
// Sigma Point Parameters
// ============================================================================

/// @brief Parameters for sigma point generation
template <typename Scalar = double>
struct SigmaPointParams {
    Scalar alpha = Scalar{1.0}; // Spread of sigma points (small = close to mean)
    Scalar beta  = Scalar{2};   // Prior knowledge (2 = Gaussian)
    Scalar kappa = Scalar{0};   // Secondary scaling parameter

    /// @brief Compute lambda for n-dimensional state
    [[nodiscard]] constexpr Scalar lambda(int n) const noexcept {
        return alpha * alpha * (n + kappa) - n;
    }
};

// ============================================================================
// SigmaPoints Propagation Strategy
// ============================================================================

/// @brief Unscented Transform propagation strategy
///
/// Generates 2n+1 sigma points around the mean, propagates them through
/// the nonlinear function, and reconstructs mean and covariance.
///
/// For Lie groups, sigma points are generated in the tangent space and
/// mapped to the group via the exponential map.
///
/// @tparam Params_ Sigma point parameters (default: standard UKF)
template <typename Params_ = SigmaPointParams<double>>
struct SigmaPoints {
    using Params = Params_;

    // ------------------------------------------------------------------------
    // Sigma point generation
    // ------------------------------------------------------------------------

    /// @brief Generate sigma points for a DifferentialSpace
    ///
    /// Sigma points are generated in tangent space then mapped to the manifold:
    /// χ_0 = x̂
    /// χ_i = x̂ ⊕ (√((n+λ)P))_i      for i = 1, ..., n
    /// χ_{n+i} = x̂ ⊕ (-(√((n+λ)P))_i)  for i = 1, ..., n
    ///
    /// @return Array of 2n+1 sigma points
    template <DifferentialSpace D>
    [[nodiscard]] static auto generate_sigma_points(
        const typename D::Point& mean, const SymmetricMatrix<D::FiberDim, typename D::Scalar>& P,
        const Params& params = Params{}) noexcept
        -> std::array<typename D::Point, 2 * D::FiberDim + 1> {
        using Scalar    = typename D::Scalar;
        constexpr int N = D::FiberDim;
        using Point     = typename D::Point;
        using Fiber     = typename D::Fiber;

        std::array<Point, 2 * N + 1> sigma_points;

        Scalar lambda = params.lambda(N);
        Scalar scale  = std::sqrt(N + lambda);

        // Cholesky decomposition: L where P = L L^T
        auto L_result = P.cholesky();
        typename SymmetricMatrix<N, Scalar>::MatrixType L;
        if (L_result) {
            L = *L_result;
        } else {
            // Fallback: diagonal sqrt
            L = SymmetricMatrix<N, Scalar>::MatrixType::Zero();
            for (int i = 0; i < N; ++i) {
                L(i, i) = std::sqrt(std::max(P(i, i), Scalar{0}));
            }
        }

        // Scale the Cholesky factor
        L *= scale;

        // Central sigma point
        sigma_points[0] = mean;

        // Positive and negative perturbations
        for (int i = 0; i < N; ++i) {
            typename Fiber::EigenType col = L.col(i);

            // Positive perturbation
            sigma_points[1 + i] = D::act(mean, Fiber(col));

            // Negative perturbation
            sigma_points[1 + N + i] = D::act(mean, Fiber(-col));
        }

        return sigma_points;
    }

    // ------------------------------------------------------------------------
    // Weights
    // ------------------------------------------------------------------------

    /// @brief Compute mean weights for sigma points
    template <int N, typename Scalar>
    [[nodiscard]] static auto mean_weights(const Params& params = Params{}) noexcept
        -> std::array<Scalar, 2 * N + 1> {
        std::array<Scalar, 2 * N + 1> W_m;

        Scalar lambda = params.lambda(N);

        // Weight for central point
        W_m[0] = lambda / (N + lambda);

        // Weights for perturbation points
        Scalar w = Scalar{1} / (Scalar{2} * (N + lambda));
        for (int i = 1; i <= 2 * N; ++i) {
            W_m[i] = w;
        }

        return W_m;
    }

    /// @brief Compute covariance weights for sigma points
    template <int N, typename Scalar>
    [[nodiscard]] static auto cov_weights(const Params& params = Params{}) noexcept
        -> std::array<Scalar, 2 * N + 1> {
        std::array<Scalar, 2 * N + 1> W_c;

        Scalar lambda = params.lambda(N);

        // Weight for central point (includes beta correction)
        W_c[0] = lambda / (N + lambda) + (Scalar{1} - params.alpha * params.alpha + params.beta);

        // Weights for perturbation points
        Scalar w = Scalar{1} / (Scalar{2} * (N + lambda));
        for (int i = 1; i <= 2 * N; ++i) {
            W_c[i] = w;
        }

        return W_c;
    }

    // ------------------------------------------------------------------------
    // Mean reconstruction on manifold
    // ------------------------------------------------------------------------

    /// @brief Compute weighted mean of points on a DifferentialSpace
    ///
    /// For Lie groups, we use iterative mean computation in tangent space.
    template <DifferentialSpace D, std::size_t NumPoints>
    [[nodiscard]] static auto weighted_mean(
        const std::array<typename D::Point, NumPoints>& points,
        const std::array<typename D::Scalar, NumPoints>& weights, int max_iter = 10,
        typename D::Scalar tol = 1e-9) noexcept -> typename D::Point {
        using Scalar = typename D::Scalar;
        using Point  = typename D::Point;
        using Fiber  = typename D::Fiber;

        // Initialize with first point (or weighted average for Euclidean)
        Point mean = points[0];

        // Iterative refinement for manifolds
        for (int iter = 0; iter < max_iter; ++iter) {
            // Compute weighted sum of tangent vectors to current mean
            typename Fiber::EigenType delta_sum = Fiber::EigenType::Zero();
            for (std::size_t i = 0; i < NumPoints; ++i) {
                auto delta = D::diff(mean, points[i]);
                delta_sum += weights[i] * delta.data;
            }

            // Check convergence
            if (delta_sum.norm() < tol) {
                break;
            }

            // Update mean
            mean = D::act(mean, Fiber(delta_sum));
        }

        return mean;
    }

    // ------------------------------------------------------------------------
    // Covariance reconstruction
    // ------------------------------------------------------------------------

    /// @brief Compute weighted covariance around mean
    template <DifferentialSpace D, std::size_t NumPoints>
    [[nodiscard]] static auto weighted_covariance(
        const typename D::Point& mean, const std::array<typename D::Point, NumPoints>& points,
        const std::array<typename D::Scalar, NumPoints>& weights) noexcept
        -> SymmetricMatrix<D::FiberDim, typename D::Scalar> {
        using Scalar     = typename D::Scalar;
        constexpr int N  = D::FiberDim;
        using MatrixType = Eigen::Matrix<Scalar, N, N>;

        MatrixType P = MatrixType::Zero();

        for (std::size_t i = 0; i < NumPoints; ++i) {
            auto delta = D::diff(mean, points[i]);
            P += weights[i] * (delta.data * delta.data.transpose());
        }

        return SymmetricMatrix<N, Scalar>(P);
    }

    /// @brief Compute weighted cross-covariance between two spaces
    template <DifferentialSpace D1, DifferentialSpace D2, std::size_t NumPoints>
    [[nodiscard]] static auto weighted_cross_covariance(
        const typename D1::Point& mean1, const std::array<typename D1::Point, NumPoints>& points1,
        const typename D2::Point& mean2, const std::array<typename D2::Point, NumPoints>& points2,
        const std::array<typename D1::Scalar, NumPoints>& weights) noexcept
        -> Eigen::Matrix<typename D1::Scalar, D1::FiberDim, D2::FiberDim> {
        using Scalar     = typename D1::Scalar;
        constexpr int N1 = D1::FiberDim;
        constexpr int N2 = D2::FiberDim;
        using MatrixType = Eigen::Matrix<Scalar, N1, N2>;

        MatrixType Pxy = MatrixType::Zero();

        for (std::size_t i = 0; i < NumPoints; ++i) {
            auto delta1 = D1::diff(mean1, points1[i]);
            auto delta2 = D2::diff(mean2, points2[i]);
            Pxy += weights[i] * (delta1.data * delta2.data.transpose());
        }

        return Pxy;
    }

    // ------------------------------------------------------------------------
    // Full prediction step
    // ------------------------------------------------------------------------

    /// @brief Full UKF prediction step
    template <DifferentialSpace D, typename Dynamics>
    [[nodiscard]] static auto predict(
        const typename D::Point& x, const SymmetricMatrix<D::FiberDim, typename D::Scalar>& P,
        const Dynamics& dynamics, const SymmetricMatrix<D::FiberDim, typename D::Scalar>& Q,
        const Params& params = Params{}) noexcept
        -> std::pair<typename D::Point, SymmetricMatrix<D::FiberDim, typename D::Scalar>> {
        using Scalar    = typename D::Scalar;
        constexpr int N = D::FiberDim;

        // Generate sigma points
        auto sigma_points = generate_sigma_points<D>(x, P, params);

        // Propagate sigma points through dynamics
        std::array<typename D::Point, 2 * N + 1> propagated;
        for (std::size_t i = 0; i < sigma_points.size(); ++i) {
            propagated[i] = dynamics(sigma_points[i]);
        }

        // Compute weights
        auto W_m = mean_weights<N, Scalar>(params);
        auto W_c = cov_weights<N, Scalar>(params);

        // Reconstruct mean
        auto x_new = weighted_mean<D>(propagated, W_m);

        // Reconstruct covariance and add process noise
        auto P_new = weighted_covariance<D>(x_new, propagated, W_c);
        P_new += Q;
        P_new.symmetrize_in_place();
        P_new.regularize_in_place();

        return {x_new, P_new};
    }

    // ------------------------------------------------------------------------
    // Measurement update
    // ------------------------------------------------------------------------

    /// @brief Full UKF update step
    template <DifferentialSpace D, DifferentialSpace M, typename MeasFunc>
    [[nodiscard]] static auto update(
        const typename D::Point& x, const SymmetricMatrix<D::FiberDim, typename D::Scalar>& P,
        const typename M::Point& z, const MeasFunc& h,
        const SymmetricMatrix<M::FiberDim, typename D::Scalar>& R,
        const Params& params = Params{}) noexcept
        -> std::pair<typename D::Point, SymmetricMatrix<D::FiberDim, typename D::Scalar>> {
        using Scalar        = typename D::Scalar;
        constexpr int N     = D::FiberDim;
        constexpr int M_dim = M::FiberDim;

        // Generate sigma points
        auto sigma_points = generate_sigma_points<D>(x, P, params);

        // Propagate through measurement function
        std::array<typename M::Point, 2 * N + 1> z_sigma;
        for (std::size_t i = 0; i < sigma_points.size(); ++i) {
            z_sigma[i] = h(sigma_points[i]);
        }

        // Compute weights
        auto W_m = mean_weights<N, Scalar>(params);
        auto W_c = cov_weights<N, Scalar>(params);

        // Predicted measurement mean
        auto z_pred = weighted_mean<M>(z_sigma, W_m);

        // Innovation covariance
        auto S = weighted_covariance<M>(z_pred, z_sigma, W_c);
        S += R;

        // Cross covariance
        auto Pxz = weighted_cross_covariance<D, M>(x, sigma_points, z_pred, z_sigma, W_c);

        // Kalman gain: K = Pxz * S^{-1}
        auto S_inv_result = S.inverse();
        if (!S_inv_result) {
            return {x, P}; // Fallback: no update
        }
        auto K = Pxz * S_inv_result->matrix();

        // Innovation
        auto innovation = M::diff(z_pred, z);

        // State update
        typename D::Fiber delta(K * innovation.data);
        auto x_new = D::act(x, delta);

        // Covariance update
        using MatType = typename SymmetricMatrix<N, Scalar>::MatrixType;
        MatType P_new = P.matrix() - K * S.matrix() * K.transpose();
        SymmetricMatrix<N, Scalar> P_updated(P_new);
        P_updated.symmetrize_in_place();
        P_updated.regularize_in_place();

        return {x_new, P_updated};
    }
};

// ============================================================================
// Common sigma point configurations
// ============================================================================

/// @brief Standard UKF parameters
using StandardUKF = SigmaPoints<SigmaPointParams<double>>;

/// @brief Scaled UKF (smaller spread, often more stable)
struct ScaledUKFParams {
    double alpha = 1e-3;
    double beta  = 2;
    double kappa = 0;

    [[nodiscard]] constexpr double lambda(int n) const noexcept {
        return alpha * alpha * (n + kappa) - n;
    }
};
using ScaledUKF = SigmaPoints<ScaledUKFParams>;

} // namespace uni_kf
