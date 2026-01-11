/// @file first_order.hpp
/// @brief First-order (Taylor expansion) propagation strategy for EKF/InEKF
///
/// Linearizes nonlinear functions using first-order Taylor expansion.
/// This is the classic EKF approach, also used in InEKF.

#pragma once

#include "../concepts/differential_space.hpp"
#include "../concepts/quadratic_form.hpp"

namespace uni_kf {

// ============================================================================
// FirstOrder Propagation Strategy
// ============================================================================

/// @brief First-order Taylor expansion propagation strategy
///
/// Linearizes dynamics f(x) around the current estimate x̂:
/// f(x) ≈ f(x̂) + F (x - x̂)  where F = ∂f/∂x|_{x̂}
///
/// For Lie groups, the linearization is in the tangent space:
/// f(X ⊕ δ) ≈ f(X) ⊕ (F δ)  in first order
struct FirstOrder {
    // ------------------------------------------------------------------------
    // Numerical differentiation settings
    // ------------------------------------------------------------------------

    /// Default step size for numerical Jacobian
    static constexpr double kDefaultEps = 1e-7;

    // ------------------------------------------------------------------------
    // Jacobian computation
    // ------------------------------------------------------------------------

    /// @brief Compute Jacobian of f: D1 → D2 at point x using numerical differentiation
    ///
    /// For DifferentialSpaces, the Jacobian is computed in tangent coordinates:
    /// F[i,j] = (f(act(x, eps*e_j)) ⊖ f(x)) / eps
    ///
    /// @tparam D1 Source DifferentialSpace
    /// @tparam D2 Target DifferentialSpace
    /// @tparam F Function type
    /// @param x Point at which to evaluate Jacobian
    /// @param f Function to differentiate
    /// @param eps Step size for finite differences
    template <DifferentialSpace D1, DifferentialSpace D2, typename F>
    [[nodiscard]] static auto jacobian(
        const typename D1::Point& x, F&& f,
        typename D1::Scalar eps = static_cast<typename D1::Scalar>(kDefaultEps)) noexcept
        -> Eigen::Matrix<typename D1::Scalar, D2::FiberDim, D1::FiberDim> {
        using Scalar    = typename D1::Scalar;
        constexpr int N = D1::FiberDim;
        constexpr int M = D2::FiberDim;
        using JacType   = Eigen::Matrix<Scalar, M, N>;
        using Fiber1    = typename D1::Fiber;
        using Point2    = typename D2::Point;

        JacType J;

        // f(x) at center
        Point2 fx = f(x);

        // Perturb each tangent direction
        for (int j = 0; j < N; ++j) {
            // Create perturbation in j-th direction
            typename Fiber1::EigenType delta_vec = Fiber1::EigenType::Zero();
            delta_vec(j)                         = eps;
            Fiber1 delta(delta_vec);

            // Perturbed point: x ⊕ δ
            auto x_plus = D1::act(x, delta);

            // f(x ⊕ δ)
            Point2 fx_plus = f(x_plus);

            // Finite difference in tangent space: (f(x⊕δ) ⊖ f(x)) / eps
            auto df  = D2::diff(fx, fx_plus);
            J.col(j) = df.data / eps;
        }

        return J;
    }

    /// @brief Compute Jacobian of f: D → D (self-map) at point x
    template <DifferentialSpace D, typename F>
    [[nodiscard]] static auto self_jacobian(
        const typename D::Point& x, F&& f,
        typename D::Scalar eps = static_cast<typename D::Scalar>(kDefaultEps)) noexcept
        -> Eigen::Matrix<typename D::Scalar, D::FiberDim, D::FiberDim> {
        return jacobian<D, D>(x, std::forward<F>(f), eps);
    }

    // ------------------------------------------------------------------------
    // State propagation (mean)
    // ------------------------------------------------------------------------

    /// @brief Propagate state through dynamics
    template <DifferentialSpace D, typename Dynamics>
    [[nodiscard]] static auto
        propagate_state(const typename D::Point& x, const Dynamics& dynamics) noexcept ->
        typename D::Point {
        return dynamics(x);
    }

    // ------------------------------------------------------------------------
    // Covariance propagation
    // ------------------------------------------------------------------------

    /// @brief Propagate covariance: P' = F P F^T + Q
    template <DifferentialSpace D, typename F_t, int N = D::FiberDim>
    [[nodiscard]] static auto propagate_covariance(
        const SymmetricMatrix<N, typename D::Scalar>& P, const F_t& F,
        const SymmetricMatrix<N, typename D::Scalar>& Q) noexcept
        -> SymmetricMatrix<N, typename D::Scalar> {
        using MatType = typename SymmetricMatrix<N, typename D::Scalar>::MatrixType;
        MatType P_new = F * P.matrix() * F.transpose() + Q.matrix();
        return SymmetricMatrix<N, typename D::Scalar>(P_new);
    }

    // ------------------------------------------------------------------------
    // Combined prediction step
    // ------------------------------------------------------------------------

    /// @brief Full prediction step: propagate mean and covariance
    ///
    /// @tparam D DifferentialSpace
    /// @param x Current state estimate
    /// @param P Current covariance
    /// @param dynamics State transition function x' = f(x)
    /// @param Q Process noise covariance
    /// @param eps Numerical differentiation step size
    template <DifferentialSpace D, typename Dynamics>
    [[nodiscard]] static auto predict(
        const typename D::Point& x, const SymmetricMatrix<D::FiberDim, typename D::Scalar>& P,
        const Dynamics& dynamics, const SymmetricMatrix<D::FiberDim, typename D::Scalar>& Q,
        typename D::Scalar eps = static_cast<typename D::Scalar>(kDefaultEps)) noexcept
        -> std::pair<typename D::Point, SymmetricMatrix<D::FiberDim, typename D::Scalar>> {
        // Propagate mean
        auto x_new = dynamics(x);

        // Compute Jacobian
        auto F = self_jacobian<D>(x, dynamics, eps);

        // Propagate covariance
        auto P_new = propagate_covariance<D>(P, F, Q);
        P_new.symmetrize_in_place();
        P_new.regularize_in_place();

        return {x_new, P_new};
    }

    // ------------------------------------------------------------------------
    // Measurement update helpers
    // ------------------------------------------------------------------------

    /// @brief Compute measurement Jacobian
    template <DifferentialSpace D, DifferentialSpace M, typename MeasFunc>
    [[nodiscard]] static auto measurement_jacobian(
        const typename D::Point& x, const MeasFunc& h,
        typename D::Scalar eps = static_cast<typename D::Scalar>(kDefaultEps)) noexcept
        -> Eigen::Matrix<typename D::Scalar, M::FiberDim, D::FiberDim> {
        return jacobian<D, M>(x, h, eps);
    }

    /// @brief Compute innovation (measurement residual)
    template <DifferentialSpace M>
    [[nodiscard]] static auto
        innovation(const typename M::Point& z_pred, const typename M::Point& z_actual) noexcept ->
        typename M::Fiber {
        return M::diff(z_pred, z_actual);
    }
};

// ============================================================================
// Analytical Jacobian helpers for common cases
// ============================================================================

/// @brief Trait for types that provide analytical Jacobians
template <typename T>
concept HasAnalyticalJacobian = requires(T t) {
    { t.jacobian() };
};

/// @brief Use analytical Jacobian if available, otherwise numerical
template <DifferentialSpace D, typename Dynamics>
[[nodiscard]] auto get_jacobian(
    const typename D::Point& x, const Dynamics& dynamics, typename D::Scalar eps = 1e-7) noexcept {
    if constexpr (HasAnalyticalJacobian<Dynamics>) {
        return dynamics.jacobian(x);
    } else {
        return FirstOrder::self_jacobian<D>(x, dynamics, eps);
    }
}

} // namespace uni_kf
