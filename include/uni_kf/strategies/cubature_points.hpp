/// @file cubature_points.hpp
/// @brief Cubature Kalman Filter (CKF) propagation strategy
///
/// CKF uses spherical-radial cubature rule for deterministic sampling:
/// - For n-dimensional state, uses 2n cubature points
/// - Equal weights: w_i = 1/(2n)
/// - Points: ξ_i = √n * [I | -I] (columns of [I; -I])
/// - 3rd order accuracy for polynomial nonlinearities
/// - No tuning parameters needed (unlike UKF)
///
/// Reference:
/// Arasaratnam & Haykin (2009), "Cubature Kalman Filters"
/// IEEE Transactions on Automatic Control

#pragma once

#include "../concepts/differential_space.hpp"
#include "../concepts/quadratic_form.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace uni_kf {

// ============================================================================
// CubaturePoints Strategy
// ============================================================================

/// @brief Cubature Kalman Filter propagation strategy
///
/// Propagates state and covariance using spherical-radial cubature rule.
/// Works on any DifferentialSpace (Euclidean or manifold).
struct CubaturePoints {
    /// @brief Generate cubature points in tangent space
    ///
    /// For n-dimensional state, generates 2n points:
    /// ξ_i = √n * e_i for i = 1..n
    /// ξ_{n+i} = -√n * e_i for i = 1..n
    ///
    /// @tparam D DifferentialSpace
    /// @param P Square root of covariance (Cholesky factor S where P = SS^T)
    /// @return 2n cubature points in tangent space (Dim x 2n matrix)
    template <DifferentialSpace D>
    [[nodiscard]] static auto generate_cubature_points(
        const Eigen::Matrix<typename D::Scalar, D::FiberDim, D::FiberDim>& S) noexcept
        -> Eigen::Matrix<typename D::Scalar, D::FiberDim, 2 * D::FiberDim> {

        using Scalar      = typename D::Scalar;
        constexpr int Dim = D::FiberDim;

        Eigen::Matrix<Scalar, Dim, 2 * Dim> points;

        // Cubature points: √n * [S | -S]
        const Scalar scale               = std::sqrt(static_cast<Scalar>(Dim));
        points.template leftCols<Dim>()  = scale * S;
        points.template rightCols<Dim>() = -scale * S;

        return points;
    }

    /// @brief Predict step using cubature rule
    ///
    /// @tparam D DifferentialSpace for state
    /// @param x Mean state
    /// @param P Covariance
    /// @param f Dynamics function x' = f(x)
    /// @param Q Process noise covariance
    /// @return Predicted mean and covariance
    template <DifferentialSpace D, typename Dynamics, typename CovType>
    [[nodiscard]] static auto predict(
        const typename D::Point& x, const CovType& P, Dynamics&& f, const CovType& Q) noexcept
        -> std::pair<typename D::Point, CovType> {

        using Scalar            = typename D::Scalar;
        using Point             = typename D::Point;
        using Fiber             = typename D::Fiber;
        constexpr int Dim       = D::FiberDim;
        constexpr int NumPoints = 2 * Dim;

        // 1. Compute square root of covariance (Cholesky decomposition)
        Eigen::Matrix<Scalar, Dim, Dim> S =
            P.matrix().template selfadjointView<Eigen::Lower>().llt().matrixL();

        // 2. Generate cubature points in tangent space
        auto xi_points = generate_cubature_points<D>(S);

        // 3. Transform cubature points to manifold and propagate
        std::vector<Point> X_propagated;
        X_propagated.reserve(NumPoints);

        for (int i = 0; i < NumPoints; ++i) {
            // Map tangent vector to manifold: X_i = x ⊕ ξ_i
            Fiber xi_i(xi_points.col(i));
            Point X_i = D::act(x, xi_i);

            // Propagate through dynamics: X_i' = f(X_i)
            Point X_i_prop = f(X_i);
            X_propagated.push_back(X_i_prop);
        }

        // 4. Compute predicted mean (weighted average on manifold)
        // For Euclidean space: simple average
        // For manifold: iterative mean on manifold (Karcher mean)
        Point x_pred = compute_mean<D>(X_propagated);

        // 5. Compute predicted covariance in tangent space
        Eigen::Matrix<Scalar, Dim, Dim> P_pred = Eigen::Matrix<Scalar, Dim, Dim>::Zero();

        constexpr Scalar weight = Scalar{1} / static_cast<Scalar>(NumPoints);
        for (const auto& X_i : X_propagated) {
            // Map back to tangent space: δ_i = X_i' ⊖ x_pred
            Fiber delta_i = D::diff(x_pred, X_i);
            P_pred += weight * (delta_i.data * delta_i.data.transpose());
        }

        // Add process noise
        P_pred += Q.matrix();

        return {x_pred, CovType(P_pred)};
    }

    /// @brief Compute measurement Jacobian using cubature points
    ///
    /// @tparam D State DifferentialSpace
    /// @tparam M Measurement DifferentialSpace
    /// @param x State
    /// @param P State covariance
    /// @param h Measurement function z = h(x)
    /// @return Tuple of (H, Pxy, Pzz) where:
    ///         H = measurement Jacobian (for reference)
    ///         Pxy = cross-covariance
    ///         Pzz = innovation covariance (measurement prediction cov)
    template <DifferentialSpace D, DifferentialSpace M, typename MeasFunc, typename CovType>
    [[nodiscard]] static auto measurement_covariances(
        const typename D::Point& x, const CovType& P, MeasFunc&& h,
        const SymmetricMatrix<M::FiberDim, typename D::Scalar>& R) noexcept
        -> std::tuple<
            Eigen::Matrix<typename D::Scalar, M::FiberDim, D::FiberDim>, // H (dummy)
            Eigen::Matrix<typename D::Scalar, D::FiberDim, M::FiberDim>, // Pxy
            Eigen::Matrix<typename D::Scalar, M::FiberDim, M::FiberDim>> // Pzz
    {
        using Scalar            = typename D::Scalar;
        using Point             = typename D::Point;
        using Fiber             = typename D::Fiber;
        using MPoint            = typename M::Point;
        using MFiber            = typename M::Fiber;
        constexpr int Dim       = D::FiberDim;
        constexpr int MeasDim   = M::FiberDim;
        constexpr int NumPoints = 2 * Dim;

        // 1. Compute square root of covariance
        Eigen::Matrix<Scalar, Dim, Dim> S =
            P.matrix().template selfadjointView<Eigen::Lower>().llt().matrixL();

        // 2. Generate cubature points
        auto xi_points = generate_cubature_points<D>(S);

        // 3. Transform and propagate through measurement function
        std::vector<MPoint> Z_points;
        Z_points.reserve(NumPoints);

        for (int i = 0; i < NumPoints; ++i) {
            Fiber xi_i(xi_points.col(i));
            Point X_i  = D::act(x, xi_i);
            MPoint Z_i = h(X_i);
            Z_points.push_back(Z_i);
        }

        // 4. Compute predicted measurement mean
        MPoint z_pred = compute_mean<M>(Z_points);

        // 5. Compute covariances
        Eigen::Matrix<Scalar, MeasDim, MeasDim> Pzz =
            Eigen::Matrix<Scalar, MeasDim, MeasDim>::Zero();
        Eigen::Matrix<Scalar, Dim, MeasDim> Pxy = Eigen::Matrix<Scalar, Dim, MeasDim>::Zero();

        constexpr Scalar weight = Scalar{1} / static_cast<Scalar>(NumPoints);
        for (int i = 0; i < NumPoints; ++i) {
            // State deviation in tangent space
            Fiber xi_i(xi_points.col(i));

            // Measurement deviation in tangent space
            MFiber dz_i = M::diff(z_pred, Z_points[i]);

            // Innovation covariance: Pzz
            Pzz += weight * (dz_i.data * dz_i.data.transpose());

            // Cross-covariance: Pxy
            Pxy += weight * (xi_i.data * dz_i.data.transpose());
        }

        // Add measurement noise to Pzz
        Pzz += R.matrix();

        // Dummy H (not used in CKF, but needed for interface compatibility)
        Eigen::Matrix<Scalar, MeasDim, Dim> H = Eigen::Matrix<Scalar, MeasDim, Dim>::Zero();

        return {H, Pxy, Pzz};
    }

private:
    /// @brief Compute mean on manifold (Karcher mean for Lie groups)
    ///
    /// For Euclidean space: simple average
    /// For manifold: iterative gradient descent on manifold
    template <DifferentialSpace D>
    [[nodiscard]] static auto compute_mean(const std::vector<typename D::Point>& points) noexcept ->
        typename D::Point {

        using Scalar      = typename D::Scalar;
        using Point       = typename D::Point;
        using Fiber       = typename D::Fiber;
        constexpr int Dim = D::FiberDim;

        if (points.empty()) {
            return D::identity();
        }

        // Initial guess: first point
        Point mean = points[0];

        // Iterative Karcher mean (gradient descent on manifold)
        constexpr int kMaxIter = 10;
        constexpr Scalar kTol  = Scalar{1e-8};

        for (int iter = 0; iter < kMaxIter; ++iter) {
            // Compute mean tangent vector
            Eigen::Matrix<Scalar, Dim, 1> delta_sum = Eigen::Matrix<Scalar, Dim, 1>::Zero();
            for (const auto& p : points) {
                Fiber delta = D::diff(mean, p);
                delta_sum += delta.data;
            }

            Fiber mean_delta(delta_sum / static_cast<Scalar>(points.size()));

            // Check convergence
            if (mean_delta.data.norm() < kTol) {
                break;
            }

            // Update mean: move in direction of mean tangent vector
            mean = D::act(mean, mean_delta);
        }

        return mean;
    }
};

} // namespace uni_kf
