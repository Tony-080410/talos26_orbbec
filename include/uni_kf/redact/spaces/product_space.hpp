#pragma once

#include "../concepts.hpp"
#include "../space.hpp"

#include <Eigen/Core>
#include <utility>
#include <vector>

namespace uni_kf::redact::spaces {

/// ProductSpace combines two spaces into a composite space.
/// This allows representing states like SE(3) × R³ for pose-velocity.
///
/// Supports:
/// - Basic Space operations (always)
/// - Linearizable (when both S1 and S2 are Linearizable)
/// - Sampleable (when both S1 and S2 are Sampleable)
/// - InvariantLieGroup (when both S1 and S2 are InvariantLieGroup)
///
/// @tparam S1 First space type
/// @tparam S2 Second space type
template <Space S1, Space S2>
class ProductSpace {
public:
    using Flux        = std::pair<typename S1::Flux, typename S2::Flux>;
    using Drift       = std::pair<typename S1::Drift, typename S2::Drift>;
    using DriftExtent = std::pair<typename S1::DriftExtent, typename S2::DriftExtent>;

    static constexpr int Dim  = S1::Dim + S2::Dim;
    static constexpr int Dim1 = S1::Dim;
    static constexpr int Dim2 = S2::Dim;

    // ========================================================================
    // Basic Space Operations
    // ========================================================================

    /// Zero flux (identity element for groups, zero for Euclidean)
    [[nodiscard]] static Flux zero_flux() { return {S1::zero_flux(), S2::zero_flux()}; }

    /// Identity element (alias for zero_flux, for Lie group convention)
    [[nodiscard]] static Flux identity() { return zero_flux(); }

    /// Zero drift (no correction in either space)
    [[nodiscard]] static Drift zero_drift() { return {S1::zero_drift(), S2::zero_drift()}; }

    /// Compose flux with drift: x ⊕ δ
    [[nodiscard]] static Flux compose(const Flux& x, const Drift& dx) {
        return {S1::compose(x.first, dx.first), S2::compose(x.second, dx.second)};
    }

    /// Difference between two flux states: y ⊖ x (returns y - x in tangent space)
    [[nodiscard]] static Drift diff(const Flux& x, const Flux& y) {
        return {S1::diff(x.first, y.first), S2::diff(x.second, y.second)};
    }

    /// Norm of drift (combined Euclidean norm)
    [[nodiscard]] static double norm(const Drift& dx) {
        const double n1 = S1::norm(dx.first);
        const double n2 = S2::norm(dx.second);
        return std::sqrt(n1 * n1 + n2 * n2);
    }

    /// Inverse of drift: -δ
    [[nodiscard]] static Drift inverse(const Drift& dx) {
        return {S1::inverse(dx.first), S2::inverse(dx.second)};
    }

    /// Inverse of flux (for groups): x⁻¹
    [[nodiscard]] static Flux inverse(const Flux& x) {
        return {S1::inverse(x.first), S2::inverse(x.second)};
    }

    /// Apply extent constraint to drift
    [[nodiscard]] static Drift extent(const DriftExtent& ext, const Drift& dx) {
        return {S1::extent(ext.first, dx.first), S2::extent(ext.second, dx.second)};
    }

    // ========================================================================
    // Drift ↔ Vector Conversion Utilities
    // ========================================================================

    /// Convert Drift to Eigen vector (concatenate components)
    [[nodiscard]] static Eigen::VectorXd drift_to_vector(const Drift& dx) {
        Eigen::VectorXd v(Dim);
        if constexpr (requires { dx.first.data; }) {
            v.head(Dim1) = dx.first.data;
        } else {
            v.head(Dim1) = dx.first;
        }
        if constexpr (requires { dx.second.data; }) {
            v.tail(Dim2) = dx.second.data;
        } else {
            v.tail(Dim2) = dx.second;
        }
        return v;
    }

    /// Convert Eigen vector to Drift (split components)
    [[nodiscard]] static Drift vector_to_drift(const Eigen::VectorXd& v) {
        typename S1::Drift d1;
        typename S2::Drift d2;
        if constexpr (requires { d1.data; }) {
            d1.data = v.head(Dim1);
        } else {
            d1 = v.head(Dim1);
        }
        if constexpr (requires { d2.data; }) {
            d2.data = v.tail(Dim2);
        } else {
            d2 = v.tail(Dim2);
        }
        return {d1, d2};
    }

    // ========================================================================
    // Linearizable Support (conditional)
    // ========================================================================

    /// Dynamics Jacobian: block-diagonal [J_f1, 0; 0, J_f2]
    [[nodiscard]] static Eigen::MatrixXd dynamics_jacobian(const Flux& state, double dt)
        requires Linearizable<S1> && Linearizable<S2> {
        Eigen::MatrixXd J               = Eigen::MatrixXd::Zero(Dim, Dim);
        J.topLeftCorner(Dim1, Dim1)     = S1::dynamics_jacobian(state.first, dt);
        J.bottomRightCorner(Dim2, Dim2) = S2::dynamics_jacobian(state.second, dt);
        return J;
    }

    /// Measurement Jacobian: block-diagonal [J_h1, 0; 0, J_h2]
    [[nodiscard]] static Eigen::MatrixXd measurement_jacobian(const Flux& state)
        requires Linearizable<S1> && Linearizable<S2> {
        Eigen::MatrixXd J               = Eigen::MatrixXd::Zero(Dim, Dim);
        J.topLeftCorner(Dim1, Dim1)     = S1::measurement_jacobian(state.first);
        J.bottomRightCorner(Dim2, Dim2) = S2::measurement_jacobian(state.second);
        return J;
    }

    // ========================================================================
    // LinearDynamics Support (conditional)
    // ========================================================================

    /// State transition matrix: block-diagonal [F1, 0; 0, F2]
    [[nodiscard]] static Eigen::MatrixXd state_transition_matrix()
        requires LinearDynamics<S1> && LinearDynamics<S2> {
        Eigen::MatrixXd F               = Eigen::MatrixXd::Zero(Dim, Dim);
        F.topLeftCorner(Dim1, Dim1)     = S1::state_transition_matrix();
        F.bottomRightCorner(Dim2, Dim2) = S2::state_transition_matrix();
        return F;
    }

    // ========================================================================
    // InvariantGroup Support (conditional)
    // ========================================================================

    /// Exponential map: exp([ξ₁, ξ₂]) = [exp₁(ξ₁), exp₂(ξ₂)]
    [[nodiscard]] static Flux exp(const Drift& xi) requires requires(const typename S1::Drift& d1) {
        S1::exp(d1);
    } && requires(const typename S2::Drift& d2) { S2::exp(d2); } {
        return {S1::exp(xi.first), S2::exp(xi.second)};
    }

    /// Logarithm map: log([x₁, x₂]) = [log₁(x₁), log₂(x₂)]
    [[nodiscard]] static Drift log(const Flux& x) requires requires(const typename S1::Flux& f1) {
        S1::log(f1);
    } && requires(const typename S2::Flux& f2) { S2::log(f2); } {
        return {S1::log(x.first), S2::log(x.second)};
    }

    /// Adjoint matrix: block-diagonal [Ad1, 0; 0, Ad2]
    [[nodiscard]] static Eigen::MatrixXd adjoint_matrix(const Flux& X)
        requires requires(const typename S1::Flux& f1) { S1::adjoint_matrix(f1); }
              && requires(const typename S2::Flux& f2) { S2::adjoint_matrix(f2); } {
        Eigen::MatrixXd Ad               = Eigen::MatrixXd::Zero(Dim, Dim);
        Ad.topLeftCorner(Dim1, Dim1)     = S1::adjoint_matrix(X.first);
        Ad.bottomRightCorner(Dim2, Dim2) = S2::adjoint_matrix(X.second);
        return Ad;
    }

    /// Inverse adjoint matrix: block-diagonal [Ad⁻¹_1, 0; 0, Ad⁻¹_2]
    [[nodiscard]] static Eigen::MatrixXd inv_adjoint_matrix(const Flux& X)
        requires requires(const typename S1::Flux& f1) { S1::inv_adjoint_matrix(f1); }
              && requires(const typename S2::Flux& f2) { S2::inv_adjoint_matrix(f2); } {
        Eigen::MatrixXd Ad_inv               = Eigen::MatrixXd::Zero(Dim, Dim);
        Ad_inv.topLeftCorner(Dim1, Dim1)     = S1::inv_adjoint_matrix(X.first);
        Ad_inv.bottomRightCorner(Dim2, Dim2) = S2::inv_adjoint_matrix(X.second);
        return Ad_inv;
    }

    // ========================================================================
    // Sampleable Support (conditional)
    // ========================================================================

    /// Sigma point type for ProductSpace
    struct SigmaPoint {
        Flux point;         // The sigma point
        Drift drift;        // The tangent vector
        double weight_mean; // Weight for mean computation
        double weight_cov;  // Weight for covariance computation

        SigmaPoint() = default;
        SigmaPoint(const Flux& p, const Drift& d, double wm, double wc)
            : point(p)
            , drift(d)
            , weight_mean(wm)
            , weight_cov(wc) {}
    };

    /// WeightedPoint type alias for Sampleable concept compatibility
    using WeightedPoint = SigmaPoint;

    /// Generate sigma points around mean with given covariance
    [[nodiscard]] static std::vector<WeightedPoint> sigma_points(
        const Flux& mean, const Eigen::MatrixXd& cov, double alpha, double beta, double kappa)
        requires Sampleable<S1> && Sampleable<S2> {
        const int n         = Dim;
        const double lambda = alpha * alpha * (n + kappa) - n;

        // Weights
        const double w0_m = lambda / (n + lambda);
        const double w0_c = w0_m + (1 - alpha * alpha + beta);
        const double wi   = 1.0 / (2 * (n + lambda));

        std::vector<WeightedPoint> points;
        points.reserve(2 * n + 1);

        // Point 0: mean
        points.emplace_back(mean, zero_drift(), w0_m, w0_c);

        // Compute Cholesky: (n+λ) * P = SS^T
        Eigen::LLT<Eigen::MatrixXd> llt((n + lambda) * cov);
        if (llt.info() != Eigen::Success) {
            // Fallback: return just the mean point
            return points;
        }
        const Eigen::MatrixXd S = llt.matrixL();

        // Points 1..n: mean + columns of S
        for (int i = 0; i < n; ++i) {
            Drift xi = vector_to_drift(S.col(i));
            Flux pt  = compose(mean, xi);
            points.emplace_back(pt, xi, wi, wi);
        }

        // Points n+1..2n: mean - columns of S
        for (int i = 0; i < n; ++i) {
            Drift xi = vector_to_drift(-S.col(i));
            Flux pt  = compose(mean, xi);
            points.emplace_back(pt, xi, wi, wi);
        }

        return points;
    }

    /// Compute weighted mean from sigma points
    ///
    /// **Karcher Mean on Product Space** (iterative logarithmic averaging):
    ///   μ_{k+1} = μ_k ⊕ Σ w_i * (p_i ⊖ μ_k)
    ///
    /// Reference:
    /// - Lou, Y., et al. (2020). "Differentiating through the Fréchet Mean." ICML 2020.
    ///   [PDF](http://proceedings.mlr.press/v119/lou20a/lou20a-supp.pdf)
    /// - Tron, R., et al. (2011). "Fast and scalable Karcher means." EUSIPCO 2011.
    ///   [DOI](https://doi.org/10.1109/EUSIPCO.2011.6073997)
    [[nodiscard]] static Flux mean(const std::vector<WeightedPoint>& points)
        requires Sampleable<S1> && Sampleable<S2> {
        if (points.empty()) {
            return zero_flux();
        }

        // Start from first point's position
        Flux mu = points[0].point;

        // Iterative Karcher mean: μ_{k+1} = μ_k ⊕ Σ w_i * (p_i ⊖ μ_k)
        // 3 iterations provides good balance of accuracy and performance.
        for (int iter = 0; iter < 3; ++iter) {
            Eigen::VectorXd step = Eigen::VectorXd::Zero(Dim);
            for (const auto& wp : points) {
                Drift delta = diff(mu, wp.point);
                step += wp.weight_mean * drift_to_vector(delta);
            }
            mu = compose(mu, vector_to_drift(step));
        }

        return mu;
    }

    /// Compute weighted covariance from sigma points around mean
    [[nodiscard]] static Eigen::MatrixXd
        covariance(const std::vector<WeightedPoint>& points, const Flux& mu)
            requires Sampleable<S1> && Sampleable<S2> {
        Eigen::MatrixXd P = Eigen::MatrixXd::Zero(Dim, Dim);

        for (const auto& wp : points) {
            Eigen::VectorXd d = drift_to_vector(diff(mu, wp.point));
            P += wp.weight_cov * (d * d.transpose());
        }

        return P;
    }
};

} // namespace uni_kf::redact::spaces
