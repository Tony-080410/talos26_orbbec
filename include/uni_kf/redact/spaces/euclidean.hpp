/// @file euclidean.hpp
/// @brief Euclidean space implementation for Space concept
///
/// Euclidean<N> is the trivial Space where Flux (state) = Drift (tangent).
/// This represents standard R^n vector spaces, serving as the base case
/// for the unified framework.
///
/// For Euclidean spaces:
/// - compose(x, dx) = x + dx (vector addition)
/// - diff(x, y) = y - x (vector subtraction)
/// - Jacobians are all identity matrices
///
/// @note This implementation uses the new Space concept API from redact/

#pragma once

#include "../concepts.hpp"
#include "../space.hpp"

#include <Eigen/Dense>
#include <concepts>
#include <utility>
#include <vector>

namespace uni_kf::redact::spaces {

// ============================================================================
// Euclidean Space
// ============================================================================

/// @brief Euclidean (vector) space as a Space
///
/// This is the trivial case where the state manifold is R^n and the
/// tangent space is identical to the state space itself.
///
/// @tparam N Dimension of the space (-1 for dynamic)
/// @tparam Scalar_ Floating point type (default: double)
template <int N, typename Scalar_ = double>
class Euclidean {
    static_assert(N > 0 || N == Eigen::Dynamic, "Dimension must be positive or Dynamic");

public:
    // ------------------------------------------------------------------------
    // Type definitions (required by Space concept)
    // ------------------------------------------------------------------------

    using Scalar = Scalar_;
    using Vector = Eigen::Matrix<Scalar, N, 1>;

    /// Flux: state (wrapped type to distinguish from Drift)
    struct Flux {
        Vector data;
        Flux() = default;
        explicit Flux(const Vector& v)
            : data(v) {}
        explicit Flux(Vector&& v)
            : data(std::move(v)) {}
    };

    /// Drift: tangent/error vector (wrapped type to distinguish from Flux)
    struct Drift {
        Vector data;
        Drift() = default;
        explicit Drift(const Vector& v)
            : data(v) {}
        explicit Drift(Vector&& v)
            : data(std::move(v)) {}
    };

    /// DriftExtent: bounded error extent (for clamping/filtering)
    struct DriftExtent {
        Scalar max_norm; // Maximum allowed norm
    };

    /// Weighted sigma point for UKF/CKF
    /// Uses dual weights: weight_mean for mean computation, weight_cov for covariance
    struct SigmaPoint {
        Flux point;         // The sigma point
        Drift drift;        // The tangent vector (for consistency with LieGroup interface)
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

    /// Compile-time dimension
    static constexpr int Dim = N;

    // ------------------------------------------------------------------------
    // Space interface
    // ------------------------------------------------------------------------

    /// @brief Identity element (zero vector)
    /// For Dynamic N, size parameter is required
    [[nodiscard]] static Flux zero_flux(int size = N) noexcept {
        if constexpr (N == Eigen::Dynamic) {
            return Flux{Vector::Zero(size)};
        } else {
            (void)size;
            return Flux{Vector::Zero()};
        }
    }

    /// @brief Identity element (zero drift)
    /// For Dynamic N, size parameter is required
    [[nodiscard]] static Drift zero_drift(int size = N) noexcept {
        if constexpr (N == Eigen::Dynamic) {
            return Drift{Vector::Zero(size)};
        } else {
            (void)size;
            return Drift{Vector::Zero()};
        }
    }

    /// @brief Composition: x ⊕ dx = x + dx (vector addition)
    [[nodiscard]] static Flux compose(const Flux& x, const Drift& dx) noexcept {
        return Flux{x.data + dx.data};
    }

    /// @brief Difference: y ⊖ x = y - x (vector subtraction)
    [[nodiscard]] static Drift diff(const Flux& x, const Flux& y) noexcept {
        return Drift{y.data - x.data};
    }

    /// @brief L2 norm of drift
    [[nodiscard]] static Scalar norm(const Drift& dx) noexcept { return dx.data.norm(); }

    /// @brief Inverse of drift: -dx
    [[nodiscard]] static Drift inverse(const Drift& dx) noexcept { return Drift{-dx.data}; }

    /// @brief Inverse of flux: -x
    [[nodiscard]] static Flux inverse(const Flux& x) noexcept { return Flux{-x.data}; }

    /// @brief Apply drift extent (clamp to maximum norm)
    [[nodiscard]] static Drift extent(const DriftExtent& ext, const Drift& dx) noexcept {
        const Scalar n = norm(dx);
        if (n > ext.max_norm && n > Scalar(0)) {
            return Drift{dx.data * (ext.max_norm / n)};
        }
        return dx;
    }

    // ------------------------------------------------------------------------
    // LinearDynamics interface (for KF)
    // ------------------------------------------------------------------------

    /// @brief Constant state transition matrix F = I
    /// @param size State dimension (required for Dynamic N)
    [[nodiscard]] static Eigen::Matrix<Scalar, -1, -1>
        state_transition_matrix(int size = N) noexcept {
        return Eigen::Matrix<Scalar, -1, -1>::Identity(size, size);
    }

    /// @brief Constant measurement matrix H (identity for direct measurement)
    /// @param meas_dim Measurement dimension
    /// @param size State dimension (required for Dynamic N)
    [[nodiscard]] static Eigen::Matrix<Scalar, -1, -1>
        measurement_matrix(int meas_dim = N, int size = N) noexcept {
        if constexpr (N == Eigen::Dynamic) {
            return Eigen::Matrix<Scalar, -1, -1>::Identity(meas_dim, size);
        } else {
            (void)size;
            return Eigen::Matrix<Scalar, -1, -1>::Identity(meas_dim, N);
        }
    }

    // ------------------------------------------------------------------------
    // Linearizable interface (for EKF)
    // ------------------------------------------------------------------------

    /// @brief Dynamics Jacobian: df/dx at (state, dt)
    /// For position-velocity models [x, y, vx, vy], returns:
    ///   F = [1, 0, dt,  0;
    ///        0, 1,  0, dt;
    ///        0, 0,  1,  0;
    ///        0, 0,  0,  1]
    ///
    /// This is the **constant velocity (CV) model** state transition matrix.
    /// For each position-velocity pair [p_i, v_i]:
    ///   p_i(t+dt) = p_i(t) + v_i(t) * dt
    ///   v_i(t+dt) = v_i(t)
    ///
    /// Reference:
    /// - Barfoot 2024, Section 4.2.1: Discrete-time constant velocity model
    /// - Oxford Estimation II: [Lecture Notes
    /// 2](https://www.robots.ox.ac.uk/~ian/Teaching/Estimation/LectureNotes2.pdf)
    /// - MathWorks constvel:
    /// [Documentation](https://www.mathworks.com/help/driving/ref/constvel.html)
    /// - Kalman Filter for 2D Motion: [Cookie Robotics](https://cookierobotics.com/071/)
    ///
    /// @param state Current state (unused for linear CV model)
    /// @param dt Time step
    /// @param size State dimension (required for Dynamic N)
    [[nodiscard]] static Eigen::Matrix<Scalar, -1, -1>
        dynamics_jacobian(const Flux& state, double dt, int size = N) noexcept {
        (void)state;
        // For position-velocity models (dim >= 4 and even), create proper CV Jacobian
        // Reference: Constant velocity model state transition matrix
        if (size >= 4 && size % 2 == 0) {
            Eigen::Matrix<Scalar, -1, -1> F = Eigen::Matrix<Scalar, -1, -1>::Identity(size, size);
            int pos_vel_pairs               = size / 2;
            for (int i = 0; i < pos_vel_pairs; ++i) {
                F(i, i + pos_vel_pairs) = dt; // Position depends on velocity
            }
            return F;
        }
        return Eigen::Matrix<Scalar, -1, -1>::Identity(size, size);
    }

    /// @brief Measurement Jacobian: dh/dx = H (from measurement_matrix)
    /// @param size State dimension (required for Dynamic N)
    [[nodiscard]] static Eigen::Matrix<Scalar, -1, -1>
        measurement_jacobian(const Flux&, int size = N) noexcept {
        return measurement_matrix(size, size);
    }

    // ------------------------------------------------------------------------
    // Sampleable interface (for UKF/CKF)
    // ------------------------------------------------------------------------

    /// @brief Generate sigma points using unscented transform
    ///
    /// **UKF Formulas** (Julier & Uhlmann 2004, Sec. III):
    ///   Lambda: λ = α²(n + κ) - n
    ///   Weights:
    ///     w₀^m = λ/(n+λ)           (mean weight for center point)
    ///     w₀^c = w₀^m + (1-α²+β) (covariance weight for center point)
    ///     wᵢ = 1/(2(n+λ))          (weights for remaining points)
    ///
    /// Reference:
    /// - Julier, S. J., & Uhlmann, J. K. (2004). "Unscented Filtering and Nonlinear Estimation."
    ///   *Proceedings of the IEEE*, 92(3), 401-422.
    ///   [PDF](https://www.cs.ubc.ca/~murphyk/Papers/Julier_Uhlmann_mar04.pdf)
    ///   - Sec. III: The Unscented Transform
    ///
    /// @param mean Center state
    /// @param cov State covariance
    /// @param alpha Unscented transform parameter (default: 1e-3)
    /// @param beta Unscented transform parameter (default: 2)
    /// @param kappa Unscented transform parameter (default: 0)
    [[nodiscard]] static std::vector<SigmaPoint> sigma_points(
        const Flux& mean, const Eigen::MatrixXd& cov, double alpha = 1e-3, double beta = 2.0,
        double kappa = 0.0) noexcept {

        const int n         = mean.data.size();
        const double lambda = alpha * alpha * (n + kappa) - n;

        // UKF weights (Julier & Uhlmann 2004, Sec. III)
        const double w0_m = lambda / (n + lambda);
        const double w0_c = w0_m + (1 - alpha * alpha + beta);
        const double wi   = 1.0 / (2 * (n + lambda));

        // Compute Cholesky: (n+λ) * P = SS^T
        Eigen::LLT<Eigen::MatrixXd> llt((n + lambda) * cov);
        if (llt.info() != Eigen::Success) {
            // Fallback: return just the mean point
            return {
                SigmaPoint{mean, Drift{Vector::Zero()}, w0_m, w0_c}
            };
        }
        const Eigen::MatrixXd S = llt.matrixL();

        std::vector<SigmaPoint> points;
        points.reserve(2 * n + 1);

        // Point 0: mean (with different weights for mean and covariance)
        points.emplace_back(mean, Drift{Vector::Zero()}, w0_m, w0_c);

        // Points 1..n: mean + columns of S
        for (int i = 0; i < n; ++i) {
            Vector col = S.col(i);
            points.emplace_back(Flux{mean.data + col}, Drift{col}, wi, wi);
        }

        // Points n+1..2n: mean - columns of S
        for (int i = 0; i < n; ++i) {
            Vector col = S.col(i);
            points.emplace_back(Flux{mean.data - col}, Drift{-col}, wi, wi);
        }

        return points;
    }

    /// @brief Compute weighted mean from sigma points
    [[nodiscard]] static Flux mean(const std::vector<SigmaPoint>& points) noexcept {
        Vector result = Vector::Zero(points[0].point.data.size());
        for (const auto& pt : points) {
            result += pt.weight_mean * pt.point.data;
        }
        return Flux{result};
    }

    /// @brief Compute weighted covariance from sigma points around mean
    [[nodiscard]] static Eigen::MatrixXd
        covariance(const std::vector<SigmaPoint>& points, const Flux& mean) noexcept {

        const int n            = mean.data.size();
        Eigen::MatrixXd result = Eigen::MatrixXd::Zero(n, n);

        for (const auto& pt : points) {
            const Vector diff = pt.point.data - mean.data;
            result += pt.weight_cov * (diff * diff.transpose());
        }

        return result;
    }

    // ------------------------------------------------------------------------
    // Lie group operations (for InvariantGroup compatibility)
    // ------------------------------------------------------------------------

    /// @brief Exponential map: exp(v) = v (identity for Euclidean space)
    /// For Euclidean spaces, the tangent space is identical to the manifold
    [[nodiscard]] static Flux exp(const Drift& v) noexcept { return Flux{v.data}; }

    /// @brief Logarithm map: log(x) = x.data (identity for Euclidean space)
    /// For Euclidean spaces, the manifold is identical to the tangent space
    [[nodiscard]] static Drift log(const Flux& x) noexcept { return Drift{x.data}; }

    /// @brief Identity element (alias for zero_flux)
    [[nodiscard]] static Flux identity() noexcept { return zero_flux(); }

    /// @brief Adjoint matrix: Ad_x = I (trivial for Euclidean space)
    /// Euclidean spaces are Abelian, so the Adjoint action is trivial
    [[nodiscard]] static Eigen::Matrix<Scalar, -1, -1>
        adjoint_matrix(const Flux&, int size = N) noexcept {
        return Eigen::Matrix<Scalar, -1, -1>::Identity(size, size);
    }

    /// @brief Inverse Adjoint matrix: Ad_x⁻¹ = I (trivial for Euclidean space)
    [[nodiscard]] static Eigen::Matrix<Scalar, -1, -1>
        inv_adjoint_matrix(const Flux&, int size = N) noexcept {
        return Eigen::Matrix<Scalar, -1, -1>::Identity(size, size);
    }

    // ------------------------------------------------------------------------
    // Utility functions
    // ------------------------------------------------------------------------

    /// @brief Distance between two points
    [[nodiscard]] static Scalar distance(const Flux& x, const Flux& y) noexcept {
        return diff(x, y).data.norm();
    }

    /// @brief Squared distance between two points
    [[nodiscard]] static Scalar squared_distance(const Flux& x, const Flux& y) noexcept {
        return diff(x, y).data.squaredNorm();
    }

    /// @brief Linear interpolation between two points
    [[nodiscard]] static Flux lerp(const Flux& x, const Flux& y, Scalar t) noexcept {
        return compose(x, Drift{diff(x, y).data * t});
    }

    /// @brief Create Flux from vector
    [[nodiscard]] static Flux from_vector(const Vector& v) noexcept { return Flux{v}; }

    /// @brief Convert Flux to vector
    [[nodiscard]] static Vector to_vector(const Flux& f) noexcept { return f.data; }
};

// ============================================================================
// Type Aliases for Common Euclidean Spaces
// ============================================================================

using R1  = Euclidean<1>;
using R2  = Euclidean<2>;
using R3  = Euclidean<3>;
using R4  = Euclidean<4>;
using R6  = Euclidean<6>;
using R9  = Euclidean<9>;
using R12 = Euclidean<12>;
using R15 = Euclidean<15>;
using R18 = Euclidean<18>;

} // namespace uni_kf::redact::spaces
