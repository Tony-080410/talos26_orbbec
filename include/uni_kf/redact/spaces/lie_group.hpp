#pragma once
#include "../concepts.hpp" // Use canonical concepts (LieGroup, etc.)
#include "../space.hpp"
#include <Eigen/Core>
#include <concepts>
#include <vector>

// lieplusplus headers for Lie group operations
#include <groups/SEn3.hpp>
#include <groups/SO3.hpp>

namespace uni_kf::redact::spaces {

/// @file lie_group.hpp
/// @brief Lie group state spaces (SO(3), SE(3), SE_2(3)) for Kalman filtering
///
/// Implements the matrix Lie groups commonly used in robotics state estimation:
/// - SO(3): 3D rotation group
/// - SE(3): 3D rigid-body transformations (rotation + translation)
/// - SE_2(3): Extended pose with velocity (9 DOF)
///
/// Key mathematical operations:
/// - **Exponential map**: exp(ω) = R (Rodrigues formula for SO(3))
/// - **Logarithm map**: log(R) = ω (axis-angle representation)
/// - **Adjoint**: Ad_R transforms tangent vectors between frames
/// - **Left/Right Jacobians**: J_l(ω), J_r(ω) for derivatives of exp/log
///
/// References:
/// - Barfoot 2024: *State Estimation for Robotics*, Chapter 7
///   [PDF](http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser24.pdf)
///   - Section 7.1.5: Rodrigues formula for SO(3)
///   - Section 7.1.6: Left and right Jacobians
/// - Solà, J., et al. (2018). "A micro Lie theory for state estimation in robotics."
///   *ICRA 2018*. [arXiv:1812.01537](https://arxiv.org/pdf/1812.01537)
///   - Appendix B: Derivatives of exp/log on SO(3) and SE(3)
/// - Hartley, R., et al. (2020). "On the scale of SO(3) covariance estimation."
///   *ICRA 2020*. [arXiv:1912.01360](https://arxiv.org/pdf/1912.01360)
///   - Section III: Jacobians of exponential map
///
/// Error Convention:
/// - **Right-invariant**: ξ_R = Log(X̂⁻¹X) (used by default, more common in robotics)
/// - **Left-invariant**: ξ_L = Log(XX̂⁻¹)
///
/// Import canonical LieGroup concept into this namespace for convenience
using uni_kf::redact::LieGroup;

// ============================================================================
// LieSide enum - 指定不变误差的定义方式
// ============================================================================

enum class LieSide {
    Right, // Right-invariant error: ξ = Log(X̂⁻¹X)
    Left   // Left-invariant error: ξ = Log(XX̂⁻¹)
};

// ============================================================================
// SigmaPoint - Weighted sigma point for UKF/CKF
// ============================================================================

/// @brief Weighted sigma point for sigma-point transforms
template <typename Flux_, typename Drift_>
struct SigmaPoint {
    using Flux  = Flux_;
    using Drift = Drift_;

    Flux point;         // The sigma point on the manifold
    Drift drift;        // The tangent vector (for Euclidean spaces)
    double weight_mean; // Weight for mean computation
    double weight_cov;  // Weight for covariance computation

    SigmaPoint() = default;
    SigmaPoint(const Flux& p, const Drift& d, double wm, double wc)
        : point(p)
        , drift(d)
        , weight_mean(wm)
        , weight_cov(wc) {}
};

// ============================================================================
// Linearizable LieGroup - 支持线性化 (EKF/InEKF)
// ============================================================================

/// @brief Concept for Lie groups with analytical Jacobians
///
/// Extends the canonical LieGroup concept with Jacobian methods for
/// sigma-point transforms and left/right Jacobians of exponential map.
/// This is more specific than the general DynamicsLinearizable concept.
template <typename T>
concept LinearizableLieGroup =
    LieGroup<T> && requires(typename T::Flux x, typename T::Drift xi, double dt) {
        /// Dynamics Jacobian: J_f = df/dx at current state
        { T::dynamics_jacobian(x, dt) } -> std::convertible_to<Eigen::MatrixXd>;

        /// Left Jacobian of exponential: J_l(ξ)
        { T::left_jacobian(xi) } -> std::convertible_to<Eigen::MatrixXd>;

        /// Inverse left Jacobian: J_l(ξ)⁻¹
        { T::inv_left_jacobian(xi) } -> std::convertible_to<Eigen::MatrixXd>;
    };

// ============================================================================
// Sampleable LieGroup - 支持 Sigma-Point 变换 (UKF/CKF)
// ============================================================================

/// @brief Concept for Lie groups with sigma-point sampling
///
/// This is the Lie-group-specific version of the canonical Sampleable concept.
/// It uses SigmaPoint with explicit Flux/Drift types rather than WeightedPoint.
template <typename T>
concept SampleableLieGroup =
    LieGroup<T>
    && requires(
        typename T::Flux mean, Eigen::MatrixXd cov, double alpha, double beta, double kappa) {
           /// Generate weighted sigma points around mean with given covariance
           {
               T::sigma_points(mean, cov, alpha, beta, kappa)
           } -> std::same_as<std::vector<SigmaPoint<typename T::Flux, typename T::Drift>>>;

           /// Compute weighted mean from sigma points
           {
               T::mean(std::declval<std::vector<SigmaPoint<typename T::Flux, typename T::Drift>>>())
           } -> std::same_as<typename T::Flux>;

           /// Compute weighted covariance from sigma points around mean
           {
               T::covariance(
                   std::declval<std::vector<SigmaPoint<typename T::Flux, typename T::Drift>>>(),
                   mean)
           } -> std::same_as<Eigen::MatrixXd>;
       };

// ============================================================================
// SO(3) - 3D Rotation Group
// ============================================================================

/// @brief SO(3) rotation group with full concept support
///
/// Implements:
/// - Space: compose, diff, norm, inverse
/// - LieGroup: exp, log, adjoint
/// - Linearizable: Jacobians for EKF
/// - Sampleable: Sigma points for UKF/CKF
template <LieSide Side = LieSide::Right>
struct SO3 {
    using Scalar = double;
    using Group  = group::SO3<Scalar>;

    // ------------------------------------------------------------------------
    // Type aliases
    // ------------------------------------------------------------------------
    using Flux          = Group;                       // State is a rotation matrix
    using Drift         = Eigen::Matrix<Scalar, 3, 1>; // Tangent space (axis-angle)
    using DriftExtent   = Drift;
    using WeightedPoint = SigmaPoint<Flux, Drift>;     // For Sampleable concept

    static constexpr int Dim        = 3;               // Manifold dimension
    static constexpr int DimDynamic = 3;

    // ------------------------------------------------------------------------
    // Space concept implementation
    // ------------------------------------------------------------------------

    [[nodiscard]] static Flux zero_flux() noexcept { return Group{}; }
    [[nodiscard]] static Drift zero_drift() noexcept { return Drift::Zero(); }

    /// Compose: R ⊕ ξ = R · Exp(ξ)  (right-invariant)
    [[nodiscard]] static Flux compose(const Flux& R, const Drift& xi) noexcept {
        if constexpr (Side == LieSide::Right) {
            return R * Group::exp(xi);
        } else {
            return Group::exp(xi) * R;
        }
    }

    /// Diff: R₁ ⊖ R₂ = Log(R₂⁻¹ · R₁)  (right-invariant)
    [[nodiscard]] static Drift diff(const Flux& a, const Flux& b) noexcept {
        if constexpr (Side == LieSide::Right) {
            return Group::log(a.inv() * b);
        } else {
            return Group::log(b * a.inv());
        }
    }

    [[nodiscard]] static Scalar norm(const Drift& xi) noexcept { return xi.norm(); }
    [[nodiscard]] static Drift inverse(const Drift& xi) noexcept { return -xi; }
    [[nodiscard]] static Flux inverse(const Flux& R) noexcept { return R.inv(); }
    [[nodiscard]] static Drift extent(const DriftExtent&, const Drift& d) noexcept { return d; }

    // ------------------------------------------------------------------------
    // LieGroup concept implementation
    // ------------------------------------------------------------------------

    [[nodiscard]] static Flux identity() noexcept { return Group{}; }
    [[nodiscard]] static Flux exp(const Drift& xi) noexcept { return Group::exp(xi); }
    [[nodiscard]] static Drift log(const Flux& R) noexcept { return Group::log(R); }

    [[nodiscard]] static Eigen::Matrix<Scalar, 3, 3> adjoint_matrix(const Flux& R) noexcept {
        return R.Adjoint();
    }

    [[nodiscard]] static Eigen::Matrix<Scalar, 3, 3> inv_adjoint_matrix(const Flux& R) noexcept {
        return R.invAdjoint();
    }

    // ------------------------------------------------------------------------
    // LinearizableLieGroup concept implementation
    // ------------------------------------------------------------------------

    /// Dynamics Jacobian for constant angular velocity: J_f ≈ I
    [[nodiscard]] static Eigen::Matrix<Scalar, 3, 3>
        dynamics_jacobian(const Flux&, double dt) noexcept {
        // For simple rotation model, Jacobian is approximately identity
        // For more complex models, this would depend on the dynamics
        (void)dt;
        return Eigen::Matrix<Scalar, 3, 3>::Identity();
    }

    /// Left Jacobian: J_l(ξ) where d/dξ Exp(ξ) = Exp(ξ) · J_l(ξ)
    [[nodiscard]] static Eigen::Matrix<Scalar, 3, 3> left_jacobian(const Drift& xi) noexcept {
        return Group::leftJacobian(xi);
    }

    /// Inverse left Jacobian
    [[nodiscard]] static Eigen::Matrix<Scalar, 3, 3> inv_left_jacobian(const Drift& xi) noexcept {
        return Group::invLeftJacobian(xi);
    }

    /// Right Jacobian: J_r(ξ) = J_l(-ξ)
    [[nodiscard]] static Eigen::Matrix<Scalar, 3, 3> right_jacobian(const Drift& xi) noexcept {
        return Group::rightJacobian(xi);
    }

    /// Inverse right Jacobian
    [[nodiscard]] static Eigen::Matrix<Scalar, 3, 3> inv_right_jacobian(const Drift& xi) noexcept {
        return Group::invRightJacobian(xi);
    }

    // ------------------------------------------------------------------------
    // SampleableLieGroup concept implementation (UKF/CKF)
    // ------------------------------------------------------------------------

    /// Generate sigma points using unscented transform
    [[nodiscard]] static std::vector<SigmaPoint<Flux, Drift>> sigma_points(
        const Flux& mean, const Eigen::MatrixXd& cov, double alpha, double beta,
        double kappa) noexcept {

        const int n         = Dim;
        const double lambda = alpha * alpha * (n + kappa) - n;

        // Weights
        const double w0_m = lambda / (n + lambda);
        const double w0_c = w0_m + (1 - alpha * alpha + beta);
        const double wi   = 1.0 / (2 * (n + lambda));

        std::vector<SigmaPoint<Flux, Drift>> points;
        points.reserve(2 * n + 1);

        // Point 0: mean
        points.emplace_back(mean, Drift::Zero(), w0_m, w0_c);

        // Compute Cholesky: (n+λ) * P = SS^T
        Eigen::LLT<Eigen::Matrix3d> llt((n + lambda) * cov);
        if (llt.info() != Eigen::Success) {
            // Fallback: return just the mean point
            return points;
        }
        const Eigen::Matrix3d S = llt.matrixL();

        // Points 1..n: mean + columns of S
        for (int i = 0; i < n; ++i) {
            Drift xi = S.col(i);
            Flux pt  = compose(mean, xi);
            points.emplace_back(pt, xi, wi, wi);
        }

        // Points n+1..2n: mean - columns of S
        for (int i = 0; i < n; ++i) {
            Drift xi = -S.col(i);
            Flux pt  = compose(mean, xi);
            points.emplace_back(pt, xi, wi, wi);
        }

        return points;
    }

    /// Compute weighted mean from sigma points using Karcher mean
    ///
    /// For Lie groups, the mean is defined as the point that minimizes
    /// the sum of squared distances: μ = argmin_χ Σ d(χ, p_i)²
    ///
    /// Uses iterative logarithmic averaging (3 iterations sufficient for most applications).
    /// For exact convergence, increase iterations or use specialized algorithms.
    ///
    /// Reference:
    /// - Tron, R., et al. (2011). "Fast and scalable Karcher means."
    ///   *EUSIPCO 2011*. [DOI](https://doi.org/10.1109/EUSIPCO.2011.6073997)
    /// - Hartley et al. 2020, Sec. III.C: Iterative logarithmic averaging on Lie groups
    [[nodiscard]] static Flux mean(const std::vector<SigmaPoint<Flux, Drift>>& points) noexcept {
        if (points.empty())
            return identity();

        // Karcher mean: iteratively refine using tangent space averaging
        // Start with first point as initial estimate
        Flux result = points[0].point;

        // Iteratively refine (Karcher mean):
        //   μ_{k+1} = μ_k · Exp( Σ w_i * Log(μ_k⁻¹ · p_i) )
        // 3 iterations provides good balance of accuracy and performance.
        for (size_t iter = 0; iter < 3; ++iter) {
            Drift sum = Drift::Zero();
            for (const auto& wp : points) {
                sum += wp.weight_mean * diff(result, wp.point);
            }
            result = compose(result, sum);
        }

        return result;
    }

    /// Compute weighted covariance from sigma points
    [[nodiscard]] static Eigen::MatrixXd
        covariance(const std::vector<SigmaPoint<Flux, Drift>>& points, const Flux& mean) noexcept {

        Eigen::MatrixXd P = Eigen::MatrixXd::Zero(Dim, Dim);

        for (const auto& wp : points) {
            Drift xi = diff(mean, wp.point);
            P += wp.weight_cov * (xi * xi.transpose());
        }

        return P;
    }
};

// Type aliases
using SO3Right = SO3<LieSide::Right>;
using SO3Left  = SO3<LieSide::Left>;

// ============================================================================
// SE(3) - 3D Rigid Body Transformation Group
// ============================================================================

template <LieSide Side = LieSide::Right>
struct SE3 {
    using Scalar = double;
    using Group  = group::SEn3<Scalar, 1>;

    using Flux          = Group;
    using Drift         = Eigen::Matrix<Scalar, 6, 1>;
    using DriftExtent   = Drift;
    using WeightedPoint = SigmaPoint<Flux, Drift>; // For Sampleable concept

    static constexpr int Dim        = 6;
    static constexpr int DimDynamic = 6;

    // Space concept
    [[nodiscard]] static Flux zero_flux() noexcept { return Group{}; }
    [[nodiscard]] static Drift zero_drift() noexcept { return Drift::Zero(); }

    [[nodiscard]] static Flux compose(const Flux& T, const Drift& xi) noexcept {
        if constexpr (Side == LieSide::Right) {
            return T * Group::exp(xi);
        } else {
            return Group::exp(xi) * T;
        }
    }

    [[nodiscard]] static Drift diff(const Flux& a, const Flux& b) noexcept {
        if constexpr (Side == LieSide::Right) {
            return Group::log(a.inv() * b);
        } else {
            return Group::log(b * a.inv());
        }
    }

    [[nodiscard]] static Scalar norm(const Drift& xi) noexcept { return xi.norm(); }
    [[nodiscard]] static Drift inverse(const Drift& xi) noexcept { return -xi; }
    [[nodiscard]] static Flux inverse(const Flux& T) noexcept { return T.inv(); }
    [[nodiscard]] static Drift extent(const DriftExtent&, const Drift& d) noexcept { return d; }

    // LieGroup concept
    [[nodiscard]] static Flux identity() noexcept { return Group{}; }
    [[nodiscard]] static Flux exp(const Drift& xi) noexcept { return Group::exp(xi); }
    [[nodiscard]] static Drift log(const Flux& T) noexcept { return Group::log(T); }

    [[nodiscard]] static Eigen::Matrix<Scalar, 6, 6> adjoint_matrix(const Flux& T) noexcept {
        return T.Adjoint();
    }

    [[nodiscard]] static Eigen::Matrix<Scalar, 6, 6> inv_adjoint_matrix(const Flux& T) noexcept {
        return T.invAdjoint();
    }

    // LinearizableLieGroup
    /// Dynamics Jacobian for SE(3) pure kinematics (constant velocity model)
    ///
    /// For the motion model: X' = X * exp(twist * dt)
    /// The Jacobian is Identity for pure kinematics (F = I) because the twist
    /// is an external input, not state-dependent.
    [[nodiscard]] static Eigen::Matrix<Scalar, 6, 6>
        dynamics_jacobian(const Flux&, double dt) noexcept {
        (void)dt;
        return Eigen::Matrix<Scalar, 6, 6>::Identity();
    }

    [[nodiscard]] static Eigen::Matrix<Scalar, 6, 6> left_jacobian(const Drift& xi) noexcept {
        return Group::leftJacobian(xi);
    }

    [[nodiscard]] static Eigen::Matrix<Scalar, 6, 6> inv_left_jacobian(const Drift& xi) noexcept {
        return Group::invLeftJacobian(xi);
    }

    // SampleableLieGroup
    [[nodiscard]] static std::vector<SigmaPoint<Flux, Drift>> sigma_points(
        const Flux& mean, const Eigen::MatrixXd& cov, double alpha, double beta,
        double kappa) noexcept {

        const int n         = Dim;
        const double lambda = alpha * alpha * (n + kappa) - n;
        const double w0_m   = lambda / (n + lambda);
        const double w0_c   = w0_m + (1 - alpha * alpha + beta);
        const double wi     = 1.0 / (2 * (n + lambda));

        std::vector<SigmaPoint<Flux, Drift>> points;
        points.reserve(2 * n + 1);
        points.emplace_back(mean, Drift::Zero(), w0_m, w0_c);

        Eigen::LLT<Eigen::Matrix<double, 6, 6>> llt((n + lambda) * cov);
        if (llt.info() != Eigen::Success)
            return points;

        const Eigen::Matrix<double, 6, 6> S = llt.matrixL();

        for (int i = 0; i < n; ++i) {
            Drift xi = S.col(i);
            points.emplace_back(compose(mean, xi), xi, wi, wi);
            xi = -S.col(i);
            points.emplace_back(compose(mean, xi), xi, wi, wi);
        }

        return points;
    }

    [[nodiscard]] static Flux mean(const std::vector<SigmaPoint<Flux, Drift>>& points) noexcept {
        if (points.empty())
            return identity();
        Flux result = points[0].point;

        for (size_t iter = 0; iter < 3; ++iter) {
            Drift sum = Drift::Zero();
            for (const auto& wp : points) {
                sum += wp.weight_mean * diff(result, wp.point);
            }
            result = compose(result, sum);
        }

        return result;
    }

    [[nodiscard]] static Eigen::MatrixXd
        covariance(const std::vector<SigmaPoint<Flux, Drift>>& points, const Flux& mean) noexcept {

        Eigen::MatrixXd P = Eigen::MatrixXd::Zero(Dim, Dim);
        for (const auto& wp : points) {
            Drift xi = diff(mean, wp.point);
            P += wp.weight_cov * (xi * xi.transpose());
        }
        return P;
    }
};

using SE3Right = SE3<LieSide::Right>;
using SE3Left  = SE3<LieSide::Left>;

// =========================================================================
// SE_2(3) / SE2(3) - Extended Pose+Velocity Group (SEn3 with n=2)
// =========================================================================

template <LieSide Side = LieSide::Right>
struct SE23 {
    using Scalar = double;
    using Group  = group::SEn3<Scalar, 2>;

    using Flux          = Group;
    using Drift         = Eigen::Matrix<Scalar, 9, 1>;
    using DriftExtent   = Drift;
    using WeightedPoint = SigmaPoint<Flux, Drift>; // For Sampleable concept

    static constexpr int Dim        = 9;
    static constexpr int DimDynamic = 9;

    // Space concept
    [[nodiscard]] static Flux zero_flux() noexcept { return Group{}; }
    [[nodiscard]] static Drift zero_drift() noexcept { return Drift::Zero(); }

    [[nodiscard]] static Flux compose(const Flux& T, const Drift& xi) noexcept {
        if constexpr (Side == LieSide::Right) {
            return T * Group::exp(xi);
        } else {
            return Group::exp(xi) * T;
        }
    }

    [[nodiscard]] static Drift diff(const Flux& a, const Flux& b) noexcept {
        if constexpr (Side == LieSide::Right) {
            return Group::log(a.inv() * b);
        } else {
            return Group::log(b * a.inv());
        }
    }

    [[nodiscard]] static Scalar norm(const Drift& xi) noexcept { return xi.norm(); }
    [[nodiscard]] static Drift inverse(const Drift& xi) noexcept { return -xi; }
    [[nodiscard]] static Flux inverse(const Flux& T) noexcept { return T.inv(); }
    [[nodiscard]] static Drift extent(const DriftExtent&, const Drift& d) noexcept { return d; }

    // LieGroup concept
    [[nodiscard]] static Flux identity() noexcept { return Group{}; }
    [[nodiscard]] static Flux exp(const Drift& xi) noexcept { return Group::exp(xi); }
    [[nodiscard]] static Drift log(const Flux& T) noexcept { return Group::log(T); }

    [[nodiscard]] static Eigen::Matrix<Scalar, 9, 9> adjoint_matrix(const Flux& T) noexcept {
        return T.Adjoint();
    }

    [[nodiscard]] static Eigen::Matrix<Scalar, 9, 9> inv_adjoint_matrix(const Flux& T) noexcept {
        return T.invAdjoint();
    }

    // LinearizableLieGroup
    /// Dynamics Jacobian for SE2(3) constant velocity model
    ///
    /// For SE2(3), the state is X = (R, v, p) where:
    ///   R ∈ SO(3) : rotation
    ///   v ∈ R³   : velocity
    ///   p ∈ R³   : position
    ///
    /// The tangent space has structure ξ = [ω, v, ṗ] where:
    ///   ω : angular velocity (3D)
    ///   v : linear velocity (3D)
    ///   ṗ : position rate (3D)
    ///
    /// For the constant velocity model (no acceleration):
    ///   - Position depends on velocity: p(t+dt) ≈ p(t) + v(t)·dt
    ///   - Velocity is constant: v(t+dt) = v(t)
    ///   - Rotation is constant (ω is part of the twist)
    ///
    /// In the right-invariant error frame ξ = Log(X̂⁻¹X), the error dynamics are:
    ///   - Rotation error: independent (dω'/dω = I₃)
    ///   - Position error: depends on velocity error (dp'/dv = I₃·dt)
    ///   - Velocity error: independent (dv'/dv = I₃)
    ///
    /// The Jacobian has the structure:
    ///   F = [I₃  0₃  0₃]   (rotation)
    ///       [0₃  0₃  I₃·dt] (position depends on velocity)
    ///       [0₃  0₃  I₃]   (velocity constant)
    ///
    /// This ensures proper covariance propagation:
    ///   P' = Φ·P·Φ^T + Q, where Φ = I + F·dt
    ///
    /// Reference:
    /// - Solà et al. (2018). "A micro Lie theory for state estimation."
    ///   Sec. IV: Error state dynamics on Lie groups
    [[nodiscard]] static Eigen::Matrix<Scalar, 9, 9>
        dynamics_jacobian(const Flux&, double dt) noexcept {
        Eigen::Matrix<Scalar, 9, 9> F = Eigen::Matrix<Scalar, 9, 9>::Zero();

        // Rotation block (0-2): independent
        F.block<3, 3>(0, 0) = Eigen::Matrix<Scalar, 3, 3>::Identity();

        // Velocity block (6-8): constant
        F.block<3, 3>(6, 6) = Eigen::Matrix<Scalar, 3, 3>::Identity();

        // Position block (3-5): depends on velocity
        F.block<3, 3>(3, 6) = Eigen::Matrix<Scalar, 3, 3>::Identity() * dt;

        return F;
    }

    [[nodiscard]] static Eigen::Matrix<Scalar, 9, 9> left_jacobian(const Drift& xi) noexcept {
        return Group::leftJacobian(xi);
    }

    [[nodiscard]] static Eigen::Matrix<Scalar, 9, 9> inv_left_jacobian(const Drift& xi) noexcept {
        return Group::invLeftJacobian(xi);
    }

    // SampleableLieGroup
    [[nodiscard]] static std::vector<SigmaPoint<Flux, Drift>> sigma_points(
        const Flux& mean, const Eigen::MatrixXd& cov, double alpha, double beta,
        double kappa) noexcept {

        const int n         = Dim;
        const double lambda = alpha * alpha * (n + kappa) - n;
        const double w0_m   = lambda / (n + lambda);
        const double w0_c   = w0_m + (1 - alpha * alpha + beta);
        const double wi     = 1.0 / (2 * (n + lambda));

        std::vector<SigmaPoint<Flux, Drift>> points;
        points.reserve(2 * n + 1);
        points.emplace_back(mean, Drift::Zero(), w0_m, w0_c);

        Eigen::LLT<Eigen::Matrix<double, 9, 9>> llt((n + lambda) * cov);
        if (llt.info() != Eigen::Success)
            return points;

        const Eigen::Matrix<double, 9, 9> S = llt.matrixL();

        for (int i = 0; i < n; ++i) {
            Drift xi = S.col(i);
            points.emplace_back(compose(mean, xi), xi, wi, wi);
            xi = -S.col(i);
            points.emplace_back(compose(mean, xi), xi, wi, wi);
        }

        return points;
    }

    [[nodiscard]] static Flux mean(const std::vector<SigmaPoint<Flux, Drift>>& points) noexcept {
        if (points.empty())
            return identity();
        Flux result = points[0].point;

        for (size_t iter = 0; iter < 3; ++iter) {
            Drift sum = Drift::Zero();
            for (const auto& wp : points) {
                sum += wp.weight_mean * diff(result, wp.point);
            }
            result = compose(result, sum);
        }

        return result;
    }

    [[nodiscard]] static Eigen::MatrixXd
        covariance(const std::vector<SigmaPoint<Flux, Drift>>& points, const Flux& mean) noexcept {

        Eigen::MatrixXd P = Eigen::MatrixXd::Zero(Dim, Dim);
        for (const auto& wp : points) {
            Drift xi = diff(mean, wp.point);
            P += wp.weight_cov * (xi * xi.transpose());
        }
        return P;
    }
};

using SE23Right = SE23<LieSide::Right>;
using SE23Left  = SE23<LieSide::Left>;

// ============================================================================
// Concept verification
// ============================================================================

static_assert(Space<SO3Right>, "SO3Right must satisfy Space");
static_assert(LieGroup<SO3Right>, "SO3Right must satisfy LieGroup");
static_assert(LinearizableLieGroup<SO3Right>, "SO3Right must satisfy LinearizableLieGroup");
static_assert(SampleableLieGroup<SO3Right>, "SO3Right must satisfy SampleableLieGroup");

static_assert(Space<SE3Right>, "SE3Right must satisfy Space");
static_assert(LieGroup<SE3Right>, "SE3Right must satisfy LieGroup");
static_assert(LinearizableLieGroup<SE3Right>, "SE3Right must satisfy LinearizableLieGroup");
static_assert(SampleableLieGroup<SE3Right>, "SE3Right must satisfy SampleableLieGroup");

static_assert(Space<SE23Right>, "SE23Right must satisfy Space");
static_assert(LieGroup<SE23Right>, "SE23Right must satisfy LieGroup");
static_assert(LinearizableLieGroup<SE23Right>, "SE23Right must satisfy LinearizableLieGroup");
static_assert(SampleableLieGroup<SE23Right>, "SE23Right must satisfy SampleableLieGroup");

} // namespace uni_kf::redact::spaces
