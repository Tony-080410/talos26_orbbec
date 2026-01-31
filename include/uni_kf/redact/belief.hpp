#pragma once

#include "covariance.hpp"
#include "lie_ops.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <expected>

namespace uni_kf::redact {

/// @file belief.hpp
/// @brief State belief with uncertainty on manifolds
///
/// Belief represents a state estimate (mean) with uncertainty (covariance)
/// on a Lie Group. Supports operations like:
/// - Manifold composition with covariance propagation
/// - Mahalanobis distance computation
/// - Log-determinant for numerical stability
///
/// References:
/// - Barfoot 2024: *State Estimation for Robotics*, Sections 4.2, 11.3
///   [PDF](http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser24.pdf)
/// - Mahalanobis distance: See De Maesschalck et al. 2000, Chemometrics journal
///
/// Belief represents a state estimate with uncertainty.
/// Combines a mean state (Flux) with a covariance matrix representation.
///
/// @tparam LieGroup The lieplusplus Lie group type (e.g., group::SO3<double>, group::SE3<double>)
/// @tparam Repr The covariance representation (Direct, Joseph, Sr, Udu)
template <typename LieGroup, Representation Repr = Representation::Direct>
class Belief {
public:
    using Flux               = LieGroup;                      // 直接使用 lieplusplus Lie group
    using Drift              = typename LieGroup::VectorType; // lieplusplus 切空间类型
    using Scalar             = double;
    static constexpr int Dim = LieGroup::Dim;

    using CovarianceType = Covariance<LieGroup, Repr, InvariantLinearStrategy>;
    using MatrixDD       = Eigen::MatrixXd;

    // ========================================================================
    // Constructors
    // ========================================================================

    /// Default constructor: initializes to identity (zero state) with zero uncertainty
    Belief()
        : mean_(LieGroup()) // lieplusplus 默认构造为单位元
        , cov_() {}

    /// Construct from mean and covariance
    Belief(const Flux& mean, const CovarianceType& cov)
        : mean_(mean)
        , cov_(cov) {}

    /// Construct from mean and raw covariance matrix
    Belief(const Flux& mean, const MatrixDD& P)
        : mean_(mean)
        , cov_(P) {}

    // ========================================================================
    // Factory Methods
    // ========================================================================

    /// Create belief at identity (zero state) with default covariance
    static Belief Identity() { return Belief(); }

    /// Create belief from mean and covariance matrix
    static Belief WithCovariance(const Flux& mean, const MatrixDD& P) { return Belief(mean, P); }

    /// Create belief from mean with scaled identity covariance
    static Belief WithScaledIdentity(const Flux& mean, Scalar scale) {
        int dim = Dim;
        if constexpr (Dim == Eigen::Dynamic) {
            // For dynamic dimension, infer from Drift type
            Drift tmp;
            if constexpr (requires { tmp.size(); }) {
                dim = static_cast<int>(tmp.size());
            } else {
                static_assert(
                    Dim != Eigen::Dynamic,
                    "Dynamic-dimension Belief requires Drift::size() or explicit dim parameter");
            }
        }
        return Belief(mean, MatrixDD::Identity(dim, dim) * scale);
    }

    /// Create belief from mean with diagonal covariance
    template <std::size_t N>
    static Belief WithDiagonal(const Flux& mean, const Scalar (&diag)[N]) {
        static_assert(N == Dim || Dim == Eigen::Dynamic, "Diagonal size mismatch");
        const int dim = (Dim == Eigen::Dynamic) ? static_cast<int>(N) : Dim;
        MatrixDD P    = MatrixDD::Zero(dim, dim);
        for (int i = 0; i < dim; ++i) {
            P(i, i) = diag[i];
        }
        return Belief(mean, P);
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /// Get the mean state
    [[nodiscard]] const Flux& mean() const noexcept { return mean_; }

    /// Get mutable mean state
    [[nodiscard]] Flux& mean() noexcept { return mean_; }

    /// Get the covariance matrix
    [[nodiscard]] MatrixDD covariance() const noexcept { return cov_.mat(); }

    /// Get the covariance storage object
    [[nodiscard]] const CovarianceType& uncertainty() const noexcept { return cov_; }

    /// Get mutable covariance storage object
    [[nodiscard]] CovarianceType& uncertainty() noexcept { return cov_; }

    // ========================================================================
    // Manifold Operations
    // ========================================================================

    /// Retract: apply drift correction to mean (x ← x ⊕ δ)
    [[nodiscard]] Belief retract(const Drift& delta) const {
        return Belief(lie_ops::compose(mean_, delta), cov_);
    }

    /// Difference: compute drift from this to other (η = x⁻¹ ⊗ y)
    [[nodiscard]] Drift diff_mean(const Belief& other) const {
        return lie_ops::diff<LieGroup>(mean_, other.mean_);
    }

    /// Compose two beliefs (manifold composition with combined covariance)
    ///
    /// For Lie groups with right-invariant error η_R = Log(X̂⁻¹X):
    ///   X = X1 · X2
    ///   P = Ad_{X2⁻¹} · P1 · Ad_{X2⁻¹}^T + P2
    ///
    /// This follows from the error composition formula for right-invariant error:
    ///   η_R = Ad_{X2⁻¹} · η_R1 + η_R2
    ///
    /// Reference: Barfoot 2024, Section 11.3.2; Bonnabel 2017, Appendix A
    [[nodiscard]] Belief compose(const Belief& other) const {
        Flux new_mean = mean_ * other.mean_;

        const MatrixDD P1 = cov_.mat();
        const MatrixDD P2 = other.cov_.mat();

        MatrixDD P_new;
        if constexpr (requires(const Flux& x) { x.Adjoint(); }) {
            // Right-invariant composition: P' = Ad_{X2⁻¹} · P1 · Ad_{X2⁻¹}^T + P2
            const MatrixDD Ad_X2_inv = lie_ops::inv_adjoint(other.mean_);
            P_new                    = Ad_X2_inv * P1 * Ad_X2_inv.transpose() + P2;
        } else {
            P_new = P1 + P2;
        }

        return Belief(new_mean, P_new);
    }

    /// Bias belief to new coordinate frame (SE(3) or similar Lie groups)
    ///
    /// Bias both mean and covariance to a new coordinate system:
    ///   - mean: x' = T * x
    ///   - covariance: P' = Ad_T * P * Ad_T^T
    ///
    /// This is a **deterministic coordinate bias**, NOT a composition
    /// of random variables. Use compose() when combining two beliefs with
    /// independent uncertainties.
    ///
    /// Example: When gimbal moves, bias belief to new gimbal frame
    ///   SE3 gimbal_pose(R, t);
    ///   belief = belief.biased(gimbal_pose);
    ///
    template <typename Group>
    [[nodiscard]] Belief biased(const Group& T) const noexcept
        requires(
            requires(const Flux& x, const Group& g) {
                // Requires group multiplication: x * g
                x * g;
            }
            && requires(const Group& g) {
                // Requires adjoint matrix for covariance bias
                g.Adjoint();
            }) {
        // Biased mean: x' = T * x
        const Flux new_mean = mean_ * T;

        // Biased covariance: P' = Ad_T * P * Ad_T^T
        const MatrixDD P     = cov_.mat();
        const MatrixDD Ad_T  = T.Adjoint();
        const MatrixDD P_new = Ad_T * P * Ad_T.transpose();

        return Belief(new_mean, P_new);
    }

    /// In-place bias to new coordinate frame
    ///
    /// Equivalent to: *this = this->biased(T);
    template <typename Group>
    void biased_in_place(const Group& T) noexcept {
        *this = biased(T);
    }

    // ========================================================================
    // Metric Operations
    // ========================================================================

    /// Trace of covariance matrix
    [[nodiscard]] Scalar trace() const noexcept { return cov_.mat().trace(); }

    /// Determinant of covariance matrix
    [[nodiscard]] Scalar determinant() const noexcept { return cov_.mat().determinant(); }

    /// Log determinant (more numerically stable for small matrices)
    ///
    /// Computes: log(det(P)) using LDLT decomposition
    ///   log(det(P)) = Σ log(diag(D))
    ///
    /// This avoids numerical overflow/underflow compared to direct det(P) computation.
    ///
    /// Reference:
    /// - Eigen documentation: LDLT decomposition for log-determinant
    /// - Ameli et al.: "Determinant Estimation Under Memory Constraints"
    ///   [Berkeley](https://www.stat.berkeley.edu/~mmahoney/pubs/8468_Determinant_Estimation_un.pdf)
    ///   - Uses block-wise LDLT for large-scale log-determinant
    /// - Barfoot 2024, Appendix E: Matrix Determinant Lemma
    [[nodiscard]] std::expected<Scalar, std::string> log_determinant() const noexcept {
        const MatrixDD P = cov_.mat();
        Eigen::LDLT<MatrixDD> ldlt(P);
        if (ldlt.info() != Eigen::Success) {
            return std::unexpected("Covariance matrix is not positive definite");
        }
        // log(det(P)) = sum(log(diag(D)))) where P = L * D * L^T
        // This is numerically stable: det(LDL^T) = det(D) and det(L) = 1
        Scalar log_det          = 0.0;
        const Eigen::VectorXd D = ldlt.vectorD();
        for (int i = 0; i < D.size(); ++i) {
            // Check for non-positive diagonal (non-SPD matrix)
            if (D(i) <= 0.0) {
                return std::unexpected("Covariance matrix is not positive definite");
            }
            log_det += std::log(D(i));
        }
        return log_det;
    }

    /// Standard deviation in a specific direction
    ///
    /// Computes: σ_d = sqrt(d^T * P * d) / ||d||
    ///
    /// This is the **directional standard deviation** on the manifold,
    /// representing uncertainty in a specific direction d.
    ///
    /// Reference:
    /// - Wikipedia: "Covariance matrix", Section on quadratic forms
    ///   [Link](https://en.wikipedia.org/wiki/Covariance_matrix)
    /// - Barfoot 2024, Section 4.2.2: Uncertainty on manifolds
    [[nodiscard]] Scalar std_dev(const Drift& direction) const noexcept {
        const MatrixDD P      = cov_.mat();
        const Scalar norm_dir = direction.norm();
        if (norm_dir < 1e-10) {
            return 0.0;
        }
        // σ = sqrt(d^T * P * d) / ||d|| (directional variance formula)
        const Scalar dir_sq = (direction.transpose() * P * direction)(0, 0);
        return std::sqrt(dir_sq) / norm_dir;
    }

    /// Mahalanobis distance to another belief
    ///
    /// Computes: d_M = sqrt(η^T * P^(-1) * η) where η = diff_mean(other)
    ///
    /// Reference:
    /// - De Maesschalck, R., et al. (2000). "The Mahalanobis distance."
    ///   *Chemometrics and Intelligent Laboratory Systems*, 50(1), 1-18.
    ///   DOI: [10.1016/S0169-7439(99)00047-7](https://doi.org/10.1016/S0169-7439(99)00047-7)
    /// - Barfoot 2024, Section 4.2.3: Mahalanobis distance on manifolds
    [[nodiscard]] std::expected<Scalar, std::string>
        mahalanobis(const Belief& other) const noexcept {
        const Drift eta  = diff_mean(other);
        const MatrixDD P = cov_.mat();

        // Try Cholesky decomposition
        Eigen::LLT<MatrixDD> llt(P);
        if (llt.info() != Eigen::Success) {
            return std::unexpected("Covariance matrix is not positive definite");
        }

        // Compute Mahalanobis distance: sqrt(eta^T * P^(-1) * eta)
        // Numerically stable: solve L * L^T * x = eta via Cholesky
        const Eigen::VectorXd L_eta = llt.matrixL().solve(eta);
        const Scalar dist_sq        = L_eta.squaredNorm();

        return std::sqrt(dist_sq);
    }

    // ========================================================================
    // Numerical Stability
    // ========================================================================

    /// Symmetrize covariance matrix: P ← (P + P^T) / 2
    void symmetrize() noexcept { cov_.symmetrize(); }

    /// Add small regularization to diagonal for numerical stability
    void regularize(Scalar eps = 1e-12) noexcept { cov_.regularize(eps); }

    /// Check if all values in belief are finite
    [[nodiscard]] bool is_finite() const noexcept {
        if constexpr (requires { mean_.matrix(); }) {
            if (!mean_.matrix().allFinite()) {
                return false;
            }
        }
        return cov_.is_finite();
    }

    // ========================================================================
    // Comparison
    // ========================================================================

    /// Equality check (approximate for floating point)
    [[nodiscard]] bool is_approx(const Belief& other, Scalar tol = 1e-6) const noexcept {
        // Check mean
        const Drift diff = lie_ops::diff<LieGroup>(mean_, other.mean_);
        if (diff.norm() > tol) {
            return false;
        }

        // Check covariance
        const MatrixDD P1 = cov_.mat();
        const MatrixDD P2 = other.cov_.mat();
        return (P1 - P2).cwiseAbs().maxCoeff() < tol;
    }

private:
    Flux mean_;
    CovarianceType cov_;
};

} // namespace uni_kf::redact
