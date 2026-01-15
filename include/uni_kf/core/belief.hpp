/// @file belief.hpp
/// @brief Belief state (mean + uncertainty) for Kalman filtering
///
/// A Belief represents a probability distribution over a DifferentialSpace,
/// characterized by a mean (point) and uncertainty representation.
///
/// The uncertainty can be represented in different forms:
/// - Covariance (P): Standard form, efficient for prediction
/// - Information (Y = P^-1): Dual form, efficient for sensor fusion
/// - SquareRoot (S where P = S*S^T): Numerically stable form

#pragma once

#include "../concepts/differential_space.hpp"
#include "../concepts/quadratic_form.hpp"
#include "../concepts/representation.hpp"

#include <random>

namespace uni_kf {

// ============================================================================
// Belief: State + Uncertainty
// ============================================================================

/// @brief Gaussian belief state on a DifferentialSpace
///
/// Represents a Gaussian distribution over a DifferentialSpace D with
/// pluggable uncertainty representation:
/// - mean: Most likely state (point on the manifold)
/// - uncertainty: Uncertainty in tangent/fiber space (various representations)
///
/// @tparam D DifferentialSpace
/// @tparam ReprTag_ Representation tag (CovarianceTag, InformationTag, SquareRootTag)
/// @tparam Scalar_ Floating point type
template <DifferentialSpace D, typename ReprTag_ = CovarianceTag, typename Scalar_ = double>
struct Belief {
    using Space      = D;
    using Scalar     = Scalar_;
    using Point      = typename D::Point;
    using Fiber      = typename D::Fiber;
    using ReprTag    = ReprTag_;
    using Repr       = ReprType<ReprTag, D::FiberDim, Scalar>;
    using Covariance = SymmetricMatrix<D::FiberDim, Scalar>;

    static constexpr int Dim = D::FiberDim;

    // Ensure the representation satisfies the concept
    static_assert(
        UncertaintyRepresentation<Repr>,
        "Representation must satisfy UncertaintyRepresentation concept");

    Point mean;       ///< State estimate (point on manifold)
    Repr uncertainty; ///< Uncertainty representation

    // ------------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------------

    Belief() noexcept
        : mean(D::identity())
        , uncertainty(Repr::Zero()) {}

    Belief(const Point& m, const Repr& u) noexcept
        : mean(m)
        , uncertainty(u) {}

    Belief(Point&& m, Repr&& u) noexcept
        : mean(std::move(m))
        , uncertainty(std::move(u)) {}

    /// @brief Construct from covariance (converts if needed)
    Belief(const Point& m, const Covariance& P) noexcept
        : mean(m)
        , uncertainty(repr_from_covariance(P)) {}

    // ------------------------------------------------------------------------
    // Static factory methods
    // ------------------------------------------------------------------------

    /// @brief Create belief at identity with zero uncertainty
    [[nodiscard]] static Belief Identity() noexcept { return Belief{}; }

    /// @brief Create belief at identity with given representation
    [[nodiscard]] static Belief WithUncertainty(const Repr& u) noexcept {
        return Belief{D::identity(), u};
    }

    /// @brief Create belief from covariance (converts to target representation)
    [[nodiscard]] static Belief WithCovariance(const Point& m, const Covariance& P) noexcept {
        return Belief{m, repr_from_covariance(P)};
    }

    /// @brief Create belief with scaled identity covariance
    [[nodiscard]] static Belief WithScaledIdentity(const Point& m, Scalar scale) noexcept {
        auto P = Covariance::ScaledIdentity(scale);
        return Belief{m, repr_from_covariance(P)};
    }

    // ------------------------------------------------------------------------
    // Unified access interface
    // ------------------------------------------------------------------------

    /// @brief Get covariance (converts if needed)
    [[nodiscard]] auto covariance() const noexcept {
        if constexpr (std::is_same_v<ReprTag, CovarianceTag>) {
            // Direct access for CovarianceRepr
            return uncertainty.to_covariance();
        } else {
            // May require conversion (returns std::expected for Information/SquareRoot)
            auto result = uncertainty.to_covariance();
            if constexpr (requires { result.value(); }) {
                // Has std::expected interface
                return result.value_or(Covariance::Identity());
            } else {
                // Direct return
                return result;
            }
        }
    }

    /// @brief Get underlying representation (mutable)
    [[nodiscard]] Repr& representation() noexcept { return uncertainty; }

    /// @brief Get underlying representation (const)
    [[nodiscard]] const Repr& representation() const noexcept { return uncertainty; }

    // ------------------------------------------------------------------------
    // Operations
    // ------------------------------------------------------------------------

    /// @brief Apply correction in tangent space: mean ← mean ⊕ delta
    [[nodiscard]] Belief retract(const Fiber& delta) const noexcept {
        return Belief{D::act(mean, delta), uncertainty};
    }

    /// @brief Apply correction with new uncertainty
    [[nodiscard]] Belief retract(const Fiber& delta, const Repr& u_new) const noexcept {
        return Belief{D::act(mean, delta), u_new};
    }

    /// @brief Compute difference to another belief (in tangent space)
    [[nodiscard]] Fiber diff_mean(const Belief& other) const noexcept {
        return D::diff(mean, other.mean);
    }

    // ------------------------------------------------------------------------
    // Uncertainty metrics
    // ------------------------------------------------------------------------

    /// @brief Trace of covariance (total variance)
    [[nodiscard]] Scalar trace() const noexcept {
        auto P = covariance();
        return P.trace();
    }

    /// @brief Determinant of covariance
    [[nodiscard]] Scalar determinant() const noexcept {
        auto P = covariance();
        return P.determinant();
    }

    /// @brief Log determinant (more numerically stable)
    [[nodiscard]] auto log_determinant() const noexcept -> std::expected<Scalar, std::string> {
        auto P = covariance();
        return P.log_determinant();
    }

    /// @brief Standard deviation in a given direction
    [[nodiscard]] Scalar std_dev(const Fiber& direction) const noexcept {
        auto P            = covariance();
        auto d_normalized = direction.data.normalized();
        return std::sqrt(d_normalized.dot(P.matrix() * d_normalized));
    }

    /// @brief Mahalanobis distance to a point
    [[nodiscard]] auto mahalanobis(const Point& p) const noexcept
        -> std::expected<Scalar, std::string> {
        auto P      = covariance();
        auto delta  = D::diff(mean, p);
        auto result = P.solve(delta.data);
        if (!result) {
            return std::unexpected(result.error());
        }
        return std::sqrt(delta.data.dot(*result));
    }

    // ------------------------------------------------------------------------
    // Sampling (for particle filters, debugging)
    // ------------------------------------------------------------------------

    /// @brief Sample a point from the belief distribution
    /// @note Requires external random generator
    template <typename RNG>
    [[nodiscard]] Point sample(RNG& rng) const noexcept {
        // Generate standard normal samples
        typename Fiber::EigenType z;
        std::normal_distribution<Scalar> dist(0, 1);
        for (int i = 0; i < Dim; ++i) {
            z(i) = dist(rng);
        }

        // Get covariance and compute Cholesky
        auto P        = covariance();
        auto L_result = P.cholesky();
        if (!L_result) {
            return mean; // Fallback: return mean
        }

        typename Fiber::EigenType delta = (*L_result) * z;
        return D::act(mean, Fiber(delta));
    }

    // ------------------------------------------------------------------------
    // Numerical stability
    // ------------------------------------------------------------------------

    /// @brief Symmetrize uncertainty representation
    void symmetrize() noexcept { uncertainty.symmetrize(); }

    /// @brief Add regularization to uncertainty representation
    void regularize(Scalar eps = Scalar{1e-12}) noexcept { uncertainty.regularize(eps); }

    /// @brief Check if uncertainty is finite
    [[nodiscard]] bool is_finite() const noexcept { return uncertainty.is_finite(); }

    // ------------------------------------------------------------------------
    // Projection (for constrained estimation)
    // ------------------------------------------------------------------------

    /// @brief Project covariance to enforce constraints
    /// @param constrained_indices Indices of constrained dimensions (set their variance to
    /// near-zero)
    template <typename IndexArray>
    void enforce_constraints(
        const IndexArray& constrained_indices, Scalar eps = Scalar{1e-12}) noexcept {
        // For non-covariance representations, convert temporarily
        if constexpr (!std::is_same_v<ReprTag, CovarianceTag>) {
            auto P = covariance();
            for (auto idx : constrained_indices) {
                for (int j = 0; j < Dim; ++j) {
                    P(idx, j) = Scalar{0};
                    P(j, idx) = Scalar{0};
                }
                P(idx, idx) = eps;
            }
            uncertainty = repr_from_covariance(P);
        } else {
            // Direct modification for covariance
            auto& P = uncertainty.covariance();
            for (auto idx : constrained_indices) {
                for (int j = 0; j < Dim; ++j) {
                    P(idx, j) = Scalar{0};
                    P(j, idx) = Scalar{0};
                }
                P(idx, idx) = eps;
            }
        }
    }

private:
    // ------------------------------------------------------------------------
    // Helper: Convert covariance to target representation
    // ------------------------------------------------------------------------

    [[nodiscard]] static Repr repr_from_covariance(const Covariance& P) noexcept {
        if constexpr (std::is_same_v<ReprTag, CovarianceTag>) {
            return Repr(P);
        } else if constexpr (std::is_same_v<ReprTag, InformationTag>) {
            auto result = Repr::from_covariance(P);
            return result.value_or(Repr::Identity());
        } else if constexpr (std::is_same_v<ReprTag, SquareRootTag>) {
            auto result = Repr::from_covariance(P);
            return result.value_or(Repr::Identity());
        } else {
            static_assert(std::is_same_v<ReprTag, CovarianceTag>, "Unknown representation tag");
        }
    }
};

// ============================================================================
// Belief utilities
// ============================================================================

/// @brief Compute KL divergence between two beliefs (approximate for non-Gaussian)
template <DifferentialSpace D, typename ReprTag, typename Scalar>
[[nodiscard]] auto
    kl_divergence(const Belief<D, ReprTag, Scalar>& p, const Belief<D, ReprTag, Scalar>& q) noexcept
    -> std::expected<Scalar, std::string> {
    constexpr int N = D::FiberDim;

    auto P = p.covariance();
    auto Q = q.covariance();

    // KL(p || q) = 0.5 * (tr(Q^{-1} P) + (m_q - m_p)^T Q^{-1} (m_q - m_p) - n + log(|Q|/|P|))

    auto Q_inv_result = Q.inverse();
    if (!Q_inv_result) {
        return std::unexpected(Q_inv_result.error());
    }

    auto delta = D::diff(p.mean, q.mean);

    // tr(Q^{-1} P)
    Scalar trace_term = (Q_inv_result->matrix() * P.matrix()).trace();

    // (m_q - m_p)^T Q^{-1} (m_q - m_p)
    Scalar quad_term = delta.data.dot(Q_inv_result->multiply(delta.data));

    // log(|Q|/|P|)
    auto log_det_q = Q.log_determinant();
    auto log_det_p = P.log_determinant();
    if (!log_det_q || !log_det_p) {
        return std::unexpected("Failed to compute log determinant");
    }
    Scalar log_ratio = *log_det_q - *log_det_p;

    return Scalar{0.5} * (trace_term + quad_term - N + log_ratio);
}

/// @brief Fuse two independent beliefs using information form
template <DifferentialSpace D, typename ReprTag, typename Scalar>
[[nodiscard]] auto fuse_beliefs(
    const Belief<D, ReprTag, Scalar>& b1, const Belief<D, ReprTag, Scalar>& b2) noexcept
    -> std::expected<Belief<D, ReprTag, Scalar>, std::string> {
    auto P1 = b1.covariance();
    auto P2 = b2.covariance();

    // Convert to information matrices: Y = P^{-1}
    auto Y1_result = P1.inverse();
    auto Y2_result = P2.inverse();
    if (!Y1_result || !Y2_result) {
        return std::unexpected("Failed to invert covariance for fusion");
    }

    // Fuse information matrices: Y = Y1 + Y2
    auto Y_fused        = *Y1_result + *Y2_result;
    auto P_fused_result = Y_fused.inverse();
    if (!P_fused_result) {
        return std::unexpected("Failed to invert fused information matrix");
    }

    // Information vectors: η = Y * m
    auto eta1      = Y1_result->multiply(D::diff(D::identity(), b1.mean).data);
    auto eta2      = Y2_result->multiply(D::diff(D::identity(), b2.mean).data);
    auto eta_fused = eta1 + eta2;

    // Fused mean: m = P * η
    auto mean_delta_result = P_fused_result->solve(eta_fused);
    if (!mean_delta_result) {
        return std::unexpected("Failed to solve for fused mean");
    }

    auto mean_fused = D::act(D::identity(), typename D::Fiber(*mean_delta_result));
    return Belief<D, ReprTag, Scalar>::WithCovariance(mean_fused, *P_fused_result);
}

} // namespace uni_kf
