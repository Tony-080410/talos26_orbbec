/// @file belief.hpp
/// @brief Belief state (mean + uncertainty) for Kalman filtering
///
/// A Belief represents a probability distribution over a DifferentialSpace,
/// characterized by a mean (point) and uncertainty (covariance).

#pragma once

#include "../concepts/differential_space.hpp"
#include "../concepts/quadratic_form.hpp"

#include <random>

namespace uni_kf {

// ============================================================================
// Belief: State + Uncertainty
// ============================================================================

/// @brief Gaussian belief state on a DifferentialSpace
///
/// Represents a Gaussian distribution over a DifferentialSpace D:
/// - mean: Most likely state (point on the manifold)
/// - covariance: Uncertainty in tangent/fiber space
///
/// @tparam D DifferentialSpace
/// @tparam Scalar_ Floating point type
template <DifferentialSpace D, typename Scalar_ = double>
struct Belief {
    using Space      = D;
    using Scalar     = Scalar_;
    using Point      = typename D::Point;
    using Fiber      = typename D::Fiber;
    using Covariance = SymmetricMatrix<D::FiberDim, Scalar>;

    static constexpr int Dim = D::FiberDim;

    Point mean;            ///< State estimate (point on manifold)
    Covariance covariance; ///< Uncertainty (in tangent space)

    // ------------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------------

    Belief() noexcept
        : mean(D::identity())
        , covariance(Covariance::Zero()) {}

    Belief(const Point& m, const Covariance& P) noexcept
        : mean(m)
        , covariance(P) {}

    Belief(Point&& m, Covariance&& P) noexcept
        : mean(std::move(m))
        , covariance(std::move(P)) {}

    // ------------------------------------------------------------------------
    // Static factory methods
    // ------------------------------------------------------------------------

    /// @brief Create belief at identity with zero covariance
    [[nodiscard]] static Belief Identity() noexcept { return Belief{}; }

    /// @brief Create belief at identity with given covariance
    [[nodiscard]] static Belief WithCovariance(const Covariance& P) noexcept {
        return Belief{D::identity(), P};
    }

    /// @brief Create belief with scaled identity covariance
    [[nodiscard]] static Belief WithScaledIdentity(const Point& m, Scalar scale) noexcept {
        return Belief{m, Covariance::ScaledIdentity(scale)};
    }

    // ------------------------------------------------------------------------
    // Operations
    // ------------------------------------------------------------------------

    /// @brief Apply correction in tangent space: mean ← mean ⊕ delta
    [[nodiscard]] Belief retract(const Fiber& delta) const noexcept {
        return Belief{D::act(mean, delta), covariance};
    }

    /// @brief Apply correction with new covariance
    [[nodiscard]] Belief retract(const Fiber& delta, const Covariance& P_new) const noexcept {
        return Belief{D::act(mean, delta), P_new};
    }

    /// @brief Compute difference to another belief (in tangent space)
    [[nodiscard]] Fiber diff_mean(const Belief& other) const noexcept {
        return D::diff(mean, other.mean);
    }

    // ------------------------------------------------------------------------
    // Uncertainty metrics
    // ------------------------------------------------------------------------

    /// @brief Trace of covariance (total variance)
    [[nodiscard]] Scalar trace() const noexcept { return covariance.trace(); }

    /// @brief Determinant of covariance
    [[nodiscard]] Scalar determinant() const noexcept { return covariance.determinant(); }

    /// @brief Log determinant (more numerically stable)
    [[nodiscard]] auto log_determinant() const noexcept -> std::expected<Scalar, std::string> {
        return covariance.log_determinant();
    }

    /// @brief Standard deviation in a given direction
    [[nodiscard]] Scalar std_dev(const Fiber& direction) const noexcept {
        auto d_normalized = direction.data.normalized();
        return std::sqrt(d_normalized.dot(covariance.matrix() * d_normalized));
    }

    /// @brief Mahalanobis distance to a point
    [[nodiscard]] auto mahalanobis(const Point& p) const noexcept
        -> std::expected<Scalar, std::string> {
        auto delta  = D::diff(mean, p);
        auto result = covariance.solve(delta.data);
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

        // Transform to covariance
        auto L_result = covariance.cholesky();
        if (!L_result) {
            return mean; // Fallback: return mean
        }

        typename Fiber::EigenType delta = (*L_result) * z;
        return D::act(mean, Fiber(delta));
    }

    // ------------------------------------------------------------------------
    // Numerical stability
    // ------------------------------------------------------------------------

    /// @brief Symmetrize covariance
    void symmetrize() noexcept { covariance.symmetrize_in_place(); }

    /// @brief Add regularization to covariance diagonal
    void regularize(Scalar eps = Scalar{1e-12}) noexcept { covariance.regularize_in_place(eps); }

    /// @brief Check if covariance is finite
    [[nodiscard]] bool is_finite() const noexcept { return covariance.is_finite(); }

    // ------------------------------------------------------------------------
    // Projection (for constrained estimation)
    // ------------------------------------------------------------------------

    /// @brief Project covariance to enforce constraints
    /// @param mask Indices of constrained dimensions (set their variance to near-zero)
    template <typename IndexArray>
    void enforce_constraints(
        const IndexArray& constrained_indices, Scalar eps = Scalar{1e-12}) noexcept {
        for (auto idx : constrained_indices) {
            // Zero out row and column
            for (int j = 0; j < Dim; ++j) {
                covariance(idx, j) = Scalar{0};
                covariance(j, idx) = Scalar{0};
            }
            // Small diagonal for numerical stability
            covariance(idx, idx) = eps;
        }
    }
};

// ============================================================================
// Belief utilities
// ============================================================================

/// @brief Compute KL divergence between two beliefs (approximate for non-Gaussian)
template <DifferentialSpace D, typename Scalar>
[[nodiscard]] auto kl_divergence(const Belief<D, Scalar>& p, const Belief<D, Scalar>& q) noexcept
    -> std::expected<Scalar, std::string> {
    constexpr int N = D::FiberDim;

    // KL(p || q) = 0.5 * (tr(Q^{-1} P) + (m_q - m_p)^T Q^{-1} (m_q - m_p) - n + log(|Q|/|P|))

    auto Q_inv_result = q.covariance.inverse();
    if (!Q_inv_result) {
        return std::unexpected(Q_inv_result.error());
    }

    auto delta = D::diff(p.mean, q.mean);

    // tr(Q^{-1} P)
    Scalar trace_term = (Q_inv_result->matrix() * p.covariance.matrix()).trace();

    // (m_q - m_p)^T Q^{-1} (m_q - m_p)
    Scalar quad_term = delta.data.dot(Q_inv_result->multiply(delta.data));

    // log(|Q|/|P|)
    auto log_det_q = q.covariance.log_determinant();
    auto log_det_p = p.covariance.log_determinant();
    if (!log_det_q || !log_det_p) {
        return std::unexpected("Failed to compute log determinant");
    }
    Scalar log_ratio = *log_det_q - *log_det_p;

    return Scalar{0.5} * (trace_term + quad_term - N + log_ratio);
}

/// @brief Fuse two independent beliefs using information form
template <DifferentialSpace D, typename Scalar>
[[nodiscard]] auto fuse_beliefs(const Belief<D, Scalar>& b1, const Belief<D, Scalar>& b2) noexcept
    -> std::expected<Belief<D, Scalar>, std::string> {
    // Convert to information matrices: Y = P^{-1}
    auto Y1_result = b1.covariance.inverse();
    auto Y2_result = b2.covariance.inverse();
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
    return Belief<D, Scalar>{mean_fused, *P_fused_result};
}

} // namespace uni_kf
