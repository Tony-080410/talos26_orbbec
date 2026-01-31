#pragma once

#include <Eigen/Dense>
#include <groups/SEn3.hpp>
#include <groups/SO3.hpp>
#include <span>
#include <vector>

namespace uni_kf::redact::lie_ops {

using namespace group; // for SO3, SEn3 type aliases

/// @brief Lie group utility operations for Kalman filtering
///
/// Provides commonly-used combinations of lieplusplus operations to
/// reduce verbosity in filter code. All operations use right-invariant
/// error convention: η_R = Log(X̂⁻¹X)

/// @brief Compose state with tangent space vector: x ⊕ ξ = x · exp(ξ)
///
/// This is the **retraction** operation that maps a tangent vector
/// back to the manifold.
///
/// @param x State on manifold
/// @param xi Tangent space vector (error/velocity)
/// @return Composed state
template <typename LieGroup>
[[nodiscard]] auto compose(const LieGroup& x, const typename LieGroup::VectorType& xi) {
    return x * LieGroup::exp(xi);
}

/// @brief Compute right-invariant error: η = Log(a⁻¹ · b)
///
/// This is the **difference** operation that computes the tangent space
/// vector from state a to state b, using right-invariant error convention.
///
/// @param a From state
/// @param b To state
/// @return Tangent space vector representing error
template <typename LieGroup>
[[nodiscard]] auto diff(const LieGroup& a, const LieGroup& b) {
    return LieGroup::log(a.inv() * b);
}

/// @brief Compute Adjoint: Ad_X
///
/// Returns the adjoint matrix of x, used for transforming tangent space
/// vectors/covariances under the group action.
///
/// @param x Lie group element
/// @return Adjoint matrix of x
template <typename LieGroup>
[[nodiscard]] auto adjoint(const LieGroup& x) -> Eigen::MatrixXd {
    return x.Adjoint();
}

/// @brief Compute Adjoint of inverse: Ad_X⁻¹
///
/// Used for covariance transformation when coordinate frame changes.
/// Equivalent to adjoint(x.inv()) but may be more efficient for some groups.
///
/// @param x Lie group element
/// @return Adjoint matrix of x⁻¹
template <typename LieGroup>
[[nodiscard]] auto inv_adjoint(const LieGroup& x) -> Eigen::MatrixXd {
    return x.inv().Adjoint();
}

/// @brief Karcher mean (Riemannian mean) on Lie group
///
/// Iteratively computes the mean of points on a manifold using:
///   μ_{k+1} = μ_k · Exp(Σ w_i · Log(μ_k⁻¹ · z_i))
///
/// @param points Points to average (must have *, inv, log operations)
/// @param weights Weights for each point (must sum to 1)
/// @param iterations Number of iterations (2-3 sufficient for most cases)
/// @return Mean state on manifold
template <typename LieGroup>
[[nodiscard]] LieGroup karcher_mean(
    const std::vector<LieGroup>& points, std::span<const double> weights, int iterations = 3) {
    using VectorType = typename LieGroup::VectorType;

    if (points.empty()) {
        return LieGroup{};
    }

    // Initialize with first point
    LieGroup mu = points[0];

    for (int iter = 0; iter < iterations; ++iter) {
        VectorType step = VectorType::Zero();
        for (size_t i = 0; i < points.size(); ++i) {
            const auto delta = diff(mu, points[i]);
            step += weights[i] * delta;
        }
        mu = compose(mu, step);
    }

    return mu;
}

/// @brief Weighted covariance on Lie group tangent space
///
/// Computes: P = Σ w_i · (μ⁻¹ ⊕ z_i) · (μ⁻¹ ⊕ z_i)^T
/// where ⊕ denotes compose (retraction to tangent space).
///
/// @param points Points on manifold
/// @param mean Mean state (from karcher_mean or similar)
/// @param weights Weights for each point
/// @return Covariance matrix in tangent space at mean
template <typename LieGroup>
[[nodiscard]] Eigen::MatrixXd tangent_covariance(
    const std::vector<LieGroup>& points, const LieGroup& mean, std::span<const double> weights) {
    using VectorType = typename LieGroup::VectorType;
    const int dim    = VectorType::SizeAtCompileTime;

    if (dim == Eigen::Dynamic || points.empty()) {
        return Eigen::MatrixXd{};
    }

    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(dim, dim);

    for (size_t i = 0; i < points.size(); ++i) {
        const VectorType xi = diff(mean, points[i]);
        P += weights[i] * (xi * xi.transpose());
    }

    return P;
}

} // namespace uni_kf::redact::lie_ops
