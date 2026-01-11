/// @file differential_space.hpp
/// @brief Unified DifferentialSpace concept - the core abstraction of uni_kf
///
/// This file defines the central abstraction that unifies vector spaces and
/// Lie groups under a single categorical framework. Both Euclidean spaces
/// and Lie groups (SE(3), SO(3), etc.) are instances of DifferentialSpace.
///
/// Key insight: Every DifferentialSpace has:
/// - Point: The state manifold (could be R^n or a Lie group)
/// - Fiber: The tangent/error space (always a vector space)
/// - act(p, v): Retraction - move from point p along tangent v
/// - diff(p, q): Local coordinates - difference between p and q in tangent space
///
/// For vector spaces: Point = Fiber, act = +, diff = -
/// For Lie groups: Point = G, Fiber = Lie algebra, act = ·Exp, diff = Log

#pragma once

#include "linear.hpp"

#include <concepts>
#include <type_traits>

namespace uni_kf {

// ============================================================================
// DifferentialSpace Concept
// ============================================================================

/// @brief Concept for differentiable state spaces
///
/// A DifferentialSpace is an abstract space where:
/// 1. Points represent states (could be vectors or group elements)
/// 2. Fibers (tangent spaces) are vector spaces representing local perturbations
/// 3. act(p, v) moves from point p along tangent vector v (retraction)
/// 4. diff(p, q) computes the tangent vector from p to q (logarithm)
///
/// Axioms (Torsor-like properties):
/// - diff(p, act(p, v)) = v (right inverse)
/// - act(p, diff(p, q)) = q (left inverse)
/// - diff(p, p) = zero (reflexivity)
///
/// @note This unifies Euclidean spaces (trivial case) and Lie groups
template <typename D>
concept DifferentialSpace =
    requires {
        /// Point type - the state manifold
        typename D::Point;

        /// Fiber type - the tangent/error space (must be a vector space)
        typename D::Fiber;

        /// Scalar type (inherited from Fiber)
        typename D::Scalar;

        /// Compile-time dimension of the fiber (tangent space)
        { D::FiberDim } -> std::convertible_to<int>;
    } && Linear<typename D::Fiber>
    && requires(typename D::Point p, typename D::Point q, typename D::Fiber v) {
           /// Identity/origin element
           { D::identity() } -> std::same_as<typename D::Point>;

           /// Retraction: p ⊕ v = act(p, v)
           /// For R^n: p + v
           /// For Lie groups: p · Exp(v) or Exp(v) · p depending on convention
           { D::act(p, v) } -> std::same_as<typename D::Point>;

           /// Local coordinates: q ⊖ p = diff(p, q)
           /// For R^n: q - p
           /// For Lie groups: Log(p^{-1} · q) or Log(q · p^{-1}) depending on convention
           { D::diff(p, q) } -> std::same_as<typename D::Fiber>;
       };

// ============================================================================
// DifferentialSpaceWithJacobian Concept
// ============================================================================

/// @brief Extended concept for spaces with analytical Jacobians
///
/// Lie groups have non-trivial Jacobians for the exponential map.
/// This concept captures spaces that provide analytical Jacobian computation,
/// which is essential for efficient covariance propagation.
template <typename D>
concept DifferentialSpaceWithJacobian = DifferentialSpace<D> && requires(typename D::Fiber v) {
    /// Left Jacobian of exponential map: d/dv (Exp(v))
    { D::left_jacobian(v) };

    /// Right Jacobian of exponential map
    { D::right_jacobian(v) };

    /// Inverse left Jacobian
    { D::inv_left_jacobian(v) };

    /// Inverse right Jacobian
    { D::inv_right_jacobian(v) };
};

/// @brief Concept for spaces with Adjoint representation
///
/// Lie groups have an Adjoint representation that maps tangent vectors
/// between different tangent spaces. This is essential for InEKF.
template <typename D>
concept DifferentialSpaceWithAdjoint =
    DifferentialSpace<D> && requires(typename D::Point p, typename D::Fiber v) {
        /// Adjoint representation: Ad_p(v)
        /// Maps tangent vector at identity to tangent vector at p
        { D::adjoint(p, v) } -> std::same_as<typename D::Fiber>;

        /// Adjoint matrix at point p
        { D::adjoint_matrix(p) };
    };

// ============================================================================
// Helper type traits
// ============================================================================

/// @brief Extract Point type from DifferentialSpace
template <DifferentialSpace D>
using point_t = typename D::Point;

/// @brief Extract Fiber type from DifferentialSpace
template <DifferentialSpace D>
using fiber_t = typename D::Fiber;

/// @brief Extract Scalar type from DifferentialSpace
template <DifferentialSpace D>
using scalar_t = typename D::Scalar;

/// @brief Get fiber dimension at compile time
template <DifferentialSpace D>
constexpr int fiber_dim_v = D::FiberDim;

/// @brief Check if a DifferentialSpace is trivial (Point = Fiber)
template <typename D>
concept TrivialDifferentialSpace =
    DifferentialSpace<D> && std::is_same_v<typename D::Point, typename D::Fiber>;

/// @brief Check if a DifferentialSpace is a Lie group
template <typename D>
concept LieGroupSpace = DifferentialSpace<D> && !TrivialDifferentialSpace<D>
                     && DifferentialSpaceWithJacobian<D> && DifferentialSpaceWithAdjoint<D>;

// ============================================================================
// DifferentialMap Concept
// ============================================================================

/// @brief Concept for smooth maps between DifferentialSpaces
///
/// A DifferentialMap F: D1 → D2 is a map that:
/// 1. Maps points: F.map(p) : D1::Point → D2::Point
/// 2. Maps tangent vectors (differential): F.fiber_map(p, v) : D1::Fiber → D2::Fiber
///
/// The fiber map is the "Jacobian" of F at point p, acting on tangent vector v.
template <typename F, typename D1, typename D2>
concept DifferentialMap = DifferentialSpace<D1> && DifferentialSpace<D2>
                       && requires(F f, typename D1::Point p, typename D1::Fiber v) {
                              /// Point map
                              { f.map(p) } -> std::same_as<typename D2::Point>;

                              /// Fiber map (differential/Jacobian)
                              { f.fiber_map(p, v) } -> std::same_as<typename D2::Fiber>;
                          };

// ============================================================================
// Retraction and Local Coordinates utilities
// ============================================================================

/// @brief Apply retraction: p ⊕ v
template <DifferentialSpace D>
[[nodiscard]] constexpr auto
    retract(const typename D::Point& p, const typename D::Fiber& v) noexcept -> typename D::Point {
    return D::act(p, v);
}

/// @brief Compute local coordinates: q ⊖ p
template <DifferentialSpace D>
[[nodiscard]] constexpr auto local(const typename D::Point& p, const typename D::Point& q) noexcept
    -> typename D::Fiber {
    return D::diff(p, q);
}

/// @brief Parallel transport (default: no-op for flat spaces, needs override for Lie groups)
template <DifferentialSpace D>
[[nodiscard]] constexpr auto parallel_transport(
    const typename D::Point& /*from*/, const typename D::Point& /*to*/,
    const typename D::Fiber& v) noexcept -> typename D::Fiber {
    // Default implementation: identity (valid for Euclidean spaces)
    // Lie groups should provide their own implementation using Adjoint
    return v;
}

// ============================================================================
// Axiom verification utilities (for testing)
// ============================================================================

namespace detail {

/// @brief Check right-inverse axiom: diff(p, act(p, v)) ≈ v
template <DifferentialSpace D>
[[nodiscard]] bool check_right_inverse(
    const typename D::Point& p, const typename D::Fiber& v,
    typename D::Scalar tol = 1e-10) noexcept {
    auto q      = D::act(p, v);
    auto v_back = D::diff(p, q);
    return (v.eigen() - v_back.eigen()).norm() < tol;
}

/// @brief Check left-inverse axiom: act(p, diff(p, q)) ≈ q
/// Note: This is harder to check for Lie groups since Point comparison isn't trivial
template <DifferentialSpace D>
requires TrivialDifferentialSpace<D> [[nodiscard]] bool check_left_inverse(
    const typename D::Point& p, const typename D::Point& q,
    typename D::Scalar tol = 1e-10) noexcept {
    auto v      = D::diff(p, q);
    auto q_back = D::act(p, v);
    return (q.eigen() - q_back.eigen()).norm() < tol;
}

/// @brief Check reflexivity: diff(p, p) ≈ 0
template <DifferentialSpace D>
[[nodiscard]] bool
    check_reflexivity(const typename D::Point& p, typename D::Scalar tol = 1e-10) noexcept {
    auto v = D::diff(p, p);
    return v.norm() < tol;
}

} // namespace detail

} // namespace uni_kf
