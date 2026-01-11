/// @file lie_manifold.hpp
/// @brief Lie group adapter for DifferentialSpace
///
/// LieManifold<G, Side> wraps a Lie group from lieplusplus as a DifferentialSpace.
/// This enables using Lie groups (SO(3), SE(3), SE₂(3), etc.) with the unified
/// Kalman filter framework.
///
/// Supports both left-invariant and right-invariant error conventions:
/// - Right-invariant (default): ξ = Log(X̂⁻¹X), X = X̂·Exp(ξ)
/// - Left-invariant: ξ = Log(XX̂⁻¹), X = Exp(ξ)·X̂

#pragma once

#include "../concepts/differential_space.hpp"
#include "../concepts/linear.hpp"
#include "../core/types.hpp"

// lieplusplus headers
#include <groups/SEn3.hpp>
#include <groups/SO3.hpp>

namespace uni_kf {

// ============================================================================
// Lie group invariance side
// ============================================================================

/// @brief Specifies the invariance convention for error-state definition
enum class LieSide {
    Right, ///< Right-invariant: ξ = Log(X̂⁻¹X), X = X̂·Exp(ξ) [default]
    Left   ///< Left-invariant: ξ = Log(XX̂⁻¹), X = Exp(ξ)·X̂
};

// ============================================================================
// LieManifold adapter
// ============================================================================

/// @brief Lie group as a DifferentialSpace
///
/// Wraps a Lie group G from lieplusplus, providing the DifferentialSpace
/// interface (act, diff, Jacobians, Adjoint).
///
/// @tparam G_ Lie group type from lieplusplus (e.g., group::SO3<double>)
/// @tparam Side Invariance convention (Right or Left)
template <typename G_, LieSide Side = LieSide::Right>
struct LieManifold {
    // ------------------------------------------------------------------------
    // Type definitions
    // ------------------------------------------------------------------------

    using G          = G_;
    using Point      = G;                      // State is a group element
    using Scalar     = typename G::Scalar;     // Floating point type
    using EigenFiber = typename G::VectorType; // Lie algebra as Eigen vector

    /// Fiber type wrapping the Lie algebra
    using Fiber = VecN<EigenFiber::RowsAtCompileTime, Scalar>;

    /// Compile-time fiber dimension
    static constexpr int FiberDim = EigenFiber::RowsAtCompileTime;

    /// Jacobian matrix type
    using JacobianType = typename G::TMatrixType;

    // ------------------------------------------------------------------------
    // DifferentialSpace interface
    // ------------------------------------------------------------------------

    /// @brief Identity element of the group
    [[nodiscard]] static Point identity() noexcept { return G{}; }

    /// @brief Retraction: X ⊕ ξ
    /// Right: X̂ ⊕ ξ = X̂ · Exp(ξ)
    /// Left: X̂ ⊕ ξ = Exp(ξ) · X̂
    [[nodiscard]] static Point act(const Point& p, const Fiber& v) noexcept {
        if constexpr (Side == LieSide::Right) {
            return p * G::exp(v.data);
        } else {
            return G::exp(v.data) * p;
        }
    }

    /// @brief Local coordinates: X ⊖ X̂
    /// Right: X ⊖ X̂ = Log(X̂⁻¹ · X)
    /// Left: X ⊖ X̂ = Log(X · X̂⁻¹)
    [[nodiscard]] static Fiber diff(const Point& p, const Point& q) noexcept {
        if constexpr (Side == LieSide::Right) {
            return Fiber(G::log(p.inv() * q));
        } else {
            return Fiber(G::log(q * p.inv()));
        }
    }

    // ------------------------------------------------------------------------
    // Jacobians of the exponential map
    // ------------------------------------------------------------------------

    /// @brief Left Jacobian: J_l(ξ) where d/dξ Exp(ξ) = Exp(ξ) · J_l(ξ)
    [[nodiscard]] static JacobianType left_jacobian(const Fiber& v) noexcept {
        return G::leftJacobian(v.data);
    }

    /// @brief Right Jacobian: J_r(ξ) = J_l(-ξ)
    [[nodiscard]] static JacobianType right_jacobian(const Fiber& v) noexcept {
        return G::rightJacobian(v.data);
    }

    /// @brief Inverse left Jacobian
    [[nodiscard]] static JacobianType inv_left_jacobian(const Fiber& v) noexcept {
        return G::invLeftJacobian(v.data);
    }

    /// @brief Inverse right Jacobian
    [[nodiscard]] static JacobianType inv_right_jacobian(const Fiber& v) noexcept {
        return G::invRightJacobian(v.data);
    }

    // ------------------------------------------------------------------------
    // Adjoint representation
    // ------------------------------------------------------------------------

    /// @brief Adjoint action: Ad_X(ξ)
    /// Maps tangent vector at identity to tangent vector at X
    [[nodiscard]] static Fiber adjoint(const Point& p, const Fiber& v) noexcept {
        return Fiber((p.Adjoint() * v.data).eval());
    }

    /// @brief Adjoint matrix at point X
    [[nodiscard]] static JacobianType adjoint_matrix(const Point& p) noexcept {
        return p.Adjoint();
    }

    /// @brief Inverse Adjoint matrix
    [[nodiscard]] static JacobianType inv_adjoint_matrix(const Point& p) noexcept {
        return p.invAdjoint();
    }

    // ------------------------------------------------------------------------
    // Lie algebra operations
    // ------------------------------------------------------------------------

    /// @brief Lie algebra adjoint: ad_ξ (small adjoint)
    [[nodiscard]] static JacobianType lie_adjoint(const Fiber& v) noexcept {
        return G::adjoint(v.data);
    }
};

// ============================================================================
// Type aliases for common Lie groups
// ============================================================================

// SO(3) - 3D rotations
using SO3d = LieManifold<group::SO3<double>, LieSide::Right>;
using SO3f = LieManifold<group::SO3<float>, LieSide::Right>;

using SO3d_left = LieManifold<group::SO3<double>, LieSide::Left>;
using SO3f_left = LieManifold<group::SO3<float>, LieSide::Left>;

// SE(3) - 3D rigid transformations (n=1)
using SE3d = LieManifold<group::SEn3<double, 1>, LieSide::Right>;
using SE3f = LieManifold<group::SEn3<float, 1>, LieSide::Right>;

using SE3d_left = LieManifold<group::SEn3<double, 1>, LieSide::Left>;
using SE3f_left = LieManifold<group::SEn3<float, 1>, LieSide::Left>;

// SE₂(3) - Extended pose (rotation + velocity + position, n=2)
using SE23d = LieManifold<group::SEn3<double, 2>, LieSide::Right>;
using SE23f = LieManifold<group::SEn3<float, 2>, LieSide::Right>;

using SE23d_left = LieManifold<group::SEn3<double, 2>, LieSide::Left>;
using SE23f_left = LieManifold<group::SEn3<float, 2>, LieSide::Left>;

// SE₃(3) - Triple extended pose (n=3)
using SE33d = LieManifold<group::SEn3<double, 3>, LieSide::Right>;
using SE33f = LieManifold<group::SEn3<float, 3>, LieSide::Right>;

// ============================================================================
// Concept verification
// ============================================================================

static_assert(DifferentialSpace<SO3d>, "SO3d must satisfy DifferentialSpace");
static_assert(DifferentialSpace<SE3d>, "SE3d must satisfy DifferentialSpace");
static_assert(DifferentialSpace<SE23d>, "SE23d must satisfy DifferentialSpace");

static_assert(!TrivialDifferentialSpace<SO3d>, "SO3d is not a trivial space");
static_assert(!TrivialDifferentialSpace<SE3d>, "SE3d is not a trivial space");

static_assert(DifferentialSpaceWithJacobian<SO3d>, "SO3d must have Jacobians");
static_assert(DifferentialSpaceWithJacobian<SE3d>, "SE3d must have Jacobians");

static_assert(DifferentialSpaceWithAdjoint<SO3d>, "SO3d must have Adjoint");
static_assert(DifferentialSpaceWithAdjoint<SE3d>, "SE3d must have Adjoint");

// ============================================================================
// Helper functions for Lie group operations
// ============================================================================

/// @brief Compute geodesic distance on a Lie group
/// d(X, Y) = ||Log(X⁻¹Y)||
template <typename G, LieSide Side>
[[nodiscard]] auto geodesic_distance(
    const typename LieManifold<G, Side>::Point& p,
    const typename LieManifold<G, Side>::Point& q) noexcept -> typename G::Scalar {
    return LieManifold<G, Side>::diff(p, q).norm();
}

/// @brief Geodesic interpolation (SLERP-like for Lie groups)
/// γ(t) = X · Exp(t · Log(X⁻¹Y))
template <typename G, LieSide Side>
[[nodiscard]] auto geodesic_interpolate(
    const typename LieManifold<G, Side>::Point& p, const typename LieManifold<G, Side>::Point& q,
    typename G::Scalar t) noexcept -> typename LieManifold<G, Side>::Point {
    using Space = LieManifold<G, Side>;
    auto v      = Space::diff(p, q);
    return Space::act(p, typename Space::Fiber(v.data * t));
}

} // namespace uni_kf
