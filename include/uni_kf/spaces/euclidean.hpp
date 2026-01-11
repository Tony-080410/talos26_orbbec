/// @file euclidean.hpp
/// @brief Euclidean space adapter for DifferentialSpace
///
/// Euclidean<N> is the trivial DifferentialSpace where Point = Fiber.
/// This represents standard R^n vector spaces, and serves as the base case
/// for the unified framework.
///
/// For Euclidean spaces:
/// - act(p, v) = p + v (vector addition)
/// - diff(p, q) = q - p (vector subtraction)
/// - Jacobians are all identity matrices

#pragma once

#include "../concepts/differential_space.hpp"
#include "../concepts/linear.hpp"
#include "../core/types.hpp"

namespace uni_kf {

// ============================================================================
// Euclidean Space
// ============================================================================

/// @brief Euclidean (vector) space as a DifferentialSpace
///
/// This is the trivial case where the state manifold is R^n and the
/// tangent space is identical to the state space itself.
///
/// @tparam N Dimension of the space
/// @tparam Scalar_ Floating point type (default: double)
template <int N, typename Scalar_ = double>
struct Euclidean {
    static_assert(N > 0 || N == Eigen::Dynamic, "Dimension must be positive or Dynamic");

    // ------------------------------------------------------------------------
    // Type definitions (required by DifferentialSpace concept)
    // ------------------------------------------------------------------------

    using Scalar = Scalar_;
    using Point  = VecN<N, Scalar>; // State is a vector
    using Fiber  = VecN<N, Scalar>; // Tangent space = State space

    /// Compile-time fiber dimension
    static constexpr int FiberDim = N;

    // ------------------------------------------------------------------------
    // DifferentialSpace interface
    // ------------------------------------------------------------------------

    /// @brief Identity element (zero vector)
    [[nodiscard]] static Point identity() noexcept { return Point::Zero(); }

    /// @brief Retraction: p ⊕ v = p + v (vector addition)
    [[nodiscard]] static Point act(const Point& p, const Fiber& v) noexcept {
        return Point(p.data + v.data);
    }

    /// @brief Local coordinates: q ⊖ p = q - p (vector subtraction)
    [[nodiscard]] static Fiber diff(const Point& p, const Point& q) noexcept {
        return Fiber(q.data - p.data);
    }

    // ------------------------------------------------------------------------
    // Jacobians (all trivial for Euclidean space)
    // ------------------------------------------------------------------------

    /// @brief Left Jacobian = Identity
    [[nodiscard]] static auto left_jacobian(const Fiber& /*v*/) noexcept -> SquareMatN<N, Scalar> {
        return SquareMatN<N, Scalar>::Identity();
    }

    /// @brief Right Jacobian = Identity
    [[nodiscard]] static auto right_jacobian(const Fiber& /*v*/) noexcept -> SquareMatN<N, Scalar> {
        return SquareMatN<N, Scalar>::Identity();
    }

    /// @brief Inverse Left Jacobian = Identity
    [[nodiscard]] static auto inv_left_jacobian(const Fiber& /*v*/) noexcept
        -> SquareMatN<N, Scalar> {
        return SquareMatN<N, Scalar>::Identity();
    }

    /// @brief Inverse Right Jacobian = Identity
    [[nodiscard]] static auto inv_right_jacobian(const Fiber& /*v*/) noexcept
        -> SquareMatN<N, Scalar> {
        return SquareMatN<N, Scalar>::Identity();
    }

    // ------------------------------------------------------------------------
    // Adjoint representation (trivial for Euclidean)
    // ------------------------------------------------------------------------

    /// @brief Adjoint action: Ad_p(v) = v (identity for Euclidean)
    [[nodiscard]] static Fiber adjoint(const Point& /*p*/, const Fiber& v) noexcept { return v; }

    /// @brief Adjoint matrix = Identity
    [[nodiscard]] static auto adjoint_matrix(const Point& /*p*/) noexcept -> SquareMatN<N, Scalar> {
        return SquareMatN<N, Scalar>::Identity();
    }

    // ------------------------------------------------------------------------
    // Point utilities (specific to Euclidean space)
    // ------------------------------------------------------------------------

    /// @brief Distance between two points
    [[nodiscard]] static Scalar distance(const Point& p, const Point& q) noexcept {
        return diff(p, q).norm();
    }

    /// @brief Squared distance between two points
    [[nodiscard]] static Scalar squared_distance(const Point& p, const Point& q) noexcept {
        return diff(p, q).squaredNorm();
    }

    /// @brief Linear interpolation between two points
    [[nodiscard]] static Point lerp(const Point& p, const Point& q, Scalar t) noexcept {
        return act(p, diff(p, q) * t);
    }
};

// ============================================================================
// Common Euclidean space type aliases
// ============================================================================

using R1  = Euclidean<1>;
using R2  = Euclidean<2>;
using R3  = Euclidean<3>;
using R4  = Euclidean<4>;
using R6  = Euclidean<6>;
using R9  = Euclidean<9>;
using R12 = Euclidean<12>;

using R1f  = Euclidean<1, float>;
using R2f  = Euclidean<2, float>;
using R3f  = Euclidean<3, float>;
using R4f  = Euclidean<4, float>;
using R6f  = Euclidean<6, float>;
using R9f  = Euclidean<9, float>;
using R12f = Euclidean<12, float>;

// ============================================================================
// Concept verification
// ============================================================================

static_assert(DifferentialSpace<R3>, "R3 must satisfy DifferentialSpace");
static_assert(DifferentialSpace<R6>, "R6 must satisfy DifferentialSpace");
static_assert(TrivialDifferentialSpace<R3>, "R3 must be a TrivialDifferentialSpace");
static_assert(DifferentialSpaceWithJacobian<R3>, "R3 must have Jacobians");
static_assert(DifferentialSpaceWithAdjoint<R3>, "R3 must have Adjoint");

} // namespace uni_kf
