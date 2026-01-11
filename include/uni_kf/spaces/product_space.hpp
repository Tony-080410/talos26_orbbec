/// @file product_space.hpp
/// @brief Product of DifferentialSpaces for augmented state estimation
///
/// ProductSpace<D1, D2> combines two DifferentialSpaces into a single space.
/// This is essential for InEKF where the state is SE₂(3) × ℝⁿ (Lie group plus
/// extra Euclidean parameters like yaw rate, radius, height).
///
/// The fiber of the product space is the concatenation of the individual fibers.

#pragma once

#include "../concepts/differential_space.hpp"
#include "../concepts/linear.hpp"
#include "euclidean.hpp"
#include "lie_manifold.hpp"

#include <utility>

namespace uni_kf {

// ============================================================================
// Product Point type
// ============================================================================

/// @brief Point in a product space = pair of points
template <typename Point1, typename Point2>
struct ProductPoint {
    Point1 first;
    Point2 second;

    ProductPoint() noexcept = default;

    ProductPoint(const Point1& p1, const Point2& p2) noexcept
        : first(p1)
        , second(p2) {}

    ProductPoint(Point1&& p1, Point2&& p2) noexcept
        : first(std::move(p1))
        , second(std::move(p2)) {}
};

// ============================================================================
// ProductSpace
// ============================================================================

/// @brief Product of two DifferentialSpaces
///
/// Given spaces D1 and D2, the product space D1 × D2 has:
/// - Point = (D1::Point, D2::Point)
/// - Fiber = concat(D1::Fiber, D2::Fiber)
/// - act((p1, p2), (v1, v2)) = (D1::act(p1, v1), D2::act(p2, v2))
/// - diff((p1, p2), (q1, q2)) = concat(D1::diff(p1, q1), D2::diff(p2, q2))
///
/// @tparam D1 First DifferentialSpace
/// @tparam D2 Second DifferentialSpace
template <DifferentialSpace D1, DifferentialSpace D2>
requires std::is_same_v<typename D1::Scalar, typename D2::Scalar> struct ProductSpace {
    // ------------------------------------------------------------------------
    // Type definitions
    // ------------------------------------------------------------------------

    using Scalar = typename D1::Scalar;
    using Point  = ProductPoint<typename D1::Point, typename D2::Point>;

    /// Fiber dimension is sum of component dimensions
    static constexpr int Dim1     = D1::FiberDim;
    static constexpr int Dim2     = D2::FiberDim;
    static constexpr int FiberDim = Dim1 + Dim2;

    using Fiber = VecN<FiberDim, Scalar>;

    /// Jacobian matrix type (block diagonal)
    using JacobianType = SquareMatN<FiberDim, Scalar>;

    // ------------------------------------------------------------------------
    // Fiber splitting/joining utilities
    // ------------------------------------------------------------------------

    /// @brief Split fiber into components
    [[nodiscard]] static auto split_fiber(const Fiber& v) noexcept
        -> std::pair<typename D1::Fiber, typename D2::Fiber> {
        typename D1::Fiber v1(v.data.template head<Dim1>());
        typename D2::Fiber v2(v.data.template tail<Dim2>());
        return {v1, v2};
    }

    /// @brief Join component fibers
    [[nodiscard]] static Fiber
        join_fiber(const typename D1::Fiber& v1, const typename D2::Fiber& v2) noexcept {
        typename Fiber::EigenType result;
        result << v1.data, v2.data;
        return Fiber(result);
    }

    // ------------------------------------------------------------------------
    // DifferentialSpace interface
    // ------------------------------------------------------------------------

    /// @brief Identity element
    [[nodiscard]] static Point identity() noexcept { return Point{D1::identity(), D2::identity()}; }

    /// @brief Retraction: (p1, p2) ⊕ (v1, v2) = (p1 ⊕ v1, p2 ⊕ v2)
    [[nodiscard]] static Point act(const Point& p, const Fiber& v) noexcept {
        auto [v1, v2] = split_fiber(v);
        return Point{D1::act(p.first, v1), D2::act(p.second, v2)};
    }

    /// @brief Local coordinates: (q1, q2) ⊖ (p1, p2) = (q1 ⊖ p1, q2 ⊖ p2)
    [[nodiscard]] static Fiber diff(const Point& p, const Point& q) noexcept {
        auto v1 = D1::diff(p.first, q.first);
        auto v2 = D2::diff(p.second, q.second);
        return join_fiber(v1, v2);
    }

    // ------------------------------------------------------------------------
    // Jacobians (block diagonal)
    // ------------------------------------------------------------------------

    /// @brief Left Jacobian = diag(J1_l, J2_l)
    [[nodiscard]] static JacobianType left_jacobian(const Fiber& v) noexcept {
        auto [v1, v2]                            = split_fiber(v);
        JacobianType J                           = JacobianType::Zero();
        J.template block<Dim1, Dim1>(0, 0)       = D1::left_jacobian(v1);
        J.template block<Dim2, Dim2>(Dim1, Dim1) = D2::left_jacobian(v2);
        return J;
    }

    /// @brief Right Jacobian = diag(J1_r, J2_r)
    [[nodiscard]] static JacobianType right_jacobian(const Fiber& v) noexcept {
        auto [v1, v2]                            = split_fiber(v);
        JacobianType J                           = JacobianType::Zero();
        J.template block<Dim1, Dim1>(0, 0)       = D1::right_jacobian(v1);
        J.template block<Dim2, Dim2>(Dim1, Dim1) = D2::right_jacobian(v2);
        return J;
    }

    /// @brief Inverse left Jacobian
    [[nodiscard]] static JacobianType inv_left_jacobian(const Fiber& v) noexcept {
        auto [v1, v2]                            = split_fiber(v);
        JacobianType J                           = JacobianType::Zero();
        J.template block<Dim1, Dim1>(0, 0)       = D1::inv_left_jacobian(v1);
        J.template block<Dim2, Dim2>(Dim1, Dim1) = D2::inv_left_jacobian(v2);
        return J;
    }

    /// @brief Inverse right Jacobian
    [[nodiscard]] static JacobianType inv_right_jacobian(const Fiber& v) noexcept {
        auto [v1, v2]                            = split_fiber(v);
        JacobianType J                           = JacobianType::Zero();
        J.template block<Dim1, Dim1>(0, 0)       = D1::inv_right_jacobian(v1);
        J.template block<Dim2, Dim2>(Dim1, Dim1) = D2::inv_right_jacobian(v2);
        return J;
    }

    // ------------------------------------------------------------------------
    // Adjoint representation (block diagonal)
    // ------------------------------------------------------------------------

    /// @brief Adjoint action
    [[nodiscard]] static Fiber adjoint(const Point& p, const Fiber& v) noexcept {
        auto [v1, v2] = split_fiber(v);
        auto adj_v1   = D1::adjoint(p.first, v1);
        auto adj_v2   = D2::adjoint(p.second, v2);
        return join_fiber(adj_v1, adj_v2);
    }

    /// @brief Adjoint matrix = diag(Ad1, Ad2)
    [[nodiscard]] static JacobianType adjoint_matrix(const Point& p) noexcept {
        JacobianType Ad                           = JacobianType::Zero();
        Ad.template block<Dim1, Dim1>(0, 0)       = D1::adjoint_matrix(p.first);
        Ad.template block<Dim2, Dim2>(Dim1, Dim1) = D2::adjoint_matrix(p.second);
        return Ad;
    }

    // ------------------------------------------------------------------------
    // Component access
    // ------------------------------------------------------------------------

    /// @brief Get first component point
    [[nodiscard]] static const typename D1::Point& first(const Point& p) noexcept {
        return p.first;
    }

    /// @brief Get second component point
    [[nodiscard]] static const typename D2::Point& second(const Point& p) noexcept {
        return p.second;
    }

    /// @brief Get first component of fiber
    [[nodiscard]] static typename D1::Fiber first_fiber(const Fiber& v) noexcept {
        return typename D1::Fiber(v.data.template head<Dim1>());
    }

    /// @brief Get second component of fiber
    [[nodiscard]] static typename D2::Fiber second_fiber(const Fiber& v) noexcept {
        return typename D2::Fiber(v.data.template tail<Dim2>());
    }
};

// ============================================================================
// Type alias for SE₂(3) × ℝⁿ (InEKF state space)
// ============================================================================

/// @brief SE₂(3) × ℝⁿ state space for InEKF with extra parameters
///
/// This is the typical state space for robot tracking:
/// - SE₂(3): rotation, velocity, position
/// - ℝⁿ: extra parameters (yaw rate, radius, height, etc.)
template <int ExtraDim, typename Scalar = double, LieSide Side = LieSide::Right>
using SE23_Rn =
    ProductSpace<LieManifold<group::SEn3<Scalar, 2>, Side>, Euclidean<ExtraDim, Scalar>>;

// Common configurations matching existing InEKF
using SE23_R4 = SE23_Rn<4, double, LieSide::Right>; // Robot: [v_yaw, r0, r1, h]
using SE23_R3 = SE23_Rn<3, double, LieSide::Right>; // Outpost: [v_yaw, dz1, dz2]

// ============================================================================
// Concept verification
// ============================================================================

static_assert(DifferentialSpace<ProductSpace<R3, R3>>, "R3 × R3 must satisfy DifferentialSpace");
static_assert(DifferentialSpace<SE23_R4>, "SE₂(3) × R⁴ must satisfy DifferentialSpace");

// Verify fiber dimension
static_assert(SE23_R4::FiberDim == 13, "SE₂(3) has 9 DOF, plus 4 extra = 13");
static_assert(ProductSpace<R3, R3>::FiberDim == 6, "R³ × R³ has 6 DOF");

} // namespace uni_kf
