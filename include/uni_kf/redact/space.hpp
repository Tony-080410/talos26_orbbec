#pragma once
#include <Eigen/Core>
#include <concepts>

/// @file space.hpp
/// @brief Core concepts for the unified Kalman filter framework
///
/// ## Type System Design
///
/// This framework uses a type-based state estimation approach where:
/// - **Flux**: State on the manifold (流形上的状态)
/// - **Drift**: Tangent/error vector (切空间误差向量)
/// - **DriftExtent**: Extent/bounds for error (误差范围)
///
/// The semantic operations are:
///   compose(x, ξ) = x ⊕ ξ  (state retraction / 组合)
///   diff(x, y) = y ⊖ x      (error computation / 差分)
///
/// For Lie groups: compose(x, ξ) = x · exp(ξ), diff(x, y) = Log(x⁻¹ · y)
///

/// @brief Core concept for all state spaces in the unified Kalman filter
///
/// Space defines the minimal interface for state estimation on manifolds.
/// All filter strategies (KF, EKF, UKF, CKF, InKF, InEKF) operate on Spaces.
///
/// ## Required Types
/// - `Flux`: State on the manifold (e.g., SO(3) rotation, SE(3) pose)
/// - `Drift`: Tangent/error vector in Lie algebra (e.g., ω ∈ ℝ³ for SO(3))
/// - `DriftExtent`: Extent/bounds for error (用于误差范围)
/// - `WeightedPoint`: Point with mean/covariance weights (for sigma-point filters)
///
/// ## Required Operations
/// - `zero_flux()`: Identity element on manifold (单位元)
/// - `zero_drift()`: Zero error in tangent space (零误差)
/// - `compose(x, ξ)`: State retraction: x ⊕ ξ (组合状态与误差)
/// - `diff(x, y)`: Error computation: y ⊖ x (计算从x到y的误差)
/// - `norm(ξ)`: Error magnitude (误差范数)
/// - `inverse(x)`: Inverse on manifold (流形上的逆)
/// - `extent(e, ξ)`: Apply extent to error (应用范围到误差)
///
/// ## Mathematical Background
///
/// For Lie groups, these operations implement:
///   compose(x, ξ) = x · exp(ξ)  (right-invariant)
///   diff(x, y) = Log(x⁻¹ · y)    (right-invariant error)
///
/// For Euclidean spaces:
///   compose(x, ξ) = x + ξ
///   diff(x, y) = y - x
template <typename T>
concept Space = requires(T) {
    // ========================================================================
    // Required Types (类型要求)
    // ========================================================================

    /// Flux: State on the manifold / 状态在流形上
    typename T::Flux;

    /// Drift: Tangent/error vector / 切空间误差向量
    typename T::Drift;

    /// DriftExtent: Extent/bounds for error / 误差范围
    typename T::DriftExtent;

    /// WeightedPoint: Point with weights for sigma-point filters
    /// Used by UKF/CKF for weighted mean and covariance computation
    typename T::WeightedPoint;

    // ========================================================================
    // Dimension (维度)
    // ========================================================================

    /// State dimension (compile-time if available, otherwise runtime)
    { T::Dim } -> std::convertible_to<int>;

    // ========================================================================
    // Core Operations (核心操作)
    // ========================================================================

    /// Zero element: Generate identity / 零元
    { T::zero_flux() } -> std::same_as<typename T::Flux>;
    { T::zero_drift() } -> std::same_as<typename T::Drift>;

    /// Compose: x ⊕ ξ / 组合状态与误差
    /// Semantic: state + error in tangent space
    {
        T::compose(std::declval<typename T::Flux>(), std::declval<typename T::Drift>())
    } -> std::same_as<typename T::Flux>;

    /// Difference: y ⊖ x / 计算从x到y的误差
    /// Semantic: error in tangent space from x to y
    {
        T::diff(std::declval<typename T::Flux>(), std::declval<typename T::Flux>())
    } -> std::same_as<typename T::Drift>;

    /// Norm: ||ξ|| / 衡量误差的大小
    { T::norm(std::declval<typename T::Drift>()) } -> std::totally_ordered;

    /// Inverse: x⁻¹ / 流形上的逆运算
    { T::inverse(std::declval<typename T::Drift>()) } -> std::same_as<typename T::Drift>;
    { T::inverse(std::declval<typename T::Flux>()) } -> std::same_as<typename T::Flux>;

    /// Extent: Apply bounds to error / 应用范围到误差
    {
        T::extent(std::declval<typename T::DriftExtent>(), std::declval<typename T::Drift>())
    } -> std::same_as<typename T::Drift>;
};
