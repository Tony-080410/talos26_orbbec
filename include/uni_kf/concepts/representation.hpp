/// @file representation.hpp
/// @brief Uncertainty representation concepts and tags
///
/// Defines the concept for uncertainty representations (Covariance, Information, SquareRoot)
/// and their corresponding tag types for template dispatch.

#pragma once

#include <concepts>
#include <expected>
#include <string>

namespace uni_kf {

// ============================================================================
// Representation Tags
// ============================================================================

/// @brief Tag for standard covariance matrix representation (P)
struct CovarianceTag {
    static constexpr const char* name = "Covariance";
};

/// @brief Tag for information matrix representation (Y = P^-1)
struct InformationTag {
    static constexpr const char* name = "Information";
};

/// @brief Tag for square root covariance representation (S where P = S*S^T)
struct SquareRootTag {
    static constexpr const char* name = "SquareRoot";
};

// ============================================================================
// Representation Concept
// ============================================================================

/// @brief Concept for uncertainty representations
///
/// All uncertainty representations must provide:
/// - Type aliases: Scalar, MatrixType
/// - Basic operations: dim(), is_finite()
/// - Conversion: to_covariance() for unified interface
/// - Numerical stability: symmetrize(), regularize()
///
/// This allows different representations to be used interchangeably
/// through a common interface.
template <typename R>
concept UncertaintyRepresentation = requires(R r, const R cr) {
    // Type requirements
    typename R::Scalar;
    typename R::MatrixType;
    { R::Dim } -> std::convertible_to<int>;

    // Basic queries
    { cr.dim() } -> std::convertible_to<int>;
    { cr.is_finite() } -> std::convertible_to<bool>;

    // Unified interface: all representations can convert to covariance
    // This returns std::expected<SymmetricMatrix<...>, std::string>
    { cr.to_covariance() };

    // Numerical stability
    { r.symmetrize() } -> std::same_as<void>;
    { r.regularize() } -> std::same_as<void>;
};

/// @brief Concept for representations that support efficient predict step
///
/// Covariance: P' = F P F^T + Q (natural)
/// SquareRoot: S' via QR decomposition (efficient)
/// Information: requires inversion (expensive)
template <typename R>
concept PredictionFriendlyRepr =
    UncertaintyRepresentation<R> && requires(const R cr, typename R::MatrixType F) {
        // Should have an efficient predict method
        // Returns std::expected<R, std::string> or R directly
        { cr.predict(F, cr) };
    };

/// @brief Concept for representations that support efficient update step
///
/// Information: Y' = Y + H^T R^-1 H (natural)
/// Covariance: requires Kalman gain computation
/// SquareRoot: requires QR decomposition
template <typename R, int MeasDim>
concept UpdateFriendlyRepr = UncertaintyRepresentation<R>
                          && requires(
                                 const R cr, Eigen::Matrix<typename R::Scalar, MeasDim, R::Dim> H,
                                 Eigen::Matrix<typename R::Scalar, MeasDim, 1> z) {
                                 // Should have an efficient update method
                                 { cr.template update<MeasDim>(H, cr) };
                             };

// ============================================================================
// Representation Dispatch Helpers
// ============================================================================

/// @brief Map tag to concrete representation type
template <typename Tag, int N, typename Scalar>
struct ReprDispatch;

// Forward declarations (will be specialized in respective headers)
template <int N, typename Scalar>
class CovarianceRepr;

template <int N, typename Scalar>
class InformationRepr;

template <int N, typename Scalar>
class SquareRootRepr;

// Specializations
template <int N, typename Scalar>
struct ReprDispatch<CovarianceTag, N, Scalar> {
    using type = CovarianceRepr<N, Scalar>;
};

template <int N, typename Scalar>
struct ReprDispatch<InformationTag, N, Scalar> {
    using type = InformationRepr<N, Scalar>;
};

template <int N, typename Scalar>
struct ReprDispatch<SquareRootTag, N, Scalar> {
    using type = SquareRootRepr<N, Scalar>;
};

/// @brief Helper alias for dispatching tag to representation type
template <typename Tag, int N, typename Scalar = double>
using ReprType = typename ReprDispatch<Tag, N, Scalar>::type;

// ============================================================================
// Representation Traits
// ============================================================================

/// @brief Traits for compile-time information about representations
template <typename Tag>
struct ReprTraits {
    static constexpr bool prediction_friendly = false;
    static constexpr bool update_friendly     = false;
    static constexpr bool requires_conversion = false;
};

template <>
struct ReprTraits<CovarianceTag> {
    static constexpr bool prediction_friendly = true;  // P' = F P F^T + Q
    static constexpr bool update_friendly     = false; // Needs Kalman gain
    static constexpr bool requires_conversion = false;
};

template <>
struct ReprTraits<InformationTag> {
    static constexpr bool prediction_friendly = false; // Needs inversion
    static constexpr bool update_friendly     = true;  // Y' = Y + H^T R^-1 H
    static constexpr bool requires_conversion = false;
};

template <>
struct ReprTraits<SquareRootTag> {
    static constexpr bool prediction_friendly = true;  // QR-based
    static constexpr bool update_friendly     = true;  // Potter-Carlson
    static constexpr bool requires_conversion = false;
};

} // namespace uni_kf
