/// @file linear.hpp
/// @brief Vector space concept for uni_kf framework
///
/// Defines the Linear concept representing finite-dimensional real vector spaces.
/// This is the algebraic foundation for the fiber (tangent/error) spaces in
/// DifferentialSpace.
///
/// Design: Uses Eigen::Matrix directly as the canonical representation,
/// avoiding wrapper overhead while maintaining type safety through concepts.

#pragma once

#include <Eigen/Core>
#include <concepts>

namespace uni_kf {

// ============================================================================
// Forward declarations
// ============================================================================

template <int N, typename Scalar = double>
struct VecN;

// ============================================================================
// Linear Concept
// ============================================================================

/// @brief Concept for finite-dimensional real vector spaces
///
/// A type V satisfies Linear if it supports:
/// - Static dimension query
/// - Zero element construction
/// - Vector addition, subtraction, scalar multiplication
/// - Inner product and norm
/// - Conversion to/from Eigen vectors
///
/// @note Eigen::Matrix<Scalar, N, 1> naturally satisfies this concept
/// through the VecN wrapper which provides the required static methods.
template <typename V>
concept Linear = requires {
    /// Compile-time dimension (returns -1 for dynamic)
    { V::Dim } -> std::convertible_to<int>;
} && requires(V v, V w, typename V::Scalar s) {
    /// Scalar type
    typename V::Scalar;

    /// Zero vector
    { V::Zero() } -> std::same_as<V>;

    /// Vector arithmetic
    { v + w } -> std::same_as<V>;
    { v - w } -> std::same_as<V>;
    { v * s } -> std::same_as<V>;
    { s * v } -> std::same_as<V>;
    { -v } -> std::same_as<V>;

    /// Inner product and norm
    { v.dot(w) } -> std::convertible_to<typename V::Scalar>;
    { v.norm() } -> std::convertible_to<typename V::Scalar>;
    { v.squaredNorm() } -> std::convertible_to<typename V::Scalar>;

    /// Element access
    { v(0) } -> std::convertible_to<typename V::Scalar>;
    { v.size() } -> std::convertible_to<int>;
};

// ============================================================================
// VecN: Eigen-based vector space implementation
// ============================================================================

/// @brief Fixed-size vector space based on Eigen
///
/// This is a thin wrapper around Eigen::Matrix that explicitly provides
/// the interface required by the Linear concept. The wrapper is necessary
/// because Eigen::Matrix doesn't have a static `Dim` member.
///
/// @tparam N Dimension (must be positive, or Eigen::Dynamic for runtime size)
/// @tparam Scalar Floating point type (default: double)
template <int N, typename Scalar_>
struct VecN {
    static_assert(N > 0 || N == Eigen::Dynamic, "Dimension must be positive or Dynamic");

    using Scalar                    = Scalar_;
    using EigenType                 = Eigen::Matrix<Scalar, N, 1>;
    static constexpr int Dim        = N;
    static constexpr bool IsDynamic = (N == Eigen::Dynamic);

    EigenType data;

    // ------------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------------

    VecN() noexcept
        : data(EigenType::Zero()) {}

    explicit VecN(const EigenType& v) noexcept
        : data(v) {}

    explicit VecN(EigenType&& v) noexcept
        : data(std::move(v)) {}

    /// Construct Vec2 from two scalars
    explicit VecN(Scalar x, Scalar y) noexcept requires(N == 2)
        : data((EigenType() << x, y).finished()) {}

    /// Construct Vec3 from three scalars
    explicit VecN(Scalar x, Scalar y, Scalar z) noexcept requires(N == 3)
        : data((EigenType() << x, y, z).finished()) {}

    /// Construct Vec4 from four scalars
    explicit VecN(Scalar x, Scalar y, Scalar z, Scalar w) noexcept requires(N == 4)
        : data((EigenType() << x, y, z, w).finished()) {}

    // ------------------------------------------------------------------------
    // Static factory methods
    // ------------------------------------------------------------------------

    [[nodiscard]] static VecN Zero() noexcept { return VecN(EigenType::Zero()); }

    [[nodiscard]] static VecN Ones() noexcept { return VecN(EigenType::Ones()); }

    [[nodiscard]] static VecN Unit(int i) noexcept { return VecN(EigenType::Unit(i)); }

    [[nodiscard]] static VecN Random() noexcept { return VecN(EigenType::Random()); }

    // For dynamic size
    [[nodiscard]] static VecN Zero(int size) noexcept requires IsDynamic {
        return VecN(EigenType::Zero(size));
    }

    // ------------------------------------------------------------------------
    // Arithmetic operations
    // ------------------------------------------------------------------------

    [[nodiscard]] VecN operator+(const VecN& other) const noexcept {
        return VecN(data + other.data);
    }

    [[nodiscard]] VecN operator-(const VecN& other) const noexcept {
        return VecN(data - other.data);
    }

    [[nodiscard]] VecN operator-() const noexcept { return VecN(-data); }

    [[nodiscard]] VecN operator*(Scalar s) const noexcept { return VecN(data * s); }

    [[nodiscard]] friend VecN operator*(Scalar s, const VecN& v) noexcept {
        return VecN(s * v.data);
    }

    VecN& operator+=(const VecN& other) noexcept {
        data += other.data;
        return *this;
    }

    VecN& operator-=(const VecN& other) noexcept {
        data -= other.data;
        return *this;
    }

    VecN& operator*=(Scalar s) noexcept {
        data *= s;
        return *this;
    }

    // ------------------------------------------------------------------------
    // Inner product and norm
    // ------------------------------------------------------------------------

    [[nodiscard]] Scalar dot(const VecN& other) const noexcept { return data.dot(other.data); }

    [[nodiscard]] Scalar norm() const noexcept { return data.norm(); }

    [[nodiscard]] Scalar squaredNorm() const noexcept { return data.squaredNorm(); }

    [[nodiscard]] VecN normalized() const noexcept { return VecN(data.normalized()); }

    // ------------------------------------------------------------------------
    // Element access
    // ------------------------------------------------------------------------

    [[nodiscard]] Scalar operator()(int i) const noexcept { return data(i); }

    [[nodiscard]] Scalar& operator()(int i) noexcept { return data(i); }

    [[nodiscard]] Scalar operator[](int i) const noexcept { return data(i); }

    [[nodiscard]] Scalar& operator[](int i) noexcept { return data(i); }

    [[nodiscard]] int size() const noexcept {
        if constexpr (IsDynamic) {
            return static_cast<int>(data.size());
        } else {
            return N;
        }
    }

    // ------------------------------------------------------------------------
    // Segment operations (for splitting/joining)
    // ------------------------------------------------------------------------

    template <int Size>
    [[nodiscard]] auto head() const noexcept {
        return VecN<Size, Scalar>(data.template head<Size>());
    }

    template <int Size>
    [[nodiscard]] auto tail() const noexcept {
        return VecN<Size, Scalar>(data.template tail<Size>());
    }

    template <int Size>
    [[nodiscard]] auto segment(int start) const noexcept {
        return VecN<Size, Scalar>(data.template segment<Size>(start));
    }

    // ------------------------------------------------------------------------
    // Eigen interop
    // ------------------------------------------------------------------------

    [[nodiscard]] const EigenType& eigen() const noexcept { return data; }

    [[nodiscard]] EigenType& eigen() noexcept { return data; }

    [[nodiscard]] static VecN from_eigen(const EigenType& v) noexcept { return VecN(v); }

    // ------------------------------------------------------------------------
    // Utility
    // ------------------------------------------------------------------------

    [[nodiscard]] bool
        isZero(Scalar eps = Eigen::NumTraits<Scalar>::dummy_precision()) const noexcept {
        return data.isZero(eps);
    }

    [[nodiscard]] bool isFinite() const noexcept { return data.allFinite(); }

    void setZero() noexcept { data.setZero(); }
};

// ============================================================================
// Type aliases for common dimensions
// ============================================================================

using Vec1  = VecN<1, double>;
using Vec2  = VecN<2, double>;
using Vec3  = VecN<3, double>;
using Vec4  = VecN<4, double>;
using Vec6  = VecN<6, double>;
using Vec9  = VecN<9, double>;
using Vec12 = VecN<12, double>;
using VecX  = VecN<Eigen::Dynamic, double>;

using Vec1f  = VecN<1, float>;
using Vec2f  = VecN<2, float>;
using Vec3f  = VecN<3, float>;
using Vec4f  = VecN<4, float>;
using Vec6f  = VecN<6, float>;
using Vec9f  = VecN<9, float>;
using Vec12f = VecN<12, float>;
using VecXf  = VecN<Eigen::Dynamic, float>;

// ============================================================================
// Concept verification (compile-time check)
// ============================================================================

static_assert(Linear<Vec3>, "Vec3 must satisfy Linear concept");
static_assert(Linear<Vec6>, "Vec6 must satisfy Linear concept");
static_assert(Linear<VecX>, "VecX must satisfy Linear concept");

// ============================================================================
// Utility: Concatenate two vectors
// ============================================================================

/// @brief Concatenate two vectors into a larger vector
template <int N1, int N2, typename Scalar>
[[nodiscard]] auto concat(const VecN<N1, Scalar>& v1, const VecN<N2, Scalar>& v2) noexcept
    -> VecN<(N1 == Eigen::Dynamic || N2 == Eigen::Dynamic) ? Eigen::Dynamic : N1 + N2, Scalar> {
    constexpr int ResultDim =
        (N1 == Eigen::Dynamic || N2 == Eigen::Dynamic) ? Eigen::Dynamic : N1 + N2;
    using ResultType = VecN<ResultDim, Scalar>;
    typename ResultType::EigenType result(v1.size() + v2.size());
    result << v1.data, v2.data;
    return ResultType(result);
}

/// @brief Split a vector into two parts at a given index
template <int N1, int N, typename Scalar>
[[nodiscard]] auto split(const VecN<N, Scalar>& v) noexcept
    -> std::pair<VecN<N1, Scalar>, VecN<N - N1, Scalar>>
    requires(N != Eigen::Dynamic && N1 > 0 && N1 < N) {
    return {v.template head<N1>(), v.template tail<N - N1>()};
}

} // namespace uni_kf
