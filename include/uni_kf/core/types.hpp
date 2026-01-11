/// @file types.hpp
/// @brief Common type aliases and matrix utilities for uni_kf
///
/// Provides Eigen-based matrix types with consistent naming conventions
/// used throughout the uni_kf framework.

#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <expected>
#include <string>

namespace uni_kf {

// ============================================================================
// Matrix type aliases
// ============================================================================

/// @brief Fixed-size matrix type
template <int Rows, int Cols, typename Scalar = double>
using MatN = Eigen::Matrix<Scalar, Rows, Cols>;

/// @brief Square matrix type
template <int N, typename Scalar = double>
using SquareMatN = Eigen::Matrix<Scalar, N, N>;

/// @brief Dynamic matrix type
template <typename Scalar = double>
using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

// Common matrix sizes
using Mat2  = SquareMatN<2>;
using Mat3  = SquareMatN<3>;
using Mat4  = SquareMatN<4>;
using Mat6  = SquareMatN<6>;
using Mat9  = SquareMatN<9>;
using Mat12 = SquareMatN<12>;

using Mat2f  = SquareMatN<2, float>;
using Mat3f  = SquareMatN<3, float>;
using Mat4f  = SquareMatN<4, float>;
using Mat6f  = SquareMatN<6, float>;
using Mat9f  = SquareMatN<9, float>;
using Mat12f = SquareMatN<12, float>;

// ============================================================================
// Identity matrix helpers
// ============================================================================

/// @brief Compile-time identity matrix
template <int N, typename Scalar = double>
[[nodiscard]] constexpr auto eye() noexcept -> SquareMatN<N, Scalar> {
    return SquareMatN<N, Scalar>::Identity();
}

/// @brief Runtime-size identity matrix
template <typename Scalar = double>
[[nodiscard]] auto eye(int n) noexcept -> MatX<Scalar> {
    return MatX<Scalar>::Identity(n, n);
}

// ============================================================================
// Symmetric matrix operations
// ============================================================================

/// @brief Symmetrize a matrix: (A + A^T) / 2
template <typename Derived>
[[nodiscard]] auto symmetrize(const Eigen::MatrixBase<Derived>& A) noexcept -> Eigen::Matrix<
    typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> {
    return 0.5 * (A + A.transpose());
}

/// @brief Add small regularization to diagonal for numerical stability
template <typename Derived>
[[nodiscard]] auto
    regularize(const Eigen::MatrixBase<Derived>& A, typename Derived::Scalar eps = 1e-12) noexcept
    -> Eigen::Matrix<
        typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> {
    auto result = A.derived().eval();
    result.diagonal().array() += eps;
    return result;
}

/// @brief Check if matrix has any NaN or Inf values
template <typename Derived>
[[nodiscard]] bool is_finite(const Eigen::MatrixBase<Derived>& A) noexcept {
    return A.allFinite();
}

/// @brief Sanitize matrix: replace NaN/Inf with zeros
template <typename Derived>
[[nodiscard]] auto sanitize(const Eigen::MatrixBase<Derived>& A) noexcept -> Eigen::Matrix<
    typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> {
    if (A.allFinite()) {
        return A.derived();
    }
    auto result = A.derived().eval();
    for (int i = 0; i < result.rows(); ++i) {
        for (int j = 0; j < result.cols(); ++j) {
            if (!std::isfinite(result(i, j))) {
                result(i, j) = typename Derived::Scalar{0};
            }
        }
    }
    return result;
}

// ============================================================================
// Matrix decomposition utilities
// ============================================================================

/// @brief LDLT decomposition result
template <int N, typename Scalar = double>
struct LDLTResult {
    SquareMatN<N, Scalar> L;
    Eigen::Matrix<Scalar, N, 1> D;
    bool success;
};

/// @brief Compute LDLT decomposition of a symmetric positive-definite matrix
template <typename Derived>
[[nodiscard]] auto ldlt_decompose(const Eigen::MatrixBase<Derived>& A) noexcept
    -> std::expected<Eigen::LDLT<Derived>, std::string> {
    Eigen::LDLT<Derived> ldlt(A);
    if (ldlt.info() != Eigen::Success) {
        return std::unexpected(
            "LDLT decomposition failed: matrix may not be positive semi-definite");
    }
    return ldlt;
}

/// @brief Solve linear system A x = b using LDLT
template <typename MatDerived, typename VecDerived>
[[nodiscard]] auto ldlt_solve(
    const Eigen::MatrixBase<MatDerived>& A, const Eigen::MatrixBase<VecDerived>& b) noexcept
    -> std::expected<
        Eigen::Matrix<typename MatDerived::Scalar, MatDerived::ColsAtCompileTime, 1>, std::string> {
    Eigen::LDLT<MatDerived> ldlt(A);
    if (ldlt.info() != Eigen::Success) {
        return std::unexpected("LDLT solve failed: matrix may not be positive semi-definite");
    }
    return ldlt.solve(b).eval();
}

/// @brief Compute matrix inverse using LDLT
template <typename Derived>
[[nodiscard]] auto ldlt_inverse(const Eigen::MatrixBase<Derived>& A) noexcept -> std::expected<
    Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>,
    std::string> {
    using MatType = Eigen::Matrix<
        typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>;
    Eigen::LDLT<MatType> ldlt(A);
    if (ldlt.info() != Eigen::Success) {
        return std::unexpected("LDLT inverse failed: matrix may not be positive semi-definite");
    }
    return ldlt.solve(MatType::Identity(A.rows(), A.cols())).eval();
}

// ============================================================================
// Cholesky utilities (for square root form)
// ============================================================================

/// @brief Compute Cholesky decomposition: A = L L^T
template <typename Derived>
[[nodiscard]] auto cholesky(const Eigen::MatrixBase<Derived>& A) noexcept -> std::expected<
    Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>,
    std::string> {
    using MatType = Eigen::Matrix<
        typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>;
    Eigen::LLT<MatType> llt(A);
    if (llt.info() != Eigen::Success) {
        return std::unexpected("Cholesky decomposition failed: matrix is not positive definite");
    }
    return MatType(llt.matrixL());
}

// ============================================================================
// Skew-symmetric matrix (for SO(3) operations)
// ============================================================================

/// @brief Construct skew-symmetric matrix from 3-vector
/// skew([x, y, z]) = [0, -z, y; z, 0, -x; -y, x, 0]
template <typename Scalar>
[[nodiscard]] auto skew(const Eigen::Matrix<Scalar, 3, 1>& v) noexcept
    -> Eigen::Matrix<Scalar, 3, 3> {
    Eigen::Matrix<Scalar, 3, 3> S;
    S << Scalar{0}, -v(2), v(1), //
        v(2), Scalar{0}, -v(0),  //
        -v(1), v(0), Scalar{0};
    return S;
}

/// @brief Extract 3-vector from skew-symmetric matrix (inverse of skew)
template <typename Scalar>
[[nodiscard]] auto unskew(const Eigen::Matrix<Scalar, 3, 3>& S) noexcept
    -> Eigen::Matrix<Scalar, 3, 1> {
    return Eigen::Matrix<Scalar, 3, 1>{S(2, 1), S(0, 2), S(1, 0)};
}

// ============================================================================
// Block matrix operations
// ============================================================================

/// @brief Set a block of matrix to specified values
template <int BlockRows, int BlockCols, typename MatDerived, typename BlockDerived>
void set_block(
    Eigen::MatrixBase<MatDerived>& M, int startRow, int startCol,
    const Eigen::MatrixBase<BlockDerived>& block) noexcept {
    M.derived().template block<BlockRows, BlockCols>(startRow, startCol) = block;
}

/// @brief Get a block of matrix
template <int BlockRows, int BlockCols, typename Derived>
[[nodiscard]] auto
    get_block(const Eigen::MatrixBase<Derived>& M, int startRow, int startCol) noexcept
    -> Eigen::Matrix<typename Derived::Scalar, BlockRows, BlockCols> {
    return M.derived().template block<BlockRows, BlockCols>(startRow, startCol);
}

} // namespace uni_kf
