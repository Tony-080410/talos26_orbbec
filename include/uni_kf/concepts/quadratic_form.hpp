/// @file quadratic_form.hpp
/// @brief Symmetric positive-definite matrix abstractions for uncertainty
///
/// Defines the QuadraticForm concept and SymmetricMatrix implementation.
/// These represent covariance/information matrices and provide operations
/// needed for Kalman filtering (similarity transform, solve, Cholesky, etc.)

#pragma once

#include "../core/types.hpp"
#include "linear.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <concepts>
#include <expected>
#include <string>

namespace uni_kf {

// ============================================================================
// QuadraticForm Concept
// ============================================================================

/// @brief Concept for symmetric positive-definite quadratic forms
///
/// A QuadraticForm represents a symmetric positive-definite matrix Q.
/// It supports operations needed for Kalman filtering:
/// - Quadratic evaluation: v^T Q v
/// - Linear solve: Q⁻¹ v
/// - Similarity transform: F Q F^T
template <typename Q>
concept QuadraticForm = requires {
    typename Q::Scalar;
    typename Q::VectorType;
    typename Q::MatrixType;

    { Q::Dim } -> std::convertible_to<int>;
} && requires(Q q, typename Q::VectorType v, typename Q::VectorType w, typename Q::MatrixType M) {
    /// Quadratic evaluation: v^T Q v
    { q.eval(v) } -> std::convertible_to<typename Q::Scalar>;

    /// Bilinear form: v^T Q w
    { q.bilinear(v, w) } -> std::convertible_to<typename Q::Scalar>;

    /// Matrix-vector multiply: Q v
    { q.multiply(v) } -> std::same_as<typename Q::VectorType>;

    /// Access underlying matrix
    { q.matrix() } -> std::same_as<const typename Q::MatrixType&>;

    /// Dimension
    { q.dim() } -> std::convertible_to<int>;
};

// ============================================================================
// SymmetricMatrix: Fixed-size symmetric positive-definite matrix
// ============================================================================

/// @brief Symmetric positive-definite matrix
///
/// Wrapper around Eigen matrix that enforces symmetry and provides
/// operations for Kalman filtering. All mutating operations maintain
/// symmetry (via symmetrize).
///
/// @tparam N Matrix dimension
/// @tparam Scalar_ Floating point type
template <int N, typename Scalar_ = double>
class SymmetricMatrix {
public:
    static_assert(N > 0 || N == Eigen::Dynamic, "Dimension must be positive or Dynamic");

    using Scalar     = Scalar_;
    using VectorType = Eigen::Matrix<Scalar, N, 1>;
    using MatrixType = Eigen::Matrix<Scalar, N, N>;

    static constexpr int Dim        = N;
    static constexpr bool IsDynamic = (N == Eigen::Dynamic);

private:
    MatrixType M_;

public:
    // ------------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------------

    /// @brief Default: zero matrix
    SymmetricMatrix() noexcept
        : M_(MatrixType::Zero()) {}

    /// @brief From Eigen matrix (symmetrizes)
    explicit SymmetricMatrix(const MatrixType& M) noexcept
        : M_(symmetrize(M)) {}

    /// @brief From Eigen matrix (move, symmetrizes)
    explicit SymmetricMatrix(MatrixType&& M) noexcept
        : M_(symmetrize(std::move(M))) {}

    // ------------------------------------------------------------------------
    // Static factory methods
    // ------------------------------------------------------------------------

    /// @brief Zero matrix
    [[nodiscard]] static SymmetricMatrix Zero() noexcept {
        return SymmetricMatrix(MatrixType::Zero());
    }

    /// @brief Identity matrix
    [[nodiscard]] static SymmetricMatrix Identity() noexcept {
        return SymmetricMatrix(MatrixType::Identity());
    }

    /// @brief Diagonal matrix from vector
    [[nodiscard]] static SymmetricMatrix Diagonal(const VectorType& diag) noexcept {
        return SymmetricMatrix(diag.asDiagonal());
    }

    /// @brief Scalar times identity
    [[nodiscard]] static SymmetricMatrix ScaledIdentity(Scalar s) noexcept {
        return SymmetricMatrix(MatrixType::Identity() * s);
    }

    // For dynamic size
    [[nodiscard]] static SymmetricMatrix Zero(int size) noexcept requires IsDynamic {
        MatrixType M = MatrixType::Zero(size, size);
        return SymmetricMatrix(M);
    }

    [[nodiscard]] static SymmetricMatrix Identity(int size) noexcept requires IsDynamic {
        MatrixType M = MatrixType::Identity(size, size);
        return SymmetricMatrix(M);
    }

    // ------------------------------------------------------------------------
    // QuadraticForm interface
    // ------------------------------------------------------------------------

    /// @brief Quadratic evaluation: v^T M v
    [[nodiscard]] Scalar eval(const VectorType& v) const noexcept { return v.dot(M_ * v); }

    /// @brief Bilinear form: v^T M w
    [[nodiscard]] Scalar bilinear(const VectorType& v, const VectorType& w) const noexcept {
        return v.dot(M_ * w);
    }

    /// @brief Matrix-vector multiply: M v
    [[nodiscard]] VectorType multiply(const VectorType& v) const noexcept {
        return (M_ * v).eval();
    }

    /// @brief Access underlying matrix
    [[nodiscard]] const MatrixType& matrix() const noexcept { return M_; }

    /// @brief Mutable access (use with care, call symmetrize after)
    [[nodiscard]] MatrixType& matrix_mut() noexcept { return M_; }

    /// @brief Dimension
    [[nodiscard]] int dim() const noexcept {
        if constexpr (IsDynamic) {
            return static_cast<int>(M_.rows());
        } else {
            return N;
        }
    }

    // ------------------------------------------------------------------------
    // Linear algebra operations
    // ------------------------------------------------------------------------

    /// @brief Solve M x = b using LDLT
    [[nodiscard]] auto solve(const VectorType& b) const noexcept
        -> std::expected<VectorType, std::string> {
        Eigen::LDLT<MatrixType> ldlt(M_);
        if (ldlt.info() != Eigen::Success) {
            return std::unexpected("LDLT solve failed: matrix may not be positive semi-definite");
        }
        return ldlt.solve(b).eval();
    }

    /// @brief Solve M X = B (matrix right-hand side)
    [[nodiscard]] auto solve_matrix(const MatrixType& B) const noexcept
        -> std::expected<MatrixType, std::string> {
        Eigen::LDLT<MatrixType> ldlt(M_);
        if (ldlt.info() != Eigen::Success) {
            return std::unexpected("LDLT solve failed");
        }
        return ldlt.solve(B).eval();
    }

    /// @brief Compute inverse using LDLT
    [[nodiscard]] auto inverse() const noexcept -> std::expected<SymmetricMatrix, std::string> {
        Eigen::LDLT<MatrixType> ldlt(M_);
        if (ldlt.info() != Eigen::Success) {
            return std::unexpected("LDLT inverse failed");
        }
        return SymmetricMatrix(ldlt.solve(MatrixType::Identity(M_.rows(), M_.cols())));
    }

    /// @brief Cholesky decomposition: M = L L^T
    [[nodiscard]] auto cholesky() const noexcept -> std::expected<MatrixType, std::string> {
        Eigen::LLT<MatrixType> llt(M_);
        if (llt.info() != Eigen::Success) {
            return std::unexpected("Cholesky failed: matrix is not positive definite");
        }
        return MatrixType(llt.matrixL());
    }

    // ------------------------------------------------------------------------
    // Similarity transforms (covariance propagation)
    // ------------------------------------------------------------------------

    /// @brief Similarity transform: F M F^T
    template <int Rows>
    [[nodiscard]] auto transform(const Eigen::Matrix<Scalar, Rows, N>& F) const noexcept
        -> SymmetricMatrix<Rows, Scalar> {
        return SymmetricMatrix<Rows, Scalar>((F * M_ * F.transpose()).eval());
    }

    /// @brief Similarity transform with additive noise: F M F^T + Q
    template <int Rows>
    [[nodiscard]] auto transform_and_add(
        const Eigen::Matrix<Scalar, Rows, N>& F,
        const SymmetricMatrix<Rows, Scalar>& Q) const noexcept -> SymmetricMatrix<Rows, Scalar> {
        return SymmetricMatrix<Rows, Scalar>((F * M_ * F.transpose() + Q.M_).eval());
    }

    /// @brief In-place similarity transform: M ← F M F^T + Q
    void transform_in_place(const MatrixType& F, const SymmetricMatrix& Q) noexcept {
        M_ = symmetrize(F * M_ * F.transpose() + Q.M_);
    }

    // ------------------------------------------------------------------------
    // Arithmetic
    // ------------------------------------------------------------------------

    [[nodiscard]] SymmetricMatrix operator+(const SymmetricMatrix& other) const noexcept {
        return SymmetricMatrix(M_ + other.M_);
    }

    [[nodiscard]] SymmetricMatrix operator-(const SymmetricMatrix& other) const noexcept {
        return SymmetricMatrix(M_ - other.M_);
    }

    [[nodiscard]] SymmetricMatrix operator*(Scalar s) const noexcept {
        return SymmetricMatrix(M_ * s);
    }

    [[nodiscard]] friend SymmetricMatrix operator*(Scalar s, const SymmetricMatrix& M) noexcept {
        return SymmetricMatrix(s * M.M_);
    }

    SymmetricMatrix& operator+=(const SymmetricMatrix& other) noexcept {
        M_ += other.M_;
        return *this;
    }

    SymmetricMatrix& operator-=(const SymmetricMatrix& other) noexcept {
        M_ -= other.M_;
        return *this;
    }

    SymmetricMatrix& operator*=(Scalar s) noexcept {
        M_ *= s;
        return *this;
    }

    // ------------------------------------------------------------------------
    // Numerical stability
    // ------------------------------------------------------------------------

    /// @brief Force symmetry: M ← (M + M^T) / 2
    void symmetrize_in_place() noexcept { M_ = symmetrize(M_); }

    /// @brief Add regularization to diagonal
    void regularize_in_place(Scalar eps = Scalar{1e-12}) noexcept { M_.diagonal().array() += eps; }

    /// @brief Return symmetrized copy
    [[nodiscard]] SymmetricMatrix symmetrized() const noexcept {
        return SymmetricMatrix(symmetrize(M_));
    }

    /// @brief Return regularized copy
    [[nodiscard]] SymmetricMatrix regularized(Scalar eps = Scalar{1e-12}) const noexcept {
        MatrixType M = M_;
        M.diagonal().array() += eps;
        return SymmetricMatrix(M);
    }

    /// @brief Check if finite (no NaN/Inf)
    [[nodiscard]] bool is_finite() const noexcept { return M_.allFinite(); }

    /// @brief Replace NaN/Inf with zeros
    void sanitize_in_place() noexcept {
        if (!M_.allFinite()) {
            for (int i = 0; i < M_.rows(); ++i) {
                for (int j = 0; j < M_.cols(); ++j) {
                    if (!std::isfinite(M_(i, j))) {
                        M_(i, j) = Scalar{0};
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------------
    // Block operations
    // ------------------------------------------------------------------------

    /// @brief Get diagonal block
    template <int BlockSize>
    [[nodiscard]] auto block(int start) const noexcept -> SymmetricMatrix<BlockSize, Scalar> {
        return SymmetricMatrix<BlockSize, Scalar>(
            M_.template block<BlockSize, BlockSize>(start, start));
    }

    /// @brief Set diagonal block
    template <int BlockSize>
    void set_block(int start, const SymmetricMatrix<BlockSize, Scalar>& block) noexcept {
        M_.template block<BlockSize, BlockSize>(start, start) = block.M_;
    }

    /// @brief Element access
    [[nodiscard]] Scalar operator()(int i, int j) const noexcept { return M_(i, j); }

    [[nodiscard]] Scalar& operator()(int i, int j) noexcept { return M_(i, j); }

    // ------------------------------------------------------------------------
    // Utility
    // ------------------------------------------------------------------------

    /// @brief Trace
    [[nodiscard]] Scalar trace() const noexcept { return M_.trace(); }

    /// @brief Determinant (use log-det for numerical stability with large matrices)
    [[nodiscard]] Scalar determinant() const noexcept { return M_.determinant(); }

    /// @brief Log determinant (more numerically stable)
    [[nodiscard]] auto log_determinant() const noexcept -> std::expected<Scalar, std::string> {
        Eigen::LDLT<MatrixType> ldlt(M_);
        if (ldlt.info() != Eigen::Success) {
            return std::unexpected("Log determinant failed: matrix not positive semi-definite");
        }
        return ldlt.vectorD().array().log().sum();
    }
};

// ============================================================================
// Type aliases
// ============================================================================

using Cov2  = SymmetricMatrix<2>;
using Cov3  = SymmetricMatrix<3>;
using Cov4  = SymmetricMatrix<4>;
using Cov6  = SymmetricMatrix<6>;
using Cov9  = SymmetricMatrix<9>;
using Cov12 = SymmetricMatrix<12>;
using Cov13 = SymmetricMatrix<13>; // SE₂(3) × R⁴
using CovX  = SymmetricMatrix<Eigen::Dynamic>;

// ============================================================================
// Concept verification
// ============================================================================

static_assert(QuadraticForm<Cov3>, "Cov3 must satisfy QuadraticForm");
static_assert(QuadraticForm<Cov6>, "Cov6 must satisfy QuadraticForm");

} // namespace uni_kf
