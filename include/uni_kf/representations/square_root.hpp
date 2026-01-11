/// @file square_root.hpp
/// @brief Square root covariance representation for uncertainty
///
/// Square root form S where P = S S^T. More numerically stable than
/// standard covariance, especially for ill-conditioned problems.

#pragma once

#include "../concepts/quadratic_form.hpp"

#include <Eigen/QR>

namespace uni_kf {

// ============================================================================
// SquareRootRepr: Square root covariance representation
// ============================================================================

/// @brief Square root covariance uncertainty representation
///
/// Stores uncertainty as lower triangular matrix S where P = S S^T.
/// Advantages:
/// - More numerically stable (positive definiteness preserved)
/// - Better for ill-conditioned problems
/// - Halves the dynamic range of matrix elements
/// Disadvantages:
/// - Update requires QR decomposition
/// - Slightly more complex operations
///
/// @tparam N State dimension
/// @tparam Scalar_ Floating point type
template <int N, typename Scalar_ = double>
class SquareRootRepr {
public:
    using Scalar     = Scalar_;
    using VectorType = Eigen::Matrix<Scalar, N, 1>;
    using MatrixType = Eigen::Matrix<Scalar, N, N>;

    static constexpr int Dim = N;

private:
    MatrixType S_; // Lower triangular: P = S S^T

public:
    // ------------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------------

    SquareRootRepr() noexcept
        : S_(MatrixType::Zero()) {}

    /// @brief From lower triangular square root matrix
    explicit SquareRootRepr(const MatrixType& S) noexcept
        : S_(S.template triangularView<Eigen::Lower>()) {}

    // ------------------------------------------------------------------------
    // Static factory methods
    // ------------------------------------------------------------------------

    [[nodiscard]] static SquareRootRepr Zero() noexcept {
        return SquareRootRepr(MatrixType::Zero());
    }

    [[nodiscard]] static SquareRootRepr Identity() noexcept {
        return SquareRootRepr(MatrixType::Identity());
    }

    [[nodiscard]] static SquareRootRepr Diagonal(const VectorType& diag) noexcept {
        MatrixType S = MatrixType::Zero();
        for (int i = 0; i < N; ++i) {
            S(i, i) = std::sqrt(std::abs(diag(i)));
        }
        return SquareRootRepr(S);
    }

    [[nodiscard]] static SquareRootRepr ScaledIdentity(Scalar s) noexcept {
        return SquareRootRepr(MatrixType::Identity() * std::sqrt(std::abs(s)));
    }

    /// @brief Create from covariance matrix (Cholesky decomposition)
    [[nodiscard]] static auto from_covariance(const SymmetricMatrix<N, Scalar>& P) noexcept
        -> std::expected<SquareRootRepr, std::string> {
        auto L_result = P.cholesky();
        if (!L_result) {
            return std::unexpected(L_result.error());
        }
        return SquareRootRepr(*L_result);
    }

    // ------------------------------------------------------------------------
    // Access
    // ------------------------------------------------------------------------

    /// @brief Square root matrix S (lower triangular, P = S S^T)
    [[nodiscard]] const MatrixType& sqrt_matrix() const noexcept { return S_; }

    [[nodiscard]] MatrixType& sqrt_matrix() noexcept { return S_; }

    [[nodiscard]] int dim() const noexcept { return N; }

    // ------------------------------------------------------------------------
    // Covariance reconstruction
    // ------------------------------------------------------------------------

    /// @brief Reconstruct covariance: P = S S^T
    [[nodiscard]] SymmetricMatrix<N, Scalar> covariance() const noexcept {
        return SymmetricMatrix<N, Scalar>((S_ * S_.transpose()).eval());
    }

    /// @brief Reconstruct covariance matrix (raw)
    [[nodiscard]] MatrixType covariance_matrix() const noexcept {
        return (S_ * S_.transpose()).eval();
    }

    // ------------------------------------------------------------------------
    // Kalman filter operations (Square Root form)
    // ------------------------------------------------------------------------

    /// @brief Predict step using square root form
    ///
    /// Given: S where P = S S^T
    /// Compute: S' where P' = F P F^T + Q = F S S^T F^T + S_Q S_Q^T
    ///
    /// Method: QR decomposition of [F S, S_Q]^T
    template <typename F_t>
    [[nodiscard]] auto predict(const F_t& F, const SquareRootRepr& sqrt_Q) const noexcept
        -> SquareRootRepr {
        // Build augmented matrix: [F*S; sqrt_Q]^T = [S^T*F^T, sqrt_Q^T]
        // Then QR decompose to get new S
        Eigen::Matrix<Scalar, 2 * N, N> augmented;
        augmented.template block<N, N>(0, 0) = (F * S_).transpose();
        augmented.template block<N, N>(N, 0) = sqrt_Q.S_.transpose();

        // QR decomposition
        Eigen::HouseholderQR<Eigen::Matrix<Scalar, 2 * N, N>> qr(augmented);

        // R^T is the new square root (lower triangular)
        MatrixType S_new = qr.matrixQR()
                               .template block<N, N>(0, 0)
                               .template triangularView<Eigen::Upper>()
                               .transpose();

        return SquareRootRepr(S_new);
    }

    /// @brief Update step using square root form
    ///
    /// Uses the Potter/Carlson square root update algorithm.
    ///
    /// @return tuple of (updated_sqrt, kalman_gain)
    template <int M, typename H_t>
    [[nodiscard]] auto update(const H_t& H, const SquareRootRepr<M, Scalar>& sqrt_R) const noexcept
        -> std::tuple<SquareRootRepr, Eigen::Matrix<Scalar, N, M>> {
        // Build augmented matrix for QR decomposition:
        // [sqrt_R,    H*S   ]
        // [  0,       S     ]
        // After QR, we get the updated square root and can extract Kalman gain

        Eigen::Matrix<Scalar, N + M, N + M> augmented;
        augmented.setZero();
        augmented.template block<M, M>(0, 0) = sqrt_R.sqrt_matrix();
        augmented.template block<M, N>(0, M) = H * S_;
        augmented.template block<N, N>(M, M) = S_;

        // QR decomposition
        Eigen::HouseholderQR<Eigen::Matrix<Scalar, N + M, N + M>> qr(augmented.transpose());
        auto R = qr.matrixQR().template triangularView<Eigen::Upper>();

        // Extract updated sqrt covariance
        MatrixType S_new = R.template block<N, N>(M, M).transpose();

        // Extract Kalman gain from the decomposition
        // K = P H^T (H P H^T + R)^{-1}
        // This requires additional computation, simplified here
        Eigen::Matrix<Scalar, N, M> K;
        auto Pxy   = S_ * (H * S_).transpose();
        auto S_inn = R.template block<M, M>(0, 0).transpose();
        // Solve: K * S_inn * S_inn^T = Pxy
        // K = Pxy * (S_inn^T)^{-1} * S_inn^{-1}
        K = Pxy
          * S_inn.template triangularView<Eigen::Lower>().solve(
              S_inn.transpose().template triangularView<Eigen::Upper>().solve(
                  Eigen::Matrix<Scalar, M, M>::Identity()));

        return {SquareRootRepr(S_new), K};
    }

    // ------------------------------------------------------------------------
    // Conversion to other representations
    // ------------------------------------------------------------------------

    /// @brief Convert to standard covariance representation
    [[nodiscard]] SymmetricMatrix<N, Scalar> to_covariance() const noexcept { return covariance(); }

    // ------------------------------------------------------------------------
    // Numerical stability
    // ------------------------------------------------------------------------

    /// @brief Re-triangularize using Cholesky
    void retriangularize() noexcept {
        Eigen::LLT<MatrixType> llt(S_ * S_.transpose());
        if (llt.info() == Eigen::Success) {
            S_ = llt.matrixL();
        }
    }

    [[nodiscard]] bool is_finite() const noexcept { return S_.allFinite(); }

    /// @brief Regularize by adding small value to diagonal
    void regularize(Scalar eps = Scalar{1e-12}) noexcept { S_.diagonal().array() += eps; }
};

// ============================================================================
// Type aliases
// ============================================================================

template <int N>
using SqrtRepr = SquareRootRepr<N, double>;

template <int N>
using SqrtReprf = SquareRootRepr<N, float>;

} // namespace uni_kf
