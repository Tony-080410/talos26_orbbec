/// @file covariance.hpp
/// @brief Covariance representation for uncertainty
///
/// Standard covariance matrix P representation. This is the most common
/// uncertainty representation, efficient for prediction steps.

#pragma once

#include "../concepts/quadratic_form.hpp"

namespace uni_kf {

// ============================================================================
// CovarianceRepr: Standard covariance matrix representation
// ============================================================================

/// @brief Covariance matrix uncertainty representation
///
/// Stores uncertainty as a covariance matrix P.
/// Advantages:
/// - Prediction is efficient: P' = F P F^T + Q
/// - Natural interpretation
/// Disadvantages:
/// - Update requires matrix inversion
/// - Less numerically stable than square root form
///
/// @tparam N State dimension
/// @tparam Scalar_ Floating point type
template <int N, typename Scalar_ = double>
class CovarianceRepr {
public:
    using Scalar     = Scalar_;
    using Covariance = SymmetricMatrix<N, Scalar>;
    using VectorType = typename Covariance::VectorType;
    using MatrixType = typename Covariance::MatrixType;

    static constexpr int Dim = N;

private:
    Covariance P_;

public:
    // ------------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------------

    CovarianceRepr() noexcept = default;

    explicit CovarianceRepr(const Covariance& P) noexcept
        : P_(P) {}

    explicit CovarianceRepr(Covariance&& P) noexcept
        : P_(std::move(P)) {}

    explicit CovarianceRepr(const MatrixType& P) noexcept
        : P_(P) {}

    // ------------------------------------------------------------------------
    // Static factory methods
    // ------------------------------------------------------------------------

    [[nodiscard]] static CovarianceRepr Zero() noexcept {
        return CovarianceRepr(Covariance::Zero());
    }

    [[nodiscard]] static CovarianceRepr Identity() noexcept {
        return CovarianceRepr(Covariance::Identity());
    }

    [[nodiscard]] static CovarianceRepr Diagonal(const VectorType& diag) noexcept {
        return CovarianceRepr(Covariance::Diagonal(diag));
    }

    [[nodiscard]] static CovarianceRepr ScaledIdentity(Scalar s) noexcept {
        return CovarianceRepr(Covariance::ScaledIdentity(s));
    }

    // ------------------------------------------------------------------------
    // Access
    // ------------------------------------------------------------------------

    [[nodiscard]] const Covariance& covariance() const noexcept { return P_; }

    [[nodiscard]] Covariance& covariance() noexcept { return P_; }

    [[nodiscard]] const MatrixType& matrix() const noexcept { return P_.matrix(); }

    [[nodiscard]] int dim() const noexcept { return P_.dim(); }

    // ------------------------------------------------------------------------
    // Kalman filter operations
    // ------------------------------------------------------------------------

    /// @brief Predict step: P' = F P F^T + Q
    template <typename F_t>
    [[nodiscard]] CovarianceRepr predict(const F_t& F, const Covariance& Q) const noexcept {
        return CovarianceRepr(Covariance((F * P_.matrix() * F.transpose() + Q.matrix()).eval()));
    }

    /// @brief Update step with Kalman gain
    ///
    /// Computes:
    /// - Innovation covariance: S = H P H^T + R
    /// - Kalman gain: K = P H^T S^{-1}
    /// - Updated covariance: P' = (I - KH) P (I - KH)^T + K R K^T (Joseph form)
    ///
    /// @return tuple of (updated_cov, kalman_gain, innovation_cov)
    template <int M, typename H_t>
    [[nodiscard]] auto update(const H_t& H, const SymmetricMatrix<M, Scalar>& R) const noexcept
        -> std::expected<
            std::tuple<CovarianceRepr, Eigen::Matrix<Scalar, N, M>, SymmetricMatrix<M, Scalar>>,
            std::string> {
        // Innovation covariance: S = H P H^T + R
        MatrixType P                      = P_.matrix();
        Eigen::Matrix<Scalar, M, M> S_mat = H * P * H.transpose() + R.matrix();
        SymmetricMatrix<M, Scalar> S(S_mat);

        // Solve for Kalman gain: K = P H^T S^{-1}
        auto S_inv_result = S.inverse();
        if (!S_inv_result) {
            return std::unexpected(S_inv_result.error());
        }

        Eigen::Matrix<Scalar, N, M> K = P * H.transpose() * S_inv_result->matrix();

        // Joseph form update: P' = (I - KH) P (I - KH)^T + K R K^T
        MatrixType I_KH  = MatrixType::Identity() - K * H;
        MatrixType P_new = I_KH * P * I_KH.transpose() + K * R.matrix() * K.transpose();

        CovarianceRepr updated{Covariance{P_new}};
        updated.P_.symmetrize_in_place();
        updated.P_.regularize_in_place();

        return std::make_tuple(updated, K, S);
    }

    /// @brief Simple update (not Joseph form, faster but less stable)
    template <int M, typename H_t, typename K_t>
    [[nodiscard]] CovarianceRepr update_simple(const H_t& H, const K_t& K) const noexcept {
        MatrixType I_KH = MatrixType::Identity() - K * H;
        return CovarianceRepr(Covariance((I_KH * P_.matrix()).eval()));
    }

    // ------------------------------------------------------------------------
    // Conversion to other representations
    // ------------------------------------------------------------------------

    /// @brief Convert to covariance (identity operation for CovarianceRepr)
    [[nodiscard]] Covariance to_covariance() const noexcept { return P_; }

    /// @brief Convert to information matrix (inverse)
    [[nodiscard]] auto to_information() const noexcept -> std::expected<Covariance, std::string> {
        return P_.inverse();
    }

    /// @brief Convert to square root (Cholesky)
    [[nodiscard]] auto to_sqrt() const noexcept -> std::expected<MatrixType, std::string> {
        return P_.cholesky();
    }

    // ------------------------------------------------------------------------
    // Numerical stability
    // ------------------------------------------------------------------------

    void symmetrize() noexcept { P_.symmetrize_in_place(); }

    void regularize(Scalar eps = Scalar{1e-12}) noexcept { P_.regularize_in_place(eps); }

    void sanitize() noexcept { P_.sanitize_in_place(); }

    [[nodiscard]] bool is_finite() const noexcept { return P_.is_finite(); }

    // ------------------------------------------------------------------------
    // Arithmetic
    // ------------------------------------------------------------------------

    [[nodiscard]] CovarianceRepr operator+(const CovarianceRepr& other) const noexcept {
        return CovarianceRepr(P_ + other.P_);
    }

    CovarianceRepr& operator+=(const CovarianceRepr& other) noexcept {
        P_ += other.P_;
        return *this;
    }
};

// ============================================================================
// Type aliases
// ============================================================================

template <int N>
using CovRepr = CovarianceRepr<N, double>;

template <int N>
using CovReprf = CovarianceRepr<N, float>;

} // namespace uni_kf
