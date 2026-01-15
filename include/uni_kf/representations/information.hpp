/// @file information.hpp
/// @brief Information matrix representation for uncertainty
///
/// Information matrix (precision matrix) Y = P^{-1} representation.
/// This is the dual of covariance, efficient for sensor fusion/update steps.

#pragma once

#include "../concepts/quadratic_form.hpp"

namespace uni_kf {

// ============================================================================
// InformationRepr: Information matrix representation
// ============================================================================

/// @brief Information matrix uncertainty representation
///
/// Stores uncertainty as information matrix Y = P^{-1} and information vector η = Y μ.
/// Advantages:
/// - Update/fusion is efficient: Y' = Y + H^T R^{-1} H
/// - Natural for sensor fusion
/// Disadvantages:
/// - Prediction requires matrix inversion
///
/// @tparam N State dimension
/// @tparam Scalar_ Floating point type
template <int N, typename Scalar_ = double>
class InformationRepr {
public:
    using Scalar     = Scalar_;
    using Matrix     = SymmetricMatrix<N, Scalar>;
    using VectorType = typename Matrix::VectorType;
    using MatrixType = typename Matrix::MatrixType;

    static constexpr int Dim = N;

private:
    Matrix Y_;       // Information matrix Y = P^{-1}
    VectorType eta_; // Information vector η = Y μ (optional, for information filter)

public:
    // ------------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------------

    InformationRepr() noexcept
        : Y_(Matrix::Zero())
        , eta_(VectorType::Zero()) {}

    explicit InformationRepr(const Matrix& Y) noexcept
        : Y_(Y)
        , eta_(VectorType::Zero()) {}

    InformationRepr(const Matrix& Y, const VectorType& eta) noexcept
        : Y_(Y)
        , eta_(eta) {}

    // ------------------------------------------------------------------------
    // Static factory methods
    // ------------------------------------------------------------------------

    [[nodiscard]] static InformationRepr Zero() noexcept { return InformationRepr(Matrix::Zero()); }

    [[nodiscard]] static InformationRepr Identity() noexcept {
        return InformationRepr(Matrix::Identity());
    }

    /// @brief Create from covariance (inverse)
    [[nodiscard]] static auto from_covariance(const Matrix& P) noexcept
        -> std::expected<InformationRepr, std::string> {
        auto Y_result = P.inverse();
        if (!Y_result) {
            return std::unexpected(Y_result.error());
        }
        return InformationRepr(*Y_result);
    }

    /// @brief Create from covariance and mean
    [[nodiscard]] static auto
        from_covariance_and_mean(const Matrix& P, const VectorType& mean) noexcept
        -> std::expected<InformationRepr, std::string> {
        auto Y_result = P.inverse();
        if (!Y_result) {
            return std::unexpected(Y_result.error());
        }
        VectorType eta = Y_result->multiply(mean);
        return InformationRepr(*Y_result, eta);
    }

    // ------------------------------------------------------------------------
    // Access
    // ------------------------------------------------------------------------

    [[nodiscard]] const Matrix& information() const noexcept { return Y_; }

    [[nodiscard]] Matrix& information() noexcept { return Y_; }

    [[nodiscard]] const VectorType& information_vector() const noexcept { return eta_; }

    [[nodiscard]] VectorType& information_vector() noexcept { return eta_; }

    [[nodiscard]] const MatrixType& matrix() const noexcept { return Y_.matrix(); }

    [[nodiscard]] int dim() const noexcept { return Y_.dim(); }

    // ------------------------------------------------------------------------
    // Kalman filter operations
    // ------------------------------------------------------------------------

    /// @brief Predict step (requires inverse for information form)
    /// Y' = (F Y^{-1} F^T + Q)^{-1}
    template <typename F_t>
    [[nodiscard]] auto predict(const F_t& F, const Matrix& Q) const noexcept
        -> std::expected<InformationRepr, std::string> {
        // Convert to covariance for prediction
        auto P_result = Y_.inverse();
        if (!P_result) {
            return std::unexpected(P_result.error());
        }

        // P' = F P F^T + Q
        MatrixType P_new = F * P_result->matrix() * F.transpose() + Q.matrix();

        // Convert back to information
        Matrix P_new_sym(P_new);
        auto Y_new_result = P_new_sym.inverse();
        if (!Y_new_result) {
            return std::unexpected(Y_new_result.error());
        }

        return InformationRepr(*Y_new_result);
    }

    /// @brief Update step (efficient in information form!)
    /// Y' = Y + H^T R^{-1} H
    /// η' = η + H^T R^{-1} z
    template <int M, typename H_t>
    [[nodiscard]] auto update(
        const H_t& H, const SymmetricMatrix<M, Scalar>& R,
        const Eigen::Matrix<Scalar, M, 1>& innovation) const noexcept
        -> std::expected<InformationRepr, std::string> {
        // R^{-1}
        auto R_inv_result = R.inverse();
        if (!R_inv_result) {
            return std::unexpected(R_inv_result.error());
        }

        // Y' = Y + H^T R^{-1} H
        MatrixType Y_new = Y_.matrix() + H.transpose() * R_inv_result->matrix() * H;

        // η' = η + H^T R^{-1} z (if using information vector)
        VectorType eta_new = eta_ + H.transpose() * R_inv_result->multiply(innovation);

        InformationRepr updated(Matrix(Y_new), eta_new);
        updated.Y_.symmetrize_in_place();
        updated.Y_.regularize_in_place();

        return updated;
    }

    /// @brief Fuse with another information representation (sensor fusion)
    /// Y_fused = Y1 + Y2
    /// η_fused = η1 + η2
    [[nodiscard]] InformationRepr fuse(const InformationRepr& other) const noexcept {
        return InformationRepr(Y_ + other.Y_, eta_ + other.eta_);
    }

    // ------------------------------------------------------------------------
    // Conversion to other representations
    // ------------------------------------------------------------------------

    /// @brief Convert to covariance matrix (Y^-1 = P)
    [[nodiscard]] auto to_covariance() const noexcept -> std::expected<Matrix, std::string> {
        return Y_.inverse();
    }

    /// @brief Extract mean from information vector: μ = Y^{-1} η
    [[nodiscard]] auto mean() const noexcept -> std::expected<VectorType, std::string> {
        return Y_.solve(eta_);
    }

    // ------------------------------------------------------------------------
    // Numerical stability
    // ------------------------------------------------------------------------

    void symmetrize() noexcept { Y_.symmetrize_in_place(); }

    void regularize(Scalar eps = Scalar{1e-12}) noexcept { Y_.regularize_in_place(eps); }

    [[nodiscard]] bool is_finite() const noexcept { return Y_.is_finite() && eta_.allFinite(); }

    // ------------------------------------------------------------------------
    // Arithmetic (fusion-friendly)
    // ------------------------------------------------------------------------

    [[nodiscard]] InformationRepr operator+(const InformationRepr& other) const noexcept {
        return fuse(other);
    }

    InformationRepr& operator+=(const InformationRepr& other) noexcept {
        Y_ += other.Y_;
        eta_ += other.eta_;
        return *this;
    }
};

// ============================================================================
// Type aliases
// ============================================================================

template <int N>
using InfoRepr = InformationRepr<N, double>;

template <int N>
using InfoReprf = InformationRepr<N, float>;

} // namespace uni_kf
