#pragma once

#include <Eigen/Dense>
#include <cassert>
#include <cmath>

namespace fcs::L3::srukf {

/// Sigma point configuration for UKF
template <typename Scalar>
struct SigmaPointConfig {
    Scalar alpha{static_cast<Scalar>(1e-3)};
    Scalar beta{static_cast<Scalar>(2.0)};
    Scalar kappa{static_cast<Scalar>(0)};
};

namespace detail {

/// Compute Cholesky factor from LDLT decomposition with jitter fallback
/// Returns lower triangular S such that S * S^T = P (approximately)
/// @param Pin Input symmetric positive semi-definite matrix
/// @param base_jitter Initial jitter value for numerical stability
/// @param max_try Maximum number of jitter increase attempts
template <int N, typename Scalar>
[[nodiscard]] inline Eigen::Matrix<Scalar, N, N> chol_from_ldlt(
    const Eigen::Matrix<Scalar, N, N>& Pin, Scalar base_jitter = Scalar(1e-12),
    int max_try = 6) noexcept {
    using Mat = Eigen::Matrix<Scalar, N, N>;

    // Symmetrize input
    Mat P = (Pin + Pin.transpose()) * Scalar(0.5);

    // 1) Try LDLT with pivoting
    Eigen::LDLT<Mat> ldlt(P);
    const bool ok     = (ldlt.info() == Eigen::Success);
    const auto& trans = ldlt.transpositionsP();

    bool no_perm = true;
    for (int i = 0; i < N; ++i) {
        if (trans.indices()(i) != i) {
            no_perm = false;
            break;
        }
    }

    Eigen::Matrix<Scalar, N, 1> D;
    if (ok) {
        D = ldlt.vectorD();
    }

    // If no permutation and D > 0: directly construct S = L * sqrt(D)
    if (ok && no_perm) {
        bool all_pos = true;
        for (int i = 0; i < N; ++i) {
            if (!(D(i) > Scalar(0))) {
                all_pos = false;
                break;
            }
        }
        if (all_pos) {
            Mat L = ldlt.matrixL();
            Mat S = L;
            for (int i = 0; i < N; ++i) {
                S.col(i) *= std::sqrt(D(i));
            }
            S.template triangularView<Eigen::StrictlyUpper>().setZero();
            // Normalize diagonal signs
            for (int i = 0; i < N; ++i) {
                if (S(i, i) < Scalar(0)) {
                    S.col(i) = -S.col(i);
                }
            }
            return S;
        }
    }

    // 2) Need permutation or D not positive: add jitter and use LLT
    Scalar jitter = base_jitter;
    Scalar min_d  = Scalar(0);
    if (ok) {
        min_d = D.minCoeff();
        if (min_d <= Scalar(0)) {
            jitter = std::max(jitter, Scalar(1e-12) + Scalar(1e-3) * std::abs(min_d));
        }
    }

    // Retry with increasing jitter
    for (int t = 0; t < max_try; ++t) {
        Mat Pj = P;
        Pj.diagonal().array() += jitter;
        Eigen::LLT<Mat> llt(Pj);
        if (llt.info() == Eigen::Success) {
            Mat S = llt.matrixL();
            for (int i = 0; i < N; ++i) {
                if (S(i, i) < Scalar(0)) {
                    S.col(i) = -S.col(i);
                }
            }
            return S;
        }
        jitter *= Scalar(10);
    }

    // Fallback: assert failure
    Eigen::LLT<Mat> llt(P);
    assert(llt.info() == Eigen::Success && "P not SPD even after jitter");
    Mat S = llt.matrixL();
    for (int i = 0; i < N; ++i) {
        if (S(i, i) < Scalar(0)) {
            S.col(i) = -S.col(i);
        }
    }
    return S;
}

/// Compose square root: S S^T = dev dev^T + noise_sqrt noise_sqrt^T
/// Uses QR decomposition for numerical stability
template <int D, typename Scalar>
[[nodiscard]] inline Eigen::Matrix<Scalar, D, D> qr_compose(
    const Eigen::Ref<const Eigen::Matrix<Scalar, D, Eigen::Dynamic>>& dev,
    const Eigen::Ref<const Eigen::Matrix<Scalar, D, Eigen::Dynamic>>& noise_sqrt) noexcept {
    const int k = static_cast<int>(dev.cols());
    const int m = static_cast<int>(noise_sqrt.cols());

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MT(k + m, D);
    if (k) {
        MT.topRows(k) = dev.transpose();
    }
    if (m) {
        MT.bottomRows(m) = noise_sqrt.transpose();
    }

    Eigen::HouseholderQR<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> qr(MT);
    Eigen::Matrix<Scalar, D, D> R = qr.matrixQR().template topLeftCorner<D, D>();
    R.template triangularView<Eigen::StrictlyLower>().setZero();

    Eigen::Matrix<Scalar, D, D> S = R.transpose();
    S.template triangularView<Eigen::StrictlyUpper>().setZero();

    // Normalize diagonal signs
    for (int i = 0; i < D; ++i) {
        if (S(i, i) < Scalar(0)) {
            S.col(i) = -S.col(i);
        }
    }
    return S;
}

/// Solve (S S^T) X = B using triangular solves
template <int D, int C, typename Scalar>
[[nodiscard]] inline Eigen::Matrix<Scalar, D, C> solve_via_chol(
    const Eigen::Matrix<Scalar, D, D>& S, const Eigen::Matrix<Scalar, D, C>& B) noexcept {
    Eigen::Matrix<Scalar, D, C> Y = S.template triangularView<Eigen::Lower>().solve(B);
    Eigen::Matrix<Scalar, D, C> X = S.transpose().template triangularView<Eigen::Upper>().solve(Y);
    return X;
}

} // namespace detail

/// Square-Root Unscented Kalman Filter
/// @tparam Model Motion/observation model with f(), h(), Q_sqrt(), R_sqrt()
/// @tparam NX State dimension
/// @tparam NZ Observation dimension
/// @tparam Scalar Floating point type
template <typename Model, int NX, int NZ, typename Scalar = double>
class SRUKF {
public:
    using VecX  = Eigen::Matrix<Scalar, NX, 1>;
    using VecZ  = Eigen::Matrix<Scalar, NZ, 1>;
    using MatXX = Eigen::Matrix<Scalar, NX, NX>;
    using MatZZ = Eigen::Matrix<Scalar, NZ, NZ>;
    using MatXZ = Eigen::Matrix<Scalar, NX, NZ>;
    using DevX  = Eigen::Matrix<Scalar, NX, Eigen::Dynamic>;
    using DevZ  = Eigen::Matrix<Scalar, NZ, Eigen::Dynamic>;

    /// Construct SRUKF
    /// @note Constructor is noexcept; invalid config will trigger assert in init_weights_
    SRUKF(Model& model, const VecX& x0, const MatXX& P0, SigmaPointConfig<Scalar> cfg = {}) noexcept
        : model_(model)
        , cfg_(cfg) {
        x_  = x0;
        Sx_ = detail::chol_from_ldlt<NX, Scalar>(P0);
        init_weights_();
        Xsig_.resize(2 * NX + 1, NX);
        Zsig_.resize(2 * NX + 1, NZ);
    }

    /// Prediction step
    void predict(Scalar dt) noexcept {
        compute_sigma_points_();

        for (int i = 0; i < Xsig_.rows(); ++i) {
            Xsig_.row(i) = model_.f(Xsig_.row(i).transpose(), dt).transpose();
        }

        x_.setZero();
        for (int i = 0; i < Xsig_.rows(); ++i) {
            x_.noalias() += wm_(i) * Xsig_.row(i).transpose();
        }

        DevX dev_x;
        dev_x.resize(NX, 2 * NX + 1);
        for (int i = 0; i < Xsig_.rows(); ++i) {
            dev_x.col(i) = wc_sqrt_(i) * (Xsig_.row(i).transpose() - x_);
        }

        const auto Qs = model_.Q_sqrt(dt);
        assert(Qs.rows() == NX && "Q_sqrt rows must be NX");
        Sx_ = detail::qr_compose<NX, Scalar>(dev_x, Qs);
    }

    /// Update step with measurement
    void update(const VecZ& z) noexcept {
        compute_sigma_points_();

        for (int i = 0; i < Xsig_.rows(); ++i) {
            Zsig_.row(i) = model_.h(Xsig_.row(i).transpose()).transpose();
        }

        VecZ z_pred = VecZ::Zero();
        for (int i = 0; i < Zsig_.rows(); ++i) {
            z_pred.noalias() += wm_(i) * Zsig_.row(i).transpose();
        }

        DevZ dev_z;
        dev_z.resize(NZ, 2 * NX + 1);
        for (int i = 0; i < Zsig_.rows(); ++i) {
            dev_z.col(i) = wc_sqrt_(i) * (Zsig_.row(i).transpose() - z_pred);
        }

        const auto Rs = model_.R_sqrt(z);
        assert(Rs.rows() == NZ && "R_sqrt rows must be NZ");
        MatZZ Sz = detail::qr_compose<NZ, Scalar>(dev_z, Rs);

        MatXZ Pxz = MatXZ::Zero();
        for (int i = 0; i < Xsig_.rows(); ++i) {
            const VecX dx = Xsig_.row(i).transpose() - x_;
            const VecZ dz = Zsig_.row(i).transpose() - z_pred;
            Pxz.noalias() += wc_(i) * (dx * dz.transpose());
        }

        // K = P_xz * P_zz^{-1}, P_zz = Sz Sz^T
        MatXZ K = detail::solve_via_chol<NZ, NX, Scalar>(Sz, Pxz.transpose()).transpose();

        const VecZ y = z - z_pred;
        x_.noalias() += K * y;

        // Covariance update: P^+ = P^- - K P_zz K^T
        MatXX P_prior = Sx_ * Sx_.transpose();
        MatZZ Pzz     = Sz * Sz.transpose();
        MatXX P_post  = P_prior - K * Pzz * K.transpose();

        Sx_ = detail::chol_from_ldlt<NX, Scalar>(P_post);
    }

    /// Combined predict + update
    void step(const VecZ& z, Scalar dt) noexcept {
        predict(dt);
        update(z);
    }

    [[nodiscard]] const VecX& x() const noexcept { return x_; }
    [[nodiscard]] VecX* x_raw_ptr() noexcept { return &x_; }
    [[nodiscard]] MatXX P() const noexcept { return Sx_ * Sx_.transpose(); }
    [[nodiscard]] const MatXX& Sx() const noexcept { return Sx_; }

private:
    Model& model_;
    SigmaPointConfig<Scalar> cfg_;

    VecX x_;
    MatXX Sx_;

    Scalar lambda_{};
    Scalar gamma_{};
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> wm_, wc_, wc_sqrt_;

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Xsig_;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Zsig_;

    void init_weights_() noexcept {
        const Scalar n = static_cast<Scalar>(NX);
        lambda_        = cfg_.alpha * cfg_.alpha * (n + cfg_.kappa) - n;

        // Ensure n + lambda is not too small
        const Scalar nlam_min = static_cast<Scalar>(1e-6);
        if (n + lambda_ <= nlam_min) {
            lambda_ = nlam_min - n;
        }
        gamma_ = std::sqrt(n + lambda_);

        wm_.setConstant(2 * NX + 1, Scalar(1) / (2 * (n + lambda_)));
        wc_    = wm_;
        wm_(0) = lambda_ / (n + lambda_);
        wc_(0) = wm_(0) + (Scalar(1) - cfg_.alpha * cfg_.alpha + cfg_.beta);

        // Covariance weights must be non-negative for SR implementation
        Scalar min_wc = wc_.minCoeff();
        assert(
            min_wc >= Scalar(-1e-12)
            && "SRUKF: covariance weights must be non-negative; adjust alpha/beta/kappa");

        wc_sqrt_.resizeLike(wc_);
        for (int i = 0; i < wc_.size(); ++i) {
            wc_sqrt_(i) = std::sqrt(std::max(wc_(i), Scalar(0)));
        }
    }

    void compute_sigma_points_() noexcept {
        Xsig_.row(0)                    = x_.transpose();
        Eigen::Matrix<Scalar, NX, NX> A = Sx_ * gamma_;
        for (int j = 0; j < NX; ++j) {
            Xsig_.row(1 + j)      = (x_ + A.col(j)).transpose();
            Xsig_.row(1 + NX + j) = (x_ - A.col(j)).transpose();
        }
    }
};

} // namespace fcs::L3::srukf
