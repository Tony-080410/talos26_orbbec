#pragma once

#include <Eigen/Dense>

#include <cassert>
#include <cmath>

template <typename Scalar>
struct SigmaPointConfig {
    Scalar alpha{static_cast<Scalar>(0.7)};
    Scalar beta{static_cast<Scalar>(2.0)};
    Scalar kappa{static_cast<Scalar>(0)};
};

namespace srukf::detail {

template <int N, typename Scalar>
inline auto chol_from_ldlt(
    const Eigen::Matrix<Scalar, N, N>& Pin, Scalar base_jitter = Scalar(1e-12), int max_try = 6)
    -> Eigen::Matrix<Scalar, N, N> {
    using Mat = Eigen::Matrix<Scalar, N, N>;

    Mat P = (Pin + Pin.transpose()) * Scalar(0.5);

    Eigen::LDLT<Mat> ldlt(P);
    const bool ok     = (ldlt.info() == Eigen::Success);
    const auto& trans = ldlt.transpositionsP();
    bool no_perm      = true;
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
            for (int i = 0; i < N; ++i) {
                if (S(i, i) < Scalar(0)) {
                    S.col(i) = -S.col(i);
                }
            }
            return S;
        }
    }

    Scalar jitter = base_jitter;
    if (ok) {
        const Scalar min_d = D.minCoeff();
        if (min_d <= Scalar(0)) {
            jitter = std::max(jitter, Scalar(1e-12) + Scalar(1e-3) * std::abs(min_d));
        }
    }

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

    Eigen::LLT<Mat> llt(P);
    assert(llt.info() == Eigen::Success);
    Mat S = llt.matrixL();
    for (int i = 0; i < N; ++i) {
        if (S(i, i) < Scalar(0)) {
            S.col(i) = -S.col(i);
        }
    }
    return S;
}

template <int D, typename Scalar>
inline auto qr_compose(
    const Eigen::Ref<const Eigen::Matrix<Scalar, D, Eigen::Dynamic>>& dev,
    const Eigen::Ref<const Eigen::Matrix<Scalar, D, Eigen::Dynamic>>& noise_sqrt)
    -> Eigen::Matrix<Scalar, D, D> {
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
    for (int i = 0; i < D; ++i) {
        if (S(i, i) < Scalar(0)) {
            S.col(i) = -S.col(i);
        }
    }
    return S;
}

template <int D, int C, typename Scalar>
inline auto
    solve_via_chol(const Eigen::Matrix<Scalar, D, D>& S, const Eigen::Matrix<Scalar, D, C>& B)
        -> Eigen::Matrix<Scalar, D, C> {
    Eigen::Matrix<Scalar, D, C> Y = S.template triangularView<Eigen::Lower>().solve(B);
    Eigen::Matrix<Scalar, D, C> X = S.transpose().template triangularView<Eigen::Upper>().solve(Y);
    return X;
}

} // namespace srukf::detail

namespace srukf {

template <typename Model, int NX, int NZ, typename Scalar = double>
class ISRCKF {
public:
    using VecX  = Eigen::Matrix<Scalar, NX, 1>;
    using VecZ  = Eigen::Matrix<Scalar, NZ, 1>;
    using MatXX = Eigen::Matrix<Scalar, NX, NX>;
    using MatZZ = Eigen::Matrix<Scalar, NZ, NZ>;
    using MatXZ = Eigen::Matrix<Scalar, NX, NZ>;
    using DevX  = Eigen::Matrix<Scalar, NX, Eigen::Dynamic>;
    using DevZ  = Eigen::Matrix<Scalar, NZ, Eigen::Dynamic>;

    explicit ISRCKF(Model& model, const VecX& x0, const MatXX& P0, SigmaPointConfig<Scalar> = {})
        : model_(model) {
        x_      = x0;
        Sx_     = detail::chol_from_ldlt<NX, Scalar>(P0);
        gamma_  = std::sqrt(static_cast<Scalar>(NX));
        w_      = Scalar(1) / (Scalar(2) * Scalar(NX));
        w_sqrt_ = std::sqrt(w_);

        Xsig_.resize(2 * NX, NX);
        Zsig_.resize(2 * NX, NZ);
    }

    void set_max_iters(int iters) { max_iters_ = (iters < 1 ? 1 : iters); }

    void set_iter_tol(Scalar tol) { iter_tol_ = (tol <= Scalar(0) ? Scalar(1e-6) : tol); }

    void predict(Scalar dt) {
        compute_cubature_points_(x_, Sx_, Xsig_);

        for (int i = 0; i < Xsig_.rows(); ++i) {
            Xsig_.row(i) = model_.f(Xsig_.row(i).transpose(), dt).transpose();
        }

        VecX x_pred = VecX::Zero();
        for (int i = 0; i < Xsig_.rows(); ++i) {
            x_pred.noalias() += w_ * Xsig_.row(i).transpose();
        }

        DevX dev_x(NX, Xsig_.rows());
        for (int i = 0; i < Xsig_.rows(); ++i) {
            dev_x.col(i) = w_sqrt_ * (Xsig_.row(i).transpose() - x_pred);
        }

        const auto Qs = model_.Q_sqrt(dt);
        assert(Qs.rows() == NX);
        Sx_ = detail::qr_compose<NX, Scalar>(dev_x, Qs);

        x_ = x_pred;
    }

    void update(const VecZ& z) {
        const VecX x_prior  = x_;
        const MatXX S_prior = Sx_;
        const MatXX P_prior = S_prior * S_prior.transpose();

        VecX x_it = x_prior;

        MatXZ Pxz_last  = MatXZ::Zero();
        MatZZ Sz_last   = MatZZ::Zero();
        VecZ zpred_last = VecZ::Zero();

        for (int it = 0; it < max_iters_; ++it) {
            compute_cubature_points_(x_it, S_prior, Xsig_);

            for (int i = 0; i < Xsig_.rows(); ++i) {
                Zsig_.row(i) = model_.h(Xsig_.row(i).transpose()).transpose();
            }

            VecZ z_pred = VecZ::Zero();
            for (int i = 0; i < Zsig_.rows(); ++i) {
                z_pred.noalias() += w_ * Zsig_.row(i).transpose();
            }

            DevZ dev_z(NZ, Zsig_.rows());
            for (int i = 0; i < Zsig_.rows(); ++i) {
                dev_z.col(i) = w_sqrt_ * (Zsig_.row(i).transpose() - z_pred);
            }

            const auto Rs = model_.R_sqrt(z);
            assert(Rs.rows() == NZ);
            MatZZ Sz = detail::qr_compose<NZ, Scalar>(dev_z, Rs);

            MatXZ Pxz = MatXZ::Zero();
            for (int i = 0; i < Xsig_.rows(); ++i) {
                const VecX dx = Xsig_.row(i).transpose() - x_it;
                const VecZ dz = Zsig_.row(i).transpose() - z_pred;
                Pxz.noalias() += w_ * (dx * dz.transpose());
            }

            MatXZ K = detail::solve_via_chol<NZ, NX, Scalar>(Sz, Pxz.transpose()).transpose();

            const VecX dxp = x_it - x_prior;
            const VecX v   = detail::solve_via_chol<NX, 1, Scalar>(S_prior, dxp);
            const VecZ Hdx = Pxz.transpose() * v;

            const VecZ innov = (z - z_pred) + Hdx;
            const VecX x_new = x_prior + K * innov;

            Pxz_last   = Pxz;
            Sz_last    = Sz;
            zpred_last = z_pred;

            const Scalar step_norm = (x_new - x_it).template lpNorm<Eigen::Infinity>();
            x_it                   = x_new;
            if (step_norm < iter_tol_) {
                break;
            }
        }

        x_ = x_it;

        const MatZZ Pzz = Sz_last * Sz_last.transpose();
        MatXZ K_last =
            detail::solve_via_chol<NZ, NX, Scalar>(Sz_last, Pxz_last.transpose()).transpose();

        MatXX P_post = P_prior - K_last * Pzz * K_last.transpose();
        Sx_          = detail::chol_from_ldlt<NX, Scalar>(P_post);
    }

    void step(const VecZ& z, Scalar dt) {
        predict(dt);
        update(z);
    }

    auto x() const noexcept -> const VecX& { return x_; }
    auto P() const -> MatXX { return Sx_ * Sx_.transpose(); }
    auto Sx() const noexcept -> const MatXX& { return Sx_; }

private:
    Model& model_;

    VecX x_{};
    MatXX Sx_{};

    Scalar gamma_{};
    Scalar w_{};
    Scalar w_sqrt_{};

    int max_iters_{3};
    Scalar iter_tol_{Scalar(1e-6)};

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Xsig_;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Zsig_;

    void compute_cubature_points_(
        const VecX& xc, const MatXX& S,
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& Xsig) {
        const Eigen::Matrix<Scalar, NX, NX> A = S * gamma_;
        for (int j = 0; j < NX; ++j) {
            Xsig.row(j)      = (xc + A.col(j)).transpose();
            Xsig.row(NX + j) = (xc - A.col(j)).transpose();
        }
    }
};

} // namespace srukf
