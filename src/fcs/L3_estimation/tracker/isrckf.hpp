#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <cmath>

namespace fcs::L3 {
namespace ckf {

template <typename Scalar>
struct SigmaPointConfig {
    Scalar alpha{static_cast<Scalar>(1e-3)}; // 0.517 - 1.932
    Scalar beta{static_cast<Scalar>(2.0)};
    Scalar kappa{static_cast<Scalar>(0)};
};

namespace detail {

// 由 LDLT 产生“可用于三角回代”的下三角平方根因子：
// 1) 尝试 LDLT（带置换）。若置换为单位且 D>0：直接构造 S = L * sqrt(D)
// 2) 否则：按 LDLT 的最小负值自适应加入 jitter，改用 LLT 得到 S（仍为下三角）
// 这样：既利用了 LDLT 的稳健性，又保持 SR-UKF 的三角结构不变。
template <int N, typename Scalar>
inline Eigen::Matrix<Scalar, N, N> chol_from_ldlt(
    const Eigen::Matrix<Scalar, N, N>& Pin, Scalar base_jitter = Scalar(1e-12), int max_try = 6) {
    using Mat = Eigen::Matrix<Scalar, N, N>;
    Mat P     = (Pin + Pin.transpose()) * Scalar(0.5); // 对称化

    // 1) LDLT（带置换）
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
    if (ok)
        D = ldlt.vectorD();

    // 若无置换且 D>0：可直接构造 S = L*sqrt(D)
    if (ok && no_perm) {
        bool all_pos = true;
        for (int i = 0; i < N; ++i) {
            if (!(D(i) > Scalar(0))) {
                all_pos = false;
                break;
            }
        }
        if (all_pos) {
            Mat L = ldlt.matrixL(); // 单位对角的下三角
            Mat S = L;              // 右乘 sqrt(D)（对角）
            for (int i = 0; i < N; ++i) {
                S.col(i) *= std::sqrt(D(i));
            }
            S.template triangularView<Eigen::StrictlyUpper>().setZero();
            // 归一化对角符号（翻转列，保持 P = S S^T 不变）
            for (int i = 0; i < N; ++i) {
                if (S(i, i) < Scalar(0)) {
                    S.col(i) = -S.col(i);
                }
            }
            return S;
        }
    }

    // 2) 需要置换或 D 非正：依据 LDLT 的最小特征提示加 jitter，然后用 LLT 得到正定的 Cholesky
    Scalar jitter = base_jitter;
    Scalar min_d  = Scalar(0);
    if (ok) {
        min_d = D.minCoeff();
        if (min_d <= Scalar(0)) {
            // 以 |min_d| 为尺度的抖动起点
            jitter = std::max(jitter, Scalar(1e-12) + Scalar(1e-3) * std::abs(min_d));
        }
    }

    // 多次退避增加 jitter
    for (int t = 0; t < max_try; ++t) {
        Mat Pj = P;
        Pj.diagonal().array() += jitter;
        Eigen::LLT<Mat> llt(Pj);
        if (llt.info() == Eigen::Success) {
            Mat S = llt.matrixL();
            // 标准化符号（翻转列）
            for (int i = 0; i < N; ++i) {
                if (S(i, i) < Scalar(0)) {
                    S.col(i) = -S.col(i);
                }
            }
            return S;
        }
        jitter *= Scalar(10);
    }

    // 3) 仍失败：特征值修复到 PSD，再做 LLT
    const Scalar scale     = std::max(P.diagonal().cwiseAbs().maxCoeff(), Scalar(1));
    const Scalar floor_eig = std::max(base_jitter * scale, Scalar(1e-12) * scale);
    Eigen::SelfAdjointEigenSolver<Mat> es(P);
    if (es.info() == Eigen::Success) {
        auto eval = es.eigenvalues();
        for (int i = 0; i < N; ++i) {
            eval(i) = std::max(eval(i), floor_eig);
        }
        Mat P_psd = es.eigenvectors() * eval.asDiagonal() * es.eigenvectors().transpose();
        P_psd     = (P_psd + P_psd.transpose()) * Scalar(0.5);

        Eigen::LLT<Mat> llt(P_psd);
        if (llt.info() == Eigen::Success) {
            Mat S = llt.matrixL();
            for (int i = 0; i < N; ++i) {
                if (S(i, i) < Scalar(0)) {
                    S.col(i) = -S.col(i);
                }
            }
            return S;
        }
    }

    // 4) 最后保底：返回一个很小的对角根，避免中断（优先保持滤波运行）
    Mat S = Mat::Zero();
    S.diagonal().setConstant(std::sqrt(std::max(base_jitter * scale, Scalar(1e-9) * scale)));
    return S;
}

// 合成平方根：S S^T = dev dev^T + noise_sqrt noise_sqrt^T
template <int D, typename Scalar>
inline Eigen::Matrix<Scalar, D, D> qr_compose(
    const Eigen::Ref<const Eigen::Matrix<Scalar, D, Eigen::Dynamic>>& dev,
    const Eigen::Ref<const Eigen::Matrix<Scalar, D, Eigen::Dynamic>>& noise_sqrt) {
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
    // 对角符号归一化（翻列）
    for (int i = 0; i < D; ++i) {
        if (S(i, i) < Scalar(0)) {
            S.col(i) = -S.col(i);
        }
    }
    return S;
}

// 解 (S S^T) X = B
template <int D, int C, typename Scalar>
inline Eigen::Matrix<Scalar, D, C>
    solve_via_chol(const Eigen::Matrix<Scalar, D, D>& S, const Eigen::Matrix<Scalar, D, C>& B) {
    Eigen::Matrix<Scalar, D, C> Y = S.template triangularView<Eigen::Lower>().solve(B);
    Eigen::Matrix<Scalar, D, C> X = S.transpose().template triangularView<Eigen::Upper>().solve(Y);
    return X;
}

template <int D, typename Scalar>
inline Eigen::Matrix<Scalar, D, Eigen::Dynamic> solve_via_chol_dyn(
    const Eigen::Matrix<Scalar, D, D>& S,
    const Eigen::Ref<const Eigen::Matrix<Scalar, D, Eigen::Dynamic>>& B) {
    Eigen::Matrix<Scalar, D, Eigen::Dynamic> Y = S.template triangularView<Eigen::Lower>().solve(B);
    Eigen::Matrix<Scalar, D, Eigen::Dynamic> X =
        S.transpose().template triangularView<Eigen::Upper>().solve(Y);
    return X;
}

template <typename Scalar>
inline Scalar wrap_angle(Scalar a) {
    return std::atan2(std::sin(a), std::cos(a));
}

template <typename Scalar>
inline Scalar shortest_angle(Scalar from, Scalar to) {
    return wrap_angle(to - from);
}

template <typename Scalar>
inline Scalar unwrap_angle(Scalar ref, Scalar raw) {
    return ref + shortest_angle(ref, raw);
}

template <int NZ, typename Scalar>
inline Eigen::Matrix<Scalar, NZ, 1> angle_aware_meas_residual(
    const Eigen::Matrix<Scalar, NZ, 1>& a, const Eigen::Matrix<Scalar, NZ, 1>& b) {
    Eigen::Matrix<Scalar, NZ, 1> r = a - b;
    if constexpr (NZ > 0) {
        r(0) = shortest_angle(b(0), a(0));   // yaw
    }
    if constexpr (NZ > 3) {
        r(3) = shortest_angle(b(3), a(3));   // armor yaw
    }
    return r;
}

template <int NZ, typename Scalar>
inline void
    unwrap_meas_sigma(Eigen::Matrix<Scalar, NZ, 1>& z, const Eigen::Matrix<Scalar, NZ, 1>& z_ref) {
    if constexpr (NZ > 0) {
        z(0) = unwrap_angle(z_ref(0), z(0)); // yaw
    }
    if constexpr (NZ > 3) {
        z(3) = unwrap_angle(z_ref(3), z(3)); // armor yaw
    }
}

template <int NX, int NZ, typename Scalar>
inline void build_observability_projection(
    const Eigen::Matrix<Scalar, NZ, NX>& H, Scalar rel_tol, Scalar abs_tol,
    Eigen::Matrix<Scalar, NX, NX>& P_obs) {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> H_dyn = H;
    Eigen::JacobiSVD<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> svd(
        H_dyn, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const auto& s = svd.singularValues();
    if (s.size() == 0) {
        P_obs.setZero();
        return;
    }

    const Scalar s_max = s(0);
    const Scalar cut   = std::max(abs_tol, rel_tol * s_max);

    int rank         = 0;
    const int s_size = static_cast<int>(s.size());
    for (; rank < s_size; ++rank) {
        if (!(s(rank) > cut)) {
            break;
        }
    }
    if (rank == 0) {
        P_obs.setZero();
        return;
    }

    const auto V_r                                                  = svd.matrixV().leftCols(rank);
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> P_obs_dyn = V_r * V_r.transpose();
    P_obs                                                           = P_obs_dyn;
}

} // namespace detail

/**
 * Iterated Square-Root Cubature Kalman Filter (ISR-SRCKF)
 * - 2N cubature 点（权重全正），避免 UKF 权重为负导致 SR 实现不适配的问题
 * - 量测更新用 Gauss-Newton / MAP 形式迭代（统计线性回归近似雅可比）
 * - public 接口风格一致，可以在ISRCKF和SRUKF之间无缝切换：predict/update/step/x/P/Sx
 *
 * Model 需提供（与 SRUKF 一致）：
 *   VecX f(const VecX& x, Scalar dt) const;
 *   VecZ h(const VecX& x) const;
 *   MatQ Q_sqrt(Scalar dt) const;           // (NX x nq)，常用 nq=NX
 *   MatR R_sqrt(const VecZ& z) const;       // (NZ x nr)，常用 nr=NZ
 */
template <typename Model, int NX, int NZ, typename Scalar = double>
class ISRCKF {
public:
    using VecX  = Eigen::Matrix<Scalar, NX, 1>;
    using VecZ  = Eigen::Matrix<Scalar, NZ, 1>;
    using MatXX = Eigen::Matrix<Scalar, NX, NX>;
    using MatZZ = Eigen::Matrix<Scalar, NZ, NZ>;
    using MatXZ = Eigen::Matrix<Scalar, NX, NZ>;
    using MatZX = Eigen::Matrix<Scalar, NZ, NX>;
    using DevX  = Eigen::Matrix<Scalar, NX, Eigen::Dynamic>;
    using DevZ  = Eigen::Matrix<Scalar, NZ, Eigen::Dynamic>;

    explicit ISRCKF(
        Model& model, const VecX& x0, const MatXX& P0, SigmaPointConfig<Scalar> /*cfg_compat*/ = {})
        : model_(model) {
        x_      = x0;
        Sx_     = detail::chol_from_ldlt<NX, Scalar>(P0);
        gamma_  = std::sqrt(static_cast<Scalar>(NX));   // CKF 固定
        w_      = Scalar(1) / (Scalar(2) * Scalar(NX)); // 2N 点，权重相等
        w_sqrt_ = std::sqrt(w_);

        Xsig_.resize(2 * NX, NX);
        Zsig_.resize(2 * NX, NZ);
    }

    // 可选：调迭代次数与收敛阈值（不影响旧代码编译）
    void set_max_iters(int iters) { max_iters_ = (iters < 1 ? 1 : iters); }
    void set_iter_tol(Scalar tol) { iter_tol_ = (tol <= Scalar(0) ? Scalar(1e-6) : tol); }
    void set_observability_truncation(Scalar rel_tol, Scalar abs_tol = Scalar(1e-8)) {
        obs_rel_tol_ = std::max(rel_tol, Scalar(0));
        obs_abs_tol_ = std::max(abs_tol, Scalar(0));
    }

    void predict(Scalar dt) {
        compute_cubature_points_(x_, Sx_, Xsig_);

        // 传播
        for (int i = 0; i < Xsig_.rows(); ++i) {
            Xsig_.row(i) = model_.f(Xsig_.row(i).transpose(), dt).transpose();
        }

        // 均值
        VecX x_pred = VecX::Zero();
        for (int i = 0; i < Xsig_.rows(); ++i)
            x_pred.noalias() += w_ * Xsig_.row(i).transpose();

        // SR 协方差
        DevX dev_x(NX, Xsig_.rows());
        for (int i = 0; i < Xsig_.rows(); ++i)
            dev_x.col(i) = w_sqrt_ * (Xsig_.row(i).transpose() - x_pred);

        const auto Qs = [&]() {
            if constexpr (requires(const Model& m, const VecX& x_cur, Scalar d) {
                              m.Q_sqrt(x_cur, d);
                          }) {
                return model_.Q_sqrt(x_, dt);
            } else {
                return model_.Q_sqrt(dt);
            }
        }();
        assert(Qs.rows() == NX && "Q_sqrt rows must be NX");
        Sx_ = detail::qr_compose<NX, Scalar>(dev_x, Qs);

        x_ = x_pred;
    }

    void update(const VecZ& z) {
        // prior 固定（迭代时不改变先验协方差）
        const VecX x_prior  = x_;
        const MatXX S_prior = Sx_;
        const MatXX P_prior = S_prior * S_prior.transpose();

        VecX x_it = x_prior;

        // 迭代中需要缓存最后一次的量测统计量，用于最终协方差更新
        MatXZ Pxz_last = MatXZ::Zero();
        MatZZ Sz_last  = MatZZ::Zero();

        for (int it = 0; it < max_iters_; ++it) {
            MatXX S_lin = S_prior;
            Eigen::Matrix<Scalar, NX, Eigen::Dynamic> N_obs;
            if constexpr (requires(const Model& m, const VecX& x_cur) {
                              m.state_observability_nullspace(x_cur);
                          }) {
                N_obs                   = model_.state_observability_nullspace(x_it);
                bool use_cov_projection = false;
                if constexpr (requires(const Model& m) { m.use_observability_cov_projection(); }) {
                    use_cov_projection = model_.use_observability_cov_projection();
                }
                if (use_cov_projection && N_obs.cols() > 0) {
                    const auto Pinv_N = detail::solve_via_chol_dyn<NX, Scalar>(S_prior, N_obs);
                    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> M =
                        N_obs.transpose() * Pinv_N;
                    M = (M + M.transpose()) * Scalar(0.5);

                    Eigen::LDLT<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> ldlt(M);
                    if (ldlt.info() == Eigen::Success) {
                        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Y =
                            ldlt.solve(N_obs.transpose());
                        MatXX P_lin = P_prior - N_obs * Y;
                        P_lin       = (P_lin + P_lin.transpose()) * Scalar(0.5);
                        S_lin       = detail::chol_from_ldlt<NX, Scalar>(P_lin);
                    }
                }
            }

            compute_cubature_points_(x_it, S_lin, Xsig_);

            // 量测传播
            const VecZ z_ref = model_.h(x_it);
            for (int i = 0; i < Xsig_.rows(); ++i) {
                VecZ zi = model_.h(Xsig_.row(i).transpose());
                detail::unwrap_meas_sigma<NZ, Scalar>(zi, z_ref);
                Zsig_.row(i) = zi.transpose();
            }

            // 量测均值
            VecZ z_pred = VecZ::Zero();
            for (int i = 0; i < Zsig_.rows(); ++i)
                z_pred.noalias() += w_ * Zsig_.row(i).transpose();

            // dev_z（加 R_sqrt）
            DevZ dev_z(NZ, Zsig_.rows());
            for (int i = 0; i < Zsig_.rows(); ++i) {
                const VecZ dz =
                    detail::angle_aware_meas_residual<NZ, Scalar>(Zsig_.row(i).transpose(), z_pred);
                dev_z.col(i) = w_sqrt_ * dz;
            }

            const auto Rs = model_.R_sqrt(z);
            assert(Rs.rows() == NZ && "R_sqrt rows must be NZ");
            MatZZ Sz = detail::qr_compose<NZ, Scalar>(dev_z, Rs);

            // Pxz：围绕 x_it 做统计线性回归（近似 H(x_it)）
            MatXZ Pxz = MatXZ::Zero();
            for (int i = 0; i < Xsig_.rows(); ++i) {
                const VecX dx = Xsig_.row(i).transpose() - x_it;
                const VecZ dz =
                    detail::angle_aware_meas_residual<NZ, Scalar>(Zsig_.row(i).transpose(), z_pred);
                Pxz.noalias() += w_ * (dx * dz.transpose());
            }

            const MatXZ Pinv_Pxz = detail::solve_via_chol<NX, NZ, Scalar>(S_lin, Pxz);
            const MatZX H_lin    = Pinv_Pxz.transpose();

            MatXZ Pxz_obs = Pxz;
            if constexpr (requires(const Model& m, const VecX& x_cur) {
                              m.state_observability_nullspace(x_cur);
                          }) {
                if (N_obs.cols() > 0) {
                    const auto Pinv_N = detail::solve_via_chol_dyn<NX, Scalar>(S_lin, N_obs);
                    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> M =
                        N_obs.transpose() * Pinv_N;
                    M = (M + M.transpose()) * Scalar(0.5);

                    Eigen::LDLT<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> ldlt(M);
                    if (ldlt.info() == Eigen::Success) {
                        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> C =
                            N_obs.transpose() * Pinv_Pxz;
                        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Y =
                            ldlt.solve(C);
                        Pxz_obs.noalias() -= N_obs * Y;
                    }
                }
            } else {
                MatXX P_obs = MatXX::Identity();
                detail::build_observability_projection<NX, NZ, Scalar>(
                    H_lin, obs_rel_tol_, obs_abs_tol_, P_obs);
                Pxz_obs = P_obs * Pxz;
            }

            // K = P_xz_obs * P_zz^{-1}, P_zz = Sz Sz^T
            MatXZ K = detail::solve_via_chol<NZ, NX, Scalar>(Sz, Pxz_obs.transpose()).transpose();

            const VecZ nu    = detail::angle_aware_meas_residual<NZ, Scalar>(z, z_pred);
            const VecZ innov = nu;
            const VecX x_new = x_prior + K * innov;

            // 保存最后一次统计量
            Pxz_last = Pxz_obs;
            Sz_last  = Sz;

            const Scalar step_norm = (x_new - x_it).template lpNorm<Eigen::Infinity>();
            x_it                   = x_new;
            if (step_norm < iter_tol_)
                break;
        }

        // 状态更新
        x_ = x_it;

        // 协方差更新（仍在协方差域做一次，最后重分解回 SR）
        MatXZ K_last =
            detail::solve_via_chol<NZ, NX, Scalar>(Sz_last, Pxz_last.transpose()).transpose();
        const MatZZ Pzz_last = Sz_last * Sz_last.transpose();
        MatXX P_post         = P_prior - K_last * Pzz_last * K_last.transpose();
        P_post               = (P_post + P_post.transpose()) * Scalar(0.5);
        Sx_                  = detail::chol_from_ldlt<NX, Scalar>(P_post);
    }

    void step(const VecZ& z, Scalar dt) {
        predict(dt);
        update(z);
    }

    const VecX& x() const noexcept { return x_; }
    VecX* x_raw_ptr() noexcept { return &x_; }
    MatXX* Sx_raw_ptr() noexcept { return &Sx_; }
    MatXX P() const { return Sx_ * Sx_.transpose(); }
    const MatXX& Sx() const noexcept { return Sx_; }

private:
    Model& model_;

    VecX x_{};
    MatXX Sx_{};

    Scalar gamma_{};
    Scalar w_{};
    Scalar w_sqrt_{};

    int max_iters_{3};                                           // 默认 3 次迭代
    Scalar iter_tol_{Scalar(1e-6)};                              // 默认无穷范数阈值
    Scalar obs_rel_tol_{Scalar(1e-3)};                           // SVD 相对截断阈值
    Scalar obs_abs_tol_{Scalar(1e-8)};                           // SVD 绝对截断阈值

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Xsig_; // (2N x NX)
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Zsig_; // (2N x NZ)

    void compute_cubature_points_(
        const VecX& xc, const MatXX& S,
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& Xsig) {
        Eigen::Matrix<Scalar, NX, NX> A = S * gamma_;
        for (int j = 0; j < NX; ++j) {
            Xsig.row(j)      = (xc + A.col(j)).transpose();
            Xsig.row(NX + j) = (xc - A.col(j)).transpose();
        }
    }
};

} // namespace ckf
} // namespace fcs::L3
