#pragma once

#include "srukf.hpp"

#include <Eigen/Dense>
#include <cassert>
#include <cmath>

namespace fcs::L3::srukf {

/// Iterated Square-Root Cubature Kalman Filter (ISR-CKF)
///
/// Features:
/// - 2N cubature points (all positive weights) - avoids negative weight issues in SR-UKF
/// - Gauss-Newton/MAP style iterated measurement update
/// - Maintains square-root form for numerical stability
///
/// Model requirements:
///   - VecX f(const VecX& x, Scalar dt) const;  // State transition
///   - VecZ h(const VecX& x) const;              // Measurement function
///   - MatQ Q_sqrt(Scalar dt) const;             // Process noise sqrt (NX x nq)
///   - MatR R_sqrt(const VecZ& z) const;         // Measurement noise sqrt (NZ x nr)
///
/// @tparam Model Motion/observation model
/// @tparam NX State dimension
/// @tparam NZ Observation dimension
/// @tparam Scalar Floating point type
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

    /// Construct ISRCKF
    /// @param model Reference to motion/observation model
    /// @param x0 Initial state estimate
    /// @param P0 Initial covariance matrix
    /// @param cfg_compat Compatibility parameter (unused, for API compatibility with SRUKF)
    ISRCKF(
        Model& model, const VecX& x0, const MatXX& P0,
        SigmaPointConfig<Scalar> /*cfg_compat*/ = {}) noexcept
        : model_(model) {
        x_      = x0;
        Sx_     = detail::chol_from_ldlt<NX, Scalar>(P0);
        gamma_  = std::sqrt(static_cast<Scalar>(NX));   // CKF uses fixed scaling
        w_      = Scalar(1) / (Scalar(2) * Scalar(NX)); // 2N points, equal weights
        w_sqrt_ = std::sqrt(w_);

        Xsig_.resize(2 * NX, NX);
        Zsig_.resize(2 * NX, NZ);
    }

    /// Set maximum iterations for measurement update
    void set_max_iters(int iters) noexcept { max_iters_ = (iters < 1 ? 1 : iters); }

    /// Set iteration convergence tolerance
    void set_iter_tol(Scalar tol) noexcept { iter_tol_ = (tol <= Scalar(0) ? Scalar(1e-6) : tol); }

    /// Prediction step
    void predict(Scalar dt) noexcept {
        compute_cubature_points_(x_, Sx_, Xsig_);

        // Propagate sigma points through dynamics
        for (int i = 0; i < Xsig_.rows(); ++i) {
            Xsig_.row(i) = model_.f(Xsig_.row(i).transpose(), dt).transpose();
        }

        // Compute predicted mean
        VecX x_pred = VecX::Zero();
        for (int i = 0; i < Xsig_.rows(); ++i) {
            x_pred.noalias() += w_ * Xsig_.row(i).transpose();
        }

        // Compute predicted covariance (SR form)
        DevX dev_x(NX, Xsig_.rows());
        for (int i = 0; i < Xsig_.rows(); ++i) {
            dev_x.col(i) = w_sqrt_ * (Xsig_.row(i).transpose() - x_pred);
        }

        const auto Qs = model_.Q_sqrt(dt);
        assert(Qs.rows() == NX && "Q_sqrt rows must be NX");
        Sx_ = detail::qr_compose<NX, Scalar>(dev_x, Qs);

        x_ = x_pred;
    }

    /// Update step with iterated Gauss-Newton refinement
    void update(const VecZ& z) noexcept {
        // Prior state and covariance (fixed during iteration)
        const VecX x_prior  = x_;
        const MatXX S_prior = Sx_;
        const MatXX P_prior = S_prior * S_prior.transpose();

        VecX x_it = x_prior;

        // Cache last iteration statistics for final covariance update
        MatXZ Pxz_last  = MatXZ::Zero();
        MatZZ Sz_last   = MatZZ::Zero();
        VecZ zpred_last = VecZ::Zero();

        for (int it = 0; it < max_iters_; ++it) {
            compute_cubature_points_(x_it, S_prior, Xsig_);

            // Propagate through measurement function
            for (int i = 0; i < Xsig_.rows(); ++i) {
                Zsig_.row(i) = model_.h(Xsig_.row(i).transpose()).transpose();
            }

            // Measurement mean
            VecZ z_pred = VecZ::Zero();
            for (int i = 0; i < Zsig_.rows(); ++i) {
                z_pred.noalias() += w_ * Zsig_.row(i).transpose();
            }

            // Measurement deviation with R_sqrt
            DevZ dev_z(NZ, Zsig_.rows());
            for (int i = 0; i < Zsig_.rows(); ++i) {
                dev_z.col(i) = w_sqrt_ * (Zsig_.row(i).transpose() - z_pred);
            }

            const auto Rs = model_.R_sqrt(z);
            assert(Rs.rows() == NZ && "R_sqrt rows must be NZ");
            MatZZ Sz = detail::qr_compose<NZ, Scalar>(dev_z, Rs);

            // Cross-covariance Pxz (statistical linear regression approximation)
            MatXZ Pxz = MatXZ::Zero();
            for (int i = 0; i < Xsig_.rows(); ++i) {
                const VecX dx = Xsig_.row(i).transpose() - x_it;
                const VecZ dz = Zsig_.row(i).transpose() - z_pred;
                Pxz.noalias() += w_ * (dx * dz.transpose());
            }

            // Kalman gain: K = P_xz * P_zz^{-1}, P_zz = Sz Sz^T
            MatXZ K = detail::solve_via_chol<NZ, NX, Scalar>(Sz, Pxz.transpose()).transpose();

            // Gauss-Newton/MAP iteration: z - h(x_it) + H(x_it)*(x_it - x_prior)
            // where H(x_it) ≈ P_xz^T * P_prior^{-1}
            const VecX dxp = x_it - x_prior;
            const VecX v   = detail::solve_via_chol<NX, 1, Scalar>(S_prior, dxp);
            const VecZ Hdx = Pxz.transpose() * v;

            const VecZ innov = (z - z_pred) + Hdx;
            const VecX x_new = x_prior + K * innov;

            // Save statistics for final update
            Pxz_last   = Pxz;
            Sz_last    = Sz;
            zpred_last = z_pred;

            const Scalar step_norm = (x_new - x_it).template lpNorm<Eigen::Infinity>();
            x_it                   = x_new;

            if (step_norm < iter_tol_) {
                break;
            }
        }

        // Final state update
        x_ = x_it;

        // Final covariance update (in covariance domain, then back to SR)
        const MatZZ Pzz = Sz_last * Sz_last.transpose();
        MatXZ K_last =
            detail::solve_via_chol<NZ, NX, Scalar>(Sz_last, Pxz_last.transpose()).transpose();

        MatXX P_post = P_prior - K_last * Pzz * K_last.transpose();
        Sx_          = detail::chol_from_ldlt<NX, Scalar>(P_post);
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

    VecX x_{};
    MatXX Sx_{};

    Scalar gamma_{};
    Scalar w_{};
    Scalar w_sqrt_{};

    int max_iters_{3};
    Scalar iter_tol_{Scalar(1e-6)};

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Xsig_; // (2N x NX)
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Zsig_; // (2N x NZ)

    /// Compute 2N cubature points
    void compute_cubature_points_(
        const VecX& xc, const MatXX& S,
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& Xsig) noexcept {
        Eigen::Matrix<Scalar, NX, NX> A = S * gamma_;
        for (int j = 0; j < NX; ++j) {
            Xsig.row(j)      = (xc + A.col(j)).transpose();
            Xsig.row(NX + j) = (xc - A.col(j)).transpose();
        }
    }
};

} // namespace fcs::L3::srukf
