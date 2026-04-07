#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <expected>
#include <limits>
#include <string>
#include <vector>

namespace fcs::L5 {

/// Specialized ADMM-based MPC solver for double-integrator systems (nx=2, nu=1).
///
/// Optimized for ARM Cortex-A55:
/// - **float** arithmetic for hardware FP (A55 has no hardware double)
/// - **Fixed-size** Eigen types — zero heap allocation in solve path
/// - Pre-computed Riccati gains (Kinf, Pinf, Quu_inv, AmBKt)
/// - Entire solve fits in L1 cache for typical horizons
///
/// ## System Model
///
/// Double-integrator dynamics:
///
///     x_{k+1} = A * x_k + B * u_k + f
///
///     A = [[1, dt], [0, 1]]    x = [position; velocity]
///     B = [[0], [dt]]          u = [acceleration]
///
/// ## Safety
///
/// - Move-only (no copy, no shared ownership)
/// - All public APIs are noexcept or return std::expected
/// - solve() performs no heap allocation; all workspace is pre-allocated at construction
/// - float→double conversion at API boundaries for interop with the rest of the system
class SmallMpcSolver {
public:
    static constexpr int NX = 2;
    static constexpr int NU = 1;

    using State = Eigen::Matrix<float, NX, 1>;
    using Input = Eigen::Matrix<float, NU, 1>;

    /// Per-axis tuning parameters.
    struct AxisConfig {
        float q_pos{9e6f};
        float q_vel{0.0f};
        float r{1.0f};
        float max_acc{50.0f};
        float state_min{-std::numeric_limits<float>::infinity()};
        float state_max{std::numeric_limits<float>::infinity()};
        bool enable_state_bound{false};
    };

    /// Construct via create(). Default ctor exists only for move-assignment target.
    SmallMpcSolver() = default;

    SmallMpcSolver(const SmallMpcSolver&)                = delete;
    SmallMpcSolver& operator=(const SmallMpcSolver&)     = delete;
    SmallMpcSolver(SmallMpcSolver&&) noexcept            = default;
    SmallMpcSolver& operator=(SmallMpcSolver&&) noexcept = default;
    ~SmallMpcSolver()                                    = default;

    /// Factory: pre-compute Riccati gains and allocate workspace.
    [[nodiscard]] static std::expected<SmallMpcSolver, std::string>
        create(float dt, int horizon, float rho, const AxisConfig& axis) {
        if (dt <= 0.0f) {
            return std::unexpected("dt must be positive");
        }
        if (horizon < 2) {
            return std::unexpected("horizon must be >= 2");
        }

        SmallMpcSolver s;
        s.N_                  = horizon;
        s.rho_                = rho;
        s.enable_state_bound_ = axis.enable_state_bound;

        // --- dynamics ---
        s.A_ << 1.0f, dt, 0.0f, 1.0f;
        s.B_ << 0.0f, dt;
        s.f_.setZero();

        // --- augmented cost ---
        const State Q_aug{axis.q_pos + rho, axis.q_vel + rho};
        const float R_aug = axis.r + rho;

        // --- Riccati recursion ---
        using Mat2 = Eigen::Matrix2f;
        using Row2 = Eigen::Matrix<float, 1, 2>;

        Mat2 P      = rho * Mat2::Identity();
        Row2 K_prev = Row2::Zero();
        Row2 K      = Row2::Zero();
        Mat2 Pinf   = Mat2::Zero();

        for (int i = 0; i < 1000; ++i) {
            const float S     = R_aug + (s.B_.transpose() * P * s.B_)(0);
            const float S_inv = 1.0f / S;
            K                 = S_inv * (s.B_.transpose() * P) * s.A_;
            Pinf = Q_aug.asDiagonal().toDenseMatrix() + s.A_.transpose() * P * (s.A_ - s.B_ * K);
            if ((K - K_prev).cwiseAbs().maxCoeff() < 1e-5f) {
                break;
            }
            K_prev = K;
            P      = Pinf;
        }

        s.Kinf_ = K;
        s.Pinf_ = Pinf;

        const float Quu = R_aug + (s.B_.transpose() * Pinf * s.B_)(0);
        s.Quu_inv_      = 1.0f / Quu;

        s.AmBKt_ = (s.A_ - s.B_ * K).transpose();
        s.APf_   = s.AmBKt_ * Pinf * s.f_;
        s.BPf_   = (s.B_.transpose() * Pinf * s.f_)(0);

        s.Q_diag_ = State{axis.q_pos, axis.q_vel};
        s.R_val_  = axis.r;

        // --- workspace ---
        s.alloc(horizon, axis);
        return s;
    }

    // ----- configuration -----

    void set_settings(float abs_tol, int max_iter) noexcept {
        abs_tol_  = abs_tol;
        max_iter_ = max_iter;
    }

    /// Set initial state (double interface for interop).
    void set_x0(double pos, double vel) noexcept {
        x_[0] = State{static_cast<float>(pos), static_cast<float>(vel)};
    }

    /// Set reference trajectory column-by-column (float interface).
    void set_ref_col(int k, float pos, float vel) noexcept { Xref_[k] = State{pos, vel}; }

    /// Bulk-set reference from a 2×N float matrix.
    void set_reference(const Eigen::Ref<const Eigen::MatrixXf>& ref) noexcept {
        for (int i = 0; i < N_; ++i) {
            Xref_[i] = ref.col(i);
        }
    }

    // ----- solve -----

    /// Run ADMM iterations. Returns true if converged within tolerance.
    bool solve() noexcept {
        converged_ = false;
        iters_     = 0;

        for (int iter = 0; iter < max_iter_; ++iter) {
            step_linear_cost();
            step_backward();
            step_forward();
            step_slack();
            step_dual();
            ++iters_;

            if ((iters_ % check_term_) == 0 && check_converged()) {
                converged_ = true;
                return true;
            }

            // swap prev slack
            std::swap(v_, vnew_);
            std::swap(z_, znew_);
        }
        return false;
    }

    // ----- accessors -----

    [[nodiscard]] float state(int dim, int k) const noexcept { return x_[k](dim); }
    [[nodiscard]] float input(int k) const noexcept { return u_[k](0); }
    [[nodiscard]] bool converged() const noexcept { return converged_; }
    [[nodiscard]] int iterations() const noexcept { return iters_; }
    [[nodiscard]] int horizon() const noexcept { return N_; }

private:
    // dynamics
    Eigen::Matrix2f A_;
    Eigen::Matrix<float, 2, 1> B_;
    State f_;

    // cost diagonal
    State Q_diag_;
    float R_val_{1.0f};

    // Riccati cache
    Eigen::Matrix<float, 1, 2> Kinf_;
    Eigen::Matrix2f Pinf_;
    float Quu_inv_{1.0f};
    Eigen::Matrix2f AmBKt_;
    State APf_;
    float BPf_{0.0f};
    float rho_{1.0f};

    // workspace vectors (pre-allocated, no allocation in solve())
    int N_{0};
    std::vector<State> x_, q_, p_, v_, vnew_, g_, Xref_;
    std::vector<Input> u_, r_, d_, z_, znew_, y_;
    std::vector<State> x_min_, x_max_;
    std::vector<Input> u_min_, u_max_;
    bool enable_state_bound_{false};

    // settings
    float abs_tol_{1e-3f};
    int max_iter_{10};
    int check_term_{1};

    // result
    bool converged_{false};
    int iters_{0};

    void alloc(int N, const AxisConfig& axis) {
        x_.assign(N, State::Zero());
        u_.assign(N - 1, Input::Zero());
        q_.assign(N, State::Zero());
        r_.assign(N - 1, Input::Zero());
        p_.assign(N, State::Zero());
        d_.assign(N - 1, Input::Zero());
        v_.assign(N, State::Zero());
        vnew_.assign(N, State::Zero());
        z_.assign(N - 1, Input::Zero());
        znew_.assign(N - 1, Input::Zero());
        g_.assign(N, State::Zero());
        y_.assign(N - 1, Input::Zero());
        Xref_.assign(N, State::Zero());

        x_min_.assign(N, State::Constant(-1e17f));
        x_max_.assign(N, State::Constant(1e17f));
        u_min_.assign(N - 1, Input::Constant(-axis.max_acc));
        u_max_.assign(N - 1, Input::Constant(axis.max_acc));

        if (axis.enable_state_bound) {
            for (auto& s : x_min_) {
                s(0) = axis.state_min;
            }
            for (auto& s : x_max_) {
                s(0) = axis.state_max;
            }
        }
    }

    void step_linear_cost() noexcept {
        for (int i = 0; i < N_; ++i) {
            q_[i].noalias() =
                -(Q_diag_.array() * Xref_[i].array()).matrix() - rho_ * (vnew_[i] - g_[i]);
        }
        for (int i = 0; i < N_ - 1; ++i) {
            r_[i](0) = -(R_val_ * 0.0f) - rho_ * (znew_[i](0) - y_[i](0));
        }
        // terminal
        p_[N_ - 1].noalias() = -(Pinf_ * Xref_[N_ - 1]) - rho_ * (vnew_[N_ - 1] - g_[N_ - 1]);
    }

    void step_backward() noexcept {
        for (int i = N_ - 2; i >= 0; --i) {
            // d[i] = Quu_inv * (B^T * p[i+1] + r[i] + BPf)
            const float Btp = B_(0) * p_[i + 1](0) + B_(1) * p_[i + 1](1);
            d_[i](0)        = Quu_inv_ * (Btp + r_[i](0) + BPf_);

            // p[i] = q[i] + AmBKt * p[i+1] - Kinf^T * r[i] + APf
            p_[i].noalias() = q_[i] + AmBKt_ * p_[i + 1] - (Kinf_.transpose() * r_[i](0)) + APf_;
        }
    }

    void step_forward() noexcept {
        for (int i = 0; i < N_ - 1; ++i) {
            // u[i] = -Kinf * x[i] - d[i]
            u_[i](0) = -(Kinf_(0) * x_[i](0) + Kinf_(1) * x_[i](1)) - d_[i](0);
            // x[i+1] = A * x[i] + B * u[i] + f
            x_[i + 1].noalias() = A_ * x_[i] + B_ * u_[i](0) + f_;
        }
    }

    void step_slack() noexcept {
        for (int i = 0; i < N_; ++i) {
            vnew_[i] = x_[i] + g_[i];
            if (enable_state_bound_) {
                vnew_[i] = vnew_[i].cwiseMax(x_min_[i]).cwiseMin(x_max_[i]);
            }
        }
        for (int i = 0; i < N_ - 1; ++i) {
            float zn    = u_[i](0) + y_[i](0);
            zn          = std::clamp(zn, u_min_[i](0), u_max_[i](0));
            znew_[i](0) = zn;
        }
    }

    void step_dual() noexcept {
        for (int i = 0; i < N_; ++i) {
            g_[i] += x_[i] - vnew_[i];
        }
        for (int i = 0; i < N_ - 1; ++i) {
            y_[i](0) += u_[i](0) - znew_[i](0);
        }
    }

    [[nodiscard]] bool check_converged() const noexcept {
        float pri_s = 0, pri_u = 0, dua_s = 0, dua_u = 0;

        for (int i = 0; i < N_; ++i) {
            pri_s = std::max(pri_s, (x_[i] - vnew_[i]).cwiseAbs().maxCoeff());
            dua_s = std::max(dua_s, (v_[i] - vnew_[i]).cwiseAbs().maxCoeff());
        }
        dua_s *= rho_;

        for (int i = 0; i < N_ - 1; ++i) {
            pri_u = std::max(pri_u, std::abs(u_[i](0) - znew_[i](0)));
            dua_u = std::max(dua_u, std::abs(z_[i](0) - znew_[i](0)));
        }
        dua_u *= rho_;

        return pri_s < abs_tol_ && pri_u < abs_tol_ && dua_s < abs_tol_ && dua_u < abs_tol_;
    }
};

} // namespace fcs::L5
