#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cmath>

namespace energy_meter {

// Sigmoid-mapped amplitude: a ∈ [0.780, 1.045]
inline double to_a(double alpha) { return 0.780 + 0.265 / (1.0 + std::exp(-alpha)); }

// Sigmoid-mapped angular frequency: ω ∈ [1.884, 2.000]
inline double to_w(double beta) { return 1.884 + 0.116 / (1.0 + std::exp(-beta)); }

// Hard constraint: b = 2.090 - a
inline double to_b(double a) { return 2.090 - a; }

enum EnergyState : uint8_t {
    XC,         // 0 - rune center X
    YC,         // 1 - rune center Y
    ZC,         // 2 - rune center Z
    YAW,        // 3 - rune yaw
    PITCH,      // 4 - rune pitch
    THETA,      // 5 - cumulative (unwrapped) rotation angle
    TAU,        // 6 - time since activation
    ALPHA,      // 7 - sigmoid-mapped amplitude parameter (unbounded)
    BETA,       // 8 - sigmoid-mapped angular frequency parameter (unbounded)
    STATE_MAX,
};

enum Measure : uint8_t {
    CENTER_X,   // 0
    CENTER_Y,   // 1
    CENTER_Z,   // 2
    RUNE_YAW,   // 3
    RUNE_PITCH, // 4
    ARMOR_ROLL, // 5
    MEASURE_MAX,
};

struct EnergyMeterModel {
    using Scalar                    = double;
    static constexpr int NX         = STATE_MAX;
    static constexpr int NZ         = MEASURE_MAX;
    static constexpr int ARMORS_NUM = 5;

    using VecX  = Eigen::Matrix<Scalar, NX, 1>;
    using VecZ  = Eigen::Matrix<Scalar, NZ, 1>;
    using MatXX = Eigen::Matrix<Scalar, NX, NX>;
    using MatZZ = Eigen::Matrix<Scalar, NZ, NZ>;

    struct Params {
        Scalar sigma_xy{1.0};
        Scalar sigma_z{1.0};
        Scalar sigma_yaw{1e-4};
        Scalar sigma_pitch{0.0};
        Scalar sigma_theta{0.01};
        Scalar sigma_alpha{1e-3};
        Scalar sigma_beta{1e-3};

        Scalar R_scale{0.1};
        Scalar meas_dist_k{0.43};
        Scalar radius{0.7};
    };

    Params params{};

    static constexpr double AMPLITUDE_SUM = 2.090;
    static constexpr double A_LOWER       = 0.780;
    static constexpr double A_UPPER       = 1.045;
    static constexpr double A_RANGE       = A_UPPER - A_LOWER; // 0.265
    static constexpr double W_LOWER       = 1.884;
    static constexpr double W_UPPER       = 2.000;
    static constexpr double W_RANGE       = W_UPPER - W_LOWER; // 0.116

    static int clamp_armor_id(int id) noexcept { return std::clamp(id, 0, ARMORS_NUM - 1); }

    // ── Templated predict/measure for Ceres Jet auto-diff ──

    template <typename S>
    static void predict_state(const S* x, const S& dt, int dir, S* xp) noexcept {
        using std::cos;
        using std::exp;

        // Copy all states
        for (int i = 0; i < STATE_MAX; ++i) {
            xp[i] = x[i];
        }

        // Sigmoid-mapped parameters
        const S a = S(A_LOWER) + S(A_RANGE) / (S(1.0) + exp(-x[ALPHA]));
        const S w = S(W_LOWER) + S(W_RANGE) / (S(1.0) + exp(-x[BETA]));
        const S b = S(AMPLITUDE_SUM) - a;

        // Time increment (exact)
        const S t_new = x[TAU] + dt;
        xp[TAU]       = t_new;

        // θ(t+dt) = θ(t) + dir · [(a/ω)·(cos(ω·t) - cos(ω·(t+dt))) + b·dt]
        const S delta_theta = (a / w) * (cos(w * x[TAU]) - cos(w * t_new)) + b * dt;
        xp[THETA]           = x[THETA] + S(static_cast<double>(dir)) * delta_theta;
    }

    template <typename S>
    static void measure_state(const S* x, int armor_id, S* z) noexcept {
        const int id         = clamp_armor_id(armor_id);
        const S angle_offset = S(2.0 * M_PI / 5.0 * static_cast<double>(id));

        z[CENTER_X]   = x[XC];
        z[CENTER_Y]   = x[YC];
        z[CENTER_Z]   = x[ZC];
        z[RUNE_YAW]   = x[YAW];
        z[RUNE_PITCH] = x[PITCH];
        z[ARMOR_ROLL] = x[THETA] + angle_offset;
    }

    // ── Non-templated convenience wrappers ──

    [[nodiscard]] VecX f(const VecX& x, double dt, int dir) const noexcept {
        VecX xp = VecX::Zero();
        predict_state<double>(x.data(), dt, dir, xp.data());
        return xp;
    }

    [[nodiscard]] VecZ h(const VecX& x, int armor_id) const noexcept {
        VecZ z = VecZ::Zero();
        measure_state<double>(x.data(), armor_id, z.data());
        return z;
    }

    [[nodiscard]] MatXX Q(double dt) const noexcept {
        MatXX Q = MatXX::Zero();
        if (dt <= 0.0) {
            return Q;
        }

        const double sxy2 = params.sigma_xy * params.sigma_xy;
        const double sz2  = params.sigma_z * params.sigma_z;
        const double sy2  = params.sigma_yaw * params.sigma_yaw;
        const double sp2  = params.sigma_pitch * params.sigma_pitch;
        const double st2  = params.sigma_theta * params.sigma_theta;
        const double sa2  = params.sigma_alpha * params.sigma_alpha;
        const double sb2  = params.sigma_beta * params.sigma_beta;

        Q(XC, XC)       = sxy2 * dt;
        Q(YC, YC)       = sxy2 * dt;
        Q(ZC, ZC)       = sz2 * dt;
        Q(YAW, YAW)     = sy2 * dt;
        Q(PITCH, PITCH) = sp2 * dt;
        Q(THETA, THETA) = st2 * dt;
        Q(TAU, TAU)     = 0.0; // exact
        Q(ALPHA, ALPHA) = sa2 * dt;
        Q(BETA, BETA)   = sb2 * dt;

        return Q;
    }

    [[nodiscard]] MatZZ R(const VecZ& z) const noexcept {
        const double x     = z[CENTER_X];
        const double y     = z[CENTER_Y];
        const double z_val = z[CENTER_Z];

        const double distance = std::sqrt(x * x + y * y + z_val * z_val);
        const double yaw      = std::atan2(y, x);

        const double delta_yaw = [&]() {
            double a = std::fmod(yaw - z[RUNE_YAW] + M_PI, 2.0 * M_PI);
            if (a <= 0.0)
                a += 2.0 * M_PI;
            return a - M_PI;
        }();

        const double K      = params.meas_dist_k;
        const double weight = std::log(K * distance + 1.0);

        MatZZ Rs = MatZZ::Zero();
        Rs.diagonal() << params.R_scale * (weight + std::abs(delta_yaw / 3 + 1)),
            params.R_scale * (weight + std::abs(delta_yaw / 3 + 1)), params.R_scale * weight,
            (std::log(std::abs(distance) + 1) / 200 + 9e-2), 0.05, 0.1;
        return Rs;
    }

    [[nodiscard]] Eigen::Matrix<Scalar, NZ, 1> R_diag(const VecZ& z) const {
        return R(z).diagonal();
    }
};

} // namespace energy_meter
