#pragma once

#include "motion_model.hpp"
#include "util.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <numbers>

namespace fcs::L3 {

struct RobotEkfMotionModel {
    using Scalar            = double;
    static constexpr int NX = STATE_MAX;
    static constexpr int NZ = MEASURE_MAX;
    static constexpr int ARMORS_NUM = 4;
    static constexpr int kHighArmorIdA = 1;
    static constexpr int kHighArmorIdB = 3;
    static constexpr int kVyawIndex = V_YAW;

    using VecX  = Eigen::Matrix<Scalar, NX, 1>;
    using VecZ  = Eigen::Matrix<Scalar, NZ, 1>;
    using MatXX = Eigen::Matrix<Scalar, NX, NX>;
    using MatZZ = Eigen::Matrix<Scalar, NZ, NZ>;

    using Params = RobotModel::Params;
    Params params{};

    [[nodiscard]] static int clamp_armor_id(int id) noexcept {
        return std::clamp(id, 0, ARMORS_NUM - 1);
    }

    template <typename T>
    static void predict_state(const T* x, const T& dt, T* xp) noexcept {
        xp[XC]    = x[XC] + x[VX] * dt;
        xp[VX]    = x[VX];
        xp[YC]    = x[YC] + x[VY] * dt;
        xp[VY]    = x[VY];
        xp[Z0]    = x[Z0] + x[VZ] * dt;
        xp[VZ]    = x[VZ];
        xp[YAW]   = x[YAW] + x[V_YAW] * dt;
        xp[V_YAW] = x[V_YAW];
        xp[R_0]   = x[R_0];
        xp[R_1]   = x[R_1];
        xp[H]     = x[H];
    }

    template <typename T>
    static void measure_state(const T* x, int armor_id, T* z) noexcept {
        using std::atan2;
        using std::cos;
        using std::sin;
        using std::sqrt;

        const int id = clamp_armor_id(armor_id);
        const bool is_high = (id == kHighArmorIdA || id == kHighArmorIdB);
        const T angle_step =
            T(2.0 * std::numbers::pi / static_cast<double>(ARMORS_NUM));
        const T yaw = x[YAW] + T(id) * angle_step;

        const T radius = is_high ? x[R_1] : x[R_0];
        const T armor_z = is_high ? (x[Z0] + x[H]) : x[Z0];
        const T armor_x = x[XC] - radius * cos(yaw);
        const T armor_y = x[YC] - radius * sin(yaw);

        const T horizontal = sqrt(armor_x * armor_x + armor_y * armor_y);
        const T distance = sqrt(horizontal * horizontal + armor_z * armor_z);

        z[ARMOR_YAW] = atan2(armor_y, armor_x);
        z[ARMOR_PITCH] = atan2(-armor_z, horizontal);
        z[ARMOR_DISTANCE] = distance;
        z[ARMOR_YAW_ARMOR] = yaw;
    }

    [[nodiscard]] VecX f(const VecX& x, double dt) const noexcept {
        VecX xp = VecX::Zero();
        predict_state<double>(x.data(), dt, xp.data());
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

        const double dt2 = dt * dt;
        const double dt3 = dt2 * dt;
        const double dt4 = dt2 * dt2;

        const double sigma_xy2 = params.sigma_a_xy * params.sigma_a_xy;
        const double sigma_z2 = params.sigma_a_z * params.sigma_a_z;
        const double sigma_yaw2 = params.sigma_a_yaw * params.sigma_a_yaw;

        auto fill_cv = [&](int idx, double q) {
            Q(idx, idx) = q * dt4 / 4.0;
            Q(idx, idx + 1) = q * dt3 / 2.0;
            Q(idx + 1, idx) = q * dt3 / 2.0;
            Q(idx + 1, idx + 1) = q * dt2;
        };

        fill_cv(XC, sigma_xy2);
        fill_cv(YC, sigma_xy2);
        fill_cv(Z0, sigma_z2);
        fill_cv(YAW, sigma_yaw2);

        Q(R_0, R_0) = params.sigma_r0 * params.sigma_r0 * dt2;
        Q(R_1, R_1) = params.sigma_r1 * params.sigma_r1 * dt2;
        Q(H, H) = params.sigma_h * params.sigma_h * dt2;
        return Q;
    }

    [[nodiscard]] MatZZ R(const VecZ& z) const noexcept {
        MatZZ R = MatZZ::Zero();
        const double delta_angle = shortest_rad(z[ARMOR_YAW], z[ARMOR_YAW_ARMOR]);
        R(0, 0) = std::max(1e-8, params.meas_yaw_var_floor);
        R(1, 1) = std::max(1e-8, params.meas_pitch_var_floor);
        R(2, 2) = std::max(
            1e-8,
            params.meas_dist_var_floor
                + params.meas_dist_delta_angle_k * std::log(std::abs(delta_angle) + 1.0));
        R(3, 3) = std::max(
            1e-8,
            params.meas_armor_yaw_var_floor
                + params.meas_armor_yaw_range_k * std::log(std::abs(z[ARMOR_DISTANCE]) + 1.0));
        return R;
    }
};

struct OutpostEkfMotionModel {
    using Scalar            = double;
    static constexpr int NX = O_STATE_MAX;
    static constexpr int NZ = MEASURE_MAX;
    static constexpr int ARMORS_NUM = 3;
    static constexpr int kVyawIndex = O_VYAW;

    using VecX  = Eigen::Matrix<Scalar, NX, 1>;
    using VecZ  = Eigen::Matrix<Scalar, NZ, 1>;
    using MatXX = Eigen::Matrix<Scalar, NX, NX>;
    using MatZZ = Eigen::Matrix<Scalar, NZ, NZ>;

    static constexpr double OUTPOST_RADIUS = OutpostModel::OUTPOST_RADIUS;
    using Params = OutpostModel::Params;
    Params params{};

    [[nodiscard]] static int clamp_armor_id(int id) noexcept {
        return std::clamp(id, 0, ARMORS_NUM - 1);
    }

    template <typename T>
    static void predict_state(const T* x, const T& dt, T* xp) noexcept {
        // Keep consistent with ISRCKF OutpostModel state transition.
        xp[O_XC] = x[O_XC];
        xp[O_YC] = x[O_YC];
        xp[O_YAW] = x[O_YAW] + x[O_VYAW] * dt;
        xp[O_VYAW] = x[O_VYAW];
        xp[O_Z0] = x[O_Z0];
        xp[O_Z1] = x[O_Z1];
        xp[O_Z2] = x[O_Z2];
    }

    template <typename T>
    static void measure_state(const T* x, int armor_id, T* z) noexcept {
        using std::atan2;
        using std::cos;
        using std::sin;
        using std::sqrt;

        const int id = clamp_armor_id(armor_id);
        const T angle_step =
            T(2.0 * std::numbers::pi / static_cast<double>(ARMORS_NUM));
        const T yaw = x[O_YAW] + T(id) * angle_step;

        const T armor_x = x[O_XC] - T(OUTPOST_RADIUS) * cos(yaw);
        const T armor_y = x[O_YC] - T(OUTPOST_RADIUS) * sin(yaw);
        const T armor_z = x[O_Z0 + id];

        const T horizontal = sqrt(armor_x * armor_x + armor_y * armor_y);
        const T distance = sqrt(horizontal * horizontal + armor_z * armor_z);

        z[ARMOR_YAW] = atan2(armor_y, armor_x);
        z[ARMOR_PITCH] = atan2(-armor_z, horizontal);
        z[ARMOR_DISTANCE] = distance;
        z[ARMOR_YAW_ARMOR] = yaw;
    }

    [[nodiscard]] VecX f(const VecX& x, double dt) const noexcept {
        VecX xp = VecX::Zero();
        predict_state<double>(x.data(), dt, xp.data());
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

        const double dt2 = dt * dt;
        const double dt3 = dt2 * dt;
        const double dt4 = dt2 * dt2;

        const double sigma_xy2 = params.sigma_q_xy * params.sigma_q_xy;
        const double sigma_z2 = params.sigma_q_z * params.sigma_q_z;
        const double sigma_yaw2 = params.sigma_a_yaw * params.sigma_a_yaw;

        // Keep process model aligned with ISRCKF OutpostModel.
        Q(O_XC, O_XC) = sigma_xy2 * dt2;
        Q(O_YC, O_YC) = sigma_xy2 * dt2;

        Q(O_YAW, O_YAW) = sigma_yaw2 * dt4 / 4.0;
        Q(O_YAW, O_VYAW) = sigma_yaw2 * dt3 / 2.0;
        Q(O_VYAW, O_YAW) = Q(O_YAW, O_VYAW);
        Q(O_VYAW, O_VYAW) = sigma_yaw2 * dt2;

        Q(O_Z0, O_Z0) = sigma_z2 * dt2;
        Q(O_Z1, O_Z1) = sigma_z2 * dt2;
        Q(O_Z2, O_Z2) = sigma_z2 * dt2;
        return Q;
    }

    [[nodiscard]] MatZZ R(const VecZ& z) const noexcept {
        MatZZ R = MatZZ::Zero();
        const double delta_angle = shortest_rad(z[ARMOR_YAW], z[ARMOR_YAW_ARMOR]);
        // Keep measurement model aligned with ISRCKF OutpostModel::R_diag.
        R(0, 0) = 4e-3;
        R(1, 1) = 4e-3;
        R(2, 2) = std::log(std::abs(delta_angle) + 1.0) + 0.3;
        R(3, 3) = std::log(std::abs(z[ARMOR_DISTANCE]) + 1.0) / 200.0 + 9e-2;
        return R;
    }
};

} // namespace fcs::L3
