#pragma once

#include "isrckf.hpp"
#include "toml_helper.hpp"
#include "util.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <expected>
#include <numbers>

namespace fcs::L3 {

// ============================================================================
// MotionModel Concept
// ============================================================================

/// Concept for motion models compatible with SRUKF/ISRCKF
template <typename M>
concept MotionModel = requires(M m, typename M::VecX x, typename M::Scalar dt, int id) {
    typename M::Scalar;
    typename M::VecX;
    typename M::VecZ;
    { M::NX } -> std::convertible_to<int>;
    { M::NZ } -> std::convertible_to<int>;
    { M::ARMORS_NUM } -> std::convertible_to<int>;
    { m.f(x, dt) } -> std::same_as<typename M::VecX>;
    { m.h(x) } -> std::same_as<typename M::VecZ>;
    { m.h(x, id) } -> std::same_as<typename M::VecZ>;
    { m.Q_sqrt(dt) } -> std::convertible_to<Eigen::Matrix<typename M::Scalar, M::NX, M::NX>>;
    { m.R_sqrt(std::declval<typename M::VecZ>()) };
    { m.R_diag(std::declval<typename M::VecZ>()) };
};

// ============================================================================
// RobotModel: 4-armor Robot Target (11-state)
// ============================================================================

/// Motion model for tracking 4-armor robot targets
/// State: [xc, vx, yc, vy, z0, vz, yaw, v_yaw, r0, r1, h]
/// Measurement: [yaw, pitch, distance, armor_yaw] (spherical coordinates)
struct RobotModel {
    using Scalar            = double;
    static constexpr int NX = STATE_MAX;   // 11
    static constexpr int NZ = MEASURE_MAX; // 4

    using VecX  = Eigen::Matrix<Scalar, NX, 1>;
    using VecZ  = Eigen::Matrix<Scalar, NZ, 1>;
    using MatXX = Eigen::Matrix<Scalar, NX, NX>;

    // Measurement vector layout: [yaw, pitch, distance, armor_yaw]
    // The underlying indices are defined in `util.hpp` (enum `Measure`), but the names there
    // are historically confusing; use explicit aliases here for readability.
    static constexpr int kMeasYaw      = ARMOR_YAW;
    static constexpr int kMeasPitch    = ARMOR_PITCH;
    static constexpr int kMeasDistance = ARMOR_DISTANCE;
    static constexpr int kMeasArmorYaw = ARMOR_YAW_ARMOR;

    // For a 4-armor robot, armors 1 and 3 are mounted at z0 + h, while 0 and 2 are at z0.
    static constexpr int kHighArmorIdA = 1;
    static constexpr int kHighArmorIdB = 3;

    /// Process noise parameters
    struct Params {
        Scalar sigma_a_xy  = 10.0; // XY acceleration noise (match reference)
        Scalar sigma_a_z   = 1.0;  // Z acceleration noise
        Scalar sigma_a_yaw = 1.0;  // Yaw acceleration noise
        Scalar sigma_r0    = 1.0;  // Radius R0 drift noise
        Scalar sigma_r1    = 1.0;  // Radius R1 drift noise (note: reference uses sigma_r0 for R_1)
        Scalar sigma_h     = 0.5;  // Height H drift noise
        std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
            READ(tbl, sigma_a_xy);
            READ(tbl, sigma_a_z);
            READ(tbl, sigma_a_yaw);
            READ(tbl, sigma_r0);
            READ(tbl, sigma_r1);
            READ(tbl, sigma_h);
            return {};
        }
    };

    int armor_id                    = 0;
    static constexpr int ARMORS_NUM = 4;

    Params params;

    RobotModel() noexcept = default;
    explicit RobotModel(const Params& p) noexcept
        : params(p) {}

    /// State transition function (constant velocity model)
    [[nodiscard]] VecX f(const VecX& x, Scalar dt) const noexcept {
        VecX xn;
        // clang-format off
        xn << x(XC) + x(VX) * dt,
              x(VX),
              x(YC) + x(VY) * dt,
              x(VY),
              x(Z0) + x(VZ) * dt,
              x(VZ),
              x(YAW) + x(V_YAW) * dt,
              x(V_YAW),
              x(R_0),
              x(R_1),
              x(H);
        // clang-format on
        return xn;
    }

    /// Measurement function (using current armor_id)
    [[nodiscard]] VecZ h(const VecX& x) const noexcept { return h(x, armor_id); }

    /// Measurement function for specific armor
    [[nodiscard]] VecZ h(const VecX& x, int id) const noexcept {
        VecZ z;
        assert(id >= 0 && id <= ARMORS_NUM - 1);

        constexpr Scalar angle_step = 2.0 * std::numbers::pi / static_cast<Scalar>(ARMORS_NUM);

        const Scalar tmp_yaw = x(YAW) + static_cast<Scalar>(id) * angle_step;

        Scalar radius;
        Scalar armor_z;
        if (id == kHighArmorIdA || id == kHighArmorIdB) {
            armor_z = x(Z0) + x(H);
            radius  = x(R_1);
        } else {
            armor_z = x(Z0);
            radius  = x(R_0);
        }

        const Scalar armor_x = x(XC) - radius * std::cos(tmp_yaw);
        const Scalar armor_y = x(YC) - radius * std::sin(tmp_yaw);

        const auto ypd   = xyz2ypd({armor_x, armor_y, armor_z});
        z[kMeasYaw]      = ypd.x();
        z[kMeasPitch]    = ypd.y();
        z[kMeasDistance] = ypd.z();
        z[kMeasArmorYaw] = tmp_yaw;

        return z;
    }

    /// Nullspace basis of one-armor instantaneous observability (dimension = 3).
    [[nodiscard]] Eigen::Matrix<Scalar, NX, 3>
        state_observability_nullspace(const VecX& x) const noexcept {
        constexpr Scalar angle_step = 2.0 * std::numbers::pi / static_cast<Scalar>(ARMORS_NUM);
        const int id                = std::clamp(armor_id, 0, ARMORS_NUM - 1);
        const bool is_high          = (id == kHighArmorIdA || id == kHighArmorIdB);
        const Scalar psi            = x(YAW) + static_cast<Scalar>(id) * angle_step;
        const Scalar c              = std::cos(psi);
        const Scalar s              = std::sin(psi);

        Eigen::Matrix<Scalar, NX, 3> N = Eigen::Matrix<Scalar, NX, 3>::Zero();

        // Gauge 1: radial center/radius trade-off of active armor pair.
        N(XC, 0) = c;
        N(YC, 0) = s;
        N(is_high ? R_1 : R_0, 0) = Scalar(1);

        // Gauge 2: opposite pair radius is unseen in current measurement.
        N(is_high ? R_0 : R_1, 1) = Scalar(1);

        // Gauge 3: vertical ambiguity.
        if (is_high) {
            // High armor sees z0 + h; null direction keeps the sum invariant.
            N(Z0, 2) = Scalar(1);
            N(H, 2)  = Scalar(-1);
        } else {
            // Low armor does not depend on h.
            N(H, 2) = Scalar(1);
        }
        return N;
    }

    [[nodiscard]] constexpr bool use_observability_cov_projection() const noexcept { return true; }

    /// Process noise covariance square root (state-aware signature for ISRCKF compatibility).
    [[nodiscard]] Eigen::Matrix<Scalar, NX, NX> Q_sqrt(const VecX& x, Scalar dt) const noexcept {
        using MatX = Eigen::Matrix<Scalar, NX, NX>;
        MatX Q     = MatX::Zero();

        if (dt <= Scalar(0)) {
            return Q;
        }

        const Scalar sigma_a_xy2 = params.sigma_a_xy * params.sigma_a_xy;
        const Scalar sigma_a_z2  = params.sigma_a_z * params.sigma_a_z;
        // Use lower yaw acceleration noise near static and recover configured noise while spinning.
        const Scalar sigma_a_yaw_eff =
            std::min(params.sigma_a_yaw, Scalar(1.0) + Scalar(4.0) * std::abs(x(V_YAW)));
        const Scalar sigma_a_yaw2 = sigma_a_yaw_eff * sigma_a_yaw_eff;

        const Scalar dt2 = dt * dt;
        const Scalar dt3 = dt2 * dt;
        const Scalar dt4 = dt2 * dt2;

        // CV model noise for position-velocity pairs
        auto fill_block = [&](int idx_p, Scalar q) {
            Q(idx_p, idx_p)         = q * dt4 / 4.0;
            Q(idx_p, idx_p + 1)     = q * dt3 / 2.0;
            Q(idx_p + 1, idx_p)     = q * dt3 / 2.0;
            Q(idx_p + 1, idx_p + 1) = q * dt2;
        };

        fill_block(XC, sigma_a_xy2);
        fill_block(YC, sigma_a_xy2);
        fill_block(Z0, sigma_a_z2);
        fill_block(YAW, sigma_a_yaw2);

        // Random walk noise for R0, R1, H
        const Scalar sigma_r0_2 = params.sigma_r0 * params.sigma_r0;
        const Scalar sigma_r1_2 = params.sigma_r1 * params.sigma_r1;
        const Scalar sigma_h_2  = params.sigma_h * params.sigma_h;

        Q(R_0, R_0) += sigma_r0_2 * dt2;
        Q(R_1, R_1) += sigma_r1_2 * dt2;
        Q(H, H) += sigma_h_2 * dt2;

        return ckf::detail::chol_from_ldlt<NX, Scalar>(Q);
    }

    /// Process noise covariance square root (legacy signature).
    [[nodiscard]] Eigen::Matrix<Scalar, NX, NX> Q_sqrt(Scalar dt) const noexcept {
        return Q_sqrt(VecX::Zero(), dt);
    }

    /// Measurement noise diagonal (match reference: +0.3 not +1.0)
    [[nodiscard]] Eigen::Matrix<Scalar, NZ, 1> R_diag(const VecZ& z) const noexcept {
        const Scalar delta_angle = shortest_rad(z[kMeasYaw], z[kMeasArmorYaw]);
        Eigen::Matrix<Scalar, NZ, 1> R_dig;
        R_dig << 4e-3, 4e-3, std::log(std::abs(delta_angle) + 1.0) + 1.0,
            std::log(std::abs(z[kMeasDistance]) + 1.0) / 200.0 + 9e-2;
        return R_dig;
    }

    /// Measurement noise covariance square root
    [[nodiscard]] Eigen::Matrix<Scalar, NZ, Eigen::Dynamic> R_sqrt(const VecZ& z) const noexcept {
        Eigen::Matrix<Scalar, NZ, NZ> Rs = Eigen::Matrix<Scalar, NZ, NZ>::Zero();
        Rs.diagonal()                    = R_diag(z).cwiseSqrt();
        return Rs;
    }
};

static_assert(MotionModel<RobotModel>);

// ============================================================================
// OutpostModel: 3-armor Outpost Target (7-state)
// ============================================================================

/// Motion model for tracking 3-armor outpost targets
/// State: [xc, yc, yaw, v_yaw, z0, z1, z2]
/// Measurement: [yaw, pitch, distance, armor_yaw]
struct OutpostModel {
    using Scalar            = double;
    static constexpr int NX = O_STATE_MAX; // 7
    static constexpr int NZ = MEASURE_MAX; // 4

    using VecX = Eigen::Matrix<Scalar, NX, 1>;
    using VecZ = Eigen::Matrix<Scalar, NZ, 1>;

    // Measurement vector layout: [yaw, pitch, distance, armor_yaw]
    static constexpr int kMeasYaw      = ARMOR_YAW;
    static constexpr int kMeasPitch    = ARMOR_PITCH;
    static constexpr int kMeasDistance = ARMOR_DISTANCE;
    static constexpr int kMeasArmorYaw = ARMOR_YAW_ARMOR;

    // State layout: Z states are stored contiguously starting at O_Z0 (z0, z1, z2).
    static constexpr int kFirstArmorZState = O_Z0;

    /// Process noise parameters
    struct Params {

        Scalar sigma_q_xy  = 10.0;
        Scalar sigma_q_z   = 1.0;
        Scalar sigma_a_yaw = 1.0;

        // 观测噪声（和距离、yaw 误差有关）
        Scalar meas_dist_k = 0.43;  // 距离权重的系数 K
        Scalar yaw_log_k   = 0.005; // yaw 噪声前的系数
        std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
            READ(tbl, sigma_q_xy);
            READ(tbl, sigma_q_z);
            READ(tbl, sigma_a_yaw);
            READ_OPT(tbl, meas_dist_k);
            READ_OPT(tbl, yaw_log_k);
            return {};
        }
    };
    // todo
    int armor_id                           = 0;
    static constexpr Scalar OUTPOST_RADIUS = 0.2765; // 详见机械图纸
    static constexpr Scalar OUTPOST_DZ     = 0.1;    // 两个装甲板之间的铅锤高度差
    static constexpr int ARMORS_NUM        = 3;

    Params params;

    OutpostModel() noexcept = default;
    explicit OutpostModel(const Params& p) noexcept
        : params(p) {}

    /// State transition function
    [[nodiscard]] VecX f(const VecX& x, Scalar dt) const noexcept {
        VecX xn;
        // clang-format off
        xn << x[O_XC],
              x[O_YC],
              x[O_YAW] + x[O_VYAW] * dt,
              x[O_VYAW],
              x[O_Z0],
              x[O_Z1],
              x[O_Z2];
        // clang-format on
        return xn;
    }

    /// Measurement function (using current armor_id)
    [[nodiscard]] VecZ h(const VecX& x) const noexcept { return h(x, armor_id); }

    /// Measurement function for specific armor
    [[nodiscard]] VecZ h(const VecX& x, int id) const noexcept {
        VecZ z;
        assert(id >= 0 && id <= ARMORS_NUM - 1);

        constexpr Scalar angle_step = 2.0 * std::numbers::pi / static_cast<Scalar>(ARMORS_NUM);

        const Scalar tmp_yaw = x(O_YAW) + angle_step * static_cast<Scalar>(id);

        const Scalar armor_x = x(O_XC) - OUTPOST_RADIUS * std::cos(tmp_yaw);
        const Scalar armor_y = x(O_YC) - OUTPOST_RADIUS * std::sin(tmp_yaw);
        const Scalar armor_z = x(kFirstArmorZState + id);

        const auto ypd   = xyz2ypd({armor_x, armor_y, armor_z});
        z[kMeasYaw]      = ypd.x();
        z[kMeasPitch]    = ypd.y();
        z[kMeasDistance] = ypd.z();
        z[kMeasArmorYaw] = tmp_yaw;

        return z;
    }

    /// Nullspace basis of one-armor instantaneous observability (dimension = 3):
    /// [v_yaw, z_unseen_1, z_unseen_2].
    [[nodiscard]] Eigen::Matrix<Scalar, NX, 3>
        state_observability_nullspace(const VecX& x) const noexcept {
        (void)x;
        Eigen::Matrix<Scalar, NX, 3> N = Eigen::Matrix<Scalar, NX, 3>::Zero();
        const int id                   = std::clamp(armor_id, 0, ARMORS_NUM - 1);

        // Instantaneous single-frame measurement does not directly observe yaw rate.
        N(O_VYAW, 0) = Scalar(1);

        int col = 1;
        for (int i = 0; i < ARMORS_NUM; ++i) {
            if (i == id) {
                continue;
            }
            N(kFirstArmorZState + i, col) = Scalar(1);
            ++col;
        }
        return N;
    }

    /// Process noise covariance square root
    [[nodiscard]] Eigen::Matrix<Scalar, NX, NX> Q_sqrt(Scalar dt) const noexcept {
        using MatX = Eigen::Matrix<Scalar, NX, NX>;
        MatX Q     = MatX::Zero();

        if (dt <= Scalar(0)) {
            return Q;
        }

        const Scalar sigma_xy2  = params.sigma_q_xy * params.sigma_q_xy;
        const Scalar sigma_z2   = params.sigma_q_z * params.sigma_q_z;
        const Scalar sigma_yaw2 = params.sigma_a_yaw * params.sigma_a_yaw;

        const Scalar dt2 = dt * dt;
        const Scalar dt3 = dt2 * dt;
        const Scalar dt4 = dt2 * dt2;

        // 1）X、Y：只有位置，没有速度 “加性随机游走”
        //    噪声大小与 dt^2 成正比即可
        Q(O_XC, O_XC) = sigma_xy2 * dt2;
        Q(O_YC, O_YC) = sigma_xy2 * dt2;

        // 2）YAW / VYAW：是 位置+速度 对 CV模型
        //    [ yaw, vyaw ] 的连续白噪声加速度模型
        Q(O_YAW, O_YAW)   = sigma_yaw2 * dt4 / 4.0;
        Q(O_YAW, O_VYAW)  = sigma_yaw2 * dt3 / 2.0;
        Q(O_VYAW, O_YAW)  = Q(O_YAW, O_VYAW);
        Q(O_VYAW, O_VYAW) = sigma_yaw2 * dt2;

        // 3）Z0, Z1, Z2：各自随机游走（整体高度可能缓慢漂移）
        Q(O_Z0, O_Z0) = sigma_z2 * dt2;
        Q(O_Z1, O_Z1) = sigma_z2 * dt2;
        Q(O_Z2, O_Z2) = sigma_z2 * dt2;

        return ckf::detail::chol_from_ldlt<NX, Scalar>(Q);
    }

    Eigen::Matrix<Scalar, NZ, 1> R_diag(const VecZ& z) const {
        const double delta_angle = shortest_rad(z[0], z[3]);
        Eigen::Matrix<Scalar, NZ, 1> R_dig;
        R_dig << 4e-3, 4e-3, std::log(std::abs(delta_angle) + 1.0) + 1.0,
            std::log(std::abs(z[kMeasDistance]) + 1.0) / 200.0 + 9e-2;
        return R_dig;
    }

    /// Measurement noise covariance square root
    [[nodiscard]] Eigen::Matrix<Scalar, NZ, Eigen::Dynamic> R_sqrt(const VecZ& z) const noexcept {
        Eigen::Matrix<Scalar, NZ, NZ> Rs = Eigen::Matrix<Scalar, NZ, NZ>::Zero();
        Rs.diagonal()                    = R_diag(z).cwiseSqrt();
        return Rs;
    }
};

static_assert(MotionModel<OutpostModel>);

// ============================================================================
// Type Aliases
// ============================================================================

using RoboCKF    = ckf::ISRCKF<RobotModel, RobotModel::NX, RobotModel::NZ>;
using OutpostCKF = ckf::ISRCKF<OutpostModel, OutpostModel::NX, OutpostModel::NZ>;

} // namespace fcs::L3
