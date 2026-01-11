#pragma once

#include "toml_helper.hpp"
#include "util.hpp"

// Use uni_kf SRCKF instead of old ISRCKF
#include <uni_kf/presets/tracker_srckf.hpp>

#include <Eigen/Dense>
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

    using VecX = Eigen::Matrix<Scalar, NX, 1>;
    using VecZ = Eigen::Matrix<Scalar, NZ, 1>;

    // Measurement vector layout: [yaw, pitch, distance, armor_yaw]
    // The underlying indices are defined in `util.hpp` (enum `Measure`), but the names there
    // are historically confusing; use explicit aliases here for readability.
    static constexpr int kMeasYaw      = ARMOR_YAW;
    static constexpr int kMeasPitch    = ARMOR_PITCH;
    static constexpr int kMeasDistance = ARMOR_DISTANCE;
    static constexpr int kMeasArmorYaw = ARMOR_YAW_ARMOR;

    // For a 4-armor robot, armors 1 and 3 are mounted at z0 + h, while 0 and 2 are at z0.
    // NOTE: h is allowed to be negative in this implementation (clamped to [-0.2, 0.2]); a negative
    // h effectively swaps which armor pair is "higher" in the model.
    static constexpr int kHighArmorIdA = 1;
    static constexpr int kHighArmorIdB = 3;

    /// Process-noise tuning parameters for the ISRCKF.
    ///
    /// Problem:
    ///   The motion/geometry model is intentionally simple (CV + constant yaw-rate + random-walk
    ///   geometry). Real targets accelerate, spin, and their geometry is not perfectly known.
    ///
    /// Intuition:
    ///   Larger noise ⇒ prediction covariance grows faster ⇒ (1) the update becomes more
    ///   responsive, and (2) association gating becomes effectively looser. Smaller noise does the
    ///   opposite (stable but laggy and easier to drop on sharp maneuvers).
    ///
    /// Analogy:
    ///   Like a spring stiffness knob: stiff (small noise) is stable; soft (large noise) is agile
    ///   but wobbly.
    ///
    /// Precise:
    ///   Each `sigma_*` is a standard deviation. `Q_sqrt(dt)` squares it and discretizes it with
    ///   dt-scaling (dt^2..dt^4 for CV blocks, dt for 1D random walks).
    struct Params {
        /// XY acceleration process-noise std-dev (≈ m/s^2).
        ///
        /// Problem: the state assumes constant velocity in x/y; real targets accelerate.
        /// Intuition: larger → follow fast maneuvers (but noisier); smaller → smoother but laggier.
        /// Analogy: "agility" in the horizontal plane.
        /// Precise: used as q = sigma_a_xy^2 in the CV blocks (xc,vx) and (yc,vy) in `Q_sqrt(dt)`.
        Scalar sigma_a_xy = 10.0;
        /// Z acceleration process-noise std-dev (≈ m/s^2).
        ///
        /// Problem/Intuition/Analogy: same as sigma_a_xy, but only for z (z0,vz).
        /// Precise: used as q = sigma_a_z^2 in the CV block (z0,vz) in `Q_sqrt(dt)`.
        Scalar sigma_a_z = 1.0;
        /// Yaw acceleration noise std-dev (≈ rad/s^2) for the [yaw, v_yaw] CV model.
        ///
        /// Problem: constant yaw-rate cannot explain sudden spin-up / spin-down.
        /// Intuition: larger → track fast turns (but yaw becomes noisier); smaller → smoother yaw
        ///           but more lag under quick turns.
        /// Analogy: "turn-rate agility" knob.
        /// Precise: used as q = sigma_a_yaw^2 in the CV block (yaw,v_yaw) in `Q_sqrt(dt)`.
        Scalar sigma_a_yaw = 1.0;
        /// Radius r0 drift noise std-dev (≈ m/√s) for random walk.
        ///
        /// Problem: geometry parameters (center→armor radius) are not perfectly known and may
        ///          drift due to calibration bias / model mismatch.
        /// Intuition: larger → adapt geometry faster (but may drift); smaller → keep geometry
        /// stable. Analogy: allowing the "shape" knob to wander slowly. Precise: enters the
        /// random-walk term Q(R_0,R_0) += sigma_r0^2 * dt in `Q_sqrt(dt)`.
        Scalar sigma_r0 = 1.0;
        /// Radius r1 drift noise std-dev (≈ m/√s) for random walk.
        ///
        /// Problem/Intuition/Analogy: same as sigma_r0, but for the other armor pair radius r1.
        /// Precise: enters Q(R_1,R_1) += sigma_r1^2 * dt in `Q_sqrt(dt)`.
        Scalar sigma_r1 = 1.0;
        /// Height h drift noise std-dev (≈ m/√s) for random walk.
        ///
        /// Problem: the vertical offset between armor pairs can be biased and slowly varying.
        /// Intuition: larger → h adapts faster (but can wander); smaller → more stable height.
        /// Analogy: letting a slow "elevation bias" parameter move.
        /// Precise: enters Q(H,H) += sigma_h^2 * dt in `Q_sqrt(dt)`.
        Scalar sigma_h = 0.5;
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

    /// Process noise covariance square root
    [[nodiscard]] Eigen::Matrix<Scalar, NX, NX> Q_sqrt(Scalar dt) const noexcept {
        using MatX = Eigen::Matrix<Scalar, NX, NX>;
        MatX Q     = MatX::Zero();

        if (dt <= Scalar(0)) {
            return Q;
        }

        const Scalar sigma_a_xy2  = params.sigma_a_xy * params.sigma_a_xy;
        const Scalar sigma_a_z2   = params.sigma_a_z * params.sigma_a_z;
        const Scalar sigma_a_yaw2 = params.sigma_a_yaw * params.sigma_a_yaw;

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

        Q(R_0, R_0) += sigma_r0_2 * dt;
        Q(R_1, R_1) += sigma_r1_2 * dt;
        Q(H, H) += sigma_h_2 * dt;

        Eigen::LLT<MatX> llt(Q);
        return llt.matrixL();
    }

    /// Measurement noise diagonal (variance, not std-dev).
    ///
    /// Problem:
    ///   The ISRCKF (and the associator gate) need a measurement-noise model to weight residuals.
    ///   Instead of a calibrated camera/PnP covariance, this project uses a hand-tuned heuristic.
    ///
    /// Intuition:
    ///   - Treat yaw/pitch as "roughly equally noisy" all the time.
    ///   - Trust distance less when the armor is more side-facing (bearing yaw differs from armor
    ///   yaw).
    ///   - Trust armor_yaw less when the target is far (PnP yaw becomes unstable).
    ///
    /// Analogy:
    ///   Dynamic confidence: the sensor "gets worse" when viewing geometry is poor.
    ///
    /// Precise:
    ///   - yaw, pitch: fixed variance 4e-3.
    ///   - distance: variance = log(|delta_angle| + 1) + 1,
    ///               delta_angle = shortest_rad(z[yaw], z[armor_yaw]).
    ///   - armor_yaw: variance = log(|distance| + 1) / 200 + 9e-2.
    ///
    /// Note:
    ///   This is NOT configurable from TOML. It affects both:
    ///   - ISRCKF update (via `R_sqrt`)
    ///   - Data association gating: d^2 = nu^T (Pzz + R)^{-1} nu
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

    /// Process-noise tuning parameters for the Outpost ISRCKF.
    ///
    /// Problem:
    ///   The outpost model does not include vx/vy states (x/y are constant in `f`). Without
    ///   process noise, the filter cannot adapt to slow motion, TF bias, or measurement drift.
    ///
    /// Intuition:
    ///   Larger noise → allow the center position / heights to move more freely (responsive but
    ///   noisier and with looser association); smaller noise → "sticky" estimates (stable but can
    ///   lag and drop matches when the world shifts).
    ///
    /// Analogy:
    ///   Like adding a random "wind" that can push the outpost center a bit each frame.
    ///
    /// Precise:
    ///   Naming can be misleading here: `sigma_q_xy` and `sigma_q_z` behave like drift-velocity
    ///   scales because Q ∝ dt^2 (see `Q_sqrt`).
    struct Params {
        /// XY drift-velocity process-noise std-dev (≈ m/s).
        ///
        /// Problem:
        ///   With no vx/vy states, x/y would otherwise be "frozen" unless measurements pull it.
        ///
        /// Intuition:
        ///   Larger → position follows measurements faster (but jitters and matches more loosely);
        ///   smaller → position is stable (but can lag or fail under frame/TF shifts).
        ///
        /// Analogy:
        ///   Allowing a random horizontal "drift speed" that accumulates over dt.
        ///
        /// Precise (why the field name can be misleading):
        ///   In `Q_sqrt(dt)`, Q(O_XC,O_XC) = sigma_q_xy^2 * dt^2 and same for Y, which corresponds
        ///   to x_{k+1} = x_k + v_drift * dt with v_drift ~ N(0, sigma_q_xy^2).
        Scalar sigma_q_xy = 10.0;
        /// Z drift-velocity process-noise std-dev (≈ m/s), applied to z0/z1/z2.
        ///
        /// Problem/Intuition/Analogy: same as sigma_q_xy, but for the per-armor height states.
        /// Precise: in `Q_sqrt(dt)`, Q(z_i,z_i) = sigma_q_z^2 * dt^2 for i ∈ {0,1,2}.
        Scalar sigma_q_z = 1.0;
        /// Yaw acceleration noise std-dev (≈ rad/s^2) for the [yaw, v_yaw] CV model.
        ///
        /// Problem/Intuition/Analogy: same as RobotModel::Params::sigma_a_yaw.
        /// Precise: used as q = sigma_a_yaw^2 in the CV block (yaw,v_yaw) in `Q_sqrt(dt)`.
        Scalar sigma_a_yaw = 1.0;
        std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
            READ(tbl, sigma_q_xy);
            READ(tbl, sigma_q_z);
            READ(tbl, sigma_a_yaw);
            return {};
        }
    };

    int armor_id                           = 0;
    static constexpr Scalar OUTPOST_RADIUS = 0.2765; // From mechanical drawing
    // NOTE: kept for reference; the filter currently estimates z0/z1/z2 from measurements
    // and does not use a fixed OUTPOST_DZ in the model.
    static constexpr Scalar OUTPOST_DZ = 0.1; // Height difference between armors
    static constexpr int ARMORS_NUM    = 3;

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

        // X, Y: drift-like random walk (Q ∝ dt^2, see Params::sigma_q_xy note)
        Q(O_XC, O_XC) = sigma_xy2 * dt2;
        Q(O_YC, O_YC) = sigma_xy2 * dt2;

        // YAW/VYAW: CV model
        Q(O_YAW, O_YAW)   = sigma_yaw2 * dt4 / 4.0;
        Q(O_YAW, O_VYAW)  = sigma_yaw2 * dt3 / 2.0;
        Q(O_VYAW, O_YAW)  = Q(O_YAW, O_VYAW);
        Q(O_VYAW, O_VYAW) = sigma_yaw2 * dt2;

        // Z0, Z1, Z2: drift-like random walk (Q ∝ dt^2, see Params::sigma_q_z note)
        Q(O_Z0, O_Z0) = sigma_z2 * dt2;
        Q(O_Z1, O_Z1) = sigma_z2 * dt2;
        Q(O_Z2, O_Z2) = sigma_z2 * dt2;

        Eigen::LLT<MatX> llt(Q);
        return llt.matrixL();
    }

    /// Measurement noise diagonal (variance, not std-dev).
    ///
    /// Problem/Intuition/Analogy:
    ///   Same heuristic as `RobotModel::R_diag`; see that function for rationale and tuning impact.
    ///
    /// Precise:
    ///   Used by ISRCKF update (via `R_sqrt`) and by DataAssociator gating (via S = Pzz + R).
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

static_assert(MotionModel<OutpostModel>);

// ============================================================================
// Type Aliases
// ============================================================================

using RoboCKF    = uni_kf::SRCKF<RobotModel, RobotModel::NX, RobotModel::NZ>;
using OutpostCKF = uni_kf::SRCKF<OutpostModel, OutpostModel::NX, OutpostModel::NZ>;

} // namespace fcs::L3
