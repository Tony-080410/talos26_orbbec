#pragma once

#include "toml_helper.hpp"
#include <expected>
#include <toml++/toml.hpp>

namespace fcs::L4 {

// ============================================================================
// Trajectory Configuration (L4 Layer - Master)
// ============================================================================

// ----------------------------------------------------------------------------
// Planner tuning guide
// ----------------------------------------------------------------------------
//
// Problem:
//   The L4 gimbal planner converts the L3 tracker state into yaw/pitch commands and a fire
//   decision. The core difficulty is time alignment + ballistics: you must aim where the target
//   will be when the bullet arrives.
//
// Intuition:
//   - Get timing right first (`plan_to_muzzle_delay`, `bullet_speed`), otherwise the lead is wrong
//     and everything else looks like "magic".
//   - Then tune armor selection and mode switching (`side_angle`, `min_switching_v_yaw`,
//     `max_tracking_v_yaw`, `transfer_thresh`).
//   - Finally tune the ballistics model details (`resistance` or RK4 parameters).
//
// Analogy:
//   Like throwing a ball at a moving target: lead by (reaction time + flight time).
//
// Precise (implementation facts; see `planner.hpp::GimbalPlanner::plan`):
//   - dt = (now - measurement_ts) + flying_time + plan_to_muzzle_delay
//   - `now` comes from `clock::now_ns()` (real-time), not from the measurement timestamp.
//   - `flying_time` is first estimated cheaply using LOS-to-center, then refined by the configured
//     trajectory compensator (Ideal/Resistance/RK4).
//   - TOML parsing is strict: `READ(...)` makes keys required; only fields read with
//   `READ_OPT(...)`
//     can be omitted and fall back to struct defaults.

enum class TrajectoryCompensatorType {
    Ideal,
    Resistance,
    RK4,
};

struct TrajectoryConfig {
    /// Number of iterations for pitch solving / compensation loop.
    ///
    /// Problem:
    ///   The compensator solves for pitch such that the simulated impact height matches the target
    ///   height. This is an iterative root-finding loop.
    ///
    /// Intuition:
    ///   Larger → more chances to converge (robust but slower); smaller → faster but may fail to
    ///   converge, especially at long range / steep pitch.
    ///
    /// Analogy:
    ///   Like "max iterations" in an optimizer.
    ///
    /// Precise:
    ///   - Ideal/Resistance: used as the loop bound in `TrajectoryCompensator::compensate(pitch)`.
    ///   - RK4: with a lookup table, runtime uses table inversion + up to 1~2 RK4 refinements;
    ///          without a table, the fallback loop uses `min(iteration_times, 6)` to avoid
    ///          expensive real-time RK4 iterations.
    int iteration_times{20};

    /// Bullet velocity at muzzle (m/s).
    ///
    /// Problem:
    ///   Lead and drop both depend strongly on bullet speed. A wrong value makes the aim wrong at
    ///   all distances.
    ///
    /// Intuition:
    ///   Higher speed → less flight time → less lead and less drop; lower speed → more lead/drop.
    ///
    /// Analogy:
    ///   Like how fast you can throw a ball: faster throws need less "leading".
    ///
    /// Precise:
    ///   - Used by all compensators as `velocity`.
    ///   - Also used in `GimbalPlanner::plan` for a cheap initial flying-time estimate:
    ///       t ≈ range / (max(1e-3, bullet_speed) * cos(pitch_los_center)).
    double bullet_speed{25.0};

    /// Gravity acceleration (m/s^2)
    ///
    /// Problem:
    ///   Gravity drives the vertical drop term; an incorrect value biases long-range pitch.
    ///
    /// Intuition:
    ///   Larger gravity → more drop → need higher pitch; smaller gravity → less drop.
    ///
    /// Analogy:
    ///   Like playing the same throw on Earth vs the Moon.
    ///
    /// Precise:
    ///   Used in all compensators in the vertical acceleration term (ideal and with drag).
    double gravity{9.8};

    /// Linearized air resistance coefficient (used by ResistanceCompensator only).
    ///
    /// Problem:
    ///   The simplified "Resistance" model uses an exponential term to approximate drag without
    ///   running RK4. This parameter must be fitted to your projectile + environment.
    ///
    /// Intuition:
    ///   Larger resistance → more deceleration → longer flight time and more drop (need more
    ///   pitch).
    ///
    /// Analogy:
    ///   Like a coarse "air thickness" knob for a simplified model.
    ///
    /// Precise:
    ///   Used only by `ResistanceCompensator`. The implementation clamps it to >= 1e-4, so using
    ///   0.0 does NOT disable drag in this model.
    double resistance{0.001};

    /// Drag coefficient Cd (dimensionless), used by RK4Compensator only.
    ///
    /// Problem:
    ///   RK4 uses a quadratic drag model. Cd is part of the drag constant k.
    ///
    /// Intuition:
    ///   Larger Cd → more drag → more drop and longer TOF.
    ///
    /// Analogy:
    ///   Like how "blunt" the projectile is in the air.
    ///
    /// Precise:
    ///   Used in `RK4Compensator::drag_k()`:
    ///     k = 0.5 * cd * rho * A / m, with A = pi*(diameter/2)^2.
    double cd{0.47};

    /// Air density rho (kg/m^3), used by RK4Compensator only.
    ///
    /// Problem:
    ///   Air density changes with altitude/temperature and affects drag in the RK4 model.
    ///
    /// Intuition:
    ///   Higher rho → more drag; lower rho → less drag.
    ///
    /// Analogy:
    ///   Like running in thick fog vs thin air.
    ///
    /// Precise:
    ///   Used in `RK4Compensator::drag_k()` as part of k.
    double rho{1.225};

    /// Bullet diameter (meters), used by RK4Compensator only to compute cross-sectional area A.
    ///
    /// Problem:
    ///   Drag scales with cross-sectional area A.
    ///
    /// Intuition:
    ///   Larger diameter → larger A → more drag.
    ///
    /// Analogy:
    ///   Like a parachute: more area catches more air.
    ///
    /// Precise:
    ///   Used in `RK4Compensator::drag_k()` via A = pi*(diameter/2)^2 (with a small lower bound).
    double diameter{0.017};

    /// Bullet mass (kg), used by RK4Compensator only.
    ///
    /// Problem:
    ///   Drag acceleration is inversely proportional to mass.
    ///
    /// Intuition:
    ///   Smaller mass → slows down faster under the same drag; larger mass → keeps speed better.
    ///
    /// Analogy:
    ///   Like pushing a ping-pong ball vs a stone through the air.
    ///
    /// Precise:
    ///   Used in `RK4Compensator::drag_k()` in the denominator (with a small lower bound).
    double mass{0.0032};

    /// Desired RK4 integration dt (seconds), used by RK4Compensator only.
    ///
    /// Problem:
    ///   RK4 integration accuracy depends on step size, but compute cost grows with the number of
    ///   steps.
    ///
    /// Intuition:
    ///   Smaller → more accurate but slower; larger → faster but less accurate (especially at long
    ///   range / steep pitch / high drag).
    ///
    /// Analogy:
    ///   Like choosing how fine the timestep is in a physics simulation.
    ///
    /// Precise:
    ///   The implementation clamps `rk4_dt` into [1e-5, 0.02] and computes:
    ///     steps = clamp(ceil(t_est / desired_dt), rk4_min_steps, rk4_max_steps)
    ///     dt    = t_est / steps
    ///   where t_est ≈ range / (max(1e-3, velocity) * max(0.05, |cos(pitch)|)).
    double rk4_dt{0.001};

    /// Minimum RK4 steps per integration, used by RK4Compensator only.
    ///
    /// Problem:
    ///   For short flight times, a naive `t_est / rk4_dt` can produce too few steps, making RK4
    ///   numerically unstable.
    ///
    /// Intuition:
    ///   Larger → more stable/accurate at short range but higher compute cost.
    ///
    /// Analogy:
    ///   A minimum "resolution" setting.
    ///
    /// Precise:
    ///   Used in `RK4Compensator::select_dt()` as the lower bound on `steps`.
    int rk4_min_steps{50};

    /// Maximum RK4 steps per integration, used by RK4Compensator only.
    ///
    /// Problem:
    ///   Very long flight times can require many RK4 steps, which may be too expensive for a
    ///   real-time loop.
    ///
    /// Intuition:
    ///   Smaller → cap compute cost but risk poor accuracy at long range; larger → better accuracy
    ///   but potentially slower.
    ///
    /// Analogy:
    ///   A hard "CPU budget" cap.
    ///
    /// Precise:
    ///   Used in `RK4Compensator::select_dt()` as the upper bound on `steps` (and thus on runtime).
    int rk4_max_steps{2000};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ(tbl, iteration_times);
        READ(tbl, bullet_speed);
        READ(tbl, gravity);
        READ(tbl, resistance);
        READ_OPT(tbl, cd);
        READ_OPT(tbl, rho);
        READ_OPT(tbl, diameter);
        READ_OPT(tbl, mass);
        READ_OPT(tbl, rk4_dt);
        READ_OPT(tbl, rk4_min_steps);
        READ_OPT(tbl, rk4_max_steps);
        return {};
    }
};

// ============================================================================
// Gimbal Planner Configuration (L4 Layer)
// ============================================================================

/// Gimbal planning configuration
struct GimbalPlannerConfig {
    /// Fire-advice window width (meters), projected to yaw tolerance by atan2(w/2, distance).
    ///
    /// Problem:
    ///   We need a practical "close enough to shoot" test for fire advice.
    ///
    /// Intuition:
    ///   This is a physical window size, not a distance/range. The same window corresponds to a
    ///   larger angular tolerance when the target is near and a smaller tolerance when far.
    ///
    /// Analogy:
    ///   Like defining the size of a bullseye on the target, then converting it to an angle from
    ///   the muzzle.
    ///
    /// Precise (naming pitfall):
    ///   Used in `GimbalPlanner::is_on_target()` as:
    ///     yaw_tol = atan2(shooting_range_w/2, distance)
    ///   with an additional lower bound of 1 degree to avoid "never fire" at extreme distances.
    double shooting_range_w{0.135};

    /// Fire-advice window height (meters), projected to pitch tolerance by atan2(h/2, distance).
    ///
    /// Problem/Intuition/Analogy:
    ///   Same as shooting_range_w, but for pitch.
    ///
    /// Precise:
    ///   Used in `is_on_target()` as:
    ///     pitch_tol = atan2(shooting_range_h/2, distance)
    ///   with the same 1-degree minimum.
    double shooting_range_h{0.135};

    /// Threshold yaw-rate for switching TrackingArmor <-> TrackingCenter (rad/s).
    ///
    /// Problem:
    ///   When the target spins fast, individual armors swap quickly and "tracking an armor" can
    ///   jitter. A center-tracking mode is more stable.
    ///
    /// Intuition:
    ///   Larger threshold → stay in armor-tracking longer (more aggressive but can jitter on fast
    ///   spins); smaller threshold → switch to center-tracking more easily (stable but may aim less
    ///   optimally at a specific armor).
    ///
    /// Analogy:
    ///   Like switching from "track a wheel spoke" to "track the wheel hub" when the wheel spins.
    ///
    /// Precise (naming pitfall):
    ///   This is a MODE SWITCH THRESHOLD (TrackingArmor <-> TrackingCenter), not a "max allowed yaw
    ///   rate". The condition is applied with a consecutive-frame counter (see transfer_thresh).
    double max_tracking_v_yaw{6.0};

    /// Extra latency from `now()` to bullet leaving muzzle (seconds).
    ///
    /// Problem:
    ///   The bullet does not leave the muzzle instantly after "now". Control/actuation delays add
    ///   extra time that must be included in lead.
    ///
    /// Intuition:
    ///   Larger → aim further ahead (more lead); smaller → aim less ahead.
    ///
    /// Analogy:
    ///   Like accounting for your reaction time before you throw a ball.
    ///
    /// Precise:
    ///   In `GimbalPlanner::plan`, dt includes:
    ///     dt = (now - measurement_ts) + flying_time + plan_to_muzzle_delay
    ///   This term does NOT include flying_time; flying_time is added separately.
    double plan_to_muzzle_delay{0.0};

    /// Armor-selection bias angle (degrees) used in select_best_armor().
    ///
    /// Problem:
    ///   When the target rotates, the "best" armor at this instant may not be the best armor at
    ///   bullet-arrival time. A directional bias helps pick the armor that is likely to remain
    ///   visible/centered.
    ///
    /// Intuition:
    ///   Larger → switch armors earlier in the rotation direction (more predictive but can switch
    ///   too aggressively); smaller → less predictive (more stable but may lag behind rotation).
    ///
    /// Analogy:
    ///   Like aiming slightly ahead of a moving doorway so the ball passes through when it arrives.
    ///
    /// Precise:
    ///   Used in `select_best_armor()` as an angular bias theta applied with the sign of v_yaw:
    ///     theta = +/- side_angle (degrees → radians). When |v_yaw| is small, it is disabled by
    ///     `min_switching_v_yaw` to avoid jitter near boundaries.
    double side_angle{10.0};

    /// Minimum |v_yaw| to enable armor-selection bias (rad/s).
    ///
    /// Problem:
    ///   At low yaw rates, directional bias can cause unnecessary armor switching near the
    ///   boundary.
    ///
    /// Intuition:
    ///   Larger → bias is disabled more often (more stable, less predictive); smaller → bias is
    ///   enabled more often (more predictive, can jitter).
    ///
    /// Analogy:
    ///   Like adding hysteresis: do not "predict" when the motion is too slow to matter.
    ///
    /// Precise:
    ///   In `select_best_armor()`, if |v_yaw| < min_switching_v_yaw then theta is forced to 0.
    double min_switching_v_yaw{1.0};

    /// Consecutive-frame threshold for planner state transitions.
    ///
    /// Problem:
    ///   Mode switching based on a single frame is noisy. We need a debounce counter.
    ///
    /// Intuition:
    ///   Larger → fewer mode flips but slower response; smaller → quicker response but can flicker.
    ///
    /// Analogy:
    ///   Like requiring N consecutive frames before a state change.
    ///
    /// Precise (easy-to-miss detail):
    ///   Implemented as `overflow_count_ > transfer_thresh` (strict '>'), so it effectively
    ///   requires `transfer_thresh + 1` consecutive frames meeting the condition.
    int transfer_thresh{5};

    /// Trajectory compensator configuration
    TrajectoryConfig trajectory;

    /// Compensator type (case-insensitive): "Ideal", "Resistance", or "RK4".
    ///
    /// Problem:
    ///   You must trade accuracy vs compute cost and calibration complexity.
    ///
    /// Intuition:
    ///   - Ideal: fastest but least accurate at long range.
    ///   - Resistance: fast and tunable with one fitted parameter.
    ///   - RK4: most physically faithful (quadratic drag) but more parameters and heavier compute
    ///          (mitigated by a lookup table rebuilt on parameter updates).
    ///
    /// Analogy:
    ///   Like choosing between a straight-line estimate, a fitted curve, or a full physics
    ///   simulation.
    ///
    /// Precise:
    ///   Selects the concrete compensator implementation created by `create_compensator(...)`.
    TrajectoryCompensatorType compensator_type = TrajectoryCompensatorType::Resistance;

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ(tbl, shooting_range_w);
        READ(tbl, shooting_range_h);
        READ(tbl, max_tracking_v_yaw);
        READ(tbl, plan_to_muzzle_delay);
        READ(tbl, side_angle);
        READ(tbl, min_switching_v_yaw);
        READ(tbl, transfer_thresh);
        READ(tbl, trajectory);
        READ(tbl, compensator_type);
        return {};
    }
};

// ============================================================================
// L4 Planning Full Config
// ============================================================================

struct L4Config {
    GimbalPlannerConfig planner{};

    /// Get shared_ptr for atomic config distribution
    [[nodiscard]] std::shared_ptr<const GimbalPlannerConfig> planner_ptr() const {
        return std::make_shared<const GimbalPlannerConfig>(planner);
    }

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ(tbl, planner);
        return {};
    }
};

} // namespace fcs::L4
