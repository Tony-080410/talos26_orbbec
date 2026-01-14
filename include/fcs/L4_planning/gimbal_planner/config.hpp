#pragma once

#include <expected>
#include <memory>

#include "fcs/L4_planning/mpc_planner/config.hpp"
#include "fcs/L4_planning/trajectory_config.hpp"

// Forward declaration to avoid including toml++ in config headers.
namespace toml {
inline namespace v3 {
class table;
}
} // namespace toml

namespace fcs::L4 {

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

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

// ============================================================================
// L4 Planning Full Config
// ============================================================================

struct L4Config {
    GimbalPlannerConfig planner{};
    MpcPlannerConfig mpc_planner{};

    /// Get shared_ptr for atomic config distribution
    [[nodiscard]] std::shared_ptr<const GimbalPlannerConfig> planner_ptr() const {
        return std::make_shared<const GimbalPlannerConfig>(planner);
    }

    [[nodiscard]] std::shared_ptr<const MpcPlannerConfig> mpc_planner_ptr() const {
        return std::make_shared<const MpcPlannerConfig>(mpc_planner);
    }

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

} // namespace fcs::L4
