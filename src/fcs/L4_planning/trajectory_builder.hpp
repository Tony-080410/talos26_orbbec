#pragma once

#include "L4_planning/aimer/aimer.hpp"
#include "L4_planning/gimbal_planner/types.hpp"
#include "core/trajectory/reference_trajectory.hpp"
#include "core/trajectory/solver/solver_interfaces.hpp"

#include <expected>
#include <string>

namespace fcs::L4 {

/// Build a full MPC reference trajectory from a planner seed.
///
/// Iterates over the MPC horizon, calling `Aimer::aim()` at each step with
/// appropriate prediction delay offsets. Produces a `ReferenceTrajectory`
/// with the state matrix [yaw, yaw_rate, pitch, pitch_rate] × horizon plus
/// metadata for fire gate computation.
///
/// ## Preconditions
/// - `seed` must contain valid robot or outpost state
/// - `horizon_ahead` and `horizon_back` must be >= 0
/// - `dt` must be > 0
///
/// ## Thread safety
/// NOT thread-safe. Do not call concurrently on shared data.
///
/// ## Parameters
/// - `seed`: Planner seed containing raw L3 target state
/// - `aimer`: L4 aimer for computing aim points at each horizon step
/// - `horizon_ahead`: Forward prediction steps
/// - `horizon_back`: Backward history steps
/// - `dt`: Time step between horizon samples (seconds)
/// - `pitch_min`: Minimum pitch clamp (radians)
/// - `pitch_max`: Maximum pitch clamp (radians)
/// - `current_ns`: Current timestamp (nanoseconds)
/// - `gimbal`: Gimbal transform (odom → gimbal_pitch)
/// - `muzzle`: Muzzle transform (odom → muzzle_link)
/// - `trajectory_solver`: Ballistic trajectory solver
/// - `bullet_speed`: Bullet velocity (m/s)
[[nodiscard]] std::expected<core::trajectory::ReferenceTrajectory, std::string>
    build_reference_trajectory(
        const PlannerSeed& seed, const Aimer& aimer, int horizon_ahead, int horizon_back, double dt,
        double pitch_min, double pitch_max, uint64_t current_ns,
        const Aimer::GimbalTransform& gimbal, const Aimer::MuzzleTransform& muzzle,
        const core::trajectory::solver::TrajectorySolver& trajectory_solver,
        double bullet_speed) noexcept;

} // namespace fcs::L4
