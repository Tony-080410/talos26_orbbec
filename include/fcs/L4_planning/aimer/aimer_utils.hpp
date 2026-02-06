#pragma once

#include <Eigen/Core>

#include "fcs/L4_planning/gimbal_planner/trajectory_compensator.hpp"

namespace fcs::L4 {

/// Maximum iterations for ballistic flying time refinement
inline constexpr int kMaxFlyingTimeRefineIter = 10;

/// Tolerance for flying time convergence (seconds)
inline constexpr double kFlyingTimeRefineTolerance = 1e-3;

/// Compute aim point (yaw, pitch) for given target position
///
/// @param compensator Ballistic trajectory compensator
/// @param target_pos_muzzle Target position in muzzle frame
/// @param bullet_speed Bullet speed (m/s)
/// @return {yaw, pitch} in radians
[[nodiscard]] inline Eigen::Vector2d compute_aim_point(
    const std::unique_ptr<TrajectoryCompensator>& compensator,
    const Eigen::Vector3d& target_pos_muzzle, double bullet_speed) noexcept {
    const double yaw = std::atan2(target_pos_muzzle.y(), target_pos_muzzle.x());
    double pitch     = 0.0;
    double yaw_out   = yaw;
    [[maybe_unused]] bool ok =
        compensator->compensate(target_pos_muzzle, yaw_out, pitch, bullet_speed);
    return {yaw, pitch};
}

/// Perform iterative flying time refinement
///
/// @param compensator Ballistic trajectory compensator
/// @param predict_fn Function to predict position at given time (seconds)
/// @param initial_delay Initial delay to start from (seconds)
/// @return Refined total delay (initial_delay + flying_time)
template <class PredictFn>
[[nodiscard]] double refine_flying_time(
    const std::unique_ptr<TrajectoryCompensator>& compensator, PredictFn&& predict_fn,
    double initial_delay) noexcept {
    double flying_time = 0.0;
    for (int i = 0; i < kMaxFlyingTimeRefineIter; ++i) {
        const Eigen::Vector3d predicted_pos = predict_fn(initial_delay + flying_time);
        const double t_curr                 = compensator->get_flying_time(predicted_pos);
        if (std::abs(t_curr - flying_time) < kFlyingTimeRefineTolerance) {
            break;
        }
        flying_time = t_curr;
    }
    return initial_delay + flying_time;
}

} // namespace fcs::L4
