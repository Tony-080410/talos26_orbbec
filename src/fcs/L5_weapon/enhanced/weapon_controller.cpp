#include "fcs/L5_weapon/enhanced/weapon_controller.hpp"

#include <Eigen/Core>
#include <cmath>

namespace fcs::L5 {

WeaponCommand EnhancedWeaponController::generate_command(
    const fcs::L4::GimbalPlan& plan, double target_distance) noexcept {

    WeaponCommand cmd;
    cmd.timestamp_ns = plan.timestamp_ns;
    cmd.yaw          = plan.yaw;
    cmd.pitch        = plan.pitch;
    cmd.pitch_rate   = plan.opt_pitch_vel[plan.mpc_current_idx];
    cmd.yaw_rate     = plan.opt_yaw_vel[plan.mpc_current_idx];
    cmd.ref_yaw      = plan.opt_yaw[plan.mpc_current_idx];
    cmd.ref_pitch    = plan.opt_pitch[plan.mpc_current_idx];
    cmd.track_center = plan.track_center;
    cmd.fire         = plan.fire_advice;
    cmd.distance     = plan.distance;

    // Calculate time of flight using trajectory compensator
    cmd.tof = calculate_time_of_flight(target_distance);

    // Calculate angular rates (for predictive control)
    // TODO: Implement proper angular rate calculation with history
    cmd.yaw_rate   = 0.0;
    cmd.pitch_rate = 0.0;

    // Apply bullet compensation
    // TODO: Implement more sophisticated bullet timing compensation
    // based on weapon characteristics and target movement prediction

    return cmd;
}

double EnhancedWeaponController::calculate_time_of_flight(double distance) const noexcept {
    if (!compensator_) {
        return 0.0;
    }

    // Simple estimation assuming straight-line trajectory
    // In reality, this would use the compensator's get_flying_time method
    // with proper target position calculation
    Eigen::Vector3d target_pos(distance, 0, 0);
    return compensator_->get_flying_time(target_pos);
}

std::pair<double, double> EnhancedWeaponController::calculate_angular_rates(
    [[maybe_unused]] const fcs::L4::GimbalPlan& plan, [[maybe_unused]] double dt) const noexcept {
    // TODO: Implement proper angular rate calculation based on target movement
    // For now, return zeros
    return {0.0, 0.0};
}

} // namespace fcs::L5
