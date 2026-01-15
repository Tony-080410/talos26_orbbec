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
    cmd.yaw_vel      = plan.yaw_vel;
    cmd.pitch_vel    = plan.pitch_vel;
    cmd.yaw_accel    = plan.yaw_acc;
    cmd.pitch_accel  = plan.pitch_acc;
    cmd.fire         = plan.fire_advice;
    cmd.distance     = plan.distance;

    // Calculate time of flight using trajectory compensator
    cmd.tof = calculate_time_of_flight(target_distance);

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

} // namespace fcs::L5
