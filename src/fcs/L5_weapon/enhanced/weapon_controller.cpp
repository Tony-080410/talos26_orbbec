#include "fcs/L5_weapon/enhanced/weapon_controller.hpp"

#include <cmath>

namespace fcs::L5 {

WeaponCommand EnhancedWeaponController::generate_command(
    const fcs::L4::GimbalPlan& plan, double target_distance) noexcept {

    WeaponCommand cmd;
    cmd.timestamp_ns = plan.timestamp_ns;
    cmd.yaw          = plan.yaw;
    cmd.pitch        = plan.pitch;
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

} // namespace fcs::L5