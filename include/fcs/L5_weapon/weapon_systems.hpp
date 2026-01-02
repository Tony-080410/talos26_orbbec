#pragma once

#include "ecs/execution_policy.hpp"
#include "fire_control.hpp"

#include "fcs/L3_estimation/armor_solver/types.hpp"
#include "fcs/core/channel_tags.hpp"
#include "scheduler/ecs/channel_types.hpp"

namespace fcs::L5 {

/// Register L5 weapon system
///
/// Pipeline:
///   L3::GimbalCommand [SPMC] -> WeaponSystem [pool_compute]
///                                    ↓
///                             L5::WeaponCommand [SPMC]
///
/// This system performs a simple 1:1 conversion from L3::GimbalCommand to L5::WeaponCommand,
/// replicating the behavior of rm_gimbal's send_to_device() function.
template <typename App>
void register_weapon_system(App& app) {
    app.template add_system<talos::exclusive<250>>(
        "gimbal_to_weapon", [](talos::spmc<L3::GimbalCommand, SolverOutputChannelTag> gimbal_in,
                               talos::spmc_mut<WeaponCommand, WeaponCommandChannelTag> weapon_out) {
            auto gimbal_cmd = gimbal_in.read();
            if (!gimbal_cmd) {
                return;
            }

            // Convert L3::GimbalCommand -> L5::WeaponCommand
            // Replicates rm_gimbal's GimbalNode::send_to_device() logic
            WeaponCommand weapon_cmd;
            weapon_cmd.yaw      = gimbal_cmd->yaw;         // Target yaw (radians)
            weapon_cmd.pitch    = gimbal_cmd->pitch;       // Target pitch (radians)
            weapon_cmd.fire     = gimbal_cmd->fire_advice; // Shoot decision from solver
            weapon_cmd.distance = gimbal_cmd->distance;    // Target distance (meters)
            weapon_cmd.valid    = true;

            // Note: yaw_rate, pitch_rate, tof are not used in rm_gimbal
            // (ref_yaw_v, ref_pitch_v, ref_yaw_a, ref_pitch_a are all set to 0)
            weapon_cmd.yaw_rate   = 0.0;
            weapon_cmd.pitch_rate = 0.0;
            weapon_cmd.tof        = 0.0;

            weapon_out.write(weapon_cmd);
        });
}

} // namespace fcs::L5
