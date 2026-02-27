#include "L5_weapon/enhanced/weapon_systems.hpp"

#include "L4_planning/config.hpp"
#include "L4_planning/gimbal_planner/types.hpp"
#include "L5_weapon/config.hpp"
#include "L5_weapon/fire_control.hpp"
#include "core/channel_topics.hpp"
#include "core/trajectory/resource.hpp"
#include "scheduler/scheduler.hpp"

#include <functional>
#include <memory>

namespace fcs::L5 {
void register_enhanced_weapon_system(talos::Scheduler& scheduler, WeaponControllerConfig&& config) {
    scheduler.world().insert_resource(std::move(config));
    scheduler.add_system<talos::fixed_rate<250>>(
        "enhanced_weapon_control",
        [&](talos::res<WeaponControllerConfig> _config, core::trajectory::trajectory_solver solver,
            talos::spmc<fcs::L4::GimbalPlan, GimbalPlanChannelTopic> plan_in,
            talos::spmc_mut<WeaponCommand, WeaponCommandChannelTopic> weapon_out) mutable {
            auto plan = plan_in.read();
            if (!plan) {
                return;
            }

            WeaponCommand cmd;
            cmd.timestamp_ns = plan->timestamp_ns;
            cmd.yaw          = plan->yaw;
            cmd.pitch        = plan->pitch;
            cmd.yaw_vel      = plan->yaw_vel;
            cmd.pitch_vel    = plan->pitch_vel;
            cmd.yaw_accel    = plan->yaw_acc;
            cmd.pitch_accel  = plan->pitch_acc;
            cmd.fire         = plan->fire_advice;
            cmd.distance     = plan->distance;

            // Calculate time of flight using trajectory solver
            cmd.tof = 0.0;

            weapon_out.write(cmd);
        });
}

} // namespace fcs::L5
