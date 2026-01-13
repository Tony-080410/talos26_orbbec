#include "fcs/L5_weapon/enhanced/weapon_systems.hpp"

#include "fcs/L4_planning/gimbal_planner/trajectory_compensator.hpp"
#include "fcs/L4_planning/gimbal_planner/types.hpp"
#include "fcs/L5_weapon/config.hpp"
#include "fcs/L5_weapon/enhanced/weapon_controller.hpp"
#include "fcs/core/channel_tags.hpp"
#include "scheduler/ecs/channel_types.hpp"
#include "scheduler/ecs/execution_policy.hpp"
#include "scheduler/scheduler/scheduler.hpp"

#include <memory>

namespace fcs::L5 {

void register_enhanced_weapon_system_with_config_and_compensator(
    talos::Scheduler& scheduler, const WeaponControllerConfig& config,
    fcs::L4::TrajectoryCompensatorType compensator_type) {
    auto compensator = fcs::L4::create_compensator(compensator_type);
    auto controller  = std::make_shared<EnhancedWeaponController>(config, std::move(compensator));

    scheduler.add_system<talos::exclusive<250>>(
        "enhanced_weapon_control",
        [controller = std::move(controller)](
            talos::spmc<fcs::L4::GimbalPlan, GimbalPlanChannelTag> plan_in,
            talos::spmc_mut<WeaponCommand, WeaponCommandChannelTag> weapon_out) {
            auto plan = plan_in.read();
            if (!plan) {
                return;
            }

            double target_distance   = plan->distance;
            WeaponCommand weapon_cmd = controller->generate_command(*plan, target_distance);
            weapon_out.write(weapon_cmd);
        });
}

} // namespace fcs::L5
