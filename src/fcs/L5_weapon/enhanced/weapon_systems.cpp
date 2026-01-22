#include "fcs/L5_weapon/enhanced/weapon_systems.hpp"

#include "fcs/L4_planning/gimbal_planner/trajectory_compensator.hpp"
#include "fcs/L4_planning/gimbal_planner/types.hpp"
#include "fcs/L4_planning/mpc_planner/config.hpp"
#include "fcs/L5_weapon/config.hpp"
#include "fcs/L5_weapon/enhanced/weapon_controller.hpp"
#include "fcs/core/channel_tags.hpp"
#include "scheduler/ecs/channel_types.hpp"
#include "scheduler/ecs/execution_policy.hpp"
#include "scheduler/scheduler/scheduler.hpp"

#include <memory>

namespace fcs::L5 {

void register_enhanced_weapon_system(talos::Scheduler& scheduler, WeaponControllerConfig&& config) {
    scheduler.world().insert_resource(std::move(config));
    scheduler.add_system<talos::exclusive<250>>(
        "enhanced_weapon_control",
        [controller = std::shared_ptr<EnhancedWeaponController>{},
         comp_type  = fcs::L4::TrajectoryCompensatorType::Ideal](
            talos::res<fcs::L4::MpcPlannerConfig> l4_config,
            talos::res<WeaponControllerConfig> config,
            talos::spmc<fcs::L4::GimbalPlan, GimbalPlanChannelTag> plan_in,
            talos::spmc_mut<WeaponCommand, WeaponCommandChannelTag> weapon_out) mutable {
            if (!controller) {
                auto comp_type   = l4_config->compensator_type;
                auto compensator = fcs::L4::create_compensator(comp_type);
                controller =
                    std::make_shared<EnhancedWeaponController>(*config, std::move(compensator));
            }

            if (config.is_changed()) {
                controller->set_config(*config);
            }

            // Only recreate compensator if type actually changed
            if (l4_config.is_changed() && l4_config->compensator_type != comp_type) {
                const auto old_type = comp_type;
                comp_type           = l4_config->compensator_type;

                SPDLOG_INFO(
                    "EnhancedWeaponController: switching compensator type from {} to {}",
                    magic_enum::enum_name(old_type).data(),
                    magic_enum::enum_name(comp_type).data());

                controller->sync_config(fcs::L4::create_compensator(comp_type));
            }

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
