#pragma once

#include "ecs/execution_policy.hpp"
#include "weapon_controller.hpp"

#include "fcs/L4_planning/gimbal_planner/trajectory_compensator.hpp"
#include "fcs/L4_planning/gimbal_planner/types.hpp"
#include "fcs/core/channel_tags.hpp"
#include "scheduler/ecs/channel_types.hpp"

#include <memory>

namespace fcs::L5 {

/// Register enhanced L5 weapon system
///
/// Pipeline:
///   L4::GimbalPlan [SPMC] → EnhancedWeaponSystem [pool_compute]
///                                    ↓
///                             L5::WeaponCommand [SPMC]
///
/// This system performs enhanced weapon control with bullet compensation
/// and sophisticated firing logic.
template <typename App>
void register_enhanced_weapon_system(
    App& app, std::shared_ptr<EnhancedWeaponController> weapon_controller) {

    app.template add_system<talos::exclusive<250>>(
        "enhanced_weapon_control",
        [controller = std::move(weapon_controller)](
            talos::spmc<fcs::L4::GimbalPlan, GimbalPlanChannelTag> plan_in,
            talos::spmc_mut<WeaponCommand, WeaponCommandChannelTag> weapon_out) {
            auto plan = plan_in.read();

            if (!plan) {
                return;
            }

            // Generate enhanced weapon command
            double target_distance   = plan->distance;
            WeaponCommand weapon_cmd = controller->generate_command(*plan, target_distance);

            weapon_out.write(weapon_cmd);
        });
}

/// Register enhanced weapon system with trajectory compensator
template <typename App>
void register_enhanced_weapon_system_with_compensator(
    App& app, std::unique_ptr<fcs::L4::TrajectoryCompensator> compensator) {

    // Use default config when none provided
    WeaponControllerConfig default_config;
    auto controller =
        std::make_shared<EnhancedWeaponController>(default_config, std::move(compensator));
    register_enhanced_weapon_system(app, controller);
}

/// Register enhanced weapon system with config and trajectory compensator
template <typename App>
void register_enhanced_weapon_system_with_config_and_compensator(
    App& app, const WeaponControllerConfig& config,
    std::unique_ptr<fcs::L4::TrajectoryCompensator> compensator) {

    auto controller = std::make_shared<EnhancedWeaponController>(config, std::move(compensator));
    register_enhanced_weapon_system(app, controller);
}

} // namespace fcs::L5
