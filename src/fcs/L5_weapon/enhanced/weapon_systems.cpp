#include "L5_weapon/enhanced/weapon_systems.hpp"

#include "L4_planning/config.hpp"
#include "L4_planning/gimbal_planner/types.hpp"
#include "L5_weapon/config.hpp"
#include "L5_weapon/enhanced/trajectory_optimizer.hpp"
#include "L5_weapon/fire_control.hpp"
#include "core/channel_topics.hpp"
#include "core/time.hpp"
#include "scheduler/scheduler.hpp"

#include <memory>

namespace fcs::L5 {
void register_enhanced_weapon_system(talos::Scheduler& scheduler, WeaponControllerConfig&& config) {
    // Read trajectory config from L4 config resource for horizon/pitch parameters
    const auto& trajectory_cfg = scheduler.world().get_res<L4::L4Config>()->reference_trajectory;
    auto optimizer = std::make_shared<TinyMpcTrajectoryOptimizer>(config, trajectory_cfg);
    scheduler.world().insert_resource(std::move(config));
    scheduler.add_system<talos::fixed_rate<250>>(
        "enhanced_weapon_control",
        [optimizer](
            [[maybe_unused]] talos::res<WeaponControllerConfig> _config,
            talos::res<L4::L4Config> l4_config, talos::res<fast_tf::CoordinateSystem> tf_buffer,
            talos::spmc<fcs::L4::GimbalPlan, GimbalPlanChannelTopic> plan_in,
            talos::spmc_mut<WeaponCommand, WeaponCommandChannelTopic> weapon_out) mutable {
            auto plan = plan_in.read();
            if (!plan) {
                return;
            }
            const uint64_t command_timestamp_ns = fcs::clock::now_ns();

            if (plan->distance < 0) {
                weapon_out.write(optimizer->passthrough(*plan, command_timestamp_ns));
                return;
            }

            const auto optimized = optimizer->optimize(
                *plan, l4_config->fire_decision, l4_config->aimer, *tf_buffer,
                command_timestamp_ns);

            WeaponCommand cmd =
                optimized ? *optimized : optimizer->passthrough(*plan, command_timestamp_ns);
            if (!optimized) {
                SPDLOG_ERROR(
                    "Optimization failed, using passthrough command: {}", optimized.error());
            }

            weapon_out.write(cmd);
        });
}

} // namespace fcs::L5
