#include "L5_weapon/enhanced/weapon_systems.hpp"

#include "L4_planning/aimer/aimer.hpp"
#include "L4_planning/config.hpp"
#include "L4_planning/gimbal_planner/types.hpp"
#include "L5_weapon/config.hpp"
#include "L5_weapon/enhanced/trajectory_optimizer.hpp"
#include "L5_weapon/fire_control.hpp"
#include "core/channel_topics.hpp"
#include "core/time.hpp"
#include "core/trajectory/resource.hpp"
#include "scheduler/scheduler.hpp"

#include <functional>
#include <memory>

namespace fcs::L5 {
void register_enhanced_weapon_system(talos::Scheduler& scheduler, WeaponControllerConfig&& config) {
    auto optimizer = std::make_shared<TinyMpcTrajectoryOptimizer>(config);
    scheduler.world().insert_resource(std::move(config));
    scheduler.add_system<talos::fixed_rate<250>>(
        "enhanced_weapon_control",
        [optimizer](
            [[maybe_unused]] talos::res<WeaponControllerConfig> _config,
            talos::res<L4::L4Config> l4_config, talos::res<L4::Aimer> aimer,
            core::trajectory::trajectory_solver solver,
            core::trajectory::bullet_speed bullet_speed_,
            talos::res<fast_tf::CoordinateSystem> tf_buffer,
            talos::spmc<fcs::L4::GimbalPlan, GimbalPlanChannelTopic> plan_in,
            talos::spmc_mut<WeaponCommand, WeaponCommandChannelTopic> weapon_out) mutable {
            auto plan = plan_in.read();
            if (!plan) {
                return;
            }

            const uint64_t command_timestamp_ns = fcs::clock::now_ns();
            WeaponCommand cmd = optimizer->passthrough(*plan, command_timestamp_ns);

            if (solver->get()) {
                const auto optimized = optimizer->optimize(
                    *plan, *aimer, l4_config->fire_decision, *tf_buffer, **solver,
                    bullet_speed_->bullet_speed, command_timestamp_ns);
                if (optimized) {
                    cmd = std::move(*optimized);
                }
            }

            weapon_out.write(cmd);
        });
}

} // namespace fcs::L5
