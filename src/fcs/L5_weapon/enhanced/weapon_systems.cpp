#include "L5_weapon/enhanced/weapon_systems.hpp"

#include "L4_planning/common/transform_utils.hpp"
#include "L5_weapon/config.hpp"
#include "L5_weapon/enhanced/trajectory_optimizer.hpp"
#include "L5_weapon/fire_control.hpp"
#include "L5_weapon/fire_decision.hpp"
#include "core/channel_topics.hpp"
#include "core/time.hpp"
#include "scheduler/scheduler.hpp"

#include <memory>

namespace fcs::L5 {
void register_enhanced_weapon_system(talos::Scheduler& scheduler, L5Config&& config) {
    // Read trajectory config from L4 config resource for horizon/pitch parameters
    const auto& trajectory_cfg = scheduler.world().get_res<L4::L4Config>()->reference_trajectory;
    scheduler.world().insert_resource(config);
    auto optimizer =
        std::make_shared<TinyMpcTrajectoryOptimizer>(config.mpc_weapon, trajectory_cfg);
    scheduler.add_system<talos::fixed_rate<250>>(
        "enhanced_weapon_control",
        [optimizer](
            talos::res<L5::L5Config> l5_config, talos::res<fast_tf::CoordinateSystem> tf_buffer,
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

            const auto optimized =
                optimizer->optimize(*plan, l5_config->fire_decision, command_timestamp_ns);

            WeaponCommand cmd =
                optimized ? *optimized : optimizer->passthrough(*plan, command_timestamp_ns);
            auto tf = L4::lookup_gimbal_transform(*tf_buffer, command_timestamp_ns);
            if (tf) {
                auto [r, p, y] = tf->euler_rot().rpy();
                auto result    = is_on_target(
                    l5_config->fire_decision, y, -p, cmd.plan_yaw, cmd.plan_pitch,
                    cmd.plan_distance);
                cmd.fire                 = result.fire;
                cmd.yaw_error            = result.yaw_error;
                cmd.pitch_error          = result.pitch_error;
                cmd.shooting_range_yaw   = result.shooting_range_yaw;
                cmd.shooting_range_pitch = result.shooting_range_pitch;
                cmd.ref_yaw              = cmd.plan_yaw;
                cmd.ref_pitch            = cmd.plan_pitch;
            } else {
                SPDLOG_ERROR("Gimbal transform lookup failed: {}", tf.error());
            }

            if (!optimized) {
                SPDLOG_WARN(
                    "Optimization failed, using passthrough command: {}", optimized.error());
            }

            weapon_out.write(cmd);
        });
}

} // namespace fcs::L5
