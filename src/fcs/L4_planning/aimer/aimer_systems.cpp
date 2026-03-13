#include "L4_planning/aimer/aimer_systems.hpp"
#include "L4_planning/aimer/aimer.hpp"

#include "L3_estimation/energy_meter/types.hpp"
#include "L3_estimation/tracker/types.hpp"
#include "L4_planning/aimer/types.hpp"
#include "L4_planning/common/transform_utils.hpp"
#include "L4_planning/config.hpp"
#include "L4_planning/gimbal_planner/types.hpp"
#include "L4_planning/planning/fire_decision.hpp"
#include "core/channel_topics.hpp"
#include "core/time.hpp"
#include "core/trajectory/resource.hpp"
#include "euler.hpp"
#include "scheduler/scheduler.hpp"

#include <spdlog/spdlog.h>

namespace fcs::L4 {
static constexpr double kMaxVisualizationDistance = 15.0;

void register_aimer_systems(talos::Scheduler& scheduler, L4Config&& config) {
    scheduler.world().insert_resource(config);

    // Create Aimer instance with config (does NOT own solver)
    Aimer aimer(config.aimer);

    scheduler.add_system<talos::pool_compute>(
        "l4_aimer",
        [aimer](
            talos::res<L4Config> cfg, core::trajectory::trajectory_solver solver,
            core::trajectory::trajectory_config traj_conf,
            talos::spmc<L3::TrackerOutput, TrackerOutputChannelTopic> tracker_in,
            talos::spmc<energy_meter::EnergyMeterState, EnergyMeterStateChannelTopic> rune_in,
            talos::res<std::shared_ptr<fast_tf::CoordinateSystem>> tf_buffer,
            talos::spmc_mut<GimbalPlan, GimbalPlanChannelTopic> aimer_out) mutable {
            std::expected<TargetPrediction, std::string> aim_result = std::unexpected("none");

            const uint64_t current_ns = fcs::clock::now_ns();

            auto transforms = lookup_gimbal_muzzle_transforms(*tf_buffer, current_ns);
            if (!transforms) {
                SPDLOG_WARN("Transform lookup failed: {}", transforms.error());
                return;
            }
            const auto& [gimbal, muzzle, timestamp_ns] = *transforms;
            (void)timestamp_ns;

            if (!solver->get()) {
                SPDLOG_ERROR("Aimer: trajectory solver not available");
                return;
            }
            const auto& solver_ref    = **solver;
            const double bullet_speed = traj_conf->bullet_speed;

            // Try Robot/Outpost from tracker
            const auto tracker_output = tracker_in.read();
            if (tracker_output && tracker_output->is_tracking()) {
                if (tracker_output->is_robot()) {
                    if (const auto robot_state = tracker_output->robot_state()) {
                        aim_result = aimer.aim(
                            *robot_state, gimbal, muzzle, tracker_output->timestamp_ns, current_ns,
                            bullet_speed, solver_ref);
                    }
                } else if (tracker_output->is_outpost()) {
                    if (const auto outpost_state = tracker_output->outpost_state()) {
                        aim_result = aimer.aim(
                            *outpost_state, gimbal, muzzle, tracker_output->timestamp_ns,
                            current_ns, bullet_speed, solver_ref);
                    }
                }
            }

            // Try Rune (fallback)
            if (!aim_result) {
                const auto rune_state = rune_in.read();
                if (rune_state && rune_state->tracking_valid) {
                    aim_result = aimer.aim(
                        *rune_state, gimbal, muzzle, current_ns, bullet_speed, solver_ref);
                }
            }

            if (aim_result) {
                const auto euler     = gimbal.euler_rot();
                const auto cur_pitch = euler.pitch;
                const auto cur_yaw   = euler.yaw;

                aim_result->aimer_fire_advice = is_on_target(
                    cfg->fire_decision, cur_yaw, cur_pitch, aim_result->aim_yaw,
                    -aim_result->aim_pitch, aim_result->distance);

                // SPDLOG_INFO(
                //     "[FireDecision] cur_yaw={:.6f}, cur_pitch={:.6f}, aim_yaw={:.6f}, "
                //     "aim_pitch={:.6f}, "
                //     "distance={:.3f}, fire_thresh={:.6f}, shooting_range_yaw={:.6f}, "
                //     "shooting_range_pitch={:.6f}, "
                //     "yaw_diff={:.6f}, pitch_diff={:.6f}, on_target={}",
                //     cur_yaw, cur_pitch, aim_result->aim_yaw, aim_result->aim_pitch,
                //     aim_result->distance, cfg->fire_decision.fire_thresh,
                //     std::abs(
                //         std::atan2(
                //             cfg->fire_decision.shooting_range_w / 2.0, aim_result->distance)),
                //     std::abs(
                //         std::atan2(
                //             cfg->fire_decision.shooting_range_h / 2.0, aim_result->distance)),
                //     std::abs(cur_yaw - aim_result->aim_yaw),
                //     std::abs(cur_pitch - (-aim_result->aim_pitch)),
                //     aim_result->aimer_fire_advice);

                GimbalPlan plan;
                plan.timestamp_ns       = aim_result->timestamp_ns;
                plan.yaw                = aim_result->aim_yaw;
                plan.pitch              = aim_result->aim_pitch;
                plan.aim_yaw            = aim_result->aim_yaw;
                plan.aim_pitch          = aim_result->aim_pitch;
                plan.distance           = aim_result->distance;
                plan.aimer_fire_advice  = aim_result->aimer_fire_advice;
                plan.fire_advice        = aim_result->aimer_fire_advice;
                plan.predicted_future   = aim_result->predicted_future_ns;
                plan.predicted_position = aim_result->predicted_position;
                VisualizationDebugData viz_debug;
                viz_debug.trajectory = solver->get()->generate_trajectory(
                    aim_result->aim_pitch, bullet_speed, kMaxVisualizationDistance);
                for (auto& p : viz_debug.trajectory) {
                    const double x = p.first;
                    const double z = p.second;
                    p.first        = x * std::cos(plan.pitch) + z * std::sin(plan.pitch);
                    p.second       = -x * std::sin(plan.pitch) + z * std::cos(plan.pitch);
                }
                plan.viz_debug = viz_debug;
                aimer_out.write(std::move(plan));
            }
        });
}

} // namespace fcs::L4
