#include "fcs/L4_planning/aimer/aimer_systems.hpp"

#include "fcs/L3_estimation/energy_meter/types.hpp"
#include "fcs/L3_estimation/tracker/types.hpp"
#include "fcs/L4_planning/aimer/aimer.hpp"
#include "fcs/L4_planning/aimer/types.hpp"
#include "fcs/L4_planning/common/transform_utils.hpp"
#include "fcs/L4_planning/gimbal_planner/trajectory_compensator.hpp"
#include "fcs/L4_planning/planning/fire_decision.hpp"
#include "fcs/core/channel_topics.hpp"
#include "fcs/core/time.hpp"
#include "math/euler.hpp"
#include "scheduler/ecs/channel_types.hpp"
#include "scheduler/ecs/execution_policy.hpp"
#include "scheduler/scheduler/scheduler.hpp"

#include <spdlog/spdlog.h>

#include <memory>

namespace fcs::L4 {

void register_aimer_systems(talos::Scheduler& scheduler, L4Config&& config) {
    // Store configuration in world

    // Create shared trajectory compensator for all aimers
    auto compensator = create_compensator(config.mpc_planner.compensator_type);

    // Sync compensator parameters from trajectory config
    auto sync_compensator = [&](decltype(compensator)& compensator_) {
        compensator_->velocity        = config.mpc_planner.trajectory.bullet_speed;
        compensator_->gravity         = config.mpc_planner.trajectory.gravity;
        compensator_->iteration_times = config.mpc_planner.trajectory.iteration_times;
        compensator_->cd              = config.mpc_planner.trajectory.cd;
        compensator_->rho             = config.mpc_planner.trajectory.rho;
        compensator_->diameter        = config.mpc_planner.trajectory.diameter;
        compensator_->mass            = config.mpc_planner.trajectory.mass;
        compensator_->rk4_dt          = config.mpc_planner.trajectory.rk4_dt;
        compensator_->rk4_min_steps   = config.mpc_planner.trajectory.rk4_min_steps;
        compensator_->rk4_max_steps   = config.mpc_planner.trajectory.rk4_max_steps;
        compensator_->on_params_updated();
    };
    sync_compensator(compensator);
    scheduler.world().insert_resource(config);

    // =========================================================================
    // Unified Aimer System (Robot/Outpost/Rune)
    // =========================================================================
    // Single system to avoid MultipleWriters violation
    // Processes all target types and outputs to single AimerOutputChannelTopic

    scheduler.add_system<talos::pool_compute>(
        "l4_aimer",
        [aimer = std::make_shared<Aimer>(config.aimer, std::move(compensator))](
            talos::res<L4Config> cfg,
            talos::spmc<L3::TrackerOutput, TrackerOutputChannelTopic> tracker_in,
            talos::spmc<energy_meter::EnergyMeterState, EnergyMeterStateChannelTopic> rune_in,
            talos::res<std::shared_ptr<fast_tf::CoordinateSystem<1024>>> tf_buffer,
            talos::spmc_mut<TargetPrediction, AimerOutputChannelTopic> aimer_out) mutable {
            std::expected<TargetPrediction, std::string> aim_result = std::unexpected("none");

            // FIXME timestamp unaligned
            const uint64_t current_ns = fcs::clock::now_ns();

            // Use common transform lookup utility
            auto transforms = lookup_gimbal_muzzle_transforms(*tf_buffer, current_ns);
            if (!transforms) {
                SPDLOG_WARN("Transform lookup failed: {}", transforms.error());
                return;
            }
            const auto& [gimbal, muzzle, timestamp_ns] = *transforms;

            // Try Robot/Outpost from tracker
            const auto tracker_output = tracker_in.read();
            if (tracker_output && tracker_output->is_tracking()) {
                if (tracker_output->is_robot()) {
                    if (const auto robot_state = tracker_output->robot_state()) {
                        aim_result = aimer->aim(
                            *robot_state, gimbal, muzzle, tracker_output->timestamp_ns, current_ns,
                            cfg->mpc_planner.trajectory.bullet_speed);
                    }
                } else if (tracker_output->is_outpost()) {
                    if (const auto outpost_state = tracker_output->outpost_state()) {
                        aim_result = aimer->aim(
                            *outpost_state, gimbal, muzzle, tracker_output->timestamp_ns,
                            current_ns, cfg->mpc_planner.trajectory.bullet_speed);
                    }
                }
            }

            // Try Rune (fallback)
            if (!aim_result) {
                const auto rune_state = rune_in.read();
                if (rune_state && rune_state->tracking_valid) {
                    aim_result = aimer->aim(
                        *rune_state, gimbal, muzzle, current_ns,
                        cfg->mpc_planner.trajectory.bullet_speed);
                }
            }

            if (aim_result) {
                // Compute aimer_fire_advice using current gimbal attitude
                // Use euler_rot() method which directly returns Ros2EulerRot
                const auto euler     = gimbal->euler_rot();
                const auto cur_roll  = euler.roll;
                const auto cur_pitch = euler.pitch;
                const auto cur_yaw   = euler.yaw;

                // Note: aim_pitch uses opposite sign convention from gimbal pitch
                // (rune_planner uses -plan.pitch for comparison, so we use -aim_pitch here)
                aim_result->aimer_fire_advice = is_on_target(
                    cfg->fire_decision, cur_yaw, cur_pitch, aim_result->aim_yaw,
                    -aim_result->aim_pitch, aim_result->distance);

                // Debug: print fire decision parameters
                SPDLOG_INFO(
                    "[FireDecision] cur_yaw={:.6f}, cur_pitch={:.6f}, aim_yaw={:.6f}, "
                    "aim_pitch={:.6f}, "
                    "distance={:.3f}, fire_thresh={:.6f}, shooting_range_yaw={:.6f}, "
                    "shooting_range_pitch={:.6f}, "
                    "yaw_diff={:.6f}, pitch_diff={:.6f}, on_target={}",
                    cur_yaw, cur_pitch, aim_result->aim_yaw, aim_result->aim_pitch,
                    aim_result->distance, cfg->fire_decision.fire_thresh,
                    std::abs(
                        std::atan2(
                            cfg->fire_decision.shooting_range_w / 2.0, aim_result->distance)),
                    std::abs(
                        std::atan2(
                            cfg->fire_decision.shooting_range_h / 2.0, aim_result->distance)),
                    std::abs(cur_yaw - aim_result->aim_yaw),
                    std::abs(cur_pitch - (-aim_result->aim_pitch)), aim_result->aimer_fire_advice);

                aimer_out.write(*aim_result);
            }
        });
}

} // namespace fcs::L4
