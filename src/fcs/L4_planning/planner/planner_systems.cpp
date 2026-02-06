#include "fcs/L4_planning/planner/planner_systems.hpp"

#include "fcs/L4_planning/aimer/types.hpp"
#include "fcs/L4_planning/common/transform_utils.hpp"
#include "fcs/L4_planning/gimbal_planner/trajectory_compensator.hpp"
#include "fcs/L4_planning/gimbal_planner/types.hpp"
#include "fcs/L4_planning/planner/mpc_planner.hpp"
#include "fcs/core/channel_topics.hpp"
#include "scheduler/ecs/channel_types.hpp"
#include "scheduler/ecs/execution_policy.hpp"
#include "scheduler/scheduler/scheduler.hpp"

#include <spdlog/spdlog.h>

#include <memory>

namespace fcs::L4 {

namespace {

// Maximum visualization distance for trajectory (meters)
constexpr double kMaxVisualizationDistance = 15.0;

// Helper function to sync compensator parameters from trajectory config
auto sync_compensator(TrajectoryCompensator& compensator, const TrajectoryConfig& traj_cfg) noexcept
    -> void {
    compensator.velocity        = traj_cfg.bullet_speed;
    compensator.gravity         = traj_cfg.gravity;
    compensator.iteration_times = traj_cfg.iteration_times;
    compensator.cd              = traj_cfg.cd;
    compensator.rho             = traj_cfg.rho;
    compensator.diameter        = traj_cfg.diameter;
    compensator.mass            = traj_cfg.mass;
    compensator.rk4_dt          = traj_cfg.rk4_dt;
    compensator.rk4_min_steps   = traj_cfg.rk4_min_steps;
    compensator.rk4_max_steps   = traj_cfg.rk4_max_steps;
    compensator.on_params_updated();
}

} // namespace

void register_planner_systems(talos::Scheduler& scheduler) {
    // Note: L4Config is already in world from register_aimer_systems

    // =========================================================================
    // MPC Planner System (consumes TargetPrediction, produces GimbalPlan)
    // =========================================================================

    scheduler.add_system<talos::pool_compute>(
        "l4_mpc_planner",
        [planner               = std::make_shared<MpcPlanner>(MpcPlannerConfig{}),
         compensator           = std::shared_ptr<TrajectoryCompensator>{},
         last_compensator_type = TrajectoryCompensatorType{}](
            talos::res<L4Config> cfg,
            talos::spmc<TargetPrediction, AimerOutputChannelTopic> aimer_in,
            talos::res<std::shared_ptr<fast_tf::CoordinateSystem<1024>>> tf_buffer,
            talos::spmc_mut<GimbalPlan, ArmorGimbalPlanChannelTopic> armor_plan_out) mutable {
            const auto prediction = aimer_in.read();
            if (!prediction) {
                return;
            }

            // Update compensator when config changes or not initialized
            if (!compensator || last_compensator_type != cfg->mpc_planner.compensator_type) {
                compensator = std::shared_ptr<TrajectoryCompensator>(
                    create_compensator(cfg->mpc_planner.compensator_type));
                last_compensator_type = cfg->mpc_planner.compensator_type;
            }
            sync_compensator(*compensator, cfg->mpc_planner.trajectory);

            // Check if MPC is enabled
            if (!cfg->mpc_planner.enabled) {
                // MPC disabled: directly convert aimer output to GimbalPlan
                GimbalPlan plan;
                plan.timestamp_ns   = prediction->timestamp_ns;
                plan.yaw            = prediction->aim_yaw;
                plan.pitch          = prediction->aim_pitch;
                plan.yaw_vel        = 0.0;
                plan.pitch_vel      = 0.0;
                plan.yaw_acc        = 0.0;
                plan.pitch_acc      = 0.0;
                plan.fire_advice    = prediction->aimer_fire_advice;
                plan.tracking_error = 0.0;

                // Target info (odom frame)
                plan.distance           = prediction->distance;
                plan.predicted_position = prediction->predicted_position;
                plan.predicted_future   = prediction->predicted_future_ns;

                // Aimer's original advice
                plan.aimer_fire_advice = prediction->aimer_fire_advice;
                plan.aim_yaw           = prediction->aim_yaw;
                plan.aim_pitch         = prediction->aim_pitch;

                // Fill visualization debug data (trajectory for Foxglove)
                // Note: Don't set viz_debug.predicted_position because:
                // 1. It has timestamp = predicted_future_ns (future time)
                // 2. TF interpolation uses current timestamp_ns
                // 3. For moving targets, time mismatch causes visual misalignment
                // 4. The trajectory itself is sufficient for visualization
                VisualizationDebugData viz_debug;
                // viz_debug.predicted_position left as std::nullopt (not set)
                viz_debug.predicted_future = prediction->predicted_future_ns;
                viz_debug.trajectory =
                    compensator->get_trajectory(kMaxVisualizationDistance, plan.pitch);
                // Rotate trajectory points by pitch angle
                for (auto& p : viz_debug.trajectory) {
                    const double x = p.first;
                    const double z = p.second;
                    p.first        = x * std::cos(plan.pitch) + z * std::sin(plan.pitch);
                    p.second       = -x * std::sin(plan.pitch) + z * std::cos(plan.pitch);
                }
                plan.viz_debug = viz_debug;

                armor_plan_out.write(plan);
                return;
            }

            // Update planner config from resource (lightweight operation)
            planner->set_config(cfg->mpc_planner);

            // Lookup gimbal transform
            auto gimbal_tf = lookup_gimbal_transform(*tf_buffer, prediction->timestamp_ns);
            if (!gimbal_tf) {
                SPDLOG_WARN("Gimbal transform lookup failed: {}", gimbal_tf.error());
                return;
            }

            // Plan trajectory
            auto plan_result = planner->plan(
                *prediction, prediction->timestamp_ns, cfg->mpc_planner.trajectory.bullet_speed,
                *gimbal_tf);

            if (!plan_result) {
                SPDLOG_WARN("MPC planning failed: {}", plan_result.error());
                return;
            }

            armor_plan_out.write(*plan_result);
        });
}

} // namespace fcs::L4
