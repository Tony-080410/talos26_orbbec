#include "fcs/L4_planning/planner/planner_systems.hpp"

#include "fcs/L4_planning/aimer/types.hpp"
#include "fcs/L4_planning/common/transform_utils.hpp"
#include "fcs/L4_planning/gimbal_planner/types.hpp"
#include "fcs/L4_planning/planner/mpc_planner.hpp"
#include "fcs/L4_planning/trajectory/solver_interfaces.hpp"
#include "fcs/core/channel_topics.hpp"
#include "scheduler/ecs/channel_types.hpp"
#include "scheduler/ecs/execution_policy.hpp"
#include "scheduler/scheduler/scheduler.hpp"

#include <spdlog/spdlog.h>

#include <functional>
#include <memory>

namespace fcs::L4 {

namespace {

// Maximum visualization distance for trajectory (meters)
constexpr double kMaxVisualizationDistance = 15.0;

[[nodiscard]] size_t solver_config_hash(const MpcPlannerConfig& cfg) noexcept {
    size_t hash             = 0;
    const auto hash_combine = [&hash](auto value) {
        hash ^= std::hash<decltype(value)>{}(value) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    };

    hash_combine(static_cast<int>(cfg.compensator_type));
    const auto& traj = cfg.trajectory;
    hash_combine(traj.iteration_times);
    hash_combine(traj.bullet_speed);
    hash_combine(traj.gravity);
    hash_combine(traj.resistance);
    hash_combine(traj.cd);
    hash_combine(traj.rho);
    hash_combine(traj.diameter);
    hash_combine(traj.mass);
    hash_combine(traj.rk4_dt);
    hash_combine(traj.rk4_min_steps);
    hash_combine(traj.rk4_max_steps);
    hash_combine(traj.enable_lookup_table);
    hash_combine(traj.table_range_max);
    hash_combine(traj.table_range_step);
    hash_combine(traj.table_pitch_min);
    hash_combine(traj.table_pitch_max);
    hash_combine(traj.table_pitch_step);
    return hash;
}

} // namespace

void register_planner_systems(talos::Scheduler& scheduler) {
    // Note: L4Config is already in world from register_aimer_systems

    // =========================================================================
    // MPC Planner System (consumes TargetPrediction, produces GimbalPlan)
    // =========================================================================

    scheduler.add_system<talos::pool_compute>(
        "l4_mpc_planner",
        [planner = std::make_shared<MpcPlanner>(MpcPlannerConfig{}),
         solver = std::shared_ptr<trajectory::IBallisticSolver>{}, last_config_hash = size_t{0}](
            talos::res<L4Config> cfg,
            talos::spmc<TargetPrediction, AimerOutputChannelTopic> aimer_in,
            talos::res<std::shared_ptr<fast_tf::CoordinateSystem<1024>>> tf_buffer,
            talos::spmc_mut<GimbalPlan, ArmorGimbalPlanChannelTopic> armor_plan_out) mutable {
            const auto prediction = aimer_in.read();
            if (!prediction) {
                return;
            }

            const size_t current_config_hash = solver_config_hash(cfg->mpc_planner);

            // Recreate solver when type changes or config changes significantly
            if (!solver || last_config_hash != current_config_hash) {
                solver = std::shared_ptr<trajectory::IBallisticSolver>(trajectory::create_solver(
                    cfg->mpc_planner.compensator_type, cfg->mpc_planner.trajectory));
                last_config_hash = current_config_hash;
            }

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
                viz_debug.trajectory       = solver->generate_trajectory(
                    plan.pitch, cfg->mpc_planner.trajectory.bullet_speed,
                    kMaxVisualizationDistance);
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
