#include "fcs/L4_planning/planning_systems.hpp"

#include "fcs/L3_estimation/tracker/types.hpp"
#include "fcs/L4_planning/gimbal_planner/types.hpp"
#include "fcs/L4_planning/mpc_planner/config.hpp"
#include "fcs/L4_planning/mpc_planner/mpc_planner.hpp"
#include "fcs/core/channel_tags.hpp"
#include "fcs/core/time.hpp"
#include "math/fast_tf/error.hpp"
#include "math/fast_tf/frame.hpp"
#include "scheduler/ecs/channel_types.hpp"
#include "scheduler/ecs/execution_policy.hpp"
#include "scheduler/scheduler/scheduler.hpp"

namespace fcs::L4 {

void register_mpc_planning_systems(
    talos::Scheduler& scheduler, std::shared_ptr<MpcGimbalPlanner> planner, double bullet_speed) {
    scheduler.add_system<talos::pool_compute>(
        "l4_mpc_planner",
        [planner_instance = std::move(planner), bullet_speed](
            talos::spmc<fcs::L3::TrackerOutput, TrackerOutputChannelTag> tracker_in,
            talos::res<std::shared_ptr<fast_tf::CoordinateSystem<1024>>> tf_buffer,
            talos::spmc_mut<GimbalPlan, GimbalPlanChannelTag> plan_out) {
            auto target = tracker_in.read();
            if (!target || !target->is_tracking()) {
                return;
            }

            const uint64_t current_time_ns = clock::now_ns();

            // FIXME timestamp unaligned
            auto muzzle_tf =
                fast_tf::lookup_clamped<fast_tf::odom_fuxk_frame, fast_tf::muzzle_link_fuxk_frame>(
                    **tf_buffer, fast_tf::FuckTransform<fast_tf::muzzle_link_fuxk_frame>{},
                    current_time_ns);
            auto gimbal_tf =
                fast_tf::lookup_clamped<fast_tf::odom_fuxk_frame, fast_tf::gimbal_link_fuxk_frame>(
                    **tf_buffer, fast_tf::FuckTransform<fast_tf::gimbal_link_fuxk_frame>{},
                    current_time_ns);
            if (!muzzle_tf) {
                SPDLOG_WARN("muzzle_tf not found for {}", fast_tf::to_string(muzzle_tf.error()));
                return;
            }
            if (!gimbal_tf) {
                SPDLOG_WARN("gimbal_tf not found for {}", fast_tf::to_string(gimbal_tf.error()));
                return;
            }

            if (auto result = planner_instance->plan(
                    *target, current_time_ns, bullet_speed, *muzzle_tf, *gimbal_tf)) {
                GimbalPlan plan;
                plan.timestamp_ns       = result->timestamp_ns;
                plan.yaw                = result->yaw;
                plan.pitch              = result->pitch;
                plan.yaw_vel            = result->yaw_vel;
                plan.pitch_vel          = result->pitch_vel;
                plan.yaw_acc            = result->yaw_acc;
                plan.pitch_acc          = result->pitch_acc;
                plan.distance           = result->distance;
                plan.fire_advice        = result->fire_advice;
                plan.tracking_error     = result->tracking_error;
                plan.predicted_future   = result->predicted_future;
                plan.predicted_position = result->predicted_position;
                plan.trajectory         = std::move(result->planned_trajectory);

                // Copy MPC debug data for Foxglove visualization
                plan.mpc_dt          = result->mpc_dt;
                plan.mpc_current_idx = result->mpc_current_idx;
                plan.ref_yaw         = std::move(result->ref_yaw);
                plan.ref_yaw_vel     = std::move(result->ref_yaw_vel);
                plan.ref_pitch       = std::move(result->ref_pitch);
                plan.ref_pitch_vel   = std::move(result->ref_pitch_vel);
                plan.opt_yaw         = std::move(result->opt_yaw);
                plan.opt_yaw_vel     = std::move(result->opt_yaw_vel);
                plan.opt_pitch       = std::move(result->opt_pitch);
                plan.opt_pitch_vel   = std::move(result->opt_pitch_vel);
                plan.ctrl_yaw_acc    = std::move(result->ctrl_yaw_acc);
                plan.ctrl_pitch_acc  = std::move(result->ctrl_pitch_acc);

                // Copy extended MPC debug data
                plan.mpc_iterations              = result->mpc_iterations;
                plan.mpc_solve_time_us           = result->mpc_solve_time_us;
                plan.mpc_converged               = result->mpc_converged;
                plan.cost_yaw_pos                = std::move(result->cost_yaw_pos);
                plan.cost_yaw_vel                = std::move(result->cost_yaw_vel);
                plan.cost_pitch_pos              = std::move(result->cost_pitch_pos);
                plan.cost_pitch_vel              = std::move(result->cost_pitch_vel);
                plan.total_cost                  = result->total_cost;
                plan.constraint_yaw_acc_margin   = std::move(result->constraint_yaw_acc_margin);
                plan.constraint_pitch_acc_margin = std::move(result->constraint_pitch_acc_margin);

                plan_out.write(plan);
            }
        });
}

void register_mpc_planning_systems_with_config(
    talos::Scheduler& scheduler, const MpcPlannerConfig& config) {
    auto planner = std::make_shared<MpcGimbalPlanner>(config);
    register_mpc_planning_systems(scheduler, std::move(planner), config.trajectory.bullet_speed);
}

} // namespace fcs::L4
