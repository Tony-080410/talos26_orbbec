#include "fcs/L4_planning/planning_systems.hpp"

#include "fcs/L3_estimation/tracker/types.hpp"
#include "fcs/L4_planning/gimbal_planner/planner.hpp"
#include "fcs/L4_planning/gimbal_planner/trajectory_compensator.hpp"
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
#include "scheduler/spin.hpp"

#include <spdlog/spdlog.h>

namespace fcs::L4 {

void register_planning_systems(
    talos::Scheduler& scheduler, std::shared_ptr<GimbalPlanner> planner) {
    scheduler.add_system<talos::pool_compute>(
        "l4_planner", [planner_instance = std::move(planner)](
                          talos::spmc<fcs::L3::TrackerOutput, TrackerOutputChannelTag> tracker_in,
                          talos::spmc_mut<GimbalPlan, GimbalPlanChannelTag> plan_out,
                          talos::res<std::shared_ptr<TrajectoryCompensator>> compensator,
                          talos::res<std::shared_ptr<fast_tf::CoordinateSystem<1024>>> tf_system) {
            auto target = tracker_in.read();
            if (!target || !target->is_tracking()) {
                return;
            }

            const uint64_t current_time_ns = clock::now_ns();

            while (true) {
                auto tf_result =
                    fast_tf::lookup<fast_tf::odom_fuxk_frame, fast_tf::muzzle_link_fuxk_frame>(
                        **tf_system,
                        fast_tf::FuckTransform<fast_tf::muzzle_link_fuxk_frame>{
                            fast_tf::TransformMatrix4d{}},
                        current_time_ns);

                if (!tf_result) {
                    if (tf_result.error() != fast_tf::TransformError::FutureExtrapolationRequired) {
                        SPDLOG_ERROR(
                            "Planner TF lookup failed at {} ns: {}", target->timestamp_ns,
                            fast_tf::to_string(tf_result.error()));
                        return;
                    }
                    SPIN_HINT();
                    continue;
                }

                if (auto result = planner_instance->plan(*target, current_time_ns, **tf_result)) {
                    plan_out.write(*result);
                }
                return;
            }
        });
}

void register_planning_systems_with_config(
    talos::Scheduler& scheduler, const GimbalPlannerConfig& config) {
    auto planner = std::make_shared<GimbalPlanner>(config);
    register_planning_systems(scheduler, std::move(planner));
}

void register_mpc_planning_systems(
    talos::Scheduler& scheduler, std::shared_ptr<MpcGimbalPlanner> planner, double bullet_speed) {
    scheduler.add_system<talos::pool_compute>(
        "l4_mpc_planner",
        [planner_instance = std::move(planner), bullet_speed](
            talos::spmc<fcs::L3::TrackerOutput, TrackerOutputChannelTag> tracker_in,
            talos::spmc_mut<GimbalPlan, GimbalPlanChannelTag> plan_out,
            talos::res<std::shared_ptr<fast_tf::CoordinateSystem<1024>>> tf_system) {
            auto target = tracker_in.read();
            if (!target || !target->is_tracking()) {
                return;
            }

            const uint64_t current_time_ns = clock::now_ns();

            // Wait for TF to be available (for yaw_diff/pitch_diff computation)
            while (true) {
                auto tf_result =
                    fast_tf::lookup<fast_tf::odom_fuxk_frame, fast_tf::muzzle_link_fuxk_frame>(
                        **tf_system,
                        fast_tf::FuckTransform<fast_tf::muzzle_link_fuxk_frame>{
                            fast_tf::TransformMatrix4d{}},
                        current_time_ns);

                if (!tf_result) {
                    if (tf_result.error() != fast_tf::TransformError::FutureExtrapolationRequired) {
                        SPDLOG_ERROR(
                            "MPC Planner TF lookup failed at {} ns: {}", target->timestamp_ns,
                            fast_tf::to_string(tf_result.error()));
                        return;
                    }
                    SPIN_HINT();
                    continue;
                }

                // Get current gimbal angles from TF
                const auto euler           = (**tf_result).euler_rot();
                const double current_yaw   = euler.yaw;
                const double current_pitch = euler.pitch;

                // Plan using MPC (pass current gimbal state)
                if (auto result = planner_instance->plan(
                        *target, current_time_ns, bullet_speed, current_yaw, current_pitch)) {
                    GimbalPlan plan;
                    plan.timestamp_ns       = result->timestamp_ns;
                    plan.yaw                = result->yaw;
                    plan.pitch              = result->pitch;
                    plan.yaw_diff           = result->yaw - current_yaw;
                    plan.pitch_diff         = result->pitch - current_pitch;
                    plan.yaw_vel            = result->yaw_vel;
                    plan.pitch_vel          = result->pitch_vel;
                    plan.yaw_acc            = result->yaw_acc;
                    plan.pitch_acc          = result->pitch_acc;
                    plan.distance           = result->distance;
                    plan.fire_advice        = result->fire_advice;
                    plan.tracking_error     = result->tracking_error;
                    plan.track_center       = result->track_center;
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

                    plan_out.write(plan);
                }
                return;
            }
        });
}

void register_mpc_planning_systems_with_config(
    talos::Scheduler& scheduler, const MpcPlannerConfig& config) {
    auto planner = std::make_shared<MpcGimbalPlanner>(config);
    register_mpc_planning_systems(scheduler, std::move(planner), config.trajectory.bullet_speed);
}

} // namespace fcs::L4
