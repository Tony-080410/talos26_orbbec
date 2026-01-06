#pragma once

#include "ecs/execution_policy.hpp"
#include "gimbal_planner/planner.hpp"
#include "gimbal_planner/trajectory_compensator.hpp"
#include "gimbal_planner/types.hpp"

#include "fcs/L3_estimation/tracker/types.hpp"
#include "fcs/core/channel_tags.hpp"
#include "fcs/core/time.hpp"
#include "math/fast_tf/error.hpp"
#include "scheduler/ecs/channel_types.hpp"

#include "scheduler/spin.hpp"
#include <memory>
#include <spdlog/spdlog.h>

namespace fcs::L4 {

/// Register L4 planning pipeline systems
///
/// Pipeline:
///   L3::TrackerOutput [SPMC] -→ PlanningSystem [pool_compute]
///                                    ↓
///                            L4::GimbalPlan [SPMC]
///
template <typename App>
void register_planning_systems(App& app, std::shared_ptr<GimbalPlanner> planner) {
    // PlanningSystem: L3::TrackerOutput → L4::GimbalPlan
    app.template add_system<talos::pool_compute>(
        "l4_planner", [planner_instance = std::move(planner)](
                          talos::spmc<fcs::L3::TrackerOutput, TrackerOutputChannelTag> tracker_in,
                          talos::spmc_mut<GimbalPlan, GimbalPlanChannelTag> plan_out,
                          talos::res<std::shared_ptr<TrajectoryCompensator>> compensator,
                          talos::res<std::shared_ptr<fast_tf::CoordinateSystem<1024>>> tf_system) {
            auto target = tracker_in.read();
            if (!target || !target->is_tracking()) {
                return;
            }

            // Get current real-time for prediction (not measurement timestamp!)
            // This matches at_vision's: solver_->solve(target_snapshot, now(), tf_buffer)
            const uint64_t current_time_ns = clock::now_ns();

            while (true) {
                // Lookup muzzle transform in odom frame
                // Use current time (not measurement timestamp!) to get latest gimbal pose
                // This matches at_vision's: lookupTransform(..., tf2::TimePointZero)
                auto tf_result =
                    fast_tf::lookup<fast_tf::odom_fuxk_frame, fast_tf::muzzle_link_fuxk_frame>(
                        **tf_system,
                        fast_tf::FuckTransform<fast_tf::muzzle_link_fuxk_frame>{
                            fast_tf::TransformMatrix4d{}},
                        current_time_ns);

                if (!tf_result) {
                    if (tf_result.error() != fast_tf::TransformError::FutureExtrapolationRequired) {
                        spdlog::error(
                            "Planner TF lookup failed at {} ns: {}", target->timestamp_ns,
                            fast_tf::to_string(tf_result.error()));
                        return;
                    }
                    SPIN_HINT();
                    continue;
                }

                // Plan gimbal command with real-time current_time_ns
                if (auto result = planner_instance->plan(*target, current_time_ns, **tf_result)) {
                    plan_out.write(*result);
                }
                return;
            }
        });
}

/// Register L4 planning systems with gimbal planner config
template <typename App>
void register_planning_systems_with_config(App& app, const GimbalPlannerConfig& config) {
    auto planner = std::make_shared<GimbalPlanner>(config);
    register_planning_systems(app, planner);
}

} // namespace fcs::L4
