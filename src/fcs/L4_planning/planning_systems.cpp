#include "fcs/L4_planning/planning_systems.hpp"

#include "fcs/L3_estimation/tracker/types.hpp"
#include "fcs/L4_planning/gimbal_planner/planner.hpp"
#include "fcs/L4_planning/gimbal_planner/trajectory_compensator.hpp"
#include "fcs/L4_planning/gimbal_planner/types.hpp"
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

} // namespace fcs::L4
