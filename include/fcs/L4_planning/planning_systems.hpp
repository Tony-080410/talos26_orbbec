#pragma once

#include <memory>

namespace talos {
class Scheduler;
} // namespace talos

namespace fcs::L4 {

class GimbalPlanner;
struct GimbalPlannerConfig;

/// Register L4 planning pipeline systems
///
/// Pipeline:
///   L3::TrackerOutput [SPMC] -→ PlanningSystem [pool_compute]
///                                    ↓
///                            L4::GimbalPlan [SPMC]
///
void register_planning_systems(
    ::talos::Scheduler& scheduler, std::shared_ptr<GimbalPlanner> planner);

/// Register L4 planning systems with gimbal planner config
void register_planning_systems_with_config(
    ::talos::Scheduler& scheduler, const GimbalPlannerConfig& config);

} // namespace fcs::L4
