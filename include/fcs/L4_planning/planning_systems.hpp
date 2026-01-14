#pragma once

#include <memory>

namespace talos {
class Scheduler;
} // namespace talos

namespace fcs::L4 {

class GimbalPlanner;
struct GimbalPlannerConfig;
class MpcGimbalPlanner;
struct MpcPlannerConfig;

/// Register L4 planning pipeline systems (legacy GimbalPlanner)
///
/// Pipeline:
///   L3::TrackerOutput [SPMC] -→ PlanningSystem [pool_compute]
///                                    ↓
///                            L4::GimbalPlan [SPMC]
///
void register_planning_systems(
    ::talos::Scheduler& scheduler, std::shared_ptr<GimbalPlanner> planner);

/// Register L4 planning systems with gimbal planner config (legacy)
void register_planning_systems_with_config(
    ::talos::Scheduler& scheduler, const GimbalPlannerConfig& config);

/// Register L4 MPC planning pipeline systems
///
/// Pipeline:
///   L3::TrackerOutput [SPMC] -→ MpcPlanningSystem [pool_compute]
///                                    ↓
///                            L4::GimbalPlan [SPMC]
///
/// Uses TinyMPC (ADMM-based QP solver) for optimal gimbal trajectory planning.
void register_mpc_planning_systems(
    ::talos::Scheduler& scheduler, std::shared_ptr<MpcGimbalPlanner> planner, double bullet_speed);

/// Register L4 MPC planning systems with config
void register_mpc_planning_systems_with_config(
    ::talos::Scheduler& scheduler, const MpcPlannerConfig& config);

} // namespace fcs::L4
