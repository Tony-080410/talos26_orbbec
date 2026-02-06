#pragma once

namespace talos {
class Scheduler;
} // namespace talos

namespace fcs::L4 {

struct L4Config;

/// Register L4 Aimer + Planner pipeline systems
///
/// Unified architecture replacing the legacy monolithic planner:
/// - Aimer layer: RobotAimer, OutpostAimer, RuneAimer (pure prediction)
/// - Planner layer: MpcPlanner (trajectory optimization)
///
/// @param scheduler ECS scheduler
/// @param config Complete L4 configuration (aimers + planner)
void register_l4_planning_systems(::talos::Scheduler& scheduler, L4Config&& config);

} // namespace fcs::L4
