#pragma once

#include "fcs/L4_planning/config.hpp"

namespace talos {
class Scheduler;
}

namespace fcs::L4 {

/// Register planner systems with the scheduler
///
/// This function registers the MPC planner system that consumes AimerOutput
/// and produces optimized GimbalPlan results.
///
/// Note: L4Config must already be inserted into world (by register_aimer_systems).
///
/// @param scheduler ECS scheduler
void register_planner_systems(talos::Scheduler& scheduler);

} // namespace fcs::L4
