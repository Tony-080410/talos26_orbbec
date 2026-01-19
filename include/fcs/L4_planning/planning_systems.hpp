#pragma once

#include <memory>

namespace talos {
class Scheduler;
} // namespace talos

namespace fcs::L4 {

class MpcGimbalPlanner;
struct MpcPlannerConfig;

/// Register L4 MPC planning pipeline systems
void register_mpc_planning_systems(::talos::Scheduler& scheduler, MpcPlannerConfig&& config);

} // namespace fcs::L4
