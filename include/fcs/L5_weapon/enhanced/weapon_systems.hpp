#pragma once

#include "fcs/L4_planning/trajectory_config.hpp"

namespace talos {
class Scheduler;
} // namespace talos

namespace fcs::L4 {
struct MpcPlannerConfig;
} // namespace fcs::L4

namespace fcs::L5 {

struct WeaponControllerConfig;

/// Register enhanced L5 weapon system (GimbalPlan -> WeaponCommand).
///
/// ## Parameters
///
/// - `scheduler`: the scheduler to register systems with
/// - `config`: weapon controller configuration (will be moved into resource)
/// - `l4_config`: L4 MPC config (for compensator type, will be moved into resource)
void register_enhanced_weapon_system(
    talos::Scheduler& scheduler, WeaponControllerConfig&& config, fcs::L4::MpcPlannerConfig&& l4_config);

} // namespace fcs::L5
