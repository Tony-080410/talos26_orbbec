#pragma once

#include "fcs/L4_planning/trajectory_config.hpp"

namespace talos {
class Scheduler;
} // namespace talos

namespace fcs::L5 {

struct WeaponControllerConfig;

/// Register enhanced L5 weapon system (GimbalPlan -> WeaponCommand).
/// Implemented in `src/fcs/L5_weapon/enhanced/weapon_systems.cpp`.
void register_enhanced_weapon_system_with_config_and_compensator(
    talos::Scheduler& scheduler, const WeaponControllerConfig& config,
    fcs::L4::TrajectoryCompensatorType compensator_type);

} // namespace fcs::L5
