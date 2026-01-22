#pragma once

#include "fcs/L4_planning/trajectory_config.hpp"

namespace talos {
class Scheduler;
} // namespace talos

namespace fcs::L5 {

struct WeaponControllerConfig;

void register_enhanced_weapon_system(talos::Scheduler& scheduler, WeaponControllerConfig&& config);

} // namespace fcs::L5
