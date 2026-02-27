#pragma once

#include "scheduler/thin.hpp"

namespace fcs::L5 {

struct WeaponControllerConfig;

void register_enhanced_weapon_system(talos::Scheduler& scheduler, WeaponControllerConfig&& config);

} // namespace fcs::L5
