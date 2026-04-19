#pragma once

#include "L5_weapon/config.hpp"
#include "scheduler/thin.hpp"

namespace fcs::L5 {

struct WeaponControllerConfig;

void register_enhanced_weapon_system(talos::Scheduler& scheduler, L5Config&& config);

} // namespace fcs::L5
