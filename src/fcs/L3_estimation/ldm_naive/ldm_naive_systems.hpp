#pragma once

#include "ldm_tracker.hpp"
#include "scheduler/scheduler.hpp"

namespace fcs::L3::ldm {

/// Register naive LDM estimation system with the scheduler.
void register_naive_ldm_systems(talos::Scheduler& scheduler, const NaiveLdmConfig& config);

} // namespace fcs::L3::ldm
