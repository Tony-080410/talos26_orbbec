#pragma once

#include "scheduler/scheduler.hpp"

namespace fcs::L3::ldm {

/// Naive LDM estimation: solver output directly through L4 to weapon control.
struct NaiveLdmConfig {};

/// Register naive LDM estimation system with the scheduler.
void register_naive_ldm_systems(talos::Scheduler& scheduler, const NaiveLdmConfig& config);

} // namespace fcs::L3::ldm
