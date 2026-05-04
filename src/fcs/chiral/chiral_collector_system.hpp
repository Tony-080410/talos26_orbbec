#pragma once

#include "scheduler/thin.hpp"

namespace fcs::chiral {

/// Register chiral data collection system
///
/// Pipeline:
///   SelectedTargetSnapshot [SPMC] -> ChiralCollectorSystem [pool_compute]
///                                      ↓
///                               Shared Memory (TalosData)
///
/// ## Parameters
///
/// - `scheduler`: the scheduler to register systems with
void register_chiral_collector_system(talos::Scheduler& scheduler);

} // namespace fcs::chiral
