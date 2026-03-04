#pragma once

#include "scheduler/scheduler.hpp"

namespace fcs::chiral {

/// Register chiral data collection system
///
/// Pipeline:
///   TrackerOutput [SPMC] -> ChiralCollectorSystem [pool_compute]
///                                ↓
///                         Shared Memory (TalosData)
///
/// ## Parameters
///
/// - `scheduler`: the scheduler to register systems with
void register_chiral_collector_system(talos::Scheduler& scheduler);

} // namespace fcs::chiral
