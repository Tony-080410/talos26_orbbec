#pragma once

// Keep this header as a thin API surface.
// The implementation lives in `src/fcs/L3_estimation/tracker_systems.cpp`.

#include <memory>

namespace talos {
class Scheduler;
} // namespace talos

namespace fcs::L3 {

class Tracker;
struct TrackerConfig;

/// Register L3 tracker pipeline systems
///
/// Pipeline:
///   AtLegacyArmorMeasurementBatch [SPMC] -> TrackerSystem [pool_compute]
///                                               ↓
///                                        TrackerOutput [SPMC]
///
/// ## Parameters
///
/// - `scheduler`: the scheduler to register systems with
/// - `config`: tracker configuration (will be moved into resource)
void register_tracker_systems(talos::Scheduler& scheduler, TrackerConfig&& config);

/// Create tracker resource from config (implementation in .cpp).
[[nodiscard]] std::shared_ptr<Tracker> create_tracker(const TrackerConfig& config);

} // namespace fcs::L3
