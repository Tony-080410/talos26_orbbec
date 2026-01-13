#pragma once

// Keep this header as a thin API surface.
// The implementation lives in `src/fcs/L2_perception/at_legacy_systems.cpp`.

#include <expected>
#include <memory>
#include <string>

namespace talos {
class Scheduler;
} // namespace talos

namespace fcs {

struct AtLegacyConfig;
class AtLegacyDetectorBackend;
class AtLegacyPnPSolver;
struct CameraConfig;

// ============================================================================
// AT Legacy Armor Detection Pipeline Systems
// ============================================================================

/// Register AT Legacy detection pipeline systems with the scheduler
///
/// Pipeline architecture:
///   Camera System (external) → Image [SPMC]
///         ↓
///   ArmorDetectorSystem (pool_compute) → Detection [SPMC]
///         ↓
///   PnPSolverSystem (pool_compute) → Measurement [SPMC]
///         ↓
///   TrackerSystem (external)
///
/// Usage:
///   auto app = talos::App::create();
///
///   // Register resources
///   app.insert_resource<AtLegacyDetectorBackend>(std::move(backend));
///   app.insert_resource<AtLegacyPnPSolver>(std::move(solver));
///
///   // Register channels
///   app.init_spmc_channel<AtLegacyImageFrame, AtLegacyImageChannelTag>();
///   app.init_spmc_channel<AtLegacyArmorDetectionBatch, AtLegacyDetectionChannelTag>();
///   app.init_spmc_channel<AtLegacyArmorMeasurementBatch, AtLegacyMeasurementChannelTag>();
///
///   // Register systems
///   register_at_legacy_detection_systems(app);

void register_at_legacy_detection_systems(talos::Scheduler& scheduler);

// ============================================================================
// Configuration Helpers
// ============================================================================

struct DetectorBackendHandle {
    std::shared_ptr<AtLegacyDetectorBackend> backend;
    std::string name;
};

/// Create detector backend resource from unified config.
/// Returns the backend pointer plus a human-readable backend name.
[[nodiscard]] std::expected<DetectorBackendHandle, std::string>
    create_detector_backend_handle(const AtLegacyConfig& config);

/// Create PnP solver from camera info
/// @param config camera config
[[nodiscard]] std::shared_ptr<AtLegacyPnPSolver> create_pnp_solver(const CameraConfig& config);

} // namespace fcs
