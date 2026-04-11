#pragma once

// Keep this header as a thin API surface.
// The implementation lives in `src/fcs/L2_perception/at_legacy_systems.cpp`.

#include "scheduler/thin.hpp"
#include <expected>
#include <memory>
#include <string>

namespace fcs {
struct CameraConfig;
};
namespace fcs::L2 {

struct AtLegacyConfig;
class AtLegacyDetectorBackend;
class AtLegacyPnPSolver;

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
/// ## Parameters
///
/// - `scheduler`: the scheduler to register systems with
/// - `config`: L2 perception configuration (will be moved into resource)
void register_at_legacy_detection_systems(talos::Scheduler& scheduler) noexcept;

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
    create_detector_backend_handle(const AtLegacyConfig& config) noexcept;

/// Create PnP solver from camera info
/// @param config camera config
[[nodiscard]] std::shared_ptr<AtLegacyPnPSolver>
    create_pnp_solver(const CameraConfig& config) noexcept;

} // namespace fcs::L2
