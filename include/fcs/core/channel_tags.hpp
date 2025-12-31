#pragma once

namespace fcs {

// ============================================================================
// AT Legacy Armor Detection Pipeline Channel Tags
// ============================================================================

/// @brief Image frame channel (Camera → Detector)
/// Data type: AtLegacyImageFrame
/// Mode: SPMC (single producer, multiple consumers)
struct AtLegacyImageChannelTag {};

/// @brief Detection result channel (Detector → PnPSolver)
/// Data type: AtLegacyArmorDetectionBatch
/// Mode: SPMC (single producer, multiple consumers)
struct AtLegacyDetectionChannelTag {};

/// @brief Measurement result channel (PnPSolver → Tracker)
/// Data type: AtLegacyArmorMeasurementBatch
/// Mode: SPMC (single producer, multiple consumers)
struct AtLegacyMeasurementChannelTag {};

// ============================================================================
// L3 Estimation Pipeline Channel Tags
// ============================================================================

/// @brief Tracker output channel (Tracker → Solver)
/// Data type: fcs::L3::TrackerOutput
/// Mode: SPMC (single producer, multiple consumers)
struct TrackerOutputChannelTag {};

/// @brief Solver output channel (Solver → Gimbal Control)
/// Data type: fcs::L3::GimbalCommand
/// Mode: SPMC (single producer, multiple consumers)
struct SolverOutputChannelTag {};

} // namespace fcs
