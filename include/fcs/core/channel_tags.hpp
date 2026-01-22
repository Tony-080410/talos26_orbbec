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

struct RuneObservationChannelTag {};
struct RuneDebugFrameChannelTag {};

struct EnergyMeterStateChannelTag {};

// ============================================================================
// L4 Planning Pipeline Channel Tags
// ============================================================================

/// @brief Candidate gimbal plan from armor pipeline (Planner → PlanMux)
/// Data type: fcs::L4::GimbalPlan
/// Mode: SPMC (single producer, multiple consumers)
struct ArmorGimbalPlanChannelTag {};

/// @brief Candidate gimbal plan from rune pipeline (Planner → PlanMux)
/// Data type: fcs::L4::GimbalPlan
/// Mode: SPMC (single producer, multiple consumers)
struct RuneGimbalPlanChannelTag {};

/// @brief Final gimbal plan channel (PlanMux → Weapon System)
/// Data type: fcs::L4::GimbalPlan
/// Mode: SPMC (single producer, multiple consumers)
struct GimbalPlanChannelTag {};

// ============================================================================
// L5 Weapon System Channel Tags
// ============================================================================

/// @brief Weapon command channel (Fire Control → Output Interface)
/// Data type: fcs::L5::WeaponCommand
/// Mode: SPMC (single producer, multiple consumers)
struct WeaponCommandChannelTag {};

} // namespace fcs