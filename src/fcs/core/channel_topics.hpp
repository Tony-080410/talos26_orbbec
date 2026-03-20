#pragma once

namespace fcs {

// ============================================================================
// AT Legacy Armor Detection Pipeline Channel Topics
// ============================================================================

/// @brief Image frame channel (Camera → Detector)
/// Data type: AtLegacyImageFrame
/// Mode: SPMC (single producer, multiple consumers)
struct AtLegacyImageChannelTopic {};

/// @brief Detection result channel (Detector → PnPSolver)
/// Data type: AtLegacyArmorDetectionBatch
/// Mode: SPMC (single producer, multiple consumers)
struct AtLegacyDetectionChannelTopic {};

/// @brief Measurement result channel (PnPSolver → Tracker)
/// Data type: AtLegacyArmorMeasurementBatch
/// Mode: SPMC (single producer, multiple consumers)
struct AtLegacyMeasurementChannelTopic {};

// ============================================================================
// L3 Estimation Pipeline Channel Topics
// ============================================================================

/// @brief Tracker output channel (Tracker → Solver)
/// Data type: fcs::L3::TrackerOutput
/// Mode: SPMC (single producer, multiple consumers)
struct TrackerOutputChannelTopic {};

struct RuneObservationChannelTopic {};
struct RuneDebugFrameChannelTopic {};

struct EnergyMeterStateChannelTopic {};

// ============================================================================
// L4 Planning Pipeline Channel Topics
// ============================================================================

/// @brief Final gimbal plan channel (PlanMux → Weapon System)
/// Data type: fcs::L4::GimbalPlan
/// Mode: SPMC (single producer, multiple consumers)
struct GimbalPlanChannelTopic {};

// ============================================================================
// L5 Weapon System Channel Topics
// ============================================================================

/// @brief Weapon command channel (Fire Control → Output Interface)
/// Data type: fcs::L5::WeaponCommand
/// Mode: SPMC (single producer, multiple consumers)
struct WeaponCommandChannelTopic {};

/// @brief Runtime control-resource snapshot channel (IMU/resource owner -> capturer)
/// Data type: fcs::core::ControlResourceSnapshot
/// Mode: SPMC (single producer, multiple consumers)
struct RuntimeControlStateChannelTopic {};

} // namespace fcs
