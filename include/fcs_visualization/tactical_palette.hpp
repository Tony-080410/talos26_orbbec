#pragma once

#include <cmath>
#include <foxglove/schemas.hpp>

namespace fcs::visualization::tactical {

// ============================================================================
// SEMANTIC NAMESPACES BY FCS LAYER
// ============================================================================
/// Design Philosophy:
/// - **Temporal Coding**: Past (ghost silver) → Present (neon cyan) → Future (amber)
/// - **Certainty Coding**: High confidence (bright, saturated) → Low confidence (dim, transparent)
/// - **Decision Coding**: Fire (luminous lime) → Hold (muted slate)
///
/// Usage Example:
/// ```cpp
/// using namespace tactical;
/// entity.color = L3::TRACKING_LOCKED;  // Compiler-enforced semantic color
/// entity.size = L3::TARGET_SIZE;       // Layer-appropriate sizing
/// ```

namespace L2 {
// =========================================================================
// L2 (Perception) - Measurement layer (raw sensor data)
// =========================================================================

/// Past measurements - ghost silver, low opacity
constexpr foxglove::schemas::Color MEASUREMENT_GHOST = {0.75, 0.75, 0.75, 0.25};

/// Measurements with confidence - higher visibility
constexpr foxglove::schemas::Color MEASUREMENT_CONFIDENCE = {0.60, 0.60, 0.60, 0.50};

/// Armor plate size for measurement visualization
constexpr double ARMOR_SIZE = 0.06;

/// Label font size for measurements
constexpr double LABEL_FONT_SIZE = 0.05;

} // namespace L2

namespace L3 {
// =========================================================================
// L3 (Estimation) - Tracking layer (state estimation)
// =========================================================================

/// Tracking locked - neon cyan, high saturation, high opacity
constexpr foxglove::schemas::Color TRACKING_LOCKED = {0.00, 1.00, 0.95, 0.90};

/// Acquiring target - muted yellow-green, medium opacity
constexpr foxglove::schemas::Color TRACKING_ACQUIRING = {0.80, 0.80, 0.40, 0.60};

/// Temporarily lost - warning amber-orange, medium-low opacity
constexpr foxglove::schemas::Color TRACKING_WARNING = {1.00, 0.60, 0.20, 0.50};

/// Lost target - crimson red, low opacity
constexpr foxglove::schemas::Color TRACKING_LOST = {1.00, 0.20, 0.20, 0.40};

/// Predicted position - electric amber
constexpr foxglove::schemas::Color PREDICTION_AMBER = {1.00, 0.75, 0.00, 0.80};

/// Uncertainty visualization - orange, very low opacity
constexpr foxglove::schemas::Color UNCERTAINTY_ORANGE = {1.00, 0.50, 0.00, 0.20};

/// Target sphere size
constexpr double TARGET_SIZE = 0.10;

/// Armor plate size for predicted positions
constexpr double ARMOR_PLATE_SIZE = 0.10;

/// Uncertainty sphere scale factor
constexpr double UNCERTAINTY_SCALE = 0.20;

/// Robot center marker size
constexpr double ROBOT_CENTER_SIZE = 0.18;

/// Outpost center marker size
constexpr double OUTPOST_CENTER_SIZE = 0.20;

/// Label offset above armor plates
constexpr double LABEL_OFFSET_Z = 0.12;

} // namespace L3

namespace L4 {
// =========================================================================
// L4 (Planning) - Prediction layer (MPC, gimbal control)
// =========================================================================

/// Gimbal aim point - fire ready (luminous lime)
constexpr foxglove::schemas::Color GIMBAL_AIM_FIRE = {0.50, 1.00, 0.30, 0.95};

/// Gimbal aim point - holding (muted slate)
constexpr foxglove::schemas::Color GIMBAL_AIM_HOLD = {0.40, 0.40, 0.45, 0.60};

/// MPC present state - cyan, high saturation
constexpr foxglove::schemas::Color MPC_PRESENT = {0.00, 1.00, 0.80, 0.85};

/// MPC reference trajectory - amber
constexpr foxglove::schemas::Color MPC_REFERENCE = {1.00, 0.70, 0.00, 0.70};

/// Bullet trajectory - fire mode
constexpr foxglove::schemas::Color TRAJECTORY_FIRE = {0.50, 1.00, 0.30, 0.80};

/// Bullet trajectory - hold mode
constexpr foxglove::schemas::Color TRAJECTORY_HOLD = {0.30, 0.40, 0.50, 0.50};

/// Selection marker size
constexpr double SELECTION_SIZE = 0.12;

/// Prediction marker size
constexpr double PREDICTION_SIZE = 0.10;

/// Trajectory dot size
constexpr double TRAJECTORY_DOT = 0.03;

} // namespace L4

namespace L5 {
// =========================================================================
// L5 (Weapon) - Action layer (fire decisions)
// =========================================================================

/// Firing now - bright green, full opacity
constexpr foxglove::schemas::Color FIRE_EXECUTE = {0.30, 1.00, 0.20, 1.00};

/// Ready to fire - luminous lime
constexpr foxglove::schemas::Color FIRE_READY = {0.50, 1.00, 0.30, 0.90};

/// Cooldown state - amber-orange
constexpr foxglove::schemas::Color FIRE_COOLDOWN = {0.80, 0.60, 0.20, 0.70};

/// Fire aborted - crimson red
constexpr foxglove::schemas::Color FIRE_ABORTED = {1.00, 0.20, 0.20, 0.80};

} // namespace L5

// ============================================================================
// CROSS-LAYER STYLES
// ============================================================================

namespace Velocity {
// =========================================================================
// Velocity vectors (used across L2/L3/L4)
// =========================================================================

/// Linear velocity color - cyan-green
constexpr foxglove::schemas::Color LINEAR = {0.00, 0.90, 0.70, 0.80};

/// Angular velocity color - amber-yellow
constexpr foxglove::schemas::Color ANGULAR = {1.00, 0.70, 0.00, 0.80};

/// Arrow shaft diameter
constexpr double ARROW_SHAFT_DIAMETER = 0.02;

/// Arrow head diameter
constexpr double ARROW_HEAD_DIAMETER = 0.04;

/// Minimum arrow length (prevents zero-length arrows)
constexpr double ARROW_MIN_LENGTH = 0.0001;

/// Shaft length ratio (of total arrow length)
constexpr double ARROW_SHAFT_RATIO = 0.8;

/// Head length ratio (of total arrow length)
constexpr double ARROW_HEAD_RATIO = 0.2;

} // namespace Velocity

// ============================================================================
// TEXT STYLES
// ============================================================================

namespace Text {
// =========================================================================
// Typography - Labels and annotations
// =========================================================================

/// Primary text - white, high opacity
constexpr foxglove::schemas::Color PRIMARY = {1.00, 1.00, 1.00, 0.95};

/// Secondary text - light gray
constexpr foxglove::schemas::Color SECONDARY = {0.80, 0.80, 0.80, 0.80};

/// Warning text - amber
constexpr foxglove::schemas::Color WARNING = {1.00, 0.70, 0.20, 0.95};

/// Error text - red
constexpr foxglove::schemas::Color ERROR = {1.00, 0.30, 0.30, 1.00};

/// Small font size
constexpr double SIZE_SMALL = 0.04;

/// Medium font size
constexpr double SIZE_MEDIUM = 0.06;

/// Large font size
constexpr double SIZE_LARGE = 0.08;

/// Default font size
constexpr double SIZE_DEFAULT = SIZE_MEDIUM;

/// Billboard mode (text always faces camera)
constexpr bool BILLBOARD_ENABLED = true;

} // namespace Text

// ============================================================================
// GEOMETRY CONSTANTS
// ============================================================================

namespace Geometry {
// =========================================================================
// Physical dimensions for armor plates
// =========================================================================

/// Armor cube thickness
constexpr double ARMOR_THICKNESS = 0.03;

/// Small armor height (balance, forward)
constexpr double ARMOR_HEIGHT_SMALL = 0.135;

/// Big armor height (engine, base)
constexpr double ARMOR_HEIGHT_BIG = 0.23;

/// Armor width
constexpr double ARMOR_WIDTH = 0.125;

/// Armor tilt angle (15 degrees in radians)
constexpr double ARMOR_TILT_ANGLE = 0.2618;

} // namespace Geometry

// ============================================================================
// TEMPORAL CONSTANTS
// ============================================================================

namespace Temporal {
// =========================================================================
// Time-related visualization constants
// =========================================================================

/// Scene entity lifetime (nanoseconds) - 100ms default
constexpr uint64_t ENTITY_LIFETIME_NS = 100'000'000ULL;

/// MPC trajectory extrapolation velocity (m/s)
constexpr double MPC_EXTRAPOLATION_VEL = 5.0;

} // namespace Temporal

// ============================================================================
// LEGACY ALIASES (for gradual migration)
// ============================================================================

namespace Legacy {
// These aliases help with incremental migration from old code
// Will be removed in future versions

constexpr foxglove::schemas::Color COLOR_YELLOW = L4::GIMBAL_AIM_HOLD;
constexpr foxglove::schemas::Color COLOR_CYAN   = L3::TRACKING_LOCKED;
constexpr foxglove::schemas::Color COLOR_GREEN  = L4::GIMBAL_AIM_FIRE;
constexpr foxglove::schemas::Color COLOR_WHITE  = L2::MEASUREMENT_GHOST;
constexpr foxglove::schemas::Color COLOR_RED    = L3::TRACKING_LOST;
constexpr foxglove::schemas::Color COLOR_ORANGE = L3::PREDICTION_AMBER;

constexpr double SPHERE_SIZE_DEFAULT = L3::TARGET_SIZE;
constexpr double SPHERE_SIZE_SMALL   = L3::ARMOR_PLATE_SIZE;
constexpr double SPHERE_SIZE_TINY    = L4::TRAJECTORY_DOT;

} // namespace Legacy

// ============================================================================
// COMPOSED STYLES
// ============================================================================

/// @brief Get tracker status color based on TrackerStatus enum
/// @param status_int The tracker status as integer (Idle=0, Detecting=1, Tracking=2, TempLost=3)
/// @return Corresponding semantic color
inline foxglove::schemas::Color tracker_status_color(int status_int) noexcept {
    // fcs::L3::TrackerStatus values: Idle=0, Detecting=1, Tracking=2, TempLost=3
    switch (status_int) {
    case 2: return L3::TRACKING_LOCKED;    // Tracking
    case 1: return L3::TRACKING_ACQUIRING; // Detecting
    case 3: return L3::TRACKING_WARNING;   // TempLost
    case 0:                                // Idle
    default: return L2::MEASUREMENT_GHOST;
    }
}

/// @brief Get fire decision color
/// @param can_fire Whether fire is permitted
/// @return Green if fire, slate if hold
inline foxglove::schemas::Color fire_decision_color(bool can_fire) noexcept {
    return can_fire ? L4::TRAJECTORY_FIRE : L4::TRAJECTORY_HOLD;
}

/// @brief Get temporal fade color for trajectory visualization
/// @param base_color Base color to fade
/// @param temporal_distance Temporal distance from present (0 = now)
/// @return Color with adjusted alpha
inline foxglove::schemas::Color
    temporal_fade(const foxglove::schemas::Color& base_color, int temporal_distance) noexcept {
    const int dist = std::abs(temporal_distance);
    double alpha   = 1.0;
    if (dist == 0)
        alpha = 1.00;
    else if (dist <= 2)
        alpha = 0.80;
    else if (dist <= 4)
        alpha = 0.50;
    else
        alpha = 0.25;

    return {base_color.r, base_color.g, base_color.b, static_cast<float>(alpha * base_color.a)};
}

} // namespace fcs::visualization::tactical
