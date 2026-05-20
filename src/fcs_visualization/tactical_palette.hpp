#pragma once

#include <cmath>
#include <foxglove/schemas.hpp>

namespace fcs::visualization::tactical {

// ============================================================================
// SEMANTIC COLOR SYSTEM — 5 CANONICAL COLORS
// ============================================================================
/// Design Philosophy:
/// - **Each color means ONE thing only** — zero ambiguity
/// - **Layer namespaces derive from these** — L2::MEASUREMENT = Semantic::INERT
/// - **Alpha is per-use** — Semantic:: defines RGB at full alpha; use .alpha() to adjust
///
/// | Color  | Name               | Meaning                    |
/// |--------|--------------------|----------------------------|
/// | Cyan   | STATUS_ACTIVE      | System operating/tracking  |
/// | Lime   | DECISION_POSITIVE  | Fire / pass / good         |
/// | Crimson| DECISION_NEGATIVE  | Fail / abort / lost        |
/// | Amber  | ALERT              | Caution / hold / runner-up |
/// | Slate  | INERT              | Background / context       |
///
/// Visual hierarchy: Primary (2x Secondary) > Secondary (2x Tertiary) > Tertiary
/// Size ratio enforced: 0.14 / 0.07 / 0.035

namespace Semantic {

// RGB-only canonical hues — alpha=1.0, layers set their own alpha per-use
constexpr ::foxglove::schemas::Color STATUS_ACTIVE     = {0.00, 0.95, 0.90, 1.00};
constexpr ::foxglove::schemas::Color DECISION_POSITIVE = {0.50, 1.00, 0.30, 1.00};
constexpr ::foxglove::schemas::Color DECISION_NEGATIVE = {1.00, 0.20, 0.20, 1.00};
constexpr ::foxglove::schemas::Color ALERT             = {1.00, 0.75, 0.00, 1.00};
constexpr ::foxglove::schemas::Color INERT             = {0.50, 0.55, 0.60, 1.00};

} // namespace Semantic

// ============================================================================
// L2 — PERCEPTION (Measurement layer)
// ============================================================================
namespace L2 {

constexpr ::foxglove::schemas::Color MEASUREMENT_GHOST      = {0.75, 0.75, 0.75, 0.25};
constexpr ::foxglove::schemas::Color MEASUREMENT_CONFIDENCE = {0.60, 0.60, 0.60, 0.50};

constexpr double ARMOR_SIZE      = 0.06;
constexpr double LABEL_FONT_SIZE = 0.05;

} // namespace L2

// ============================================================================
// L3 — ESTIMATION (Tracking layer)
// ============================================================================
namespace L3 {

constexpr ::foxglove::schemas::Color TRACKING_LOCKED    = {0.00, 1.00, 0.95, 0.90};
constexpr ::foxglove::schemas::Color TRACKING_ACQUIRING = {0.80, 0.80, 0.40, 0.60};
constexpr ::foxglove::schemas::Color TRACKING_WARNING   = {1.00, 0.60, 0.20, 0.50};
constexpr ::foxglove::schemas::Color TRACKING_LOST      = {1.00, 0.20, 0.20, 0.40};
constexpr ::foxglove::schemas::Color PREDICTION_AMBER   = {1.00, 0.75, 0.00, 0.80};
constexpr ::foxglove::schemas::Color UNCERTAINTY_ORANGE = {1.00, 0.50, 0.00, 0.20};

constexpr double TARGET_SIZE         = 0.10;
constexpr double ARMOR_PLATE_SIZE    = 0.10;
constexpr double UNCERTAINTY_SCALE   = 0.20;
constexpr double ROBOT_CENTER_SIZE   = 0.18;
constexpr double OUTPOST_CENTER_SIZE = 0.20;
constexpr double LABEL_OFFSET_Z      = 0.12;

} // namespace L3

// ============================================================================
// L4 — PLANNING (Gimbal, MPC, prediction)
// ============================================================================
namespace L4 {

// Decision colors — ring COLOR encodes fire/hold
constexpr ::foxglove::schemas::Color GIMBAL_AIM_FIRE = {0.50, 1.00, 0.30, 0.95};
constexpr ::foxglove::schemas::Color GIMBAL_AIM_HOLD = {1.00, 0.75, 0.00, 0.85};

// Context colors — background elements
constexpr ::foxglove::schemas::Color FUTURE_ARMOR   = {0.45, 0.55, 0.65, 0.45};
constexpr ::foxglove::schemas::Color SPATIAL_LINK   = {0.50, 0.55, 0.60, 0.35};
constexpr ::foxglove::schemas::Color SELECTION_LINK = {0.60, 0.90, 0.40, 0.60};

// Selection trace — candidate hierarchy
constexpr ::foxglove::schemas::Color CANDIDATE_RUNNER_UP  = {1.00, 0.75, 0.20, 0.75};
constexpr ::foxglove::schemas::Color CANDIDATE_ELIMINATED = {0.60, 0.60, 0.60, 0.25};

// Trajectory colors
constexpr ::foxglove::schemas::Color TRAJECTORY_FIRE = {0.50, 1.00, 0.30, 0.10};
constexpr ::foxglove::schemas::Color TRAJECTORY_HOLD = {0.30, 0.40, 0.50, 0.05};

// MPC colors
constexpr ::foxglove::schemas::Color MPC_PRESENT   = {0.00, 1.00, 0.80, 0.40};
constexpr ::foxglove::schemas::Color MPC_REFERENCE = {1.00, 0.70, 0.00, 0.35};

// Sizes — 3-tier hierarchy (2x ratio between tiers)
constexpr double SPATIAL_LINK_THICKNESS    = 0.008;
constexpr double TRAJECTORY_LINE_THICKNESS = 0.008;
constexpr double SELECTION_SIZE            = 0.06;
constexpr double PREDICTION_SIZE           = 0.04;
constexpr double TRAJECTORY_DOT            = 0.017;
constexpr double RING_HEIGHT               = 0.005;
constexpr double RING_SCALE                = 1.6;

} // namespace L4

// ============================================================================
// L5 — WEAPON (Fire decisions)
// ============================================================================
namespace L5 {

constexpr ::foxglove::schemas::Color FIRE_EXECUTE  = {0.30, 1.00, 0.20, 1.00};
constexpr ::foxglove::schemas::Color FIRE_READY    = {0.50, 1.00, 0.30, 0.90};
constexpr ::foxglove::schemas::Color FIRE_COOLDOWN = {0.80, 0.60, 0.20, 0.70};
constexpr ::foxglove::schemas::Color FIRE_ABORTED  = {1.00, 0.20, 0.20, 0.80};

} // namespace L5

// ============================================================================
// CROSS-LAYER STYLES
// ============================================================================
namespace Velocity {

constexpr ::foxglove::schemas::Color LINEAR  = Semantic::STATUS_ACTIVE;
constexpr ::foxglove::schemas::Color ANGULAR = Semantic::ALERT;

constexpr double ARROW_SHAFT_DIAMETER = 0.02;
constexpr double ARROW_HEAD_DIAMETER  = 0.04;
constexpr double ARROW_MIN_LENGTH     = 0.0001;
constexpr double ARROW_SHAFT_RATIO    = 0.8;
constexpr double ARROW_HEAD_RATIO     = 0.2;

} // namespace Velocity

// ============================================================================
// TEXT STYLES
// ============================================================================
namespace Text {

constexpr ::foxglove::schemas::Color PRIMARY   = {1.00, 1.00, 1.00, 0.95};
constexpr ::foxglove::schemas::Color SECONDARY = {0.80, 0.80, 0.80, 0.80};
constexpr ::foxglove::schemas::Color WARNING   = Semantic::ALERT;
constexpr ::foxglove::schemas::Color ERROR     = Semantic::DECISION_NEGATIVE;

constexpr double SIZE_SMALL      = 0.04;
constexpr double SIZE_MEDIUM     = 0.06;
constexpr double SIZE_LARGE      = 0.08;
constexpr double SIZE_DEFAULT    = SIZE_MEDIUM;
constexpr bool BILLBOARD_ENABLED = true;

} // namespace Text

// ============================================================================
// GEOMETRY CONSTANTS
// ============================================================================
namespace Geometry {

constexpr double ARMOR_THICKNESS    = 0.03;
constexpr double ARMOR_HEIGHT_SMALL = 0.135;
constexpr double ARMOR_HEIGHT_BIG   = 0.23;
constexpr double ARMOR_WIDTH        = 0.125;
constexpr double ARMOR_TILT_ANGLE   = 0.2618; // 15 degrees in radians

} // namespace Geometry

// ============================================================================
// TEMPORAL CONSTANTS
// ============================================================================
namespace Temporal {

constexpr uint64_t ENTITY_LIFETIME_NS  = 100'000'000ULL; // 100ms default
constexpr double MPC_EXTRAPOLATION_VEL = 5.0;

} // namespace Temporal

// ============================================================================
// LEGACY ALIASES (for gradual migration)
// ============================================================================
namespace Legacy {

constexpr ::foxglove::schemas::Color COLOR_YELLOW = L4::GIMBAL_AIM_HOLD;
constexpr ::foxglove::schemas::Color COLOR_CYAN   = L3::TRACKING_LOCKED;
constexpr ::foxglove::schemas::Color COLOR_GREEN  = L4::GIMBAL_AIM_FIRE;
constexpr ::foxglove::schemas::Color COLOR_WHITE  = L2::MEASUREMENT_GHOST;
constexpr ::foxglove::schemas::Color COLOR_RED    = L3::TRACKING_LOST;
constexpr ::foxglove::schemas::Color COLOR_ORANGE = L3::PREDICTION_AMBER;

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
inline ::foxglove::schemas::Color tracker_status_color(int status_int) noexcept {
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
/// @return Lime if fire, amber if hold
inline ::foxglove::schemas::Color fire_decision_color(bool can_fire) noexcept {
    return can_fire ? L4::TRAJECTORY_FIRE : L4::TRAJECTORY_HOLD;
}

/// @brief Get temporal fade color for trajectory visualization
/// @param base_color Base color to fade
/// @param temporal_distance Temporal distance from present (0 = now)
/// @return Color with adjusted alpha
inline ::foxglove::schemas::Color
    temporal_fade(const ::foxglove::schemas::Color& base_color, int temporal_distance) noexcept {
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
