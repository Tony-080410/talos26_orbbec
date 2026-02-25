#pragma once

#include "L3_estimation/tracker/types.hpp"
#include "core/channel_topics.hpp"

#include <Eigen/Core>
#include <cstdint>
#include <optional>
#include <vector>

namespace fcs::L4 {

// ============================================================================
// Visualization Debug Data (Optional)
// ============================================================================

/// Visualization debug data (optional)
///
/// Contains data for visualization purposes (e.g., Foxglove plots).
/// This data is optional and should only be populated when visualization
/// is enabled.
struct VisualizationDebugData {
    std::vector<std::pair<double, double>> trajectory; ///< Ballistic trajectory (x, z)
};

/// Source of gimbal plan.
enum class GimbalPlanSource : uint8_t {
    Armor = 1,
    Rune  = 2,
};

// ============================================================================
// Gimbal Plan - Final Output from L4 Planner
// ============================================================================

/// Gimbal plan - final L4 output consumed by L5.
///
/// L4 (Aimer + plan adapter) -> GimbalPlan -> L5 (reference trajectory + MPC) -> WeaponCommand
struct GimbalPlan {
    // ========================================================================
    // Basic Information
    // ========================================================================
    uint64_t timestamp_ns{0};
    double distance{0.0};
    std::optional<Eigen::Vector3d> predicted_position;
    uint64_t predicted_future{0};
    GimbalPlanSource source;

    // ========================================================================
    // Target Angles
    // ========================================================================
    double yaw{0.0};
    double pitch{0.0};
    double yaw_vel{0.0};
    double pitch_vel{0.0};
    double yaw_acc{0.0};
    double pitch_acc{0.0};

    // ========================================================================
    // Fire Advice
    // ========================================================================
    bool fire_advice{false};
    double tracking_error{0.0};

    // ========================================================================
    // Aimer's Original Advice (for debugging/comparison)
    // ========================================================================
    bool aimer_fire_advice{false};
    double aim_yaw{0.0};
    double aim_pitch{0.0};

    // ========================================================================
    // Optional Debug Data
    // ========================================================================
    std::optional<VisualizationDebugData> viz_debug;
};

// ============================================================================
// L4 Channel Topics (for ECS)
// ============================================================================

// Re-export L3 types for convenience
using fcs::L3::TrackerOutput;
using fcs::L3::TrackerOutputChannelTopic;

} // namespace fcs::L4
