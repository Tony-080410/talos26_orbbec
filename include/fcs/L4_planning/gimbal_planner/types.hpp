#pragma once

#include "fcs/L3_estimation/tracker/types.hpp"
#include "fcs/core/channel_tags.hpp"

#include <Eigen/Core>
#include <cstdint>
#include <optional>
#include <vector>

namespace fcs::L4 {

// ============================================================================
// L4 Planning Output Types
// ============================================================================

/// @brief Gimbal plan containing targeting information
/// This struct represents the strategic decision output from the L4 planner
struct GimbalPlan {
    uint64_t timestamp_ns{0};

    // Target gimbal angles (radians, in odom frame)
    double yaw{0};
    double pitch{0};

    // Angular differences from current pose
    double yaw_diff{0};
    double pitch_diff{0};

    // Target info
    double distance{0};
    bool fire_advice{false};
    double confidence{0.0}; // Confidence level in the fire decision

    // Predicted armor position (for visualization)
    std::optional<Eigen::Vector3d> predicted_position;

    // Trajectory points (x, z) in muzzle_link frame (for visualization)
    std::vector<std::pair<double, double>> trajectory;
};

// ============================================================================
// L4 Channel Tags (for ECS)
// ============================================================================

// Re-export L3 types for convenience
using fcs::L3::TrackerOutput;
using fcs::L3::TrackerOutputChannelTag;

} // namespace fcs::L4