#pragma once

#include "L3_estimation/tracker/types.hpp"
#include "L4_planning/aimer/types.hpp"
#include "L4_planning/plan_source.hpp"
#include "core/armor_types.hpp"
#include "core/channel_topics.hpp"
#include "core/trajectory/reference_trajectory.hpp"

#include <Eigen/Core>
#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
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

/// L4 planner seed used for reference trajectory generation.
///
/// Contains the raw L3 target state, used by L4 to build the full reference
/// trajectory before passing it to L5.
struct PlannerSeed {
    using State = std::variant<std::monostate, L3::RobotTargetState, L3::OutpostTargetState>;

    uint64_t state_timestamp_ns{0};
    State state{};
    bool target_jumped{false};
    std::optional<int> tracker_last_armor_id{};
    ArmorAimPhase aim_phase{ArmorAimPhase::SingleArmor};
    int selected_armor_id{0};
    ArmorType armor_type{ArmorType::Invalid};

    [[nodiscard]] bool is_robot() const noexcept {
        return std::holds_alternative<L3::RobotTargetState>(state);
    }

    [[nodiscard]] bool is_outpost() const noexcept {
        return std::holds_alternative<L3::OutpostTargetState>(state);
    }

    [[nodiscard]] const L3::RobotTargetState* robot_state() const noexcept {
        return std::get_if<L3::RobotTargetState>(&state);
    }

    [[nodiscard]] const L3::OutpostTargetState* outpost_state() const noexcept {
        return std::get_if<L3::OutpostTargetState>(&state);
    }
};

// ============================================================================
// Gimbal Plan - Final Output from L4 Planner
// ============================================================================

/// Gimbal plan - final L4 output consumed by L5.
///
/// L4 (Aimer + plan adapter) -> GimbalPlan -> L5 (reference trajectory + solver) -> WeaponCommand
struct GimbalPlan {
    // ========================================================================
    // Basic Information
    // ========================================================================
    uint64_t timestamp_ns{0};
    double distance{0.0};
    std::optional<Eigen::Vector3d> predicted_position;
    uint64_t predicted_future{0};
    GimbalPlanSource source{GimbalPlanSource::Armor};
    std::optional<PlannerSeed> planner_seed;

    /// Pre-built reference trajectory (optional).
    /// When populated by L4, L5 skips trajectory reconstruction and uses this directly.
    std::optional<core::trajectory::ReferenceTrajectory> reference_trajectory;

    // ========================================================================
    // Target Angles
    // ========================================================================
    double yaw{0.0};
    double pitch{0.0};
    ArmorAimPhase aim_phase{ArmorAimPhase::SingleArmor};
    int selected_armor_id{0};
    int rough_selected_armor_id{0};
    ArmorType armor_type{ArmorType::Invalid};
    double yaw_vel{0.0};
    double pitch_vel{0.0};
    double yaw_acc{0.0};
    double pitch_acc{0.0};

    // ========================================================================
    // Fire Advice
    // ========================================================================
    bool fire_advice{false};

    // ========================================================================
    // Aimer's Original Advice (for debugging/comparison)
    // ========================================================================
    double aim_yaw{0.0};
    double aim_pitch{0.0};
};

// ============================================================================
// L4 Channel Topics
// ============================================================================

using fcs::L3::TrackerOutputChannelTopic;

} // namespace fcs::L4
