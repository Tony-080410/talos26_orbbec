#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <cstdint>
#include <optional>
#include <variant>

#include "fcs/core/channel_topics.hpp"
#include "fcs/core/types.hpp"
#include <fmt/core.h>
#include <magic_enum.hpp>
#include <numbers>

namespace fcs::L3 {

// ============================================================================
// Tracker Status
// ============================================================================

enum class TrackerStatus : uint8_t {
    Idle      = 0, // Not tracking, waiting for first detection
    Detecting = 1, // Short-term detection, awaiting confirmation
    Tracking  = 2, // Stable tracking
    TempLost  = 3, // Temporarily lost, using prediction
};

// ============================================================================
// Robot Target State (11-dim model output)
// ============================================================================
// Robot Target State (11-dim model output)
// ============================================================================

struct RobotTargetState {
    // Center position (in odom frame)
    Eigen::Vector3d position{0, 0, 0}; // [xc, yc, z0]
    Eigen::Vector3d velocity{0, 0, 0}; // [vx, vy, vz]

    // Rotation
    double yaw{0};   // Armor 0 yaw angle
    double v_yaw{0}; // Angular velocity

    // Geometry
    double radius0{0.22}; // Armor 0,2 radius
    double radius1{0.22}; // Armor 1,3 radius
    double z1{0};         // z0 + h (armor 1,3 height)

    int armors_num{4};

    /// Calculate all armor poses [x, y, z, yaw]
    [[nodiscard]] std::array<Eigen::Vector4d, 4> armor_poses() const noexcept;
};

// ============================================================================
// Outpost Target State (7-dim model output)
// ============================================================================

struct OutpostTargetState {
    Eigen::Vector2d position{0, 0};    // [xc, yc]
    Eigen::Vector3d velocity{0, 0, 0}; // [vx, vy, vz]
    double yaw{0};
    double v_yaw{0};
    std::array<double, 3> z{0, 0, 0};  // z0, z1, z2

    static constexpr double radius  = 0.2765;
    static constexpr int armors_num = 3;

    /// Calculate all armor poses [x, y, z, yaw]
    [[nodiscard]] std::array<Eigen::Vector4d, 3> armor_poses() const noexcept;
};

// ============================================================================
// Filter Output (for InEKF vs ISRCKF comparison)
// ============================================================================

/// Output from a single filter for comparison
struct FilterOutput {
    bool active{false};
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
    Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};
    double yaw{0};
    double v_yaw{0};
    Eigen::Matrix3d position_covariance{Eigen::Matrix3d::Zero()};
};

/// Comparison result between InEKF and ISRCKF
struct TrackerComparison {
    FilterOutput isrckf; // ISRCKF output
    FilterOutput inekf;  // InEKF output

    // Position difference
    [[nodiscard]] Eigen::Vector3d position_diff() const noexcept {
        return isrckf.position - inekf.position;
    }

    [[nodiscard]] double position_distance() const noexcept { return position_diff().norm(); }

    // Velocity difference
    [[nodiscard]] Eigen::Vector3d velocity_diff() const noexcept {
        return isrckf.velocity - inekf.velocity;
    }

    [[nodiscard]] double velocity_distance() const noexcept { return velocity_diff().norm(); }

    // Yaw difference (account for wraparound)
    [[nodiscard]] double yaw_diff() const noexcept {
        double diff = isrckf.yaw - inekf.yaw;
        while (diff > M_PI)
            diff -= 2.0 * M_PI;
        while (diff < -M_PI)
            diff += 2.0 * M_PI;
        return diff;
    }

    // Both filters agree on tracking
    [[nodiscard]] bool both_active() const noexcept { return isrckf.active && inekf.active; }
};

// ============================================================================
// Debug Information Structures (optional, for future use)
// ============================================================================

/// Association debug information (from JpdaMatcher)
struct AssociationDebug {
    // Phase probabilities (all 4 or 3 phases)
    std::vector<double> phase_probs;
    int map_phase{0};         // MAP phase estimate
    bool is_ambiguous{false}; // Top-2 phases are close

    // Permutation information
    struct PermutationInfo {
        std::vector<int> perm;
        double log_prob{0.0};
        bool gate_passed{false};
        double mahalanobis_sq{0.0};
    };
    std::vector<PermutationInfo> top_permutations;
};

/// Filter debug information
struct FilterDebug {
    Eigen::VectorXd innovation; // [dx, dy, dz, dyaw] or similar
    double covariance_trace{0.0};
    Eigen::VectorXd position_uncertainty;
    Eigen::VectorXd velocity_uncertainty;
    double yaw_uncertainty{0.0};
};

/// State machine debug information
struct StateMachineDebug {
    double time_in_state{0.0};
    int detecting_count{0};
    double lost_time{0.0};
    int consecutive_updates{0};
};

// ============================================================================
// Unified Tracker Output
// ============================================================================

struct TrackerOutput {
    uint64_t timestamp_ns{0};
    TrackerStatus status{TrackerStatus::Idle};
    ArmorName target_name{ArmorName::Invalid};

    // Variant: empty, robot, or outpost
    std::variant<std::monostate, RobotTargetState, OutpostTargetState> state;

    // Debug information (populated if available)
    std::optional<AssociationDebug> association_debug;
    std::optional<FilterDebug> filter_debug;
    std::optional<StateMachineDebug> state_machine_debug;

    [[nodiscard]] bool is_tracking() const noexcept {
        return status == TrackerStatus::Tracking || status == TrackerStatus::TempLost;
    }

    [[nodiscard]] bool is_robot() const noexcept {
        return std::holds_alternative<RobotTargetState>(state);
    }

    [[nodiscard]] bool is_outpost() const noexcept {
        return std::holds_alternative<OutpostTargetState>(state);
    }

    [[nodiscard]] const RobotTargetState* robot_state() const noexcept {
        return std::get_if<RobotTargetState>(&state);
    }

    [[nodiscard]] const OutpostTargetState* outpost_state() const noexcept {
        return std::get_if<OutpostTargetState>(&state);
    }
};

// ============================================================================
// Channel Topics (for ECS)
// ============================================================================

// Channel topics are defined in fcs/core/channel_topics.hpp:
// - fcs::TrackerOutputChannelTopic
// - fcs::SolverOutputChannelTopic

// Re-export for convenience
using ::fcs::TrackerOutputChannelTopic;

// ============================================================================
// Implementation
// ============================================================================

inline std::array<Eigen::Vector4d, 4> RobotTargetState::armor_poses() const noexcept {
    std::array<Eigen::Vector4d, 4> poses;
    constexpr double angle_step = 2.0 * std::numbers::pi / 4.0;

    for (int i = 0; i < 4; ++i) {
        const double armor_yaw = yaw + static_cast<double>(i) * angle_step;
        double r, z;

        if (i == 1 || i == 3) {
            r = radius1;
            z = z1;
        } else {
            r = radius0;
            z = position.z();
        }

        const double x = position.x() - r * std::cos(armor_yaw);
        const double y = position.y() - r * std::sin(armor_yaw);

        poses[i] = {x, y, z, armor_yaw};
    }

    return poses;
}

inline std::array<Eigen::Vector4d, 3> OutpostTargetState::armor_poses() const noexcept {
    std::array<Eigen::Vector4d, 3> poses;
    constexpr double angle_step = 2.0 * std::numbers::pi / 3.0;

    for (int i = 0; i < 3; ++i) {
        const double armor_yaw = yaw + static_cast<double>(i) * angle_step;
        const double x         = position.x() - radius * std::cos(armor_yaw);
        const double y         = position.y() - radius * std::sin(armor_yaw);

        poses[i] = {x, y, z[i], armor_yaw};
    }

    return poses;
}

} // namespace fcs::L3

// ============================================================================
// fmt::formatter specializations
// ============================================================================

namespace fmt {

template <>
struct formatter<fcs::L3::TrackerStatus> : formatter<std::string_view> {
    auto format(const fcs::L3::TrackerStatus s, format_context& ctx) const {
        return formatter<std::string_view>::format(magic_enum::enum_name(s), ctx);
    }
};

} // namespace fmt
