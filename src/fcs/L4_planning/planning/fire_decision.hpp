#pragma once

#include "L4_planning/aimer/types.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace fcs::L4 {

/// Fire decision configuration
struct FireDecisionConfig {
    /// Minimum shooting threshold (radians), used for long distances
    /// Default aligned with sp_vision_25 planner
    double fire_thresh{0.003};

    /// Maximum yaw/pitch mismatch allowed by the L5 dual-delay consistency gate.
    double delay_gate_error_rad{0.0035};

    /// Physical window dimensions (meters)
    double shooting_range_h{0.135}; // Height
    double shooting_range_w{0.135}; // Width
};

/// Check if target is within shootable range
///
/// Converts physical window (meters) to angular tolerance and checks if current gimbal
/// attitude is within range. This is the unified fire decision logic used by L4 aimers
/// and the L5 fire gate.
///
/// @param cfg Fire decision configuration
/// @param cur_yaw Current gimbal yaw (radians)
/// @param cur_pitch Current gimbal pitch (radians)
/// @param target_yaw Target yaw (radians)
/// @param target_pitch Target pitch (radians)
/// @param distance Distance to target (meters)
///
/// @return true if target is within shootable range
[[nodiscard]] inline bool is_on_target(
    const FireDecisionConfig& cfg, double cur_yaw, double cur_pitch, double target_yaw,
    double target_pitch, double distance) noexcept {

    // Convert physical window to angular tolerance
    auto shooting_range_yaw   = std::abs(std::atan2(cfg.shooting_range_w / 2.0, distance));
    auto shooting_range_pitch = std::abs(std::atan2(cfg.shooting_range_h / 2.0, distance));

    // Set minimum angle threshold
    const auto max_error = cfg.fire_thresh;
    shooting_range_yaw   = std::max(shooting_range_yaw, max_error);
    shooting_range_pitch = std::max(shooting_range_pitch, max_error);

    return std::abs(cur_yaw - target_yaw) < shooting_range_yaw
        && std::abs(cur_pitch - target_pitch) < shooting_range_pitch;
}

[[nodiscard]] inline bool is_on_target_phase_aware(
    const FireDecisionConfig& cfg, ArmorAimPhase phase, double diff_yaw, double yaw_limit_deg,
    double min_enable_yaw_deg, double cur_yaw, double cur_pitch, double target_yaw,
    double target_pitch, double distance) noexcept {
    auto shooting_range_yaw   = std::abs(std::atan2(cfg.shooting_range_w / 2.0, distance));
    auto shooting_range_pitch = std::abs(std::atan2(cfg.shooting_range_h / 2.0, distance));

    const auto max_error = cfg.fire_thresh;
    shooting_range_yaw   = std::max(shooting_range_yaw, max_error);
    shooting_range_pitch = std::max(shooting_range_pitch, max_error);

    if (phase == ArmorAimPhase::WholeCarCenter) {
        const double yaw_limit  = yaw_limit_deg * std::numbers::pi / 180.0;
        const double min_enable = min_enable_yaw_deg * std::numbers::pi / 180.0;
        if (std::abs(diff_yaw) > yaw_limit) {
            return false;
        }
        shooting_range_yaw = std::max(shooting_range_yaw, min_enable) * std::cos(diff_yaw);
    }

    return std::abs(cur_yaw - target_yaw) < shooting_range_yaw
        && std::abs(cur_pitch - target_pitch) < shooting_range_pitch;
}

} // namespace fcs::L4
