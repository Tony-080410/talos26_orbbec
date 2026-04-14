#pragma once

#include "L4_planning/aimer/types.hpp"
#include "core/armor_types.hpp"
#include "core/math/normalize.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace fcs::L4 {

/// Fire decision configuration
struct FireDecisionConfig {
    /// Minimum shooting threshold (radians), used for long distances
    /// Default aligned with sp_vision_25 planner
    double fire_thresh{0.003};

    /// Legacy mismatch threshold retained for debug/telemetry compatibility.
    double delay_gate_error_rad{0.0035};

    /// Physical window dimensions (meters)
    double shooting_range_h{0.12};
    double shooting_range_w_small{0.12};
    double shooting_range_w_large{0.24};

    /// Minimum pitch gate opening (degrees), aligned with the authoritative armor_control.
    double min_enable_pitch_deg{0.25};
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
    auto shooting_range_yaw   = std::abs(std::atan2(cfg.shooting_range_w_small / 2.0, distance));
    auto shooting_range_pitch = std::abs(std::atan2(cfg.shooting_range_h / 2.0, distance));

    // Set minimum angle threshold
    const auto max_error = cfg.fire_thresh;
    shooting_range_yaw   = std::max(shooting_range_yaw, max_error);
    shooting_range_pitch = std::max(shooting_range_pitch, max_error);

    return std::abs(core::math::normalize_angle(target_yaw - cur_yaw)) < shooting_range_yaw
        && std::abs(core::math::normalize_angle(target_pitch - cur_pitch)) < shooting_range_pitch;
}

struct FireWindow {
    double yaw{0.0};
    double pitch{0.0};
};

[[nodiscard]] inline FireWindow compute_fire_window(
    const FireDecisionConfig& cfg, ArmorType armor_type, double fire_yaw_diff, double distance,
    double min_enable_yaw_deg) noexcept {
    constexpr double kPitchFactorAngleRad = 15.0 * std::numbers::pi / 180.0;

    const double clamped_distance = std::max(distance, 1e-6);
    const double shooting_range_w =
        (armor_type == ArmorType::Large) ? cfg.shooting_range_w_large : cfg.shooting_range_w_small;
    const double min_enable_yaw   = min_enable_yaw_deg * std::numbers::pi / 180.0;
    const double min_enable_pitch = cfg.min_enable_pitch_deg * std::numbers::pi / 180.0;

    double shooting_range_yaw = std::abs(std::atan2(shooting_range_w / 2.0, clamped_distance));
    double shooting_range_pitch =
        std::abs(std::atan2(cfg.shooting_range_h / 2.0, clamped_distance));

    shooting_range_yaw *= std::cos(fire_yaw_diff);
    shooting_range_pitch *= std::cos(kPitchFactorAngleRad);

    shooting_range_yaw   = std::max(shooting_range_yaw, min_enable_yaw);
    shooting_range_pitch = std::max(shooting_range_pitch, min_enable_pitch);

    return FireWindow{
        .yaw   = std::abs(shooting_range_yaw),
        .pitch = std::abs(shooting_range_pitch),
    };
}

[[nodiscard]] inline bool is_on_target_phase_aware(
    const FireDecisionConfig& cfg, [[maybe_unused]] ArmorAimPhase phase, ArmorType armor_type,
    double fire_yaw_diff, double min_enable_yaw_deg, double cur_yaw, double cur_pitch,
    double target_yaw, double target_pitch, double distance) noexcept {
    const FireWindow window =
        compute_fire_window(cfg, armor_type, fire_yaw_diff, distance, min_enable_yaw_deg);
    return std::abs(core::math::normalize_angle(target_yaw - cur_yaw)) < window.yaw
        && std::abs(core::math::normalize_angle(target_pitch - cur_pitch)) < window.pitch;
}

} // namespace fcs::L4
