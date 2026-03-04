#pragma once

#include <algorithm>
#include <cmath>
#include <expected>
#include <numbers>
#include <string>

// Forward declaration to avoid including toml++ in config headers.
namespace toml::inline v3 {
class table;
}

namespace fcs::L4 {

/// Fire decision configuration
struct FireDecisionConfig {
    /// Minimum shooting threshold (radians), used for long distances
    /// Default: ~0.17 degrees (180.0 / pi)
    double fire_thresh{180.0 / std::numbers::pi};

    /// Physical window dimensions (meters)
    double shooting_range_h{0.135}; // Height
    double shooting_range_w{0.135}; // Width

    /// Read configuration from TOML table
    [[nodiscard]] std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

/// Check if target is within shootable range
///
/// Converts physical window (meters) to angular tolerance and checks if current gimbal
/// attitude is within range. This is the unified fire decision logic used by both
/// Aimer and Planner.
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

} // namespace fcs::L4
