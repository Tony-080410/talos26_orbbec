#pragma once

#include "new_motion_model.hpp"
#include "toml_helper.hpp"
#include <expected>
#include <memory>
#include <string>

// Forward declaration to avoid including toml++ in config headers.
namespace toml::inline v3 {
class table;
}

namespace fcs::L3 {

// ============================================================================
// Tracker Configuration
// ============================================================================

template <typename T>
struct TargetConfig {
    /// Timeout before returning to IDLE from TEMP_LOST (seconds)
    double lost_threshold;
    /// Number of consecutive frames required to transition DETECTING -> TRACKING
    uint32_t tracking_threshold;
    /// Mahalanobis distance gate for target matching
    double matcher_gate{10.0};
    /// motion model parameters
    T model;

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ(tbl, lost_threshold);
        READ(tbl, tracking_threshold);
        READ(tbl, matcher_gate);
        READ(tbl, model);
        return {};
    }
};

struct RobotInEKFConfig {
    /// Target geometry (center -> armor offsets)
    double radius0{0.23}; // Armor 0,2 radius
    double radius1{0.23}; // Armor 1,3 radius
    double height{0.0};   // z1 - z0 (armor 1,3 height offset)

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ(tbl, radius0);
        READ(tbl, radius1);
        READ(tbl, height);
        return {};
    }
};

struct TrackerConfig {
    /// Robot target parameters
    TargetConfig<RobotEkfMotionModel::Params> robot;
    /// Outpost target parameters
    TargetConfig<OutpostEkfMotionModel::Params> outpost;

    /// InEKF parameters
    RobotInEKFConfig robot_inekf;

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ(tbl, robot);
        READ(tbl, outpost);
        READ(tbl, robot_inekf);
        return {};
    }
};

// ============================================================================
// L3 Estimation Full Config
// ============================================================================

struct L3Config {
    TrackerConfig tracker{};

    /// Get shared_ptr for atomic config distribution
    [[nodiscard]] std::shared_ptr<const TrackerConfig> tracker_ptr() const {
        return std::make_shared<const TrackerConfig>(tracker);
    }

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ(tbl, tracker);
        return {};
    }
};

} // namespace fcs::L3
