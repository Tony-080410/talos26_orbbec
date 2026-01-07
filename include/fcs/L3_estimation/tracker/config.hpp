#pragma once

#include "inekf_config.hpp"
#include "motion_model.hpp"
#include "toml_helper.hpp"
#include <expected>
#include <memory>
#include <string>
#include <toml++/toml.hpp>

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

struct TrackerConfig {
    /// Robot target parameters
    TargetConfig<RobotModel::Params> robot;
    /// Outpost target parameters
    TargetConfig<OutpostModel::Params> outpost;
    /// InEKF parameters
    InEKFConfig ineff;
    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ(tbl, robot);
        READ(tbl, outpost);
        READ(tbl, ineff);
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
