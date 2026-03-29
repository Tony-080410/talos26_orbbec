#pragma once

#include "new_motion_model.hpp"
#include <memory>

namespace fcs::L3 {

// ============================================================================
// Tracker Configuration
// ============================================================================

template <typename T>
struct TargetConfig {
    /// Timeout before returning to IDLE from TEMP_LOST (seconds)
    double lost_threshold{};
    /// Number of consecutive frames required to transition DETECTING -> TRACKING
    uint32_t tracking_threshold{};
    /// Mahalanobis distance gate for target matching
    double matcher_gate{10.0};
    /// motion model parameters
    T model{};
};

struct RobotInEKFConfig {
    /// Target geometry (center -> armor offsets)
    double radius0{0.23}; // Armor 0,2 radius
    double radius1{0.23}; // Armor 1,3 radius
    double height{0.0};   // z1 - z0 (armor 1,3 height offset)
};

struct PnpQualityConfig {
    /// Reference σ_obs for quality modulation scaling.
    /// PnP reprojection RMSE below this → tighter tracking; above → more conservative.
    double sigma_ref{0.002};
    /// Minimum quality scale factor (clamp lower bound).
    double scale_min{0.5};
    /// Maximum quality scale factor (clamp upper bound).
    double scale_max{5.0};
};

struct TrackerConfig {
    /// Robot target parameters
    TargetConfig<RobotEkfMotionModel::Params> robot;
    /// Outpost target parameters
    TargetConfig<OutpostEkfMotionModel::Params> outpost;

    /// PnP observation quality modulation
    PnpQualityConfig pnp_quality;

    /// InEKF parameters
    RobotInEKFConfig robot_inekf;
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
};

} // namespace fcs::L3
