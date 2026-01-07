#pragma once

#include "toml_helper.hpp"
#include <expected>
#include <string>

namespace fcs::L3 {

/// InEKF configuration parameters
struct InEKFConfig {
    /// Process noise parameters
    double sigma_a_xy{5.0}; // XY acceleration noise
    double sigma_a_z{1.0};  // Z acceleration noise
    double sigma_yaw{20.0}; // Yaw acceleration noise (for v_yaw CV model)
    double sigma_r{1.0};    // Radius drift noise
    double sigma_h{0.5};    // Height drift noise

    /// Measurement noise (position only)
    double sigma_m{0.01}; // Position measurement noise
    double sigma_w{0.25}; // Yaw measurement noise (rad)

    // Target geometry used by InEKF measurement model (center -> armor offsets)
    double radius0{0.23}; // Armor 0,2 radius
    double radius1{0.23}; // Armor 1,3 radius
    double height{0.0};   // z1 - z0 (armor 1,3 height offset)

    /// Enable InEKF for tracking
    bool enable{false}; // Master switch

    /// Enable debug logging
    bool debug{false};

    /// Read configuration from TOML
    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ(tbl, sigma_a_xy);
        READ(tbl, sigma_a_z);
        READ(tbl, sigma_yaw);
        READ(tbl, sigma_r);
        READ(tbl, sigma_h);
        READ(tbl, sigma_m);
        READ(tbl, sigma_w);
        READ(tbl, radius0);
        READ(tbl, radius1);
        READ(tbl, height);
        READ(tbl, enable);
        READ(tbl, debug);
        return {};
    }
};

} // namespace fcs::L3
