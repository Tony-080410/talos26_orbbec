#pragma once

#include "toml_helper.hpp"
#include <expected>
#include <string>

namespace fcs::L3 {

/// InEKF configuration parameters for 4-armor robot targets.
struct RobotInEKFConfig {
    /// Process noise parameters
    double sigma_a_xy{5.0}; // XY acceleration noise
    double sigma_a_z{1.0};  // Z acceleration noise
    double sigma_yaw{20.0}; // Yaw acceleration noise (for v_yaw CV model)
    double sigma_r{1.0};    // Radius drift noise (shared for r0/r1)
    double sigma_h{0.5};    // Height drift noise (h random walk)

    /// Measurement noise
    double sigma_m{0.01}; // Position measurement noise
    double sigma_w{0.25}; // Yaw measurement noise (rad)

    /// Target geometry (center -> armor offsets)
    double radius0{0.23}; // Armor 0,2 radius
    double radius1{0.23}; // Armor 1,3 radius
    double height{0.0};   // z1 - z0 (armor 1,3 height offset)

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
        return {};
    }
};

/// InEKF configuration parameters for 3-armor outpost targets.
struct OutpostInEKFConfig {
    /// Process noise parameters
    double sigma_a_xy{5.0}; // XY acceleration noise
    double sigma_a_z{1.0};  // Z acceleration noise
    double sigma_yaw{20.0}; // Yaw acceleration noise (for v_yaw CV model)
    double sigma_h{0.5};    // dz drift noise (dz1/dz2 random walk)

    /// Measurement noise
    double sigma_m{0.01}; // Position measurement noise
    double sigma_w{0.25}; // Yaw measurement noise (rad)

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ(tbl, sigma_a_xy);
        READ(tbl, sigma_a_z);
        READ(tbl, sigma_yaw);
        READ(tbl, sigma_h);
        READ(tbl, sigma_m);
        READ(tbl, sigma_w);
        return {};
    }
};

} // namespace fcs::L3
