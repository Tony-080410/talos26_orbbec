#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstdint>

namespace fcs::energy_meter {

struct EnergyMeterState {
    uint64_t timestamp_ns{};
    bool tracking_valid{false};

    Eigen::Vector3d r_center_odom{Eigen::Vector3d::Zero()};
    double radius{0.7};

    // ISRCKF state outputs
    double yaw{0.0};
    double pitch{0.0};
    double roll{0.0}; // signed continuous angle (= THETA for selected blade)
    double t{0.0};    // time since activation

    // Model parameters (from sigmoid-mapped state)
    double a{0.0};     // amplitude
    double omega{0.0}; // angular frequency
    double b{0.0};     // = 2.090 - a

    int blade_id{0};
    bool blade_id_confirmed{false};
    int direction{0};  // +1 CW, -1 ACW, 0 unknown

    bool model_valid{false};
    bool is_big_rune{true};

    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
    Eigen::Quaterniond quat{Eigen::Quaterniond::Identity()};
};

} // namespace fcs::energy_meter
