#pragma once

#include <Eigen/Core>

#include <cstdint>

namespace fcs::energy_meter {

struct EnergyMeterState {
    uint64_t timestamp_ns{};

    bool tracking_valid{false};

    Eigen::Vector3d r_center_odom{Eigen::Vector3d::Zero()};
    double radius{};

    double roll{};
    double v_roll{};
    int blade_id{-1};

    Eigen::Vector3d predicted_position_odom{Eigen::Vector3d::Zero()};
    uint64_t predicted_future_ns{};

    bool is_big_rune{false};
    bool model_valid{false};
    double fit_error{};
};

} // namespace fcs::energy_meter
