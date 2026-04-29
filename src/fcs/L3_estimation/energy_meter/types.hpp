#pragma once

#include <Eigen/Core>

#include <cstdint>

namespace fcs::energy_meter {

struct EnergyMeterState {
    uint64_t timestamp_ns{};

    bool tracking_valid{false};

    Eigen::Vector3d r_center_odom{Eigen::Vector3d::Zero()};
    double radius{};

    double yaw{};
    double pitch{};
    double roll{};
    double v_roll{};
    int blade_id{-1};
    bool blade_id_confirmed{false};

    Eigen::Vector3d position{Eigen::Vector3d::Zero()};

    Eigen::Quaterniond quat{Eigen::Quaterniond::Identity()}; // 预测姿态

    bool is_big_rune{false};
    bool model_valid{false};
    double fit_error{};

    int direction{0};

    double sin_amplitude{0};                                 // a: 振幅 [0.780, 1.045]
    double sin_omega{0};                                     // ω: 角频率（rad/s）[1.884, 2.000]
    double sin_phase{0};                                     // φ: 速度模型相位（rad）
    double sin_offset{0};                                    // b: 速度偏移 = 2.090 - a
    double sin_const_term{0};                                // c: 常数项（rad）

    double relative_time{0.0};                               // 从开始到现在的累计时间（秒）
};

} // namespace fcs::energy_meter
