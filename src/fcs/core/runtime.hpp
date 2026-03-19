#pragma once
#include "core/armor_types.hpp"
#include "scheduler/system/components.hpp"

#include <array>
#include <cstdint>

namespace fcs::core {

struct ImuState {
    uint64_t timestamp_ns{0};
    double yaw{0.0};
    double pitch{0.0};
    double roll{0.0};
    double yaw_vel{0.0};
    double pitch_vel{0.0};
    double roll_vel{0.0};
    bool attitude_valid{false};
    bool angular_velocity_valid{false};
};

struct ControlTransformSnapshot {
    bool present{false};
    std::array<double, 3> translation{};
    std::array<double, 4> quaternion{};
};

struct ControlResourceSnapshot {
    uint64_t sample_timestamp_ns{0};
    ImuState imu{};
    ArmorColor detecting_color{ArmorColor::Blue};
    double bullet_speed_raw{0.0};
    double bullet_speed{0.0};
    ControlTransformSnapshot odom_to_gimbal_pitch{};
    ControlTransformSnapshot gimbal_to_camera_link{};
    ControlTransformSnapshot odom_to_camera_optical{};
    ControlTransformSnapshot odom_to_muzzle{};
};
using namespace talos::scheduler::system;
using imu_state     = res<ImuState>;
using imu_state_mut = res_mut<ImuState>;

using detecting_color     = res<ArmorColor>;
using detecting_color_mut = res_mut<ArmorColor>;

} // namespace fcs::core
