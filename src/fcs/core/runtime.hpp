#pragma once
#include "core/armor_types.hpp"
#include "scheduler/system/components.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fmt/base.h>

namespace fcs::core {

struct ImuState {
    uint64_t timestamp_ns{0};
    double yaw{0.0};
    double pitch{0.0};
    double roll{0.0};
    double yaw_vel{0.0};
    double pitch_vel{0.0};
    double roll_vel{0.0};
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

enum Capability {
    Armor  = 0,
    Rune   = 1,
    Ldm    = 2,
    Quanta = 3,
};

using capabilities     = res<std::vector<Capability>>;
using capabilities_mut = res_mut<std::vector<Capability>>;

inline bool capable(std::vector<Capability> ls, Capability cap) {
    return std::ranges::any_of(ls, [cap](auto x) { return x == cap; });
}

} // namespace fcs::core

namespace fmt {
template <>
struct formatter<fcs::core::Capability> : formatter<std::string_view> {
    auto format(const fcs::core::Capability c, format_context& ctx) const {
        return formatter<std::string_view>::format(magic_enum::enum_name(c), ctx);
    }
};
} // namespace fmt
