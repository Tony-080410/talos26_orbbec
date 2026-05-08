#pragma once
#include <chrono>
#include <cstdint>
#include <fmt/base.h>
#include <fmt/format.h>
#include <string_view>

#include "magic_enum.hpp"

namespace talos::chiral::gimbal {
static_assert(sizeof(bool) == sizeof(uint8_t));

// 机器人自身颜色 (来自裁判系统)
enum class Color : uint8_t {
    Red  = 0,
    Blue = 1,
};

struct McuData {
    int64_t timestamp_ns_system_clock;
    Color self_color;
    float bullet_speed;
    float yaw;
    float pitch;
    float roll;
    float yaw_vel;
    float pitch_vel;
    float roll_vel;

    McuData() = default;
    McuData(
        Color self_color_, float yaw_, float pitch_, float roll_, float yaw_vel_, float pitch_vel_,
        float roll_vel_) noexcept
        : self_color(self_color_)
        , bullet_speed(114514.0)
        , yaw(yaw_)
        , pitch(pitch_)
        , roll(roll_)
        , yaw_vel(yaw_vel_)
        , pitch_vel(pitch_vel_)
        , roll_vel(roll_vel_) {
        const auto now = std::chrono::system_clock::now();
        auto t =
            std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
        timestamp_ns_system_clock = t;
    }
};

struct McuRequest {
    int64_t timestamp_ns_system_clock;
    bool fire_advice;
    float target_yaw;
    float target_pitch;
    float ref_yaw_v;
    float ref_pitch_v;
    float ref_yaw_a;
    float ref_pitch_a;
    float distance;
    bool valid() const noexcept { return distance > 0.0; }
};

} // namespace talos::chiral::gimbal

namespace fmt {

template <>
struct formatter<talos::chiral::gimbal::Color> : formatter<std::string_view> {
    auto format(talos::chiral::gimbal::Color c, format_context& ctx) const {
        return formatter<std::string_view>::format(magic_enum::enum_name(c), ctx);
    }
};

} // namespace fmt
