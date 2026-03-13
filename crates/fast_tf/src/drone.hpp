#pragma once

#include "frame.hpp"
#include "matrix.hpp"

#include <numbers>

namespace fast_tf {

inline CoordinateSystem drone_new(timestamp_ns_t ts = 0) {
    auto system       = CoordinateSystem();
    constexpr fp_t pi = std::numbers::pi;

    update<odom_fuxk_frame>(system, {}, ts);
    update<gimbal_yaw_fuxk_frame>(system, {}, ts);
    update<gimbal_pitch_fuxk_frame>(
        system, EdgeTransform<gimbal_pitch_fuxk_frame>::from_rpy(0, 0, 0, -0.005, 0, -0.08364), ts);

    update<camera_link_fuxk_frame>(
        system, EdgeTransform<camera_link_fuxk_frame>::from_rpy(0, 0, 0, 0.1521, 0.05096, -0.01642),
        0);

    update<camera_optical_fuxk_frame>(
        system,
        EdgeTransform<camera_optical_fuxk_frame>::from_rpy(
            -pi / 2.0, 0.0, -pi / 2.0, 0.0, 0.0, 0.0),
        0);

    update<muzzle_link_fuxk_frame>(
        system, EdgeTransform<muzzle_link_fuxk_frame>::from_rpy(0.0, 0.0, 0.0, 0.07527, 0, 0.0), 0);
    return system;
}

} // namespace fast_tf
