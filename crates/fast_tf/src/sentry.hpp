#pragma once

#include "frame.hpp"
#include "matrix.hpp"

#include <numbers>

namespace fast_tf {

constexpr auto to_rad_2(double x) { return x * std::numbers::pi / 180.0; }
/**
 * @brief 创建哨兵机器人的坐标系统，初始化固定变换
 *
 * TF 树结构:
 *   world
 *     └── odom
 *           └── gimbal_link
 *                 ├── camera_link
 *                 │     └── camera_optical
 *                 └── muzzle_link
 */
inline CoordinateSystem sentry_new(timestamp_ns_t ts = 0) {
    auto system       = CoordinateSystem();
    constexpr fp_t pi = std::numbers::pi;
    // odom -> world
    update<odom_fuxk_frame>(system, EdgeTransform<odom_fuxk_frame>{}, ts);
    update<gimbal_yaw_fuxk_frame>(system, EdgeTransform<gimbal_yaw_fuxk_frame>{}, ts);
    update<gimbal_pitch_fuxk_frame>(
        system, EdgeTransform<gimbal_pitch_fuxk_frame>::from_rpy(0, 0, 0, 0, 0, 0.0791), ts);
    // NOTE: odom -> gimbal_link is dynamic (IMU); do not seed here.

    // camera_link -> gimbal_link (static)
    update<camera_link_fuxk_frame>(
        system, EdgeTransform<camera_link_fuxk_frame>::from_rpy(0.0, 0.0, 0.0, 0.0273, 0, 0.0544),
        0);

    // camera_optical -> camera_link (static)
    // xyz="0 0 0" rpy="-pi/2 0 -pi/2"
    update<camera_optical_fuxk_frame>(
        system,
        EdgeTransform<camera_optical_fuxk_frame>::from_rpy(
            -pi / 2.0, 0.0, -pi / 2.0, 0.0, 0.0, 0.0),
        0);

    // muzzle_link -> gimbal_link (static)
    // xyz="0.0874 0.0 -0.0874" rpy="0 0 0"
    update<muzzle_link_fuxk_frame>(
        system, EdgeTransform<muzzle_link_fuxk_frame>::from_rpy(0.0, 0.0, 0.0, 0.06, 0, 0), 0);
    return system;
}

} // namespace fast_tf
