#pragma once

#include "frame.hpp"
#include "matrix.hpp"

#include <numbers>
#include <opencv2/core/cvdef.h>

namespace fast_tf {
/**
 * @brief 创建步兵机器人的坐标系统，初始化固定变换
 *
 * TF 树结构:
 *   world
 *     └── odom
 *           └── gimbal_link
 *                 ├── camera_link
 *                 │     └── camera_optical
 *                 └── muzzle_link
 */
inline std::shared_ptr<CoordinateSystem> infantry_new(timestamp_ns_t ts = 0) {
    auto system       = std::make_shared<CoordinateSystem>();
    constexpr fp_t pi = std::numbers::pi;
    // odom -> world
    update<odom_fuxk_frame>(*system, {}, ts);

    // NOTE: odom -> gimbal_link is dynamic (IMU); do not seed here.
    // Leaving it empty matches tf2 behavior: lookups fail until IMU data arrives.

    // camera_link -> gimbal_link (static)
    // xyz="0.04444 0.0 -0.0363" rpy="0 0 0"
    update<camera_link_fuxk_frame>(
        *system,
        EdgeTransform<camera_link_fuxk_frame>::from_rpy(
            0.0, ((2.42 / 180.0) * CV_PI), ((1.35 / 180.0) * CV_PI), 0.04425, 0.01012, -0.05259),
        0);

    // camera_optical -> camera_link (static)
    // xyz="0 0 0" rpy="-pi/2 0 -pi/2"
    update<camera_optical_fuxk_frame>(
        *system,
        EdgeTransform<camera_optical_fuxk_frame>::from_rpy(
            -pi / 2.0, 0.0, -pi / 2.0, 0.0, 0.0, 0.0),
        0);

    // muzzle_link -> gimbal_link (static)
    // xyz="0.0874 0.0 -0.0874" rpy="0 0 0"
    update<muzzle_link_fuxk_frame>(
        *system,
        EdgeTransform<muzzle_link_fuxk_frame>::from_rpy(
            0.0, 0.0, 0.0, 0.0874, -0.009, -0.0874 * 1.28),
        0);
    return system;
}

} // namespace fast_tf
