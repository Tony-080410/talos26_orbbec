#pragma once

#include "frame.hpp"
#include "matrix.hpp"

#include <numbers>

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
template <std::size_t N>
std::shared_ptr<CoordinateSystem<N>> infantry_new(timestamp_ns_t ts = 0) {
    auto system       = std::make_shared<CoordinateSystem<N>>();
    constexpr fp_t pi = std::numbers::pi;
    // odom -> world
    update<odom_fuxk_frame>(*system, FuckTransform<odom_fuxk_frame>{TransformMatrix4d{}}, ts);

    // NOTE: gimbal_link -> odom is dynamic (IMU); do not seed an identity transform at ts=0.
    // Leaving it empty matches tf2 behavior: lookups fail until IMU data arrives.

    // camera_link -> gimbal_link
    // xyz="0.04444 0.0 -0.0363" rpy="0 0 0"
    update<camera_link_fuxk_frame>(
        *system,
        FuckTransform<camera_link_fuxk_frame>{
            TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 0.04444, 0.0, -0.0363)},
        ts);

    // camera_optical -> camera_link
    // xyz="0 0 0" rpy="-pi/2 0 -pi/2"
    update<camera_optical_fuxk_frame>(
        *system,
        FuckTransform<camera_optical_fuxk_frame>{
            TransformMatrix4d::from_rpy(-pi / 2.0, 0.0, -pi / 2.0, 0.0, 0.0, 0.0)},
        ts);

    // muzzle_link -> gimbal_link
    // xyz="0.0874 0.0 -0.0874" rpy="0 0 0"
    update<muzzle_link_fuxk_frame>(
        *system,
        FuckTransform<muzzle_link_fuxk_frame>{
            TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 0.0874, -0.009, -0.0874)},
        ts);
    return system;
}

} // namespace fast_tf