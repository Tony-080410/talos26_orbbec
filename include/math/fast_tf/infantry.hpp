#pragma once

#include "buffer.hpp"
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
std::unique_ptr<CoordinateSystem<N>> infantry_new(timestamp_ns_t ts = 0) {
    auto system       = std::make_unique<CoordinateSystem<N>>();
    constexpr fp_t pi = std::numbers::pi;
    // odom -> world
    update<odom_fuxk_frame>(system, FuckTransform<odom_fuxk_frame>{TransformMatrix4d{}}, ts);

    // gimbal_link -> odom
    update<gimbal_link_fuxk_frame>(
        system, FuckTransform<gimbal_link_fuxk_frame>{TransformMatrix4d{}}, ts);

    // camera_link -> gimbal_link
    // xyz="0.04444 0.0 -0.0363" rpy="0 0 0"
    update<camera_link_fuxk_frame>(
        system,
        FuckTransform<camera_link_fuxk_frame>{
            TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)},
        ts);

    // camera_optical -> camera_link
    // xyz="0 0 0" rpy="-pi/2 0 -pi/2"
    update<camera_optical_fuxk_frame>(
        system,
        FuckTransform<camera_optical_fuxk_frame>{
            TransformMatrix4d::from_rpy(-pi / 2.0, 0.0, -pi / 2.0, 0.0, 0.0, 0.0)},
        ts);

    // muzzle_link -> gimbal_link
    // xyz="0.0874 0.0 -0.0874" rpy="0 0 0"
    update<muzzle_link_fuxk_frame>(
        system,
        FuckTransform<muzzle_link_fuxk_frame>{
            TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 0.0, 0.0, -0.0)},
        ts);
    return system;
}

} // namespace fast_tf