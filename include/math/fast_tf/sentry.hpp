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
template <std::size_t N>
std::shared_ptr<CoordinateSystem<N>> sentry_new(timestamp_ns_t ts = 0) {
    auto system       = std::make_shared<CoordinateSystem<N>>();
    constexpr fp_t pi = std::numbers::pi;
    // odom -> world
    update<odom_fuxk_frame>(*system, FuckTransform<odom_fuxk_frame>{TransformMatrix4d{}}, ts);

    // NOTE: gimbal_link -> odom is dynamic (IMU); do not seed an identity transform at ts=0.

    // camera_link -> gimbal_link
    update<camera_link_fuxk_frame>(
        *system,
        FuckTransform<camera_link_fuxk_frame>{TransformMatrix4d::from_rpy(
            0.0, to_rad_2(-1.288), to_rad_2(-0.65), 0.0879, 0.0026, 0.0079)},
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
            TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 0.0543, 0.0026, -0.0466)},
        ts);
    return system;
}

extern template std::shared_ptr<CoordinateSystem<1024>> sentry_new<1024>(timestamp_ns_t);

} // namespace fast_tf
