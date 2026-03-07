#pragma once

#include "frame.hpp"
#include "matrix.hpp"

#include <numbers>

namespace fast_tf {

constexpr auto to_rad(double x) { return (x / 180.0) * CV_PI; }

/**
 * @brief 创建英雄机器人的坐标系统，初始化固定变换
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
std::shared_ptr<CoordinateSystem<N>> hero_new(timestamp_ns_t ts = 0) {
    auto system       = std::make_shared<CoordinateSystem<N>>();
    constexpr fp_t pi = std::numbers::pi;
    // odom -> world
    update<odom_fuxk_frame>(*system, FuckTransform<odom_fuxk_frame>{TransformMatrix4d{}}, ts);
    update<gimbal_yaw_fuxk_frame>(
        *system, FuckTransform<gimbal_yaw_fuxk_frame>{TransformMatrix4d{}}, ts);
    update<gimbal_pitch_fuxk_frame>(
        *system,
        FuckTransform<gimbal_pitch_fuxk_frame>{
            TransformMatrix4d::from_rpy(0, 0, 0, -0.06492, 0, 0.164)},
        ts);
    // NOTE: odom -> gimbal_link is dynamic (IMU); do not seed here.

    // camera_link -> gimbal_link (static)
    update<camera_link_fuxk_frame>(
        *system,
        FuckTransform<camera_link_fuxk_frame>{
            TransformMatrix4d::from_rpy(0, 0, 0, 0.25391, 0.0735, 0.00639)},
        0);

    // camera_optical -> camera_link (static)
    // xyz="0 0 0" rpy="-pi/2 0 -pi/2"
    update<camera_optical_fuxk_frame>(
        *system,
        FuckTransform<camera_optical_fuxk_frame>{
            TransformMatrix4d::from_rpy(-pi / 2.0, 0.0, -pi / 2.0, 0.0, 0.0, 0.0)},
        0);

    // muzzle_link -> gimbal_link (static)
    // xyz="0.0874 0.0 -0.0874" rpy="0 0 0"
    update<muzzle_link_fuxk_frame>(
        *system,
        FuckTransform<muzzle_link_fuxk_frame>{
            TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 0.17605, 0, -0.03332)},
        0);
    return system;
}

extern template std::shared_ptr<CoordinateSystem<1024>> hero_new<1024>(timestamp_ns_t);

} // namespace fast_tf
