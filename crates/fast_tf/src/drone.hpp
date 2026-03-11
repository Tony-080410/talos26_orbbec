#pragma once

#include "frame.hpp"
#include "matrix.hpp"

#include <numbers>

namespace fast_tf {

inline std::shared_ptr<CoordinateSystem> drone_new(timestamp_ns_t ts = 0) {
    auto system       = std::make_shared<CoordinateSystem>();
    constexpr fp_t pi = std::numbers::pi;

    update<odom_fuxk_frame>(*system, FuckTransform<odom_fuxk_frame>{TransformMatrix4d{}}, ts);
    update<gimbal_yaw_fuxk_frame>(
        *system, FuckTransform<gimbal_yaw_fuxk_frame>{TransformMatrix4d{}}, ts);
    update<gimbal_pitch_fuxk_frame>(
        *system,
        FuckTransform<gimbal_pitch_fuxk_frame>{
            TransformMatrix4d::from_rpy(0, 0, 0, -0.005, 0, -0.08364)},
        ts);

    update<camera_link_fuxk_frame>(
        *system,
        FuckTransform<camera_link_fuxk_frame>{
            TransformMatrix4d::from_rpy(0, 0, 0, 0.1521, 0.05096, -0.01642)},
        0);

    update<camera_optical_fuxk_frame>(
        *system,
        FuckTransform<camera_optical_fuxk_frame>{
            TransformMatrix4d::from_rpy(-pi / 2.0, 0.0, -pi / 2.0, 0.0, 0.0, 0.0)},
        0);

    update<muzzle_link_fuxk_frame>(
        *system,
        FuckTransform<muzzle_link_fuxk_frame>{
            TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 0.07527, 0, 0.0)},
        0);
    return system;
}

} // namespace fast_tf
