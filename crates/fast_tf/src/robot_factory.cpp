#include "robot_factory.hpp"

#include <numbers>

namespace fast_tf {

namespace {

void clear_coordinate_system(CoordinateSystem& system) {
    for_each_coordinate_buffer(system, [](auto, auto& buffer) {
        if constexpr (requires { buffer.clear(); }) {
            buffer.clear();
        }
    });
}

} // namespace

void init_infantry_system(CoordinateSystem& system, timestamp_ns_t ts) {
    clear_coordinate_system(system);
    constexpr fp_t pi = std::numbers::pi;

    update<odom_fuxk_frame>(system, {}, ts);

    update<camera_link_fuxk_frame>(
        system,
        EdgeTransform<camera_link_fuxk_frame>::from_rpy(
            0.0, (2.42 / 180.0) * pi, (1.35 / 180.0) * pi, 0.04425, 0.01012, -0.05259),
        0);

    update<camera_optical_fuxk_frame>(
        system,
        EdgeTransform<camera_optical_fuxk_frame>::from_rpy(
            -pi / 2.0, 0.0, -pi / 2.0, 0.0, 0.0, 0.0),
        0);

    update<muzzle_link_fuxk_frame>(
        system,
        EdgeTransform<muzzle_link_fuxk_frame>::from_rpy(
            0.0, 0.0, 0.0, 0.0874, -0.009, -0.0874 * 1.28),
        0);
}

void init_hero_system(CoordinateSystem& system, timestamp_ns_t ts) {
    clear_coordinate_system(system);
    constexpr fp_t pi = std::numbers::pi;

    update<odom_fuxk_frame>(system, EdgeTransform<odom_fuxk_frame>{}, ts);
    update<gimbal_yaw_fuxk_frame>(system, EdgeTransform<gimbal_yaw_fuxk_frame>{}, ts);
    update<gimbal_pitch_fuxk_frame>(
        system, EdgeTransform<gimbal_pitch_fuxk_frame>::from_rpy(0, 0, 0, -0.06492, 0, 0.164), ts);

    update<camera_link_fuxk_frame>(
        system,
        EdgeTransform<camera_link_fuxk_frame>::from_rpy(
            0, (-1.0 / 180.0) * pi, 0.0, 0.25391, 0.0735, 0.00639),
        0);

    update<camera_optical_fuxk_frame>(
        system,
        EdgeTransform<camera_optical_fuxk_frame>::from_rpy(
            -pi / 2.0, 0.0, -pi / 2.0, 0.0, 0.0, 0.0),
        0);

    update<muzzle_link_fuxk_frame>(
        system,
        EdgeTransform<muzzle_link_fuxk_frame>::from_rpy(0.0, 0.0, 0.0, 0.17605, 0, -0.03332), 0);
}

void init_sentry_system(CoordinateSystem& system, timestamp_ns_t ts) {
    clear_coordinate_system(system);
    constexpr fp_t pi = std::numbers::pi;

    update<odom_fuxk_frame>(system, EdgeTransform<odom_fuxk_frame>{}, ts);
    update<gimbal_yaw_fuxk_frame>(system, EdgeTransform<gimbal_yaw_fuxk_frame>{}, ts);
    update<gimbal_pitch_fuxk_frame>(
        system, EdgeTransform<gimbal_pitch_fuxk_frame>::from_rpy(0, 0, 0, 0, 0, 0.0791), ts);

    update<camera_link_fuxk_frame>(
        system, EdgeTransform<camera_link_fuxk_frame>::from_rpy(0.0, 0.0, 0.0, 0.0273, 0, 0.0544),
        0);

    update<camera_optical_fuxk_frame>(
        system,
        EdgeTransform<camera_optical_fuxk_frame>::from_rpy(
            -pi / 2.0, 0.0, -pi / 2.0, 0.0, 0.0, 0.0),
        0);

    update<muzzle_link_fuxk_frame>(
        system, EdgeTransform<muzzle_link_fuxk_frame>::from_rpy(0.0, 0.0, 0.0, 0.06, 0, 0), 0);
}

void init_drone_system(CoordinateSystem& system, timestamp_ns_t ts) {
    clear_coordinate_system(system);
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
}

CoordinateSystem create_infantry_system(timestamp_ns_t ts) {
    CoordinateSystem system;
    init_infantry_system(system, ts);
    return system;
}

CoordinateSystem create_hero_system(timestamp_ns_t ts) {
    CoordinateSystem system;
    init_hero_system(system, ts);
    return system;
}

CoordinateSystem create_sentry_system(timestamp_ns_t ts) {
    CoordinateSystem system;
    init_sentry_system(system, ts);
    return system;
}

CoordinateSystem create_drone_system(timestamp_ns_t ts) {
    CoordinateSystem system;
    init_drone_system(system, ts);
    return system;
}

} // namespace fast_tf
