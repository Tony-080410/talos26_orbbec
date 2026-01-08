#pragma once

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <numbers>
#include <vector>

namespace fcs::L3 {

// ============================================================================
// State Space Indices
// ============================================================================

/// Robot state space (11-dim): 4-armor target tracking
enum RoboState : uint8_t {
    XC,    // Center X position
    VX,    // X velocity
    YC,    // Center Y position
    VY,    // Y velocity
    Z0,    // Armor 0,2 Z position
    VZ,    // Z velocity
    YAW,   // Armor 0 yaw angle
    V_YAW, // Angular velocity
    R_0,   // Armor 0,2 radius
    R_1,   // Armor 1,3 radius
    H,     // Z1 - Z0 (height difference)
    STATE_MAX,
};

/// Outpost state space (7-dim): 3-armor stationary target
enum OutpostState : uint8_t {
    O_XC,
    O_YC,
    O_YAW,
    O_VYAW,
    O_Z0,
    O_Z1,
    O_Z2,
    O_STATE_MAX,
};

/// Measurement space (spherical): [yaw, pitch, distance, armor_yaw]
enum Measure : uint8_t {
    ARMOR_YAW,
    ARMOR_PITCH,
    ARMOR_DISTANCE,
    ARMOR_YAW_ARMOR,
    MEASURE_MAX,
};

// ============================================================================
// Coordinate Transforms
// ============================================================================

/// Convert Cartesian to spherical: xyz -> [yaw, pitch, distance]
[[nodiscard]] inline Eigen::Vector3d xyz2ypd(const Eigen::Vector3d& xyz) noexcept {
    const double x = xyz.x();
    const double y = xyz.y();
    const double z = xyz.z();

    const double distance   = xyz.norm();
    const double yaw        = std::atan2(y, x);
    const double horizontal = std::hypot(x, y); // More accurate and faster than sqrt(x*x + y*y)
    const double pitch      = std::atan2(z, horizontal);

    return {yaw, pitch, distance};
}

/// Convert spherical to Cartesian: [yaw, pitch, distance] -> xyz
[[nodiscard]] inline Eigen::Vector3d ypd2xyz(const Eigen::Vector3d& ypd) noexcept {
    const double yaw      = ypd.x();
    const double pitch    = ypd.y();
    const double distance = ypd.z();

    const double cp = std::cos(pitch);
    const double x  = distance * cp * std::cos(yaw);
    const double y  = distance * cp * std::sin(yaw);
    const double z  = distance * std::sin(pitch);

    return {x, y, z};
}

// ============================================================================
// Angle Utilities
// ============================================================================

/// Normalize angle to [-PI, PI)
[[nodiscard]] inline double normalize_rad(double a) noexcept {
    a = std::fmod(a + std::numbers::pi, 2.0 * std::numbers::pi);
    if (a < 0.0) {
        a += 2.0 * std::numbers::pi;
    }
    return a - std::numbers::pi;
}

/// Compute shortest angle difference: to - from, result in (-PI, PI]
[[nodiscard]] inline double shortest_rad(double from, double to) noexcept {
    return normalize_rad(to - from);
}

/// Single argument version: normalize angle
[[nodiscard]] inline double shortest_rad(double angle) noexcept { return normalize_rad(angle); }

/// Unwrap angle based on previous value (continuity)
[[nodiscard]] inline double unwrap_rad(double prev, double raw) noexcept {
    const double d = shortest_rad(prev, raw);
    return prev + d;
}

/// Normalize angle to (-180, 180] degrees
[[nodiscard]] inline double normalize_deg(double a) noexcept {
    a = std::fmod(a + 180.0, 360.0);
    if (a <= 0.0) {
        a += 360.0;
    }
    return a - 180.0;
}

/// Compute shortest angle difference in degrees
[[nodiscard]] inline double shortest_deg(double from, double to) noexcept {
    return normalize_deg(to - from);
}

[[nodiscard]] inline double shortest_deg(double angle) noexcept { return normalize_deg(angle); }

/// Unwrap angle in degrees
[[nodiscard]] inline double unwrap_deg(double prev, double raw) noexcept {
    const double d = shortest_deg(prev, raw);
    return prev + d;
}

// ============================================================================
// Armor Position Calculation
// ============================================================================

/// Calculate all armor positions for a 4-armor robot target
/// @param target_center Center position [xc, yc, z0]
/// @param target_yaw Armor 0 yaw angle
/// @param radius0 Radius for armor 0,2
/// @param radius1 Radius for armor 1,3
/// @param z0 Z position for armor 0,2
/// @param z1 Z position for armor 1,3
/// @param armors_num Number of armors (3 or 4)
/// @return Vector of [x, y, z, yaw] for each armor
[[nodiscard]] inline std::vector<Eigen::Vector4d> get_robo_armor_poses(
    const Eigen::Vector3d& target_center, double target_yaw, double radius0, double radius1,
    double z0, double z1, size_t armors_num) noexcept {
    std::vector<Eigen::Vector4d> poses;
    poses.reserve(armors_num);

    const double angle_step = 2.0 * std::numbers::pi / static_cast<double>(armors_num);

    for (size_t i = 0; i < armors_num; ++i) {
        const double armor_yaw = normalize_rad(target_yaw + static_cast<double>(i) * angle_step);
        double radius;
        double armor_z;

        if (armors_num == 4 && (i == 1 || i == 3)) {
            radius  = radius1;
            armor_z = z1;
        } else {
            radius  = radius0;
            armor_z = z0;
        }

        const double armor_x = target_center.x() - radius * std::cos(armor_yaw);
        const double armor_y = target_center.y() - radius * std::sin(armor_yaw);

        poses.emplace_back(armor_x, armor_y, armor_z, armor_yaw);
    }

    return poses;
}

/// Calculate armor positions for outpost (3 armors, fixed radius)
[[nodiscard]] inline std::vector<Eigen::Vector4d> get_outpost_armor_poses(
    const Eigen::Vector2d& target_pos, double yaw, double radius, double z0, double z1,
    double z2) noexcept {
    constexpr size_t ARMORS_NUM = 3;
    constexpr double angle_step = 2.0 * std::numbers::pi / static_cast<double>(ARMORS_NUM);

    std::vector<Eigen::Vector4d> poses;
    poses.reserve(ARMORS_NUM);

    const std::array<double, 3> zs = {z0, z1, z2};

    for (size_t i = 0; i < ARMORS_NUM; ++i) {
        const double armor_yaw = normalize_rad(yaw + static_cast<double>(i) * angle_step);
        const double armor_x   = target_pos.x() - radius * std::cos(armor_yaw);
        const double armor_y   = target_pos.y() - radius * std::sin(armor_yaw);

        poses.emplace_back(armor_x, armor_y, zs[i], armor_yaw);
    }

    return poses;
}

/// Extract position from pose [x, y, z, yaw] -> [x, y, z]
[[nodiscard]] inline Eigen::Vector3d pose_to_position(const Eigen::Vector4d& pose) noexcept {
    return {pose.x(), pose.y(), pose.z()};
}

/// Extract positions from poses
[[nodiscard]] inline std::vector<Eigen::Vector3d>
    poses_to_positions(const std::vector<Eigen::Vector4d>& poses) noexcept {
    std::vector<Eigen::Vector3d> positions;
    positions.reserve(poses.size());
    for (const auto& pose : poses) {
        positions.emplace_back(pose.x(), pose.y(), pose.z());
    }
    return positions;
}

/// Calculate armor position from state vector
[[nodiscard]] inline Eigen::Vector3d
    state_to_armor_xyz(const double* x, int id, int armors_num) noexcept {
    const double angle_step = 2.0 * std::numbers::pi / static_cast<double>(armors_num);
    const double armor_yaw  = x[YAW] + static_cast<double>(id) * angle_step;

    double radius;
    double armor_z;

    if (armors_num == 4 && (id == 1 || id == 3)) {
        radius  = x[R_1];
        armor_z = x[Z0] + x[H];
    } else {
        radius  = x[R_0];
        armor_z = x[Z0];
    }

    const double armor_x = x[XC] - radius * std::cos(armor_yaw);
    const double armor_y = x[YC] - radius * std::sin(armor_yaw);

    return {armor_x, armor_y, armor_z};
}

} // namespace fcs::L3
