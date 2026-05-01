#pragma once

#include <Eigen/Geometry>

#include <cmath>
#include <cstdint>

namespace energy_meter {

enum class Direction { UNKNOWN, CLOCKWISE, ANTI_CLOCKWISE };

inline double normalize_rad(double a) {
    a = std::fmod(a + M_PI, 2.0 * M_PI);
    if (a <= 0.0) {
        a += 2.0 * M_PI;
    }
    return a - M_PI;
}

inline double shortest_rad(double from, double to) { return normalize_rad(to - from); }

inline double unwrap_rad(double prev, double raw) {
    const double d = shortest_rad(prev, raw);
    return prev + d;
}

inline auto orientation2roll(const Eigen::Quaterniond& q) -> double {
    const Eigen::Matrix3d R = q.normalized().toRotationMatrix();
    return std::atan2(R(2, 1), R(2, 2));
}

} // namespace energy_meter
