#pragma once

#include <cmath>
#include <numbers>

namespace fcs::core::math {

/// Normalize angle to [-pi, pi].
[[nodiscard]] inline double normalize_angle(double angle) noexcept {
    return std::remainder(angle, 2.0 * std::numbers::pi);
}

} // namespace fcs::core::math
