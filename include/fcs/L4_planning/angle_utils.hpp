#pragma once

#include <cmath>
#include <numbers>

namespace fcs::L4 {

/// Normalize angle to [-pi, pi] in O(1).
[[nodiscard]] inline double normalize_rad(double angle) noexcept {
    return std::remainder(angle, 2.0 * std::numbers::pi);
}

} // namespace fcs::L4

