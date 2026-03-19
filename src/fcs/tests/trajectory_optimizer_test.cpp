#include <gtest/gtest.h>

#include <array>

#include "L5_weapon/enhanced/trajectory_optimizer.hpp"

namespace {

fcs::L5::DelayGateAngles make_angles(double yaw, double pitch = 0.0) {
    return fcs::L5::DelayGateAngles{.yaw = yaw, .pitch = pitch};
}

} // namespace

TEST(TrajectoryOptimizer, DualDelayGateUsesPerSideDelayedTrackingError) {
    const std::array reference = {
        make_angles(0.0, 0.0),
        make_angles(0.0, 0.0),
        make_angles(0.0, 0.0),
    };
    const std::array optimized = {
        make_angles(0.10, 0.0),
        make_angles(0.00, 0.0),
        make_angles(0.40, 0.0),
    };

    const auto errors = fcs::L5::compute_dual_delay_gate_errors(
        static_cast<int>(reference.size()), 1, 1,
        [&](int index) noexcept { return reference[static_cast<size_t>(index)]; },
        [&](int index) noexcept { return optimized[static_cast<size_t>(index)]; });

    EXPECT_NEAR(errors.forward, 0.40, 1e-12);
    EXPECT_NEAR(errors.backward, 0.10, 1e-12);
    EXPECT_GT(errors.forward, errors.backward);
}
