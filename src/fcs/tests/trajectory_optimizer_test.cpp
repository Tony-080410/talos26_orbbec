#include <gtest/gtest.h>

#include "L5_weapon/enhanced/trajectory_optimizer.hpp"

#include <Eigen/Core>

namespace fcs::L5 {
namespace {

using ReferenceTrajectory = core::trajectory::ReferenceTrajectory;

ReferenceTrajectory make_reference_trajectory(int horizon) {
    ReferenceTrajectory trajectory;
    trajectory.state = ReferenceTrajectory::StateMatrix::Zero(4, horizon);
    trajectory.distances.reserve(horizon);
    trajectory.time_of_flights.reserve(horizon);

    for (int i = 0; i < horizon; ++i) {
        trajectory.distances.push_back(static_cast<double>(i));
        trajectory.time_of_flights.push_back(static_cast<double>(i) * 0.01);
    }

    return trajectory;
}

L4::TrackCommand make_track(const ReferenceTrajectory& reference, uint64_t timestamp_ns) {
    return L4::TrackCommand{
        .timestamp_ns = timestamp_ns,
        .trajectory   = reference,
    };
}

TEST(TrajectoryOptimizerTest, FreshReferenceKeepsOriginalCenterSample) {
    WeaponControllerConfig config;
    config.reference_age_threshold_s = 1.0;
    config.enable_debug              = true;

    L4::ReferenceTrajectoryConfig trajectory_cfg;
    trajectory_cfg.horizon_back  = 2;
    trajectory_cfg.horizon_ahead = 3;
    trajectory_cfg.dt            = 0.1;

    TinyMpcTrajectoryOptimizer optimizer(config, trajectory_cfg);
    const auto reference =
        make_reference_trajectory(trajectory_cfg.horizon_back + trajectory_cfg.horizon_ahead + 1);
    const auto track = make_track(reference, 1'000'000'000ULL);

    auto result = optimizer.optimize(track, track.timestamp_ns);
    ASSERT_TRUE(result.has_value()) << result.error();

    const WeaponCommand& cmd = *result;
    EXPECT_DOUBLE_EQ(cmd.distance, 2.0);
    ASSERT_TRUE(cmd.viz_debug.has_value());
    EXPECT_EQ(cmd.viz_debug->center_index, 2);
    EXPECT_EQ(cmd.viz_debug->reference_plan.size(), 6U);
    EXPECT_DOUBLE_EQ(cmd.viz_debug->reference_plan.front().distance, 0.0);
}

TEST(TrajectoryOptimizerTest, ReferenceAgeShiftsWindowForwardAndShrinksDebugHorizon) {
    WeaponControllerConfig config;
    config.reference_age_threshold_s = 1.0;
    config.enable_debug              = true;

    L4::ReferenceTrajectoryConfig trajectory_cfg;
    trajectory_cfg.horizon_back  = 2;
    trajectory_cfg.horizon_ahead = 3;
    trajectory_cfg.dt            = 0.1;

    TinyMpcTrajectoryOptimizer optimizer(config, trajectory_cfg);
    const auto reference =
        make_reference_trajectory(trajectory_cfg.horizon_back + trajectory_cfg.horizon_ahead + 1);
    const auto track = make_track(reference, 1'000'000'000ULL);

    auto result = optimizer.optimize(track, 1'200'000'000ULL);
    ASSERT_TRUE(result.has_value()) << result.error();

    const WeaponCommand& cmd = *result;
    EXPECT_DOUBLE_EQ(cmd.distance, 4.0);
    EXPECT_DOUBLE_EQ(cmd.tof, 0.04);
    ASSERT_TRUE(cmd.viz_debug.has_value());
    EXPECT_EQ(cmd.viz_debug->center_index, 2);
    EXPECT_EQ(cmd.viz_debug->reference_plan.size(), 4U);
    EXPECT_DOUBLE_EQ(cmd.viz_debug->reference_plan.front().distance, 2.0);
    EXPECT_DOUBLE_EQ(cmd.viz_debug->reference_plan[2].distance, 4.0);
}

} // namespace
} // namespace fcs::L5
