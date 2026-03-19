#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <string_view>
#include <utility>
#include <vector>

#include "L3_estimation/tracker/new_tracker.hpp"
#include "L3_estimation/tracker/types.hpp"
#include "L4_planning/aimer/aimer.hpp"
#include "L4_planning/planning/fire_decision.hpp"
#include "core/trajectory/solver/solver_interfaces.hpp"

namespace {

class MockTrajectorySolver final : public fcs::core::trajectory::solver::TrajectorySolver {
public:
    [[nodiscard]] fcs::core::trajectory::solver::AimSolution
        solve(const Eigen::Vector3d& target_pos, double v0) const noexcept override {
        const double distance = target_pos.norm();
        const double xy       = std::hypot(target_pos.x(), target_pos.y());
        return {
            .yaw            = std::atan2(target_pos.y(), target_pos.x()),
            .pitch          = std::atan2(target_pos.z(), xy),
            .time_of_flight = distance / std::max(v0, 1e-6),
            .converged      = true,
            .iterations     = 1,
        };
    }

    [[nodiscard]] std::vector<std::pair<double, double>>
        generate_trajectory(double, double, double) const noexcept override {
        return {};
    }

    [[nodiscard]] std::string_view solver_name() const noexcept override { return "mock"; }

    [[nodiscard]] const fcs::core::trajectory::model::BallisticModel*
        get_model() const noexcept override {
        return nullptr;
    }
};

fcs::L4::Aimer::GimbalTransform make_gimbal() {
    return fcs::L4::Aimer::GimbalTransform::from_rpy(0.0, 0.0, 0.0);
}

fcs::L4::Aimer::MuzzleTransform make_muzzle() {
    return fcs::L4::Aimer::MuzzleTransform::from_translation(0.0, 0.0, 0.0);
}

fcs::L3::RobotTargetState make_robot_target(double yaw, double v_yaw = 0.0) {
    fcs::L3::RobotTargetState target;
    target.position   = Eigen::Vector3d(1.0, 0.0, 0.0);
    target.velocity   = Eigen::Vector3d::Zero();
    target.yaw        = yaw;
    target.v_yaw      = v_yaw;
    target.radius0    = 0.2;
    target.radius1    = 0.3;
    target.z1         = 0.4;
    target.armors_num = 4;
    return target;
}

fcs::AtLegacyArmorMeasurement make_measurement(
    const Eigen::Vector4d& armor_pose, fcs::ArmorName name, float image_center_distance = 0.0f) {
    fcs::AtLegacyArmorMeasurement measurement;
    measurement.transform = fcs::AtLegacyArmorMeasurement::Transform::from_rpy(
        0.0, 0.0, armor_pose[3], armor_pose[0], armor_pose[1], armor_pose[2]);
    measurement.name                     = name;
    measurement.color                    = fcs::ArmorColor::Blue;
    measurement.type                     = fcs::ArmorType::Small;
    measurement.confidence               = 1.0f;
    measurement.distance_to_image_center = image_center_distance;
    measurement.timestamp_ns             = 0;
    return measurement;
}

} // namespace

TEST(AimerPhase, SingleArmorLocksSymmetricCandidates) {
    fcs::L4::Aimer aimer(fcs::L4::AimerConfig{});
    MockTrajectorySolver solver;
    const auto gimbal = make_gimbal();
    const auto muzzle = make_muzzle();
    const auto target = make_robot_target(-std::numbers::pi / 4.0);

    fcs::L4::ArmorAimContext lock_left;
    lock_left.target_jumped      = true;
    lock_left.phase              = fcs::L4::ArmorAimPhase::SingleArmor;
    lock_left.preferred_armor_id = 0;

    const auto left = aimer.aim(target, lock_left, gimbal, muzzle, 0, 0, 0.0, 30.0, solver);
    ASSERT_TRUE(left.has_value()) << left.error();
    EXPECT_EQ(left->selected_armor_id, 0);

    fcs::L4::ArmorAimContext lock_right = lock_left;
    lock_right.preferred_armor_id       = 1;
    const auto right = aimer.aim(target, lock_right, gimbal, muzzle, 0, 0, 0.0, 30.0, solver);
    ASSERT_TRUE(right.has_value()) << right.error();
    EXPECT_EQ(right->selected_armor_id, 1);
}

TEST(AimerPhase, WholeCarArmorUsesRotationDirection) {
    fcs::L4::Aimer aimer(fcs::L4::AimerConfig{});
    MockTrajectorySolver solver;
    const auto gimbal = make_gimbal();
    const auto muzzle = make_muzzle();

    fcs::L4::ArmorAimContext context;
    context.target_jumped = true;
    context.phase         = fcs::L4::ArmorAimPhase::WholeCarArmor;

    const auto positive = aimer.aim(
        make_robot_target(-std::numbers::pi / 4.0, 8.0), context, gimbal, muzzle, 0, 0, 0.0, 30.0,
        solver);
    ASSERT_TRUE(positive.has_value()) << positive.error();
    EXPECT_EQ(positive->selected_armor_id, 0);

    const auto negative = aimer.aim(
        make_robot_target(-std::numbers::pi / 4.0, -8.0), context, gimbal, muzzle, 0, 0, 0.0, 30.0,
        solver);
    ASSERT_TRUE(negative.has_value()) << negative.error();
    EXPECT_EQ(negative->selected_armor_id, 1);
}

TEST(AimerPhase, WholeCarArmorFallbackStaysWithinValidCandidates) {
    fcs::L4::Aimer aimer(fcs::L4::AimerConfig{});
    MockTrajectorySolver solver;
    const auto gimbal = make_gimbal();
    const auto muzzle = make_muzzle();

    fcs::L4::ArmorAimContext context;
    context.target_jumped      = true;
    context.phase              = fcs::L4::ArmorAimPhase::WholeCarArmor;
    context.preferred_armor_id = 2;

    const auto prediction = aimer.aim(
        make_robot_target(-25.0 * std::numbers::pi / 180.0, -8.0), context, gimbal, muzzle, 0, 0,
        0.0, 30.0, solver);
    ASSERT_TRUE(prediction.has_value()) << prediction.error();
    EXPECT_NE(prediction->selected_armor_id, 2);
    EXPECT_LE(
        std::abs(prediction->diff_yaw),
        fcs::L4::AimerConfig{}.coming_angle_deg * std::numbers::pi / 180.0);
}

TEST(AimerPhase, WholeCarPairReselectsLegacySubsetWithoutLocking) {
    fcs::L4::Aimer aimer(fcs::L4::AimerConfig{});
    MockTrajectorySolver solver;
    const auto gimbal = make_gimbal();
    const auto muzzle = make_muzzle();

    fcs::L4::ArmorAimContext context;
    context.target_jumped      = true;
    context.phase              = fcs::L4::ArmorAimPhase::WholeCarPair;
    context.preferred_armor_id = 0;

    const auto prediction = aimer.aim(
        make_robot_target(std::numbers::pi, 12.0), context, gimbal, muzzle, 0, 0, 0.0, 30.0,
        solver);
    ASSERT_TRUE(prediction.has_value()) << prediction.error();
    EXPECT_EQ(prediction->selected_armor_id, 2);
}

TEST(AimerPhase, WholeCarCenterProjectsCenterAim) {
    fcs::L4::Aimer aimer(fcs::L4::AimerConfig{});
    MockTrajectorySolver solver;
    const auto gimbal = make_gimbal();
    const auto muzzle = make_muzzle();
    const auto target = make_robot_target(-std::numbers::pi / 8.0);

    fcs::L4::ArmorAimContext context;
    context.target_jumped      = true;
    context.phase              = fcs::L4::ArmorAimPhase::WholeCarCenter;
    context.preferred_armor_id = 1;

    const auto prediction = aimer.aim(target, context, gimbal, muzzle, 0, 0, 0.0, 30.0, solver);
    ASSERT_TRUE(prediction.has_value()) << prediction.error();
    EXPECT_EQ(prediction->selected_armor_id, 0);
    EXPECT_NEAR(prediction->predicted_position.x(), 0.8, 1e-6);
    EXPECT_NEAR(prediction->predicted_position.y(), 0.0, 1e-6);
    EXPECT_NEAR(prediction->predicted_position.z(), target.position.z(), 1e-6);
}

TEST(AimerPhase, PairAndCenterFireGateRejectsOutsideYawLimit) {
    const fcs::L4::FireDecisionConfig fire_cfg{};
    const bool allowed = fcs::L4::is_on_target_phase_aware(
        fire_cfg, fcs::L4::ArmorAimPhase::WholeCarPair, 50.0 * std::numbers::pi / 180.0, 40.0, 0.25,
        0.0, 0.0, 0.0, 0.0, 5.0);
    EXPECT_FALSE(allowed);
}

TEST(AimerPhase, TrackerJumpedLatchesWhenAnyVisibleArmorIsNonZero) {
    fcs::L3::TrackerConfig config;
    config.robot.lost_threshold     = 1.0;
    config.robot.tracking_threshold = 0;
    config.robot.matcher_gate       = 100.0;

    fcs::L3::TrackerNew tracker(config);

    fcs::AtLegacyArmorMeasurementBatch first_batch;
    first_batch.measurements.push_back(
        make_measurement(Eigen::Vector4d{0.8, 0.0, 0.0, 0.0}, fcs::ArmorName::Three, 0.0f));
    ASSERT_TRUE(tracker.first_meet(first_batch).has_value());

    const auto first_output = tracker.get_output();
    ASSERT_TRUE(first_output.robot_state() != nullptr);
    EXPECT_FALSE(first_output.target_jumped);

    const auto armor_poses = first_output.robot_state()->armor_poses();

    fcs::AtLegacyArmorMeasurementBatch update_batch;
    update_batch.measurements.push_back(make_measurement(armor_poses[0], fcs::ArmorName::Three));
    update_batch.measurements.push_back(make_measurement(armor_poses[1], fcs::ArmorName::Three));

    ASSERT_TRUE(tracker.update(update_batch));

    const auto output = tracker.get_output();
    ASSERT_TRUE(output.robot_state() != nullptr);
    EXPECT_TRUE(output.target_jumped);
}
