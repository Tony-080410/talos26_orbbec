#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
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

class InvalidTrajectorySolver final : public fcs::core::trajectory::solver::TrajectorySolver {
public:
    [[nodiscard]] fcs::core::trajectory::solver::AimSolution
        solve(const Eigen::Vector3d&, double) const noexcept override {
        return {
            .yaw            = 0.0,
            .pitch          = 0.0,
            .time_of_flight = 0.0,
            .converged      = false,
            .iterations     = 1,
        };
    }

    [[nodiscard]] std::vector<std::pair<double, double>>
        generate_trajectory(double, double, double) const noexcept override {
        return {};
    }

    [[nodiscard]] std::string_view solver_name() const noexcept override { return "invalid"; }

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

fcs::L4::Aimer::MuzzleTransform make_muzzle(double x, double y, double z) {
    return fcs::L4::Aimer::MuzzleTransform::from_translation(x, y, z);
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

fcs::L3::OutpostTargetState make_outpost_target(double yaw, double v_yaw = 0.0) {
    fcs::L3::OutpostTargetState target;
    target.position = Eigen::Vector2d(1.0, 0.0);
    target.velocity = Eigen::Vector3d::Zero();
    target.yaw      = yaw;
    target.v_yaw    = v_yaw;
    target.z        = {0.0, 0.0, 0.0};
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

TEST(AimerPhase, WholeCarPairLowLayerUsesZeroTwoSubset) {
    fcs::L4::Aimer aimer(fcs::L4::AimerConfig{});
    MockTrajectorySolver solver;
    const auto gimbal = make_gimbal();
    const auto muzzle = make_muzzle();

    fcs::L4::ArmorAimContext context;
    context.target_jumped = true;
    context.phase         = fcs::L4::ArmorAimPhase::WholeCarPair;

    const auto target = make_robot_target(std::numbers::pi, 12.0);

    const auto prediction = aimer.aim(target, context, gimbal, muzzle, 0, 0, 0.0, 30.0, solver);
    ASSERT_TRUE(prediction.has_value()) << prediction.error();
    EXPECT_EQ(prediction->selected_armor_id, 2);
}

TEST(AimerPhase, WholeCarPairHighLayerUsesOneThreeZeroSubset) {
    fcs::L4::Aimer aimer(fcs::L4::AimerConfig{});
    MockTrajectorySolver solver;
    const auto gimbal = make_gimbal();
    const auto muzzle = make_muzzle();

    fcs::L4::ArmorAimContext context;
    context.target_jumped = true;
    context.phase         = fcs::L4::ArmorAimPhase::WholeCarPair;

    const auto prediction = aimer.aim(
        make_robot_target(-std::numbers::pi / 2.0, 12.0), context, gimbal, muzzle, 0, 0, 0.0, 30.0,
        solver);
    ASSERT_TRUE(prediction.has_value()) << prediction.error();
    EXPECT_EQ(prediction->selected_armor_id, 1);
}

TEST(AimerPhase, WholeCarPairUsesCurrentActiveArmorInsteadOfStalePreferredLayer) {
    fcs::L4::Aimer aimer(fcs::L4::AimerConfig{});
    MockTrajectorySolver solver;
    const auto gimbal = make_gimbal();
    const auto muzzle = make_muzzle();

    fcs::L4::ArmorAimContext context;
    context.target_jumped      = true;
    context.phase              = fcs::L4::ArmorAimPhase::WholeCarPair;
    context.preferred_armor_id = 1;

    const auto prediction = aimer.aim(
        make_robot_target(std::numbers::pi, 12.0), context, gimbal, muzzle, 0, 0, 0.0, 30.0,
        solver);
    ASSERT_TRUE(prediction.has_value()) << prediction.error();
    EXPECT_EQ(prediction->selected_armor_id, 2);
}

TEST(AimerPhase, WholeCarPairOutpostUsesAllArmors) {
    fcs::L4::Aimer aimer(fcs::L4::AimerConfig{});
    MockTrajectorySolver solver;
    const auto gimbal = make_gimbal();
    const auto muzzle = make_muzzle();

    fcs::L4::ArmorAimContext context;
    context.target_jumped      = true;
    context.phase              = fcs::L4::ArmorAimPhase::WholeCarPair;
    context.preferred_armor_id = 0;

    const auto prediction = aimer.aim(
        make_outpost_target(-2.0 * std::numbers::pi / 3.0, 12.0), context, gimbal, muzzle, 0, 0,
        0.0, 30.0, solver);
    ASSERT_TRUE(prediction.has_value()) << prediction.error();
    EXPECT_EQ(prediction->selected_armor_id, 1);
}

TEST(AimerPhase, WholeCarCenterProjectsAlongMuzzleRay) {
    fcs::L4::Aimer aimer(fcs::L4::AimerConfig{});
    MockTrajectorySolver solver;
    const auto gimbal = make_gimbal();
    const auto muzzle = make_muzzle(1.0, 0.0, 0.0);

    auto target     = make_robot_target(std::atan2(5.0, 1.0));
    target.position = Eigen::Vector3d(1.0, 5.0, 0.0);
    target.velocity = Eigen::Vector3d::Zero();
    target.radius0  = 0.2;
    target.radius1  = 0.3;
    target.z1       = 0.4;

    fcs::L4::ArmorAimContext context;
    context.target_jumped      = true;
    context.phase              = fcs::L4::ArmorAimPhase::WholeCarCenter;
    context.preferred_armor_id = 1;

    const auto prediction = aimer.aim(target, context, gimbal, muzzle, 0, 0, 0.0, 30.0, solver);
    ASSERT_TRUE(prediction.has_value()) << prediction.error();
    EXPECT_EQ(prediction->selected_armor_id, 0);
    EXPECT_NEAR(prediction->predicted_position.x(), 1.0, 1e-6);
    EXPECT_NEAR(prediction->predicted_position.y(), 4.8, 1e-6);
    EXPECT_NEAR(prediction->predicted_position.z(), target.position.z(), 1e-6);
}

TEST(AimerPhase, FlightTimeTracksFinalArmorAcrossHandoff) {
    fcs::L4::Aimer aimer(fcs::L4::AimerConfig{});
    MockTrajectorySolver solver;
    const auto gimbal = make_gimbal();
    const auto muzzle = make_muzzle();

    fcs::L4::ArmorAimContext context;
    context.target_jumped = true;
    context.phase         = fcs::L4::ArmorAimPhase::SingleArmor;

    const auto prediction =
        aimer.aim(make_robot_target(-3.0, -2.0), context, gimbal, muzzle, 0, 0, 0.0, 0.6, solver);
    ASSERT_TRUE(prediction.has_value()) << prediction.error();
    EXPECT_NE(prediction->rough_selected_armor_id, prediction->selected_armor_id);
    EXPECT_NEAR(
        prediction->flying_time, prediction->predicted_position.norm() / 0.6,
        fcs::L4::kFlyingTimeRefineTolerance);
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

TEST(AimerPhase, BallisticFailureFailsClosed) {
    fcs::L4::Aimer aimer(fcs::L4::AimerConfig{});
    InvalidTrajectorySolver solver;
    const auto gimbal = make_gimbal();
    const auto muzzle = make_muzzle();

    fcs::L4::ArmorAimContext context;
    context.target_jumped = true;
    context.phase         = fcs::L4::ArmorAimPhase::SingleArmor;

    const auto prediction = aimer.aim(
        make_robot_target(-std::numbers::pi / 4.0), context, gimbal, muzzle, 0, 0, 0.0, 30.0,
        solver);
    EXPECT_FALSE(prediction.has_value());
}

TEST(AimerPhase, TrackerArmorVyawTracksSinusoidalOmegaInput) {
    constexpr double kPi       = std::numbers::pi;
    constexpr double kDt       = 0.01;
    constexpr double kDuration = 10.0;
    constexpr double kWarmup   = 2.0;

    auto omega = [](double t) noexcept { return 2.5 + 0.5 * std::sin(2.0 * kPi * t); };

    fcs::L3::TrackerConfig config;
    config.robot.lost_threshold                 = 1.0;
    config.robot.tracking_threshold             = 0;
    config.robot.matcher_gate                   = 100.0;
    config.robot.model.sigma_a_xy               = 10.0;
    config.robot.model.sigma_a_z                = 2.0;
    config.robot.model.sigma_a_yaw              = 20.0;
    config.robot.model.sigma_r0                 = 1.0;
    config.robot.model.sigma_r1                 = 1.0;
    config.robot.model.sigma_h                  = 0.8;
    config.robot.model.meas_yaw_var_floor       = 4e-3;
    config.robot.model.meas_pitch_var_floor     = 4e-3;
    config.robot.model.meas_dist_var_floor      = 0.3;
    config.robot.model.meas_dist_delta_angle_k  = 1.0;
    config.robot.model.meas_armor_yaw_var_floor = 9e-2;
    config.robot.model.meas_armor_yaw_range_k   = 0.005;
    config.robot_inekf.radius0                  = 0.23;
    config.robot_inekf.radius1                  = 0.23;
    config.robot_inekf.height                   = 0.1;

    fcs::L3::TrackerNew tracker(config);

    fcs::L3::RobotTargetState truth;
    truth.position   = Eigen::Vector3d(3.0, 0.4, 0.2);
    truth.velocity   = Eigen::Vector3d::Zero();
    truth.yaw        = 0.3;
    truth.v_yaw      = omega(0.0);
    truth.radius0    = config.robot_inekf.radius0;
    truth.radius1    = config.robot_inekf.radius1;
    truth.z1         = truth.position.z() + config.robot_inekf.height;
    truth.armors_num = 4;

    fcs::AtLegacyArmorMeasurementBatch first_batch;
    first_batch.measurements.push_back(
        make_measurement(truth.armor_poses()[0], fcs::ArmorName::Three));
    ASSERT_TRUE(tracker.first_meet(first_batch).has_value());

    double time                 = 0.0;
    double abs_err_sum          = 0.0;
    double sq_err_sum           = 0.0;
    double max_abs_err          = 0.0;
    double est_min_after_settle = std::numeric_limits<double>::infinity();
    double est_max_after_settle = -std::numeric_limits<double>::infinity();
    size_t sample_count         = 0;

    const int steps = static_cast<int>(kDuration / kDt);
    for (int step = 0; step < steps; ++step) {
        tracker.predict(kDt);

        time += kDt;
        truth.v_yaw = omega(time);
        truth.yaw += truth.v_yaw * kDt;

        fcs::AtLegacyArmorMeasurementBatch batch;
        const auto poses = truth.armor_poses();
        for (const auto& pose : poses) {
            batch.measurements.push_back(make_measurement(pose, fcs::ArmorName::Three));
        }

        ASSERT_TRUE(tracker.update(batch));

        const auto output = tracker.get_output();
        ASSERT_TRUE(output.robot_state() != nullptr);

        const double est_v_yaw = output.robot_state()->v_yaw;
        if (time >= kWarmup) {
            const double err = est_v_yaw - truth.v_yaw;
            abs_err_sum += std::abs(err);
            sq_err_sum += err * err;
            max_abs_err          = std::max(max_abs_err, std::abs(err));
            est_min_after_settle = std::min(est_min_after_settle, est_v_yaw);
            est_max_after_settle = std::max(est_max_after_settle, est_v_yaw);
            ++sample_count;
        }
    }

    ASSERT_GT(sample_count, 0u);

    const double mae  = abs_err_sum / static_cast<double>(sample_count);
    const double rmse = std::sqrt(sq_err_sum / static_cast<double>(sample_count));
    const double amp  = 0.5 * (est_max_after_settle - est_min_after_settle);

    std::cout << "sinusoidal omega tracking: "
              << "mae=" << mae << ", rmse=" << rmse << ", max_abs_err=" << max_abs_err
              << ", est_min=" << est_min_after_settle << ", est_max=" << est_max_after_settle
              << ", est_amp=" << amp << '\n';

    EXPECT_LT(mae, 0.25);
    EXPECT_LT(rmse, 0.30);
    EXPECT_LT(max_abs_err, 0.70);
    EXPECT_GT(amp, 0.15);
    EXPECT_LT(amp, 0.60);
}

TEST(AimerPhase, TrackerArmorIdealFireHitRateOnSinusoidalOmegaInput) {
    constexpr double kPi          = std::numbers::pi;
    constexpr double kDt          = 0.01;
    constexpr double kDuration    = 10.0;
    constexpr double kWarmup      = 2.0;
    constexpr double kBulletSpeed = 30.0;
    constexpr double kArmorWidth  = 0.135;
    constexpr double kArmorHeight = 0.135;

    auto omega  = [](double t) noexcept { return 2.5 + 0.5 * std::sin(2.0 * kPi * t); };
    auto yaw_at = [](double t) noexcept {
        return 0.3 + 2.5 * t + (1.0 - std::cos(2.0 * kPi * t)) / (4.0 * kPi);
    };
    auto wrap_angle = [](double angle) noexcept {
        return std::remainder(angle, 2.0 * std::numbers::pi);
    };

    fcs::L3::TrackerConfig tracker_config;
    tracker_config.robot.lost_threshold                 = 1.0;
    tracker_config.robot.tracking_threshold             = 0;
    tracker_config.robot.matcher_gate                   = 100.0;
    tracker_config.robot.model.sigma_a_xy               = 10.0;
    tracker_config.robot.model.sigma_a_z                = 2.0;
    tracker_config.robot.model.sigma_a_yaw              = 20.0;
    tracker_config.robot.model.sigma_r0                 = 1.0;
    tracker_config.robot.model.sigma_r1                 = 1.0;
    tracker_config.robot.model.sigma_h                  = 0.8;
    tracker_config.robot.model.meas_yaw_var_floor       = 4e-3;
    tracker_config.robot.model.meas_pitch_var_floor     = 4e-3;
    tracker_config.robot.model.meas_dist_var_floor      = 0.3;
    tracker_config.robot.model.meas_dist_delta_angle_k  = 1.0;
    tracker_config.robot.model.meas_armor_yaw_var_floor = 9e-2;
    tracker_config.robot.model.meas_armor_yaw_range_k   = 0.005;
    tracker_config.robot_inekf.radius0                  = 0.23;
    tracker_config.robot_inekf.radius1                  = 0.23;
    tracker_config.robot_inekf.height                   = 0.1;

    fcs::L3::TrackerNew tracker(tracker_config);
    fcs::L4::Aimer aimer(fcs::L4::AimerConfig{});
    MockTrajectorySolver solver;
    const auto gimbal = make_gimbal();
    const auto muzzle = make_muzzle();

    auto truth_state_at = [&](double t) noexcept {
        fcs::L3::RobotTargetState truth;
        truth.position   = Eigen::Vector3d(4.0, 0.0, 0.2);
        truth.velocity   = Eigen::Vector3d::Zero();
        truth.yaw        = yaw_at(t);
        truth.v_yaw      = omega(t);
        truth.radius0    = tracker_config.robot_inekf.radius0;
        truth.radius1    = tracker_config.robot_inekf.radius1;
        truth.z1         = truth.position.z() + tracker_config.robot_inekf.height;
        truth.armors_num = 4;
        return truth;
    };

    const auto truth0 = truth_state_at(0.0);
    fcs::AtLegacyArmorMeasurementBatch first_batch;
    first_batch.measurements.push_back(
        make_measurement(truth0.armor_poses()[0], fcs::ArmorName::Three));
    ASSERT_TRUE(tracker.first_meet(first_batch).has_value());

    size_t total_shots                   = 0;
    size_t total_hits                    = 0;
    size_t settled_shots                 = 0;
    size_t settled_hits                  = 0;
    double yaw_abs_err_sum               = 0.0;
    double pitch_abs_err_sum             = 0.0;
    double max_abs_yaw_err               = 0.0;
    double max_abs_pitch_err             = 0.0;
    double min_yaw_margin_after_warmup   = std::numeric_limits<double>::infinity();
    double min_pitch_margin_after_warmup = std::numeric_limits<double>::infinity();

    const int steps = static_cast<int>(kDuration / kDt);
    for (int step = 1; step <= steps; ++step) {
        const double time = static_cast<double>(step) * kDt;
        tracker.predict(kDt);

        const auto truth_now = truth_state_at(time);
        fcs::AtLegacyArmorMeasurementBatch batch;
        for (const auto& pose : truth_now.armor_poses()) {
            batch.measurements.push_back(make_measurement(pose, fcs::ArmorName::Three));
        }
        ASSERT_TRUE(tracker.update(batch));

        const auto output = tracker.get_output();
        ASSERT_TRUE(output.robot_state() != nullptr);

        fcs::L4::ArmorAimContext context;
        context.target_jumped      = output.target_jumped;
        context.phase              = fcs::L4::ArmorAimPhase::SingleArmor;
        context.preferred_armor_id = output.last_armor_id;

        const uint64_t measurement_ns = static_cast<uint64_t>(std::llround(time * 1.0e9));
        const auto prediction         = aimer.aim(
            *output.robot_state(), context, gimbal, muzzle, measurement_ns, measurement_ns, 0.0,
            kBulletSpeed, solver);
        ASSERT_TRUE(prediction.has_value()) << prediction.error();

        const double future_time =
            static_cast<double>(prediction->predicted_future_ns - measurement_ns) * 1.0e-9 + time;
        const auto truth_future = truth_state_at(future_time);
        const auto actual_pose =
            truth_future.armor_poses()[static_cast<size_t>(prediction->selected_armor_id)];
        const Eigen::Vector3d actual_pos = actual_pose.head<3>() - muzzle.translation();

        const double actual_yaw = std::atan2(actual_pos.y(), actual_pos.x());
        const double actual_pitch =
            std::atan2(actual_pos.z(), std::hypot(actual_pos.x(), actual_pos.y()));
        const double yaw_err   = wrap_angle(prediction->aim_yaw - actual_yaw);
        const double pitch_err = prediction->aim_pitch - actual_pitch;
        const double distance  = actual_pos.norm();
        const double yaw_tol   = std::atan2(kArmorWidth / 2.0, distance);
        const double pitch_tol = std::atan2(kArmorHeight / 2.0, distance);
        const bool hit         = std::abs(yaw_err) < yaw_tol && std::abs(pitch_err) < pitch_tol;

        ++total_shots;
        if (hit) {
            ++total_hits;
        }

        if (time >= kWarmup) {
            ++settled_shots;
            if (hit) {
                ++settled_hits;
            }
            yaw_abs_err_sum += std::abs(yaw_err);
            pitch_abs_err_sum += std::abs(pitch_err);
            max_abs_yaw_err   = std::max(max_abs_yaw_err, std::abs(yaw_err));
            max_abs_pitch_err = std::max(max_abs_pitch_err, std::abs(pitch_err));
            min_yaw_margin_after_warmup =
                std::min(min_yaw_margin_after_warmup, yaw_tol - std::abs(yaw_err));
            min_pitch_margin_after_warmup =
                std::min(min_pitch_margin_after_warmup, pitch_tol - std::abs(pitch_err));
        }
    }

    ASSERT_GT(total_shots, 0u);
    ASSERT_GT(settled_shots, 0u);

    const double total_hit_rate =
        static_cast<double>(total_hits) / static_cast<double>(total_shots);
    const double settled_hit_rate =
        static_cast<double>(settled_hits) / static_cast<double>(settled_shots);
    const double mean_abs_yaw_err   = yaw_abs_err_sum / static_cast<double>(settled_shots);
    const double mean_abs_pitch_err = pitch_abs_err_sum / static_cast<double>(settled_shots);

    std::cout << "ideal fire on sinusoidal omega: "
              << "total_hit_rate=" << total_hit_rate << ", settled_hit_rate=" << settled_hit_rate
              << ", mean_abs_yaw_err=" << mean_abs_yaw_err
              << ", max_abs_yaw_err=" << max_abs_yaw_err
              << ", mean_abs_pitch_err=" << mean_abs_pitch_err
              << ", max_abs_pitch_err=" << max_abs_pitch_err
              << ", min_yaw_margin=" << min_yaw_margin_after_warmup
              << ", min_pitch_margin=" << min_pitch_margin_after_warmup << '\n';

    EXPECT_GT(total_hit_rate, 0.95);
    EXPECT_GT(settled_hit_rate, 0.99);
    EXPECT_GT(min_yaw_margin_after_warmup, 0.0);
    EXPECT_GT(min_pitch_margin_after_warmup, 0.0);
}

TEST(AimerPhase, TrackerArmorIdealFireHitRateOnTwoSecondAbsSinBurst) {
    constexpr double kPi          = std::numbers::pi;
    constexpr double kDt          = 0.01;
    constexpr double kDuration    = 8.0;
    constexpr double kWarmup      = 1.0;
    constexpr double kBulletSpeed = 30.0;
    constexpr double kArmorWidth  = 0.135;
    constexpr double kArmorHeight = 0.135;
    constexpr double kBaseOmega   = 2.0;
    constexpr double kPeakOmega   = 3.0;
    constexpr double kBurstStart  = 2.0;
    constexpr double kBurstEnd    = 4.0;

    auto omega = [](double t) noexcept {
        if (t < kBurstStart || t > kBurstEnd) {
            return kBaseOmega;
        }
        const double phase = kPi * (t - kBurstStart);
        return kBaseOmega + (kPeakOmega - kBaseOmega) * std::abs(std::sin(phase));
    };

    auto yaw_at = [&](double t) noexcept {
        const double pre_burst  = std::min(t, kBurstStart);
        const double burst_t    = std::clamp(t, kBurstStart, kBurstEnd) - kBurstStart;
        const double post_burst = std::max(0.0, t - kBurstEnd);

        const double yaw_pre   = kBaseOmega * pre_burst;
        const double yaw_burst = 2.0 * burst_t + (1.0 - std::cos(kPi * burst_t)) / kPi;
        const double yaw_post  = kBaseOmega * post_burst;
        return 0.3 + yaw_pre + yaw_burst + yaw_post;
    };

    fcs::L3::TrackerConfig tracker_config;
    tracker_config.robot.lost_threshold                 = 1.0;
    tracker_config.robot.tracking_threshold             = 0;
    tracker_config.robot.matcher_gate                   = 100.0;
    tracker_config.robot.model.sigma_a_xy               = 10.0;
    tracker_config.robot.model.sigma_a_z                = 2.0;
    tracker_config.robot.model.sigma_a_yaw              = 20.0;
    tracker_config.robot.model.sigma_r0                 = 1.0;
    tracker_config.robot.model.sigma_r1                 = 1.0;
    tracker_config.robot.model.sigma_h                  = 0.8;
    tracker_config.robot.model.meas_yaw_var_floor       = 4e-3;
    tracker_config.robot.model.meas_pitch_var_floor     = 4e-3;
    tracker_config.robot.model.meas_dist_var_floor      = 0.3;
    tracker_config.robot.model.meas_dist_delta_angle_k  = 1.0;
    tracker_config.robot.model.meas_armor_yaw_var_floor = 9e-2;
    tracker_config.robot.model.meas_armor_yaw_range_k   = 0.005;
    tracker_config.robot_inekf.radius0                  = 0.23;
    tracker_config.robot_inekf.radius1                  = 0.23;
    tracker_config.robot_inekf.height                   = 0.1;

    fcs::L3::TrackerNew tracker(tracker_config);
    fcs::L4::Aimer aimer(fcs::L4::AimerConfig{});
    MockTrajectorySolver solver;
    const auto gimbal = make_gimbal();
    const auto muzzle = make_muzzle();

    auto wrap_angle = [](double angle) noexcept {
        return std::remainder(angle, 2.0 * std::numbers::pi);
    };

    auto truth_state_at = [&](double t) noexcept {
        fcs::L3::RobotTargetState truth;
        truth.position   = Eigen::Vector3d(4.0, 0.0, 0.2);
        truth.velocity   = Eigen::Vector3d::Zero();
        truth.yaw        = yaw_at(t);
        truth.v_yaw      = omega(t);
        truth.radius0    = tracker_config.robot_inekf.radius0;
        truth.radius1    = tracker_config.robot_inekf.radius1;
        truth.z1         = truth.position.z() + tracker_config.robot_inekf.height;
        truth.armors_num = 4;
        return truth;
    };

    const auto truth0 = truth_state_at(0.0);
    fcs::AtLegacyArmorMeasurementBatch first_batch;
    first_batch.measurements.push_back(
        make_measurement(truth0.armor_poses()[0], fcs::ArmorName::Three));
    ASSERT_TRUE(tracker.first_meet(first_batch).has_value());

    size_t total_shots     = 0;
    size_t total_hits      = 0;
    size_t burst_shots     = 0;
    size_t burst_hits      = 0;
    size_t settled_shots   = 0;
    size_t settled_hits    = 0;
    double yaw_abs_err_sum = 0.0;
    double max_abs_yaw_err = 0.0;

    const int steps = static_cast<int>(kDuration / kDt);
    for (int step = 1; step <= steps; ++step) {
        const double time = static_cast<double>(step) * kDt;
        tracker.predict(kDt);

        const auto truth_now = truth_state_at(time);
        fcs::AtLegacyArmorMeasurementBatch batch;
        for (const auto& pose : truth_now.armor_poses()) {
            batch.measurements.push_back(make_measurement(pose, fcs::ArmorName::Three));
        }
        ASSERT_TRUE(tracker.update(batch));

        const auto output = tracker.get_output();
        ASSERT_TRUE(output.robot_state() != nullptr);

        fcs::L4::ArmorAimContext context;
        context.target_jumped      = output.target_jumped;
        context.phase              = fcs::L4::ArmorAimPhase::SingleArmor;
        context.preferred_armor_id = output.last_armor_id;

        const uint64_t measurement_ns = static_cast<uint64_t>(std::llround(time * 1.0e9));
        const auto prediction         = aimer.aim(
            *output.robot_state(), context, gimbal, muzzle, measurement_ns, measurement_ns, 0.0,
            kBulletSpeed, solver);
        ASSERT_TRUE(prediction.has_value()) << prediction.error();

        const double future_time =
            static_cast<double>(prediction->predicted_future_ns - measurement_ns) * 1.0e-9 + time;
        const auto truth_future = truth_state_at(future_time);
        const auto actual_pose =
            truth_future.armor_poses()[static_cast<size_t>(prediction->selected_armor_id)];
        const Eigen::Vector3d actual_pos = actual_pose.head<3>() - muzzle.translation();

        const double actual_yaw = std::atan2(actual_pos.y(), actual_pos.x());
        const double actual_pitch =
            std::atan2(actual_pos.z(), std::hypot(actual_pos.x(), actual_pos.y()));
        const double yaw_err   = wrap_angle(prediction->aim_yaw - actual_yaw);
        const double pitch_err = prediction->aim_pitch - actual_pitch;
        const double distance  = actual_pos.norm();
        const double yaw_tol   = std::atan2(kArmorWidth / 2.0, distance);
        const double pitch_tol = std::atan2(kArmorHeight / 2.0, distance);
        const bool hit         = std::abs(yaw_err) < yaw_tol && std::abs(pitch_err) < pitch_tol;

        ++total_shots;
        if (hit) {
            ++total_hits;
        }
        if (time >= kBurstStart && time <= kBurstEnd) {
            ++burst_shots;
            if (hit) {
                ++burst_hits;
            }
        }
        if (time >= kWarmup) {
            ++settled_shots;
            if (hit) {
                ++settled_hits;
            }
            yaw_abs_err_sum += std::abs(yaw_err);
            max_abs_yaw_err = std::max(max_abs_yaw_err, std::abs(yaw_err));
        }
    }

    ASSERT_GT(total_shots, 0u);
    ASSERT_GT(burst_shots, 0u);
    ASSERT_GT(settled_shots, 0u);

    const double total_hit_rate =
        static_cast<double>(total_hits) / static_cast<double>(total_shots);
    const double burst_hit_rate =
        static_cast<double>(burst_hits) / static_cast<double>(burst_shots);
    const double settled_hit_rate =
        static_cast<double>(settled_hits) / static_cast<double>(settled_shots);
    const double mean_abs_yaw_err = yaw_abs_err_sum / static_cast<double>(settled_shots);

    std::cout << "ideal fire on abs-sin burst omega: "
              << "total_hit_rate=" << total_hit_rate << ", burst_hit_rate=" << burst_hit_rate
              << ", settled_hit_rate=" << settled_hit_rate
              << ", mean_abs_yaw_err=" << mean_abs_yaw_err
              << ", max_abs_yaw_err=" << max_abs_yaw_err << '\n';

    EXPECT_GT(total_hit_rate, 0.95);
    EXPECT_GT(burst_hit_rate, 0.95);
    EXPECT_GT(settled_hit_rate, 0.98);
}
