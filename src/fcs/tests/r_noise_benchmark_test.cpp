#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <optional>
#include <vector>

#include "L3_estimation/tracker/new_motion_model.hpp"
#include "L3_estimation/tracker/new_tracker.hpp"
#include "L4_planning/aimer/aimer.hpp"
#include "core/trajectory/solver/solver_interfaces.hpp"

namespace {

using namespace std::chrono_literals;

// ============================================================================
// Mock trajectory solver (from aimer_phase_test.cpp)
// ============================================================================

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

// ============================================================================
// Test helpers (from aimer_phase_test.cpp)
// ============================================================================

fcs::L4::Aimer::GimbalTransform make_gimbal() {
    return fcs::L4::Aimer::GimbalTransform::from_rpy(0.0, 0.0, 0.0);
}

fcs::L4::Aimer::MuzzleTransform make_muzzle() {
    return fcs::L4::Aimer::MuzzleTransform::from_translation(0.0, 0.0, 0.0);
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

double wrap_angle(double angle) noexcept { return std::remainder(angle, 2.0 * std::numbers::pi); }

// ============================================================================
// OLD R heuristic (reconstructed from git HEAD)
// ============================================================================
// Old params: meas_yaw_var_floor=4e-3, meas_pitch_var_floor=4e-3,
//             meas_dist_var_floor=0.3, meas_dist_delta_angle_k=1.0,
//             meas_armor_yaw_var_floor=9e-2, meas_armor_yaw_range_k=1/200

struct OldRParams {
    double meas_yaw_var_floor       = 4e-3;
    double meas_pitch_var_floor     = 4e-3;
    double meas_dist_var_floor      = 0.3;
    double meas_dist_delta_angle_k  = 1.0;
    double meas_armor_yaw_var_floor = 9e-2;
    double meas_armor_yaw_range_k   = 1.0 / 200.0;
};

/// Old R diagonal: yaw=const, pitch=const, dist=log(delta_angle), armor_yaw=log(distance)
[[nodiscard]] Eigen::Vector4d
    old_r_diag(double distance, double delta_angle, const OldRParams& p = {}) noexcept {
    Eigen::Vector4d r;
    r[0] = std::max(1e-8, p.meas_yaw_var_floor);
    r[1] = std::max(1e-8, p.meas_pitch_var_floor);
    r[2] = std::max(
        1e-8,
        p.meas_dist_var_floor + p.meas_dist_delta_angle_k * std::log(std::abs(delta_angle) + 1.0));
    r[3] = std::max(
        1e-8,
        p.meas_armor_yaw_var_floor + p.meas_armor_yaw_range_k * std::log(std::abs(distance) + 1.0));
    return r;
}

// ============================================================================
// NEW R using DistanceNoiseCoeff (current code)
// ============================================================================

struct NewRParams {
    double r_dist_ref = 5.0;
    fcs::L3::DistanceNoiseCoeff r_yaw{2e-3, 2e-3};
    fcs::L3::DistanceNoiseCoeff r_pitch{2e-3, 2e-3};
    fcs::L3::DistanceNoiseCoeff r_dist{0.15, 0.15};
    fcs::L3::DistanceNoiseCoeff r_armor_yaw{0.045, 0.045};
};

[[nodiscard]] Eigen::Vector4d new_r_diag(double distance, const NewRParams& p = {}) noexcept {
    Eigen::Vector4d r;
    r[0] = p.r_yaw.at(distance, p.r_dist_ref);
    r[1] = p.r_pitch.at(distance, p.r_dist_ref);
    r[2] = p.r_dist.at(distance, p.r_dist_ref);
    r[3] = p.r_armor_yaw.at(distance, p.r_dist_ref);
    return r;
}

// ============================================================================
// Tracker config builders
// ============================================================================

fcs::L3::TrackerConfig make_config(bool use_old_r) {
    fcs::L3::TrackerConfig config;
    config.robot.lost_threshold     = 1.0;
    config.robot.tracking_threshold = 0;
    config.robot.matcher_gate       = 100.0;
    config.robot.model.sigma_a_xy   = 10.0;
    config.robot.model.sigma_a_z    = 2.0;
    config.robot.model.sigma_a_yaw  = 20.0;
    config.robot.model.sigma_r0     = 1.0;
    config.robot.model.sigma_r1     = 1.0;
    config.robot.model.sigma_h      = 0.8;
    config.robot.model.r_dist_ref   = 5.0;

    if (use_old_r) {
        // Old heuristic: constant yaw/pitch, log-based dist/armor_yaw
        // Set z=0 to make DistanceNoiseCoeff return just the base value (constant)
        config.robot.model.r_yaw       = {4e-3, 0.0};
        config.robot.model.r_pitch     = {4e-3, 0.0};
        config.robot.model.r_dist      = {0.3, 0.0};
        config.robot.model.r_armor_yaw = {9e-2, 0.0};
    } else {
        config.robot.model.r_yaw       = {2e-3, 2e-3};
        config.robot.model.r_pitch     = {2e-3, 2e-3};
        config.robot.model.r_dist      = {0.15, 0.15};
        config.robot.model.r_armor_yaw = {0.045, 0.045};
    }

    config.robot_inekf.radius0 = 0.23;
    config.robot_inekf.radius1 = 0.23;
    config.robot_inekf.height  = 0.1;
    return config;
}

// ============================================================================
// Test 1: R noise profile table
// ============================================================================

TEST(RNoiseBenchmark, RProfileTable) {
    constexpr double kDeltaAngle = 0.1; // typical delta_angle between armor_yaw and observation yaw

    std::cout << "\n";
    std::cout << "====================================================================\n";
    std::cout << "=== R Noise Profile: Old (constant/log) vs New (d^2 scaling) ===\n";
    std::cout << "====================================================================\n";
    std::cout << "  Old: R_yaw=4e-3(const), R_pitch=4e-3(const), R_dist=0.3+log(da), "
                 "R_ayaw=0.09+log(d)\n";
    std::cout << "  New: R(i) = base + z*(d/5)^2\n";
    std::cout << "       R_yaw={2e-3,2e-3}, R_pitch={2e-3,2e-3}, R_dist={0.15,0.15}, "
                 "R_ayaw={0.045,0.045}\n";
    std::cout << "\n";

    // Header
    std::printf(
        "  d(m)  |              OLD               |              NEW               |       ratio "
        "(new/old)\n");
    std::printf(
        "        |   r_yaw   r_pit   r_dst   r_ayaw |   r_yaw   r_pit   r_dst   r_ayaw |   yaw    "
        "pit    dst   ayaw\n");
    std::printf(
        "  "
        "------+---------+---------+---------+---------+---------+---------+---------+---------+---"
        "------+---------+---------+-------\n");

    const double distances[] = {1.0, 2.0, 3.0, 5.0, 8.0, 10.0};
    for (const double d : distances) {
        const auto old_r = old_r_diag(d, kDeltaAngle);
        const auto new_r = new_r_diag(d);

        std::printf(
            "  %4.1f  | %7.5f %7.5f %7.5f %7.5f | %7.5f %7.5f %7.5f %7.5f | %5.3f  %5.3f  %5.3f  "
            "%5.3f\n",
            d, old_r[0], old_r[1], old_r[2], old_r[3], new_r[0], new_r[1], new_r[2], new_r[3],
            new_r[0] / old_r[0], new_r[1] / old_r[1], new_r[2] / old_r[2], new_r[3] / old_r[3]);
    }
    std::cout << "\n";
    std::cout << "  Key insight:\n";
    std::cout << "    Old R is CONSTANT for yaw/pitch, weakly log-dependent for dist/armor_yaw.\n";
    std::cout << "    New R scales quadratically with distance for ALL dimensions.\n";
    std::cout << "    At d=5m (calibration point): old ~ new (by design).\n";
    std::cout << "    At d<5m: new R is LOWER  -> tighter tracking (more responsive).\n";
    std::cout << "    At d>5m: new R is HIGHER -> more conservative (rejects bad matches).\n";
}

// ============================================================================
// Test 2: Tracker accuracy vs distance
// ============================================================================

struct TrackingMetrics {
    double v_yaw_mae;
    double v_yaw_rmse;
    double pos_rmse;
    double yaw_rmse;
    size_t sample_count;
};

TrackingMetrics run_tracker_at_distance(double target_distance, bool use_old_r) {
    constexpr double kPi       = std::numbers::pi;
    constexpr double kDt       = 0.01;
    constexpr double kDuration = 10.0;
    constexpr double kWarmup   = 2.0;

    auto omega  = [](double t) noexcept { return 2.5 + 0.5 * std::sin(2.0 * kPi * t); };
    auto yaw_at = [](double t) noexcept {
        return 0.3 + 2.5 * t + (1.0 - std::cos(2.0 * kPi * t)) / (4.0 * kPi);
    };

    fcs::L3::TrackerNew tracker(make_config(use_old_r));

    auto truth_state_at = [&](double t) noexcept {
        fcs::L3::RobotTargetState truth;
        truth.position   = Eigen::Vector3d(target_distance, 0.0, 0.2);
        truth.velocity   = Eigen::Vector3d::Zero();
        truth.yaw        = yaw_at(t);
        truth.v_yaw      = omega(t);
        truth.radius0    = 0.23;
        truth.radius1    = 0.23;
        truth.z1         = truth.position.z() + 0.1;
        truth.armors_num = 4;
        return truth;
    };

    const auto truth0 = truth_state_at(0.0);
    fcs::AtLegacyArmorMeasurementBatch first_batch;
    first_batch.measurements.push_back(
        make_measurement(truth0.armor_poses()[0], fcs::ArmorName::Three));
    if (!tracker.first_meet(first_batch).has_value()) {
        return {};
    }

    double v_yaw_abs_err_sum = 0.0;
    double v_yaw_sq_err_sum  = 0.0;
    double pos_sq_err_sum    = 0.0;
    double yaw_sq_err_sum    = 0.0;
    size_t sample_count      = 0;

    const int steps = static_cast<int>(kDuration / kDt);
    for (int step = 1; step <= steps; ++step) {
        const double time = static_cast<double>(step) * kDt;
        tracker.predict(kDt);

        const auto truth_now = truth_state_at(time);
        fcs::AtLegacyArmorMeasurementBatch batch;
        for (const auto& pose : truth_now.armor_poses()) {
            batch.measurements.push_back(make_measurement(pose, fcs::ArmorName::Three));
        }

        if (!tracker.update(batch))
            continue;

        const auto output = tracker.get_output();
        if (output.robot_state() == nullptr)
            continue;

        if (time < kWarmup)
            continue;

        const double v_yaw_err = output.robot_state()->v_yaw - truth_now.v_yaw;
        const double pos_err   = (Eigen::Vector3d(
                                      output.robot_state()->position.x() - truth_now.position.x(),
                                      output.robot_state()->position.y() - truth_now.position.y(),
                                      output.robot_state()->position.z() - truth_now.position.z()))
                                     .norm();
        const double yaw_err   = wrap_angle(output.robot_state()->yaw - truth_now.yaw);

        v_yaw_abs_err_sum += std::abs(v_yaw_err);
        v_yaw_sq_err_sum += v_yaw_err * v_yaw_err;
        pos_sq_err_sum += pos_err * pos_err;
        yaw_sq_err_sum += yaw_err * yaw_err;
        ++sample_count;
    }

    if (sample_count == 0)
        return {};

    return {
        .v_yaw_mae    = v_yaw_abs_err_sum / static_cast<double>(sample_count),
        .v_yaw_rmse   = std::sqrt(v_yaw_sq_err_sum / static_cast<double>(sample_count)),
        .pos_rmse     = std::sqrt(pos_sq_err_sum / static_cast<double>(sample_count)),
        .yaw_rmse     = std::sqrt(yaw_sq_err_sum / static_cast<double>(sample_count)),
        .sample_count = sample_count,
    };
}

TEST(RNoiseBenchmark, TrackerAccuracyVsDistance) {
    constexpr std::array<double, 3> kDistances = {2.0, 5.0, 8.0};

    std::cout << "\n";
    std::cout << "========================================================================\n";
    std::cout << "=== Tracker Accuracy: Old (constant R) vs New (distance-dependent R) ===\n";
    std::cout << "========================================================================\n";
    std::cout << "  Scenario: 4-armor robot, sinusoidal omega (2.5+0.5*sin), 10s, dt=0.01s\n";
    std::cout << "\n";

    // Header
    std::printf(
        "  d(m) Ver | v_yaw_MAE v_yaw_RMSE  pos_RMSE  yaw_RMSE | v_yaw_MAE v_yaw_RMSE  pos_RMSE  "
        "yaw_RMSE\n");
    std::printf(
        "  -----+-----+------------------ OLD ------------------+------------------ NEW "
        "------------------\n");

    for (const double d : kDistances) {
        auto old_m = run_tracker_at_distance(d, true);
        auto new_m = run_tracker_at_distance(d, false);

        ASSERT_GT(old_m.sample_count, 0u) << "No samples at d=" << d << " (old)";
        ASSERT_GT(new_m.sample_count, 0u) << "No samples at d=" << d << " (new)";

        std::printf(
            "  %4.1f OLD | %9.6f %10.6f %9.6f %9.6f |\n", d, old_m.v_yaw_mae, old_m.v_yaw_rmse,
            old_m.pos_rmse, old_m.yaw_rmse);
        std::printf(
            "  %4.1f NEW | %9.6f %10.6f %9.6f %9.6f |\n", d, new_m.v_yaw_mae, new_m.v_yaw_rmse,
            new_m.pos_rmse, new_m.yaw_rmse);
        std::printf(
            "         | v_yaw_MAE ratio (new/old) = %.3f\n\n", new_m.v_yaw_mae / old_m.v_yaw_mae);

        // Regression guard: new tracker must maintain tracking quality
        EXPECT_LT(new_m.v_yaw_mae, 0.35) << "New v_yaw MAE too high at d=" << d;
        EXPECT_LT(new_m.v_yaw_rmse, 0.40) << "New v_yaw RMSE too high at d=" << d;
    }
}

// ============================================================================
// Test 3: Fire hit rate vs distance
// ============================================================================

struct HitRateMetrics {
    double total_hit_rate;
    double settled_hit_rate;
    double mean_abs_yaw_err;
    double mean_abs_pitch_err;
    double min_yaw_margin;
    double min_pitch_margin;
};

HitRateMetrics run_fire_at_distance(double target_distance, bool use_old_r) {
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

    fcs::L3::TrackerNew tracker(make_config(use_old_r));
    fcs::L4::Aimer aimer(fcs::L4::AimerConfig{});
    MockTrajectorySolver solver;
    const auto gimbal = make_gimbal();
    const auto muzzle = make_muzzle();

    auto truth_state_at = [&](double t) noexcept {
        fcs::L3::RobotTargetState truth;
        truth.position   = Eigen::Vector3d(target_distance, 0.0, 0.2);
        truth.velocity   = Eigen::Vector3d::Zero();
        truth.yaw        = yaw_at(t);
        truth.v_yaw      = omega(t);
        truth.radius0    = 0.23;
        truth.radius1    = 0.23;
        truth.z1         = truth.position.z() + 0.1;
        truth.armors_num = 4;
        return truth;
    };

    const auto truth0 = truth_state_at(0.0);
    fcs::AtLegacyArmorMeasurementBatch first_batch;
    first_batch.measurements.push_back(
        make_measurement(truth0.armor_poses()[0], fcs::ArmorName::Three));
    if (!tracker.first_meet(first_batch).has_value()) {
        return {};
    }

    size_t total_shots       = 0;
    size_t total_hits        = 0;
    size_t settled_shots     = 0;
    size_t settled_hits      = 0;
    double yaw_abs_err_sum   = 0.0;
    double pitch_abs_err_sum = 0.0;
    double min_yaw_margin    = std::numeric_limits<double>::infinity();
    double min_pitch_margin  = std::numeric_limits<double>::infinity();

    const int steps = static_cast<int>(kDuration / kDt);
    for (int step = 1; step <= steps; ++step) {
        const double time = static_cast<double>(step) * kDt;
        tracker.predict(kDt);

        const auto truth_now = truth_state_at(time);
        fcs::AtLegacyArmorMeasurementBatch batch;
        for (const auto& pose : truth_now.armor_poses()) {
            batch.measurements.push_back(make_measurement(pose, fcs::ArmorName::Three));
        }
        if (!tracker.update(batch))
            continue;

        const auto output = tracker.get_output();
        if (output.robot_state() == nullptr)
            continue;

        fcs::L4::ArmorAimContext context;
        context.target_jumped      = output.target_jumped;
        context.phase              = fcs::L4::ArmorAimPhase::SingleArmor;
        context.preferred_armor_id = output.last_armor_id;

        const uint64_t measurement_ns = static_cast<uint64_t>(std::llround(time * 1.0e9));
        const auto prediction         = aimer.aim(
            *output.robot_state(), context, gimbal, muzzle, measurement_ns, measurement_ns, 0.0,
            kBulletSpeed, solver);
        if (!prediction.has_value())
            continue;

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
            min_yaw_margin   = std::min(min_yaw_margin, yaw_tol - std::abs(yaw_err));
            min_pitch_margin = std::min(min_pitch_margin, pitch_tol - std::abs(pitch_err));
        }
    }

    if (total_shots == 0)
        return {};

    return {
        .total_hit_rate   = static_cast<double>(total_hits) / static_cast<double>(total_shots),
        .settled_hit_rate = static_cast<double>(settled_hits) / static_cast<double>(settled_shots),
        .mean_abs_yaw_err = yaw_abs_err_sum / static_cast<double>(settled_shots),
        .mean_abs_pitch_err = pitch_abs_err_sum / static_cast<double>(settled_shots),
        .min_yaw_margin     = min_yaw_margin,
        .min_pitch_margin   = min_pitch_margin,
    };
}

TEST(RNoiseBenchmark, FireHitRateVsDistance) {
    constexpr std::array<double, 3> kDistances = {2.0, 5.0, 8.0};

    std::cout << "\n";
    std::cout << "========================================================================\n";
    std::cout << "=== Fire Hit Rate: Old (constant R) vs New (distance-dependent R) ===\n";
    std::cout << "========================================================================\n";
    std::cout << "  Bullet speed=30 m/s, armor=0.135m x 0.135m, sinusoidal omega\n";
    std::cout << "\n";

    // Header
    std::printf(
        "  d(m) Ver | total%% settl%%  yaw_err pitch_err yaw_mrg pit_mrg | total%% settl%%  "
        "yaw_err pitch_err yaw_mrg pit_mrg\n");
    std::printf(
        "  -----+-----+--------------------------------- OLD "
        "---------------------------------+--------------------------------- NEW "
        "---------------------------------\n");

    for (const double d : kDistances) {
        auto old_m = run_fire_at_distance(d, true);
        auto new_m = run_fire_at_distance(d, false);

        ASSERT_GT(old_m.settled_hit_rate, 0.0) << "No hits at d=" << d << " (old)";
        ASSERT_GT(new_m.settled_hit_rate, 0.0) << "No hits at d=" << d << " (new)";

        std::printf(
            "  %4.1f OLD | %5.1f%% %6.1f%% %8.5f %9.5f %7.5f %7.5f |\n", d,
            old_m.total_hit_rate * 100.0, old_m.settled_hit_rate * 100.0, old_m.mean_abs_yaw_err,
            old_m.mean_abs_pitch_err, old_m.min_yaw_margin, old_m.min_pitch_margin);
        std::printf(
            "  %4.1f NEW | %5.1f%% %6.1f%% %8.5f %9.5f %7.5f %7.5f |\n", d,
            new_m.total_hit_rate * 100.0, new_m.settled_hit_rate * 100.0, new_m.mean_abs_yaw_err,
            new_m.mean_abs_pitch_err, new_m.min_yaw_margin, new_m.min_pitch_margin);

        // Hit rate delta
        const double delta = (new_m.settled_hit_rate - old_m.settled_hit_rate) * 100.0;
        const char* arrow  = delta > 0.1 ? "UP" : (delta < -0.1 ? "DN" : "==");
        std::printf("         | settled hit rate delta: %+.1f%% %s\n\n", delta, arrow);

        // Regression guard
        EXPECT_GT(new_m.settled_hit_rate, 0.95) << "New settled hit rate too low at d=" << d;
    }
}

} // namespace
