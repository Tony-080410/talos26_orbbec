#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include "L3_estimation/tracker/motion_model.hpp"
#include "L3_estimation/tracker/tracker.hpp"
#include "L3_estimation/tracker/util.hpp"
#include "core/types.hpp"
#include "frame.hpp"
#include "matrix.hpp"

namespace {

using fcs::L3::OutpostCKF;
using fcs::L3::OutpostModel;
using fcs::L3::RoboCKF;
using fcs::L3::RobotModel;
using namespace fcs::L3;

auto synth_measurement(
    const RobotModel& model, const RoboCKF::VecX& x_true, const RoboCKF::VecX& x_est, int step)
    -> RoboCKF::VecZ {
    RoboCKF::VecZ z = model.h(x_true);

    const double t = static_cast<double>(step);
    z[0] += 0.004 * std::sin(0.11 * t);
    z[1] += 0.003 * std::cos(0.07 * t);
    z[2] += 0.010 * std::sin(0.05 * t);
    z[3] += 0.004 * std::cos(0.13 * t);

    const RoboCKF::VecZ z_pred = model.h(x_est);
    z[0]                       = unwrap_rad(z_pred[0], z[0]);
    z[1]                       = unwrap_rad(z_pred[1], z[1]);
    z[3]                       = unwrap_rad(z_pred[3], z[3]);
    return z;
}

struct SpinStopResult {
    double r0_at_stop{0.0};
    double max_r0_dev{0.0};
    double min_r0{std::numeric_limits<double>::infinity()};
    bool finite_state{true};
};

struct LongRunResult {
    bool finite_state{true};
    double max_xy_err_after_settle{0.0};
    double max_r0_dev_after_settle{0.0};
};

auto run_spin_then_stop() -> SpinStopResult {
    RobotModel model;
    model.armor_id = 0;

    RoboCKF::VecX x_true;
    x_true << 2.0, 0.0, 0.8, 0.0, 0.3, 0.0, 0.0, 0.0, 0.23, 0.23, 0.0;

    RoboCKF::VecX x0 = x_true;
    x0[YAW]          = 0.1;
    x0[R_0]          = 0.24;

    RoboCKF::MatXX P0 = RoboCKF::MatXX::Identity() * 0.15;
    RoboCKF kf(model, x0, P0);
    kf.set_max_iters(3);
    kf.set_iter_tol(1e-8);

    constexpr double dt    = 0.01;
    constexpr double omega = 2.0;
    const int spin_steps   = static_cast<int>(std::ceil((4.0 * M_PI / omega) / dt));
    const int stop_steps   = 1000; // 10s

    int step = 0;
    for (int i = 0; i < spin_steps; ++i, ++step) {
        x_true[YAW] += omega * dt;
        x_true[V_YAW] = omega;
        kf.predict(dt);
        const auto z = synth_measurement(model, x_true, kf.x(), step);
        kf.update(z);
    }

    SpinStopResult result;
    result.r0_at_stop = kf.x()[R_0];
    result.min_r0     = result.r0_at_stop;

    for (int i = 0; i < stop_steps; ++i, ++step) {
        x_true[V_YAW] = 0.0;
        kf.predict(dt);
        const auto z = synth_measurement(model, x_true, kf.x(), step);
        kf.update(z);

        const auto& x = kf.x();
        result.finite_state &= x.allFinite();
        result.min_r0     = std::min(result.min_r0, x[R_0]);
        result.max_r0_dev = std::max(result.max_r0_dev, std::abs(x[R_0] - result.r0_at_stop));
    }
    return result;
}

auto run_static_single_armor() -> SpinStopResult {
    RobotModel model;
    model.armor_id = 0;

    RoboCKF::VecX x_true;
    x_true << 1.8, 0.0, 0.6, 0.0, 0.25, 0.0, 1.0, 0.0, 0.23, 0.23, 0.0;

    RoboCKF::VecX x0 = x_true;
    x0[YAW] += 0.2;
    x0[R_0] += 0.01;

    RoboCKF::MatXX P0 = RoboCKF::MatXX::Identity() * 0.10;
    RoboCKF kf(model, x0, P0);
    kf.set_max_iters(3);
    kf.set_iter_tol(1e-8);

    constexpr double dt   = 0.01;
    const int total_steps = 1200;

    SpinStopResult result;
    int step = 0;
    for (int i = 0; i < total_steps; ++i, ++step) {
        kf.predict(dt);
        const auto z = synth_measurement(model, x_true, kf.x(), step);
        kf.update(z);
        const auto& x = kf.x();
        if (i == 200) {
            result.r0_at_stop = x[R_0];
        }
        if (i >= 200) {
            result.max_r0_dev = std::max(result.max_r0_dev, std::abs(x[R_0] - result.r0_at_stop));
        }
        result.min_r0 = std::min(result.min_r0, x[R_0]);
        result.finite_state &= x.allFinite();
    }
    return result;
}

auto run_static_single_armor_long_run() -> LongRunResult {
    RobotModel model;
    model.armor_id = 0;

    RoboCKF::VecX x_true;
    x_true << 1.8, 0.0, 0.6, 0.0, 0.25, 0.0, 1.0, 0.0, 0.23, 0.23, 0.0;

    RoboCKF::VecX x0 = x_true;
    x0[YAW] += 0.2;
    x0[R_0] += 0.01;
    x0[XC] += 0.08;
    x0[YC] -= 0.06;

    RoboCKF::MatXX P0 = RoboCKF::MatXX::Identity() * 0.10;
    RoboCKF kf(model, x0, P0);
    kf.set_max_iters(3);
    kf.set_iter_tol(1e-8);

    constexpr double dt       = 0.01;
    constexpr int total_steps = 13000; // 130s
    constexpr int settle_step = 2000;  // 20s

    LongRunResult result;
    double r0_ref = 0.0;
    for (int i = 0; i < total_steps; ++i) {
        kf.predict(dt);
        const auto z = synth_measurement(model, x_true, kf.x(), i);
        kf.update(z);
        const auto& x = kf.x();
        result.finite_state &= x.allFinite();
        if (i == settle_step) {
            r0_ref = x[R_0];
        }
        if (i >= settle_step) {
            const double ex = x[XC] - x_true[XC];
            const double ey = x[YC] - x_true[YC];
            result.max_xy_err_after_settle =
                std::max(result.max_xy_err_after_settle, std::hypot(ex, ey));
            result.max_r0_dev_after_settle =
                std::max(result.max_r0_dev_after_settle, std::abs(x[R_0] - r0_ref));
        }
    }
    return result;
}

auto synth_outpost_measurement(
    const OutpostModel& model, const OutpostCKF::VecX& x_true, const OutpostCKF::VecX& x_est,
    int id, int step) -> OutpostCKF::VecZ {
    OutpostCKF::VecZ z = model.h(x_true, id);

    const double t = static_cast<double>(step);
    z[0] += 0.003 * std::sin(0.09 * t);
    z[1] += 0.002 * std::cos(0.05 * t);
    z[2] += 0.008 * std::sin(0.07 * t);
    z[3] += 0.003 * std::cos(0.11 * t);

    const OutpostCKF::VecZ z_pred = model.h(x_est, id);
    z[0]                          = unwrap_rad(z_pred[0], z[0]);
    z[3]                          = unwrap_rad(z_pred[3], z[3]);
    return z;
}

struct OutpostResult {
    bool finite_state{true};
    double max_xy_err{0.0};
};

auto run_outpost_single_armor_static() -> OutpostResult {
    OutpostModel model;
    model.armor_id = 0;

    OutpostCKF::VecX x_true;
    x_true << 1.7, -0.5, 0.8, 0.0, 0.22, 0.31, 0.27;

    OutpostCKF::VecX x0 = x_true;
    x0[O_XC] += 0.15;
    x0[O_YC] -= 0.10;
    x0[O_YAW] += 0.12;

    OutpostCKF::MatXX P0 = OutpostCKF::MatXX::Identity() * 0.15;
    OutpostCKF kf(model, x0, P0);
    kf.set_max_iters(3);
    kf.set_iter_tol(1e-8);

    constexpr double dt = 0.01;
    constexpr int steps = 1200;

    OutpostResult res;
    for (int i = 0; i < steps; ++i) {
        const int id   = 0;
        model.armor_id = id;
        kf.predict(dt);
        const auto z = synth_outpost_measurement(model, x_true, kf.x(), id, i);
        kf.update(z);
        const auto& x = kf.x();
        res.finite_state &= x.allFinite();
        const double ex = x[O_XC] - x_true[O_XC];
        const double ey = x[O_YC] - x_true[O_YC];
        res.max_xy_err  = std::max(res.max_xy_err, std::hypot(ex, ey));
    }
    return res;
}

auto run_outpost_multi_armor_static() -> OutpostResult {
    OutpostModel model;
    model.armor_id = 0;

    OutpostCKF::VecX x_true;
    x_true << 1.9, 0.4, -0.5, 0.0, 0.24, 0.33, 0.29;

    OutpostCKF::VecX x0 = x_true;
    x0[O_XC] -= 0.12;
    x0[O_YC] += 0.10;
    x0[O_YAW] -= 0.14;
    x0[O_Z0] += 0.03;
    x0[O_Z1] -= 0.02;
    x0[O_Z2] += 0.01;

    OutpostCKF::MatXX P0 = OutpostCKF::MatXX::Identity() * 0.15;
    OutpostCKF kf(model, x0, P0);
    kf.set_max_iters(3);
    kf.set_iter_tol(1e-8);

    constexpr double dt = 0.01;
    constexpr int steps = 1200;

    OutpostResult res;
    for (int i = 0; i < steps; ++i) {
        const int id   = i % 3;
        model.armor_id = id;
        kf.predict(dt);
        const auto z = synth_outpost_measurement(model, x_true, kf.x(), id, i);
        kf.update(z);
        const auto& x = kf.x();
        res.finite_state &= x.allFinite();
        const double ex = x[O_XC] - x_true[O_XC];
        const double ey = x[O_YC] - x_true[O_YC];
        res.max_xy_err  = std::max(res.max_xy_err, std::hypot(ex, ey));
    }
    return res;
}

struct TrackerLongRunResult {
    bool tracking_ok{true};
    bool finite_state{true};
    double late_xy_range{0.0};
};

struct TrackerSpinResult {
    bool tracking_ok{true};
    bool finite_state{true};
    double max_xy_err_during_startup{0.0};
    double max_xy_err_after_settle{0.0};
    double max_vyaw_err_after_settle{0.0};
};

auto make_robot_batch_from_state(
    const RobotModel& model, const RoboCKF::VecX& x_true, int armor_id, int step)
    -> fcs::AtLegacyArmorMeasurementBatch {
    const auto z_true = model.h(x_true, armor_id);
    RoboCKF::VecZ z   = z_true;

    const double t = static_cast<double>(step);
    z[0] += 0.004 * std::sin(0.11 * t);
    z[1] += 0.003 * std::cos(0.07 * t);
    z[2] += 0.010 * std::sin(0.05 * t);
    z[3] += 0.004 * std::cos(0.13 * t);

    const Eigen::Vector3d xyz = ypd2xyz({z[0], z[1], z[2]});

    fcs::AtLegacyArmorMeasurement meas;
    meas.name                     = fcs::ArmorName::Two;
    meas.color                    = fcs::ArmorColor::Red;
    meas.type                     = fcs::ArmorType::Small;
    meas.distance_to_image_center = 0.0f;
    meas.transform                = fast_tf::FuckTransform<fast_tf::odom_fuxk_frame>{
        fast_tf::TransformMatrix4d::from_rpy(0.0, 0.0, z[3], xyz.x(), xyz.y(), xyz.z())};

    fcs::AtLegacyArmorMeasurementBatch batch;
    batch.measurements.push_back(meas);
    return batch;
}

auto run_tracker_robot_single_armor_long_run() -> TrackerLongRunResult {
    TrackerConfig cfg{};
    cfg.robot.lost_threshold       = 1.0;
    cfg.robot.tracking_threshold   = 1;
    cfg.robot.matcher_gate         = 10.0;
    cfg.robot.model.sigma_a_xy     = 10.0;
    cfg.robot.model.sigma_a_z      = 2.0;
    cfg.robot.model.sigma_a_yaw    = 20.0;
    cfg.robot.model.sigma_r0       = 1.0;
    cfg.robot.model.sigma_r1       = 1.0;
    cfg.robot.model.sigma_h        = 0.8;
    cfg.outpost.lost_threshold     = 2.0;
    cfg.outpost.tracking_threshold = 1;
    cfg.outpost.matcher_gate       = 4.0;

    Tracker tracker(cfg);

    RobotModel truth_model;
    truth_model.armor_id = 0;
    RoboCKF::VecX x_true;
    x_true << 1.8, 0.0, 0.6, 0.0, 0.25, 0.0, 1.0, 0.0, 0.23, 0.23, 0.0;

    auto batch0                       = make_robot_batch_from_state(truth_model, x_true, 0, 0);
    [[maybe_unused]] const auto first = tracker.first_meet(batch0);

    constexpr double dt       = 0.01;
    constexpr int total_steps = 13000; // 130s
    constexpr int late_begin  = 7000;  // 70s

    double x_min = std::numeric_limits<double>::infinity();
    double x_max = -std::numeric_limits<double>::infinity();
    double y_min = std::numeric_limits<double>::infinity();
    double y_max = -std::numeric_limits<double>::infinity();

    TrackerLongRunResult res;
    for (int i = 1; i < total_steps; ++i) {
        tracker.predict(dt);
        auto batch    = make_robot_batch_from_state(truth_model, x_true, 0, i);
        const bool ok = tracker.update(batch);
        res.tracking_ok &= ok;

        const auto out = tracker.get_output();
        if (!out.is_robot()) {
            res.tracking_ok = false;
            continue;
        }
        const auto* st  = out.robot_state();
        const auto& pos = st->position;
        res.finite_state &= std::isfinite(pos.x()) && std::isfinite(pos.y())
                         && std::isfinite(pos.z()) && std::isfinite(st->yaw)
                         && std::isfinite(st->radius0) && std::isfinite(st->radius1);

        if (i >= late_begin) {
            x_min = std::min(x_min, pos.x());
            x_max = std::max(x_max, pos.x());
            y_min = std::min(y_min, pos.y());
            y_max = std::max(y_max, pos.y());
        }
    }
    if (std::isfinite(x_min) && std::isfinite(x_max) && std::isfinite(y_min)
        && std::isfinite(y_max)) {
        res.late_xy_range = std::hypot(x_max - x_min, y_max - y_min);
    }
    return res;
}

auto run_tracker_robot_spin_6p28() -> TrackerSpinResult {
    TrackerConfig cfg{};
    cfg.robot.lost_threshold       = 1.0;
    cfg.robot.tracking_threshold   = 1;
    cfg.robot.matcher_gate         = 10.0;
    cfg.robot.model.sigma_a_xy     = 10.0;
    cfg.robot.model.sigma_a_z      = 2.0;
    cfg.robot.model.sigma_a_yaw    = 20.0;
    cfg.robot.model.sigma_r0       = 1.0;
    cfg.robot.model.sigma_r1       = 1.0;
    cfg.robot.model.sigma_h        = 0.8;
    cfg.outpost.lost_threshold     = 2.0;
    cfg.outpost.tracking_threshold = 1;
    cfg.outpost.matcher_gate       = 4.0;

    Tracker tracker(cfg);

    RobotModel truth_model;
    truth_model.armor_id = 0;
    RoboCKF::VecX x_true;
    x_true << 1.8, 0.0, 0.6, 0.0, 0.25, 0.0, 0.0, 0.0, 0.23, 0.23, 0.0;

    auto batch0                       = make_robot_batch_from_state(truth_model, x_true, 0, 0);
    [[maybe_unused]] const auto first = tracker.first_meet(batch0);

    constexpr double dt         = 0.01;
    constexpr double omega      = 6.28;
    constexpr int total_steps   = 4000;
    constexpr int startup_steps = 300;
    constexpr int settle_begin  = 1500;
    constexpr double angle_step = 2.0 * M_PI / 4.0;

    TrackerSpinResult res;
    for (int i = 1; i < total_steps; ++i) {
        x_true[YAW] += omega * dt;
        x_true[V_YAW] = omega;

        const int visible_id = static_cast<int>(std::llround(x_true[YAW] / angle_step)) % 4;
        const int id         = (visible_id + 4) % 4;

        tracker.predict(dt);
        auto batch    = make_robot_batch_from_state(truth_model, x_true, id, i);
        const bool ok = tracker.update(batch);
        res.tracking_ok &= ok;

        const auto out = tracker.get_output();
        if (!out.is_robot()) {
            res.tracking_ok = false;
            continue;
        }

        const auto* st  = out.robot_state();
        const auto& pos = st->position;
        res.finite_state &= std::isfinite(pos.x()) && std::isfinite(pos.y())
                         && std::isfinite(pos.z()) && std::isfinite(st->yaw)
                         && std::isfinite(st->v_yaw) && std::isfinite(st->radius0)
                         && std::isfinite(st->radius1);

        if (i >= settle_begin) {
            const double ex             = pos.x() - x_true[XC];
            const double ey             = pos.y() - x_true[YC];
            res.max_xy_err_after_settle = std::max(res.max_xy_err_after_settle, std::hypot(ex, ey));
            res.max_vyaw_err_after_settle =
                std::max(res.max_vyaw_err_after_settle, std::abs(st->v_yaw - omega));
        }
        if (i < startup_steps) {
            const double ex = pos.x() - x_true[XC];
            const double ey = pos.y() - x_true[YC];
            res.max_xy_err_during_startup =
                std::max(res.max_xy_err_during_startup, std::hypot(ex, ey));
        }
    }

    return res;
}

auto run_tracker_robot_static_then_spin_6p28() -> TrackerSpinResult {
    TrackerConfig cfg{};
    cfg.robot.lost_threshold       = 1.0;
    cfg.robot.tracking_threshold   = 1;
    cfg.robot.matcher_gate         = 10.0;
    cfg.robot.model.sigma_a_xy     = 10.0;
    cfg.robot.model.sigma_a_z      = 2.0;
    cfg.robot.model.sigma_a_yaw    = 20.0;
    cfg.robot.model.sigma_r0       = 1.0;
    cfg.robot.model.sigma_r1       = 1.0;
    cfg.robot.model.sigma_h        = 0.8;
    cfg.outpost.lost_threshold     = 2.0;
    cfg.outpost.tracking_threshold = 1;
    cfg.outpost.matcher_gate       = 4.0;

    Tracker tracker(cfg);

    RobotModel truth_model;
    truth_model.armor_id = 0;
    RoboCKF::VecX x_true;
    x_true << 1.8, 0.0, 0.6, 0.0, 0.25, 0.0, 0.0, 0.0, 0.23, 0.23, 0.0;

    auto batch0                       = make_robot_batch_from_state(truth_model, x_true, 0, 0);
    [[maybe_unused]] const auto first = tracker.first_meet(batch0);

    constexpr double dt            = 0.01;
    constexpr double omega         = 6.28;
    constexpr int static_steps     = 1200; // 12s
    constexpr int spin_steps       = 1800; // 18s
    constexpr int startup_steps    = 300;  // first 3s after spin starts
    constexpr int spin_settle_step = 300;  // 3s into spin
    constexpr double angle_step    = 2.0 * M_PI / 4.0;

    TrackerSpinResult res;
    int step = 1;

    for (int i = 0; i < static_steps; ++i, ++step) {
        x_true[V_YAW] = 0.0;
        const int id  = 0;
        tracker.predict(dt);
        auto batch    = make_robot_batch_from_state(truth_model, x_true, id, step);
        const bool ok = tracker.update(batch);
        res.tracking_ok &= ok;
    }

    for (int i = 0; i < spin_steps; ++i, ++step) {
        x_true[YAW] += omega * dt;
        x_true[V_YAW] = omega;

        const int visible_id = static_cast<int>(std::llround(x_true[YAW] / angle_step)) % 4;
        const int id         = (visible_id + 4) % 4;

        tracker.predict(dt);
        auto batch    = make_robot_batch_from_state(truth_model, x_true, id, step);
        const bool ok = tracker.update(batch);
        res.tracking_ok &= ok;

        const auto out = tracker.get_output();
        if (!out.is_robot()) {
            res.tracking_ok = false;
            continue;
        }

        const auto* st  = out.robot_state();
        const auto& pos = st->position;
        res.finite_state &= std::isfinite(pos.x()) && std::isfinite(pos.y())
                         && std::isfinite(pos.z()) && std::isfinite(st->yaw)
                         && std::isfinite(st->v_yaw) && std::isfinite(st->radius0)
                         && std::isfinite(st->radius1);

        if (i >= spin_settle_step) {
            const double ex             = pos.x() - x_true[XC];
            const double ey             = pos.y() - x_true[YC];
            res.max_xy_err_after_settle = std::max(res.max_xy_err_after_settle, std::hypot(ex, ey));
            res.max_vyaw_err_after_settle =
                std::max(res.max_vyaw_err_after_settle, std::abs(st->v_yaw - omega));
        }
        if (i < startup_steps) {
            const double ex = pos.x() - x_true[XC];
            const double ey = pos.y() - x_true[YC];
            res.max_xy_err_during_startup =
                std::max(res.max_xy_err_during_startup, std::hypot(ex, ey));
        }
    }

    return res;
}

} // namespace

TEST(TrackerISRCKFObservability, SpinThenStopRadiusRemainsBounded) {
    const auto res = run_spin_then_stop();
    EXPECT_TRUE(res.finite_state);
    EXPECT_GT(res.min_r0, 0.05);
    EXPECT_LT(res.max_r0_dev, 0.02);
}

TEST(TrackerISRCKFObservability, StaticSingleArmorRadiusDoesNotDrift) {
    const auto res = run_static_single_armor();
    EXPECT_TRUE(res.finite_state);
    EXPECT_GT(res.min_r0, 0.05);
    EXPECT_LT(res.max_r0_dev, 0.015);
}

TEST(TrackerISRCKFObservability, StaticSingleArmorLongRunNoReJitter) {
    const auto res = run_static_single_armor_long_run();
    EXPECT_TRUE(res.finite_state);
    EXPECT_LT(res.max_xy_err_after_settle, 0.06);
    EXPECT_LT(res.max_r0_dev_after_settle, 0.02);
}

TEST(TrackerISRCKFObservability, OutpostSingleArmorNoInstantDivergence) {
    const auto res = run_outpost_single_armor_static();
    EXPECT_TRUE(res.finite_state);
    EXPECT_LT(res.max_xy_err, 0.35);
}

TEST(TrackerISRCKFObservability, OutpostMultiArmorNoInstantDivergence) {
    const auto res = run_outpost_multi_armor_static();
    EXPECT_TRUE(res.finite_state);
    EXPECT_LT(res.max_xy_err, 0.25);
}

TEST(TrackerISRCKFObservability, TrackerRobotSingleArmorLongRunNoLateJitterExplosion) {
    const auto res = run_tracker_robot_single_armor_long_run();
    EXPECT_TRUE(res.tracking_ok);
    EXPECT_TRUE(res.finite_state);
    EXPECT_LT(res.late_xy_range, 0.08);
}

TEST(TrackerISRCKFObservability, TrackerRobotSpin6p28Converges) {
    const auto res = run_tracker_robot_spin_6p28();
    EXPECT_TRUE(res.tracking_ok);
    EXPECT_TRUE(res.finite_state);
    EXPECT_LT(res.max_xy_err_during_startup, 0.45);
    EXPECT_LT(res.max_xy_err_after_settle, 0.25);
    EXPECT_LT(res.max_vyaw_err_after_settle, 1.40);
}

TEST(TrackerISRCKFObservability, TrackerRobotStaticThenSpin6p28Converges) {
    const auto res = run_tracker_robot_static_then_spin_6p28();
    EXPECT_TRUE(res.tracking_ok);
    EXPECT_TRUE(res.finite_state);
    EXPECT_LT(res.max_xy_err_during_startup, 0.55);
    EXPECT_LT(res.max_xy_err_after_settle, 0.28);
    EXPECT_LT(res.max_vyaw_err_after_settle, 1.50);
}
