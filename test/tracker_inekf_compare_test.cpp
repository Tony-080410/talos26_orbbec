/// @file tracker_inekf_compare_test.cpp
/// @brief Compare redact-based InEKF tracker vs existing ISRCKF on the same synthetic stream
///
/// This is a regression test: the redact-based implementation must not diverge wildly from
/// the existing ISRCKF when fed the exact same measurements.

#include <gtest/gtest.h>

#include "fcs/L3_estimation/tracker/config.hpp"
#include "fcs/L3_estimation/tracker/tracker.hpp"
#include "fcs/L3_estimation/tracker/util.hpp"
#include "fcs/core/types.hpp"

#include "math/fast_tf/matrix.hpp"

#include <cmath>

namespace {

[[nodiscard]] fcs::AtLegacyArmorMeasurement
    make_measurement(const Eigen::Vector3d& armor_pos_odom, double armor_yaw, uint64_t ts_ns) {
    fcs::AtLegacyArmorMeasurement m;
    m.name                     = fcs::ArmorName::One;
    m.distance_to_image_center = 0.0f;
    m.timestamp_ns             = ts_ns;
    m.transform =
        fast_tf::FuckTransform<fast_tf::odom_fuxk_frame>(fast_tf::TransformMatrix4d::from_rpy(
            0.0, 0.0, armor_yaw, armor_pos_odom.x(), armor_pos_odom.y(), armor_pos_odom.z()));
    return m;
}

[[nodiscard]] fcs::AtLegacyArmorMeasurementBatch make_batch(
    const Eigen::Vector3d& armor_pos_odom, double armor_yaw, uint64_t ts_ns, uint64_t frame_id) {
    fcs::AtLegacyArmorMeasurementBatch batch;
    batch.timestamp_ns = ts_ns;
    batch.frame_id     = frame_id;
    batch.measurements.push_back(make_measurement(armor_pos_odom, armor_yaw, ts_ns));
    return batch;
}

[[nodiscard]] fcs::L3::TrackerConfig make_test_config(bool inekf_enabled) {
    fcs::L3::TrackerConfig cfg;
    cfg.inekf_enabled = inekf_enabled;

    // Keep association permissive and stable for this deterministic stream.
    cfg.jpda_matcher.enable_jpda_matcher = false;

    cfg.robot.lost_threshold     = 1.0;
    cfg.robot.tracking_threshold = 1;
    cfg.robot.matcher_gate       = 1e9;

    cfg.outpost.lost_threshold     = 1.0;
    cfg.outpost.tracking_threshold = 1;
    cfg.outpost.matcher_gate       = 1e9;

    // Robot InEKF defaults are fine; keep spherical observation on for the new backend.
    cfg.robot_inekf.use_spherical_observation = true;
    return cfg;
}

} // namespace

TEST(TrackerInEKF, RedactDoesNotDivergeFromISRCKFOnSimpleStream) {
    auto cfg_inekf = make_test_config(true);
    auto cfg_isr   = make_test_config(false);

    fcs::L3::Tracker tracker_inekf(cfg_inekf);
    fcs::L3::Tracker tracker_isr(cfg_isr);

    // Ground truth: constant velocity + constant yaw-rate.
    const Eigen::Vector3d p0{5.0, 1.0, 0.2};
    const Eigen::Vector3d v{0.8, -0.2, 0.0};
    const double yaw0  = 0.3;
    const double omega = 0.8; // rad/s
    const double r0    = 0.23;

    const double dt = 1.0 / 60.0;

    auto armor_pose_at = [&](double t) {
        const Eigen::Vector3d pc = p0 + v * t;
        const double yaw         = fcs::L3::normalize_rad(yaw0 + omega * t);
        const Eigen::Vector3d armor_pos{
            pc.x() - r0 * std::cos(yaw), pc.y() - r0 * std::sin(yaw), pc.z()};
        return std::pair<Eigen::Vector3d, double>{armor_pos, yaw};
    };

    // Initialize from the first measurement.
    const auto [armor0_pos, armor0_yaw] = armor_pose_at(0.0);
    const uint64_t t0_ns                = 1'000'000'000ULL;
    const auto init_batch               = make_batch(armor0_pos, armor0_yaw, t0_ns, 0);

    ASSERT_TRUE(tracker_inekf.first_meet(init_batch).has_value());
    ASSERT_TRUE(tracker_isr.first_meet(init_batch).has_value());

    // Replay.
    for (int k = 1; k <= 300; ++k) {
        const double t                    = static_cast<double>(k) * dt;
        const auto [armor_pos, armor_yaw] = armor_pose_at(t);
        const auto batch                  = make_batch(
            armor_pos, armor_yaw, t0_ns + static_cast<uint64_t>(t * 1e9), static_cast<uint64_t>(k));

        tracker_inekf.step(dt, batch);
        tracker_isr.step(dt, batch);

        const auto out_inekf = tracker_inekf.get_output();
        const auto out_isr   = tracker_isr.get_output();

        const auto* s_inekf = out_inekf.robot_state();
        const auto* s_isr   = out_isr.robot_state();
        ASSERT_NE(s_inekf, nullptr);
        ASSERT_NE(s_isr, nullptr);

        ASSERT_TRUE(s_inekf->position.allFinite());
        ASSERT_TRUE(s_isr->position.allFinite());
        ASSERT_TRUE(std::isfinite(s_inekf->yaw));
        ASSERT_TRUE(std::isfinite(s_isr->yaw));

        const double pos_err = (s_inekf->position - s_isr->position).norm();
        const double yaw_err = std::abs(fcs::L3::shortest_rad(s_isr->yaw, s_inekf->yaw));

        // This is a loose "sanity" bound: if the new backend is fundamentally wrong (signs,
        // wrapping, group ops), it will blow past this quickly.
        ASSERT_LT(pos_err, 1.0) << "k=" << k << " pos_err=" << pos_err;
        ASSERT_LT(yaw_err, 1.0) << "k=" << k << " yaw_err=" << yaw_err;
    }
}
