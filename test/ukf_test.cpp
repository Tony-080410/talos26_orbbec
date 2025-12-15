#include <gtest/gtest.h>
#include <cmath>
#include "ukf_logic.hpp"

using namespace spike;

class UKFTest : public ::testing::Test {
 protected:
    TrackerSoA tracker{};
    UKFState ukf{};

    void SetUp() override {
        // Reset tracker state manually (atomic member prevents copy assignment)
        tracker.state.store(TrackerState::IDLE);
        tracker.detecting_count = 0;
        tracker.lost_count = 0;
        tracker.generation = 0;
        for (int i = 0; i < 3; ++i) {
            tracker.pos[i] = 0.0f;
            tracker.vel[i] = 0.0f;
        }
        tracker.yaw = 0.0f;
        tracker.v_yaw = 0.0f;
        tracker.radius[0] = 1.0f;
        tracker.radius[1] = 1.0f;

        // Reset UKF state
        ukf = UKFState{};
    }
};

// ============== Initialization Tests ==============

TEST_F(UKFTest, InitFromDetection) {
    Vec3 pos(1.0f, 2.0f, 3.0f);
    float yaw = 0.5f;

    init_from_detection(tracker, ukf, pos, yaw);

    EXPECT_FLOAT_EQ(tracker.pos[0], 1.0f);
    EXPECT_FLOAT_EQ(tracker.pos[1], 2.0f);
    EXPECT_FLOAT_EQ(tracker.pos[2], 3.0f);
    EXPECT_FLOAT_EQ(tracker.yaw, 0.5f);

    // Velocity should be zero
    EXPECT_FLOAT_EQ(tracker.vel[0], 0.0f);
    EXPECT_FLOAT_EQ(tracker.vel[1], 0.0f);
    EXPECT_FLOAT_EQ(tracker.vel[2], 0.0f);
    EXPECT_FLOAT_EQ(tracker.v_yaw, 0.0f);
}

TEST_F(UKFTest, InitialCovarianceIsPositiveDefinite) {
    Vec3 pos(0.0f, 0.0f, 0.0f);
    init_from_detection(tracker, ukf, pos, 0.0f);

    // Check diagonal is positive
    for (int i = 0; i < 11; ++i) {
        EXPECT_GT(ukf.P(i, i), 0.0f) << "Diagonal element " << i << " not positive";
    }

    // Check symmetry
    for (int i = 0; i < 11; ++i) {
        for (int j = i + 1; j < 11; ++j) {
            EXPECT_FLOAT_EQ(ukf.P(i, j), ukf.P(j, i))
                << "P not symmetric at (" << i << "," << j << ")";
        }
    }
}

// ============== Prediction Tests ==============

TEST_F(UKFTest, PredictWithZeroVelocity) {
    Vec3 pos(1.0f, 2.0f, 3.0f);
    init_from_detection(tracker, ukf, pos, 0.0f);

    // Predict with 10ms dt
    predict(tracker, ukf, 10.0f);

    // Position should not change (velocity is zero)
    EXPECT_NEAR(tracker.pos[0], 1.0f, 1e-5f);
    EXPECT_NEAR(tracker.pos[1], 2.0f, 1e-5f);
    EXPECT_NEAR(tracker.pos[2], 3.0f, 1e-5f);
}

TEST_F(UKFTest, PredictWithConstantVelocity) {
    Vec3 pos(0.0f, 0.0f, 0.0f);
    init_from_detection(tracker, ukf, pos, 0.0f);

    // Set velocity: 1 m/s in X direction
    tracker.vel[0] = 1.0f;
    tracker.vel[1] = 0.0f;
    tracker.vel[2] = 0.0f;

    // Predict with 100ms dt
    predict(tracker, ukf, 100.0f);

    // Position X should advance by 0.1m
    EXPECT_NEAR(tracker.pos[0], 0.1f, 1e-5f);
    EXPECT_NEAR(tracker.pos[1], 0.0f, 1e-5f);
    EXPECT_NEAR(tracker.pos[2], 0.0f, 1e-5f);
}

TEST_F(UKFTest, PredictYawWithAngularVelocity) {
    Vec3 pos(0.0f, 0.0f, 0.0f);
    init_from_detection(tracker, ukf, pos, 0.0f);

    // Set angular velocity: 1 rad/s
    tracker.v_yaw = 1.0f;

    // Predict with 100ms dt
    predict(tracker, ukf, 100.0f);

    // Yaw should advance by 0.1 rad
    EXPECT_NEAR(tracker.yaw, 0.1f, 1e-5f);
}

TEST_F(UKFTest, PredictYawWrapAround) {
    Vec3 pos(0.0f, 0.0f, 0.0f);
    init_from_detection(tracker, ukf, pos, static_cast<float>(M_PI - 0.05f));

    tracker.v_yaw = 1.0f;  // 1 rad/s

    // Predict with 100ms - should wrap from ~pi to ~-pi
    predict(tracker, ukf, 100.0f);

    // Should wrap to negative
    EXPECT_LT(tracker.yaw, 0.0f);
    EXPECT_GT(tracker.yaw, static_cast<float>(-M_PI));
}

TEST_F(UKFTest, PredictIncreasesUncertainty) {
    Vec3 pos(0.0f, 0.0f, 0.0f);
    init_from_detection(tracker, ukf, pos, 0.0f);

    float P_before = ukf.P(0, 0);

    predict(tracker, ukf, 10.0f);

    // Position uncertainty should increase after prediction
    EXPECT_GT(ukf.P(0, 0), P_before);
}

// ============== Update Tests ==============

TEST_F(UKFTest, UpdateReducesUncertainty) {
    Vec3 pos(0.0f, 0.0f, 0.0f);
    init_from_detection(tracker, ukf, pos, 0.0f);

    // Increase uncertainty
    predict(tracker, ukf, 100.0f);

    float P_before = ukf.P(0, 0);

    // Update with measurement at same position
    Vec3 meas(0.0f, 0.0f, 0.0f);
    update(tracker, ukf, meas, 0.0f);

    // Uncertainty should decrease
    EXPECT_LT(ukf.P(0, 0), P_before);
}

TEST_F(UKFTest, UpdateCorrectsMeanState) {
    Vec3 pos(0.0f, 0.0f, 0.0f);
    init_from_detection(tracker, ukf, pos, 0.0f);

    // Measurement shows position at (1, 0, 0)
    Vec3 meas(1.0f, 0.0f, 0.0f);
    update(tracker, ukf, meas, 0.0f);

    // State should move toward measurement
    EXPECT_GT(tracker.pos[0], 0.0f);
    EXPECT_LT(tracker.pos[0], 1.0f);  // Partial correction
}

TEST_F(UKFTest, UpdateMaintainsCovarianceSymmetry) {
    Vec3 pos(0.0f, 0.0f, 0.0f);
    init_from_detection(tracker, ukf, pos, 0.0f);

    for (int iter = 0; iter < 10; ++iter) {
        predict(tracker, ukf, 10.0f);
        Vec3 meas(0.1f * iter, 0.0f, 0.0f);
        update(tracker, ukf, meas, 0.0f);
    }

    // Check symmetry after multiple iterations (relaxed tolerance for float32 accumulation)
    for (int i = 0; i < 11; ++i) {
        for (int j = i + 1; j < 11; ++j) {
            EXPECT_NEAR(ukf.P(i, j), ukf.P(j, i), 1e-3f)
                << "P not symmetric at (" << i << "," << j << ")";
        }
    }
}

// ============== Data Association Tests ==============

TEST_F(UKFTest, MahalanobisDistanceZeroForExactMatch) {
    Vec3 pos(1.0f, 2.0f, 3.0f);
    init_from_detection(tracker, ukf, pos, 0.0f);

    // Measurement exactly at tracker position
    float d2 = mahalanobis_distance_sq(tracker, ukf, pos);

    EXPECT_NEAR(d2, 0.0f, 1e-5f);
}

TEST_F(UKFTest, MahalanobisDistanceIncreasesWithOffset) {
    Vec3 pos(0.0f, 0.0f, 0.0f);
    init_from_detection(tracker, ukf, pos, 0.0f);

    Vec3 meas1(0.1f, 0.0f, 0.0f);
    Vec3 meas2(0.5f, 0.0f, 0.0f);

    float d1 = mahalanobis_distance_sq(tracker, ukf, meas1);
    float d2 = mahalanobis_distance_sq(tracker, ukf, meas2);

    EXPECT_GT(d2, d1);
}

TEST_F(UKFTest, IsAssociatedReturnsTrueForNearby) {
    Vec3 pos(0.0f, 0.0f, 0.0f);
    init_from_detection(tracker, ukf, pos, 0.0f);

    Vec3 nearby(0.1f, 0.1f, 0.1f);
    EXPECT_TRUE(is_associated(tracker, ukf, nearby));
}

TEST_F(UKFTest, IsAssociatedReturnsFalseForFar) {
    Vec3 pos(0.0f, 0.0f, 0.0f);
    init_from_detection(tracker, ukf, pos, 0.0f);

    Vec3 faraway(10.0f, 10.0f, 10.0f);
    EXPECT_FALSE(is_associated(tracker, ukf, faraway));
}

// ============== Integration Tests ==============

TEST_F(UKFTest, PredictUpdateCycle) {
    Vec3 pos(0.0f, 0.0f, 5.0f);  // 5m away
    init_from_detection(tracker, ukf, pos, 0.0f);

    // Simulate target moving at 1 m/s in X
    tracker.vel[0] = 1.0f;

    for (int i = 0; i < 100; ++i) {
        predict(tracker, ukf, 2.0f);  // 2ms interval (500Hz)

        // Simulate noisy measurement
        float noise = (i % 2 == 0) ? 0.01f : -0.01f;
        Vec3 meas(tracker.pos[0] + noise, tracker.pos[1], tracker.pos[2]);
        update(tracker, ukf, meas, 0.0f);
    }

    // After 200ms, position should be ~0.2m
    EXPECT_NEAR(tracker.pos[0], 0.2f, 0.05f);
}

TEST_F(UKFTest, ConvergenceToTrueState) {
    // Start with wrong initial guess
    Vec3 pos(0.0f, 0.0f, 0.0f);
    init_from_detection(tracker, ukf, pos, 0.0f);

    // True state is at (1, 1, 1)
    Vec3 true_pos(1.0f, 1.0f, 1.0f);

    // Multiple updates should converge
    for (int i = 0; i < 50; ++i) {
        update(tracker, ukf, true_pos, 0.0f);
    }

    EXPECT_NEAR(tracker.pos[0], 1.0f, 0.05f);
    EXPECT_NEAR(tracker.pos[1], 1.0f, 0.05f);
    EXPECT_NEAR(tracker.pos[2], 1.0f, 0.05f);
}
