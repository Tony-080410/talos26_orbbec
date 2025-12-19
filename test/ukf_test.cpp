#include <gtest/gtest.h>
#include "ukf_logic.hpp"

using namespace talos;

class UKFTest : public testing::Test {
protected:
    TrackerSoA tracker{};
    UKFState ukf{};

    void SetUp() override {
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
        ukf = UKFState{};
    }
};

// ============== Initialization Tests ==============

TEST_F(UKFTest, InitFromDetection) {
    const Vec3 pos(1.0f, 2.0f, 3.0f);
    init_from_detection(tracker, ukf, pos, 0.5f);

    EXPECT_FLOAT_EQ(tracker.pos[0], 1.0f);
    EXPECT_FLOAT_EQ(tracker.pos[1], 2.0f);
    EXPECT_FLOAT_EQ(tracker.pos[2], 3.0f);
    EXPECT_FLOAT_EQ(tracker.yaw, 0.5f);
    EXPECT_FLOAT_EQ(tracker.vel[0], 0.0f);
    EXPECT_FLOAT_EQ(tracker.vel[1], 0.0f);
    EXPECT_FLOAT_EQ(tracker.vel[2], 0.0f);
    EXPECT_FLOAT_EQ(tracker.v_yaw, 0.0f);
}

TEST_F(UKFTest, InitialCovarianceIsPositiveDefinite) {
    const Vec3 pos(0.0f, 0.0f, 0.0f);
    init_from_detection(tracker, ukf, pos, 0.0f);

    Cov11 P = ukf.P();
    for (int i = 0; i < 11; ++i) {
        EXPECT_GT(P(i, i), 0.0f) << "Diagonal element " << i << " not positive";
    }

    for (int i = 0; i < 11; ++i) {
        for (int j = i + 1; j < 11; ++j) {
            EXPECT_NEAR(P(i, j), P(j, i), 1e-6f) << "P not symmetric at (" << i << "," << j << ")";
        }
    }
}

// ============== Prediction Tests ==============

TEST_F(UKFTest, PredictWithZeroVelocity) {
    const Vec3 pos(1.0f, 2.0f, 3.0f);
    init_from_detection(tracker, ukf, pos, 0.0f);
    predict(tracker, ukf, 10.0f);

    EXPECT_NEAR(tracker.pos[0], 1.0f, 1e-5f);
    EXPECT_NEAR(tracker.pos[1], 2.0f, 1e-5f);
    EXPECT_NEAR(tracker.pos[2], 3.0f, 1e-5f);
}

TEST_F(UKFTest, PredictWithConstantVelocity) {
    const Vec3 pos(0.0f, 0.0f, 0.0f);
    init_from_detection(tracker, ukf, pos, 0.0f);

    tracker.vel[0] = 1.0f;
    predict(tracker, ukf, 100.0f);

    EXPECT_NEAR(tracker.pos[0], 0.1f, 1e-5f);
    EXPECT_NEAR(tracker.pos[1], 0.0f, 1e-5f);
    EXPECT_NEAR(tracker.pos[2], 0.0f, 1e-5f);
}

TEST_F(UKFTest, PredictYawWithAngularVelocity) {
    const Vec3 pos(0.0f, 0.0f, 0.0f);
    init_from_detection(tracker, ukf, pos, 0.0f);

    tracker.v_yaw = 1.0f;
    predict(tracker, ukf, 100.0f);

    EXPECT_NEAR(tracker.yaw, 0.1f, 1e-5f);
}

TEST_F(UKFTest, PredictYawWrapAround) {
    const Vec3 pos(0.0f, 0.0f, 0.0f);
    init_from_detection(tracker, ukf, pos, static_cast<float>(M_PI) - 0.05f);

    tracker.v_yaw = 1.0f;
    predict(tracker, ukf, 100.0f);

    EXPECT_LT(tracker.yaw, 0.0f);
    EXPECT_GT(tracker.yaw, static_cast<float>(-M_PI));
}

TEST_F(UKFTest, PredictIncreasesUncertainty) {
    const Vec3 pos(0.0f, 0.0f, 0.0f);
    init_from_detection(tracker, ukf, pos, 0.0f);

    const float P_before = ukf.P()(0, 0);
    predict(tracker, ukf, 10.0f);

    EXPECT_GT(ukf.P()(0, 0), P_before);
}

// ============== Update Tests ==============

TEST_F(UKFTest, UpdateReducesUncertainty) {
    const Vec3 pos(0.0f, 0.0f, 0.0f);
    init_from_detection(tracker, ukf, pos, 0.0f);

    predict(tracker, ukf, 100.0f);
    const float P_before = ukf.P()(0, 0);

    const Vec3 meas(0.0f, 0.0f, 0.0f);
    update(tracker, ukf, meas, 0.0f);

    EXPECT_LT(ukf.P()(0, 0), P_before);
}

TEST_F(UKFTest, UpdateCorrectsMeanState) {
    const Vec3 pos(0.0f, 0.0f, 0.0f);
    init_from_detection(tracker, ukf, pos, 0.0f);

    const Vec3 meas(1.0f, 0.0f, 0.0f);
    update(tracker, ukf, meas, 0.0f);

    EXPECT_GT(tracker.pos[0], 0.0f);
    EXPECT_LT(tracker.pos[0], 1.0f);
}

TEST_F(UKFTest, UpdateMaintainsCovarianceSymmetry) {
    const Vec3 pos(0.0f, 0.0f, 0.0f);
    init_from_detection(tracker, ukf, pos, 0.0f);

    for (int iter = 0; iter < 10; ++iter) {
        predict(tracker, ukf, 10.0f);
        Vec3 meas(0.1f * static_cast<float>(iter), 0.0f, 0.0f);
        update(tracker, ukf, meas, 0.0f);
    }

    Cov11 P = ukf.P();
    for (int i = 0; i < 11; ++i) {
        for (int j = i + 1; j < 11; ++j) {
            EXPECT_NEAR(P(i, j), P(j, i), 1e-3f) << "P not symmetric at (" << i << "," << j << ")";
        }
    }
}

// ============== Data Association Tests ==============

TEST_F(UKFTest, MahalanobisDistanceZeroForExactMatch) {
    const Vec3 pos(1.0f, 2.0f, 3.0f);
    init_from_detection(tracker, ukf, pos, 0.0f);

    const float d2 = mahalanobis_distance_sq(tracker, ukf, pos);
    EXPECT_NEAR(d2, 0.0f, 1e-5f);
}

TEST_F(UKFTest, MahalanobisDistanceIncreasesWithOffset) {
    const Vec3 pos(0.0f, 0.0f, 0.0f);
    init_from_detection(tracker, ukf, pos, 0.0f);

    const Vec3 meas1(0.1f, 0.0f, 0.0f);
    const Vec3 meas2(0.5f, 0.0f, 0.0f);

    const float d1 = mahalanobis_distance_sq(tracker, ukf, meas1);
    const float d2 = mahalanobis_distance_sq(tracker, ukf, meas2);

    EXPECT_GT(d2, d1);
}

TEST_F(UKFTest, IsAssociatedReturnsTrueForNearby) {
    const Vec3 pos(0.0f, 0.0f, 0.0f);
    init_from_detection(tracker, ukf, pos, 0.0f);

    const Vec3 nearby(0.1f, 0.1f, 0.1f);
    EXPECT_TRUE(is_associated(tracker, ukf, nearby));
}

TEST_F(UKFTest, IsAssociatedReturnsFalseForFar) {
    const Vec3 pos(0.0f, 0.0f, 0.0f);
    init_from_detection(tracker, ukf, pos, 0.0f);

    const Vec3 faraway(10.0f, 10.0f, 10.0f);
    EXPECT_FALSE(is_associated(tracker, ukf, faraway));
}

// ============== Integration Tests ==============

TEST_F(UKFTest, PredictUpdateCycle) {
    const Vec3 pos(0.0f, 0.0f, 5.0f);
    init_from_detection(tracker, ukf, pos, 0.0f);
    tracker.vel[0] = 1.0f;

    for (int i = 0; i < 100; ++i) {
        predict(tracker, ukf, 2.0f);
        const float noise = i % 2 == 0 ? 0.01f : -0.01f;
        Vec3 meas(tracker.pos[0] + noise, tracker.pos[1], tracker.pos[2]);
        update(tracker, ukf, meas, 0.0f);
    }

    EXPECT_NEAR(tracker.pos[0], 0.2f, 0.05f);
}

TEST_F(UKFTest, ConvergenceToTrueState) {
    const Vec3 pos(0.0f, 0.0f, 0.0f);
    init_from_detection(tracker, ukf, pos, 0.0f);

    const Vec3 true_pos(1.0f, 1.0f, 1.0f);
    for (int i = 0; i < 50; ++i) {
        update(tracker, ukf, true_pos, 0.0f);
    }

    EXPECT_NEAR(tracker.pos[0], 1.0f, 0.05f);
    EXPECT_NEAR(tracker.pos[1], 1.0f, 0.05f);
    EXPECT_NEAR(tracker.pos[2], 1.0f, 0.05f);
}
