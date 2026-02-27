#include <gtest/gtest.h>

#include "math/fast_tf/buffer.hpp"
#include "math/fast_tf/matrix.hpp"

using namespace fast_tf;

TEST(CircularTransformBufferTest, BasicPushAndLatest) {
    CircularTransformBuffer<10> buffer;

    EXPECT_TRUE(buffer.is_empty());
    EXPECT_EQ(buffer.size(), 0);

    const auto tf = TransformMatrix4d::from_translation(1.0, 2.0, 3.0);
    buffer.push(1000, tf);

    EXPECT_FALSE(buffer.is_empty());
    EXPECT_EQ(buffer.size(), 1);

    const auto result = buffer.latest();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->timestamp_ns, 1000);
}

TEST(CircularTransformBufferTest, TimeRange) {
    CircularTransformBuffer<10> buffer;

    EXPECT_FALSE(buffer.time_range().has_value());

    buffer.push(1000, TransformMatrix4d{});
    buffer.push(2000, TransformMatrix4d{});
    buffer.push(3000, TransformMatrix4d{});

    const auto range = buffer.time_range();
    ASSERT_TRUE(range.has_value());
    EXPECT_EQ(range->first, 1000);
    EXPECT_EQ(range->second, 3000);
}

TEST(CircularTransformBufferTest, InterpolationClamped) {
    CircularTransformBuffer<10> buffer;

    // Push transforms at t=0, 1000, 2000
    buffer.push(0, TransformMatrix4d::from_translation(0.0, 0.0, 0.0));
    buffer.push(1000, TransformMatrix4d::from_translation(1.0, 0.0, 0.0));
    buffer.push(2000, TransformMatrix4d::from_translation(2.0, 0.0, 0.0));

    // Test exact match
    auto result = buffer.at_interpolated_clamped(1000);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->transform.translation().x(), 1.0, 1e-6);

    // Test interpolation at t=500 (between 0 and 1000)
    result = buffer.at_interpolated_clamped(500);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->transform.translation().x(), 0.5, 1e-6);

    // Test clamping below range
    result = buffer.at_interpolated_clamped(0);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->transform.translation().x(), 0.0, 1e-6);

    // Test clamping above range
    result = buffer.at_interpolated_clamped(3000);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->transform.translation().x(), 2.0, 1e-6);
}

TEST(CircularTransformBufferTest, CircularOverwrite) {
    CircularTransformBuffer<3> buffer;

    buffer.push(1000, TransformMatrix4d::from_translation(1.0, 0.0, 0.0));
    buffer.push(2000, TransformMatrix4d::from_translation(2.0, 0.0, 0.0));
    buffer.push(3000, TransformMatrix4d::from_translation(3.0, 0.0, 0.0));

    EXPECT_TRUE(buffer.is_full());
    EXPECT_EQ(buffer.size(), 3);

    // Push one more, should overwrite oldest
    buffer.push(4000, TransformMatrix4d::from_translation(4.0, 0.0, 0.0));

    EXPECT_EQ(buffer.size(), 3);

    const auto range = buffer.time_range();
    ASSERT_TRUE(range.has_value());
    EXPECT_EQ(range->first, 2000); // oldest is now 2000
    EXPECT_EQ(range->second, 4000);
}

TEST(TransformMatrix4dTest, LerpPosition) {
    const auto a = TransformMatrix4d::from_translation(0.0, 0.0, 0.0);
    const auto b = TransformMatrix4d::from_translation(10.0, 20.0, 30.0);

    const auto mid = TransformMatrix4d::lerp(a, b, 0.5);
    EXPECT_NEAR(mid.translation().x(), 5.0, 1e-6);
    EXPECT_NEAR(mid.translation().y(), 10.0, 1e-6);
    EXPECT_NEAR(mid.translation().z(), 15.0, 1e-6);
}

TEST(TransformMatrix4dTest, LerpRotation) {
    // Identity to 90-degree rotation around Z
    const auto a = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0);
    const auto b = TransformMatrix4d::from_rpy(0.0, 0.0, M_PI / 2.0);

    const auto mid   = TransformMatrix4d::lerp(a, b, 0.5);
    const auto euler = mid.euler_rot();

    // Should be 45 degrees around Z
    EXPECT_NEAR(euler.roll, 0.0, 1e-6);
    EXPECT_NEAR(euler.pitch, 0.0, 1e-6);
    EXPECT_NEAR(euler.yaw, M_PI / 4.0, 1e-6);
}

TEST(TransformMatrix4dTest, LerpRightInvariantConsistency) {
    const auto a = TransformMatrix4d::from_rpy(0.1, -0.2, 0.3, 1.0, 2.0, 3.0);
    const auto b = TransformMatrix4d::from_rpy(-0.4, 0.5, -0.6, 4.0, 5.0, 6.0);

    constexpr double t = 0.3;
    const auto mid     = TransformMatrix4d::lerp_se3(a, b, t);

    const auto rel_expected = TransformMatrix4d::exp(a.rminus(b) * t);
    const auto mid_expected = a * rel_expected;

    const auto M  = mid.matrix();
    const auto Me = mid_expected.matrix();
    EXPECT_NEAR((M - Me).norm(), 0.0, 1e-9);
}

TEST(TransformMatrix4dTest, FromQuaternionNormalizes) {
    Eigen::Quaterniond q(2.0, 0.0, 0.0, 0.0); // non-unit quaternion (but valid after normalize)
    const auto tf = TransformMatrix4d::from_quaternion_xyz(q, 1.0, 2.0, 3.0);

    EXPECT_NEAR(tf.quaternion().norm(), 1.0, 1e-12);
    EXPECT_NEAR(tf.translation().x(), 1.0, 1e-12);
    EXPECT_NEAR(tf.translation().y(), 2.0, 1e-12);
    EXPECT_NEAR(tf.translation().z(), 3.0, 1e-12);
}
