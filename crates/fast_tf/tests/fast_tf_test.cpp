#include <gtest/gtest.h>

#include "buffer.hpp"
#include "frame.hpp"
#include "hero.hpp"
#include "matrix.hpp"

#include <type_traits>

using namespace fast_tf;

namespace {

struct test_a_frame {};
struct test_b_frame {};
struct test_c_frame {};
struct test_identity_frame {};

using TestTransform = TransformMatrixd<test_identity_frame, test_identity_frame>;

} // namespace

TEST(BufferBasicsTest, BasicPushAndLatest) {
    Buffer<TestTransform, 10, 1> buffer;

    EXPECT_TRUE(buffer.is_empty());
    EXPECT_EQ(buffer.size(), 0);

    const auto tf = TestTransform::from_translation(1.0, 2.0, 3.0);
    buffer.push(1000, tf);

    EXPECT_FALSE(buffer.is_empty());
    EXPECT_EQ(buffer.size(), 1);

    const auto result = buffer.latest();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->timestamp, 1000);
}

TEST(BufferBasicsTest, TimeRange) {
    Buffer<TestTransform, 10, 1> buffer;

    EXPECT_FALSE(buffer.time_range().has_value());

    buffer.push(1000, TestTransform{});
    buffer.push(2000, TestTransform{});
    buffer.push(3000, TestTransform{});

    const auto range = buffer.time_range();
    ASSERT_TRUE(range.has_value());
    EXPECT_EQ(range->first, 1000);
    EXPECT_EQ(range->second, 3000);
}

TEST(BufferBasicsTest, InterpolationClamped) {
    Buffer<TestTransform, 10, 1> buffer;

    // Push transforms at t=0, 1000, 2000
    buffer.push(0, TestTransform::from_translation(0.0, 0.0, 0.0));
    buffer.push(1000, TestTransform::from_translation(1.0, 0.0, 0.0));
    buffer.push(2000, TestTransform::from_translation(2.0, 0.0, 0.0));

    // Test exact match
    auto result = buffer.lookup(1000, clamped);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->value.translation().x(), 1.0, 1e-6);

    // Test interpolation at t=500 (between 0 and 1000)
    result = buffer.lookup(500, clamped);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->value.translation().x(), 0.5, 1e-6);

    // Test clamping below range
    result = buffer.lookup(0, clamped);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->value.translation().x(), 0.0, 1e-6);

    // Test clamping above range
    result = buffer.lookup(3000, clamped);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->value.translation().x(), 2.0, 1e-6);
}

TEST(BufferBasicsTest, CircularOverwrite) {
    Buffer<TestTransform, 3, 1> buffer;

    buffer.push(1000, TestTransform::from_translation(1.0, 0.0, 0.0));
    buffer.push(2000, TestTransform::from_translation(2.0, 0.0, 0.0));
    buffer.push(3000, TestTransform::from_translation(3.0, 0.0, 0.0));

    EXPECT_TRUE(buffer.is_full());
    EXPECT_EQ(buffer.size(), 3);

    // Push one more, should overwrite oldest
    buffer.push(4000, TestTransform::from_translation(4.0, 0.0, 0.0));

    EXPECT_EQ(buffer.size(), 3);

    const auto range = buffer.time_range();
    ASSERT_TRUE(range.has_value());
    EXPECT_EQ(range->first, 2000); // oldest is now 2000
    EXPECT_EQ(range->second, 4000);
}

TEST(TransformMatrixTest, LerpPosition) {
    const auto a = TestTransform::from_translation(0.0, 0.0, 0.0);
    const auto b = TestTransform::from_translation(10.0, 20.0, 30.0);

    const auto mid = TestTransform::lerp(a, b, 0.5);
    EXPECT_NEAR(mid.translation().x(), 5.0, 1e-6);
    EXPECT_NEAR(mid.translation().y(), 10.0, 1e-6);
    EXPECT_NEAR(mid.translation().z(), 15.0, 1e-6);
}

TEST(TransformMatrixTest, LerpRotation) {
    // Identity to 90-degree rotation around Z
    const auto a = TestTransform::from_rpy(0.0, 0.0, 0.0);
    const auto b = TestTransform::from_rpy(0.0, 0.0, M_PI / 2.0);

    const auto mid   = TestTransform::lerp(a, b, 0.5);
    const auto euler = mid.euler_rot();

    // Should be 45 degrees around Z
    EXPECT_NEAR(euler.roll, 0.0, 1e-6);
    EXPECT_NEAR(euler.pitch, 0.0, 1e-6);
    EXPECT_NEAR(euler.yaw, M_PI / 4.0, 1e-6);
}

TEST(TransformMatrixTest, LerpRightInvariantConsistency) {
    const auto a = TestTransform::from_rpy(0.1, -0.2, 0.3, 1.0, 2.0, 3.0);
    const auto b = TestTransform::from_rpy(-0.4, 0.5, -0.6, 4.0, 5.0, 6.0);

    constexpr double t = 0.3;
    const auto mid     = TestTransform::lerp_se3(a, b, t);

    const auto mid_expected = a.rplus(a.rminus(b) * t);

    const auto M  = mid.matrix();
    const auto Me = mid_expected.matrix();
    EXPECT_NEAR((M - Me).norm(), 0.0, 1e-9);
}

TEST(TransformMatrixTest, FromQuaternionNormalizes) {
    Eigen::Quaterniond q(2.0, 0.0, 0.0, 0.0); // non-unit quaternion (but valid after normalize)
    const auto tf = TestTransform::from_quaternion_xyz(q, 1.0, 2.0, 3.0);

    EXPECT_NEAR(tf.quaternion().norm(), 1.0, 1e-12);
    EXPECT_NEAR(tf.translation().x(), 1.0, 1e-12);
    EXPECT_NEAR(tf.translation().y(), 2.0, 1e-12);
    EXPECT_NEAR(tf.translation().z(), 3.0, 1e-12);
}

TEST(TypedTransformMatrixTest, CompositionAndInverseAreStronglyTyped) {
    const auto ab = TransformMatrixd<test_a_frame, test_b_frame>::from_translation(1.0, 0.0, 0.0);
    const auto bc = TransformMatrixd<test_b_frame, test_c_frame>::from_translation(0.0, 2.0, 0.0);

    const auto ac = ab * bc;
    static_assert(std::is_same_v<
                  std::remove_cvref_t<decltype(ac)>, TransformMatrixd<test_a_frame, test_c_frame>>);

    const auto ca = ac.inv();
    static_assert(std::is_same_v<
                  std::remove_cvref_t<decltype(ca)>, TransformMatrixd<test_c_frame, test_a_frame>>);

    EXPECT_NEAR(ac.translation().x(), 1.0, 1e-6);
    EXPECT_NEAR(ac.translation().y(), 2.0, 1e-6);
    EXPECT_NEAR(ac.translation().z(), 0.0, 1e-6);

    const auto aa = ac * ca;
    static_assert(std::is_same_v<
                  std::remove_cvref_t<decltype(aa)>, TransformMatrixd<test_a_frame, test_a_frame>>);
    EXPECT_NEAR(aa.translation().norm(), 0.0, 1e-9);
}

TEST(FrameLookupTest, TypedLookupReturnsFrameSpecificTransform) {
    const auto system = hero_new(0);
    const auto result = lookup_clamped<odom_fuxk_frame, camera_link_fuxk_frame>(system, 0);

    ASSERT_TRUE(result.has_value());
    static_assert(std::is_same_v<
                  std::remove_cvref_t<decltype(*result)>,
                  TransformMatrixd<camera_link_fuxk_frame, odom_fuxk_frame>>);

    EXPECT_NEAR(result->translation().x(), 0.18899, 1e-6);
    EXPECT_NEAR(result->translation().y(), 0.0735, 1e-6);
    EXPECT_NEAR(result->translation().z(), 0.17039, 1e-6);
}

TEST(BufferTest, ExactLookupReturnsStoredValue) {
    Buffer<TestTransform, 8, 1> buffer;

    buffer.push(1000, TestTransform::from_translation(11.0, 0.0, 0.0));

    const auto result = buffer.lookup(1000, exact);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->timestamp, 1000);
    EXPECT_NEAR(result->value.translation().x(), 11.0, 1e-6);
}

TEST(BufferTest, ExactRequiresMatchingTimestamp) {
    Buffer<TestTransform, 8, 1> buffer;

    buffer.push(1000, TestTransform::from_translation(1.0, 0.0, 0.0));
    buffer.push(2000, TestTransform::from_translation(2.0, 0.0, 0.0));

    const auto result = buffer.lookup(1500, exact);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), TransformError::ExactTimestampNotFound);
}

TEST(BufferTest, NearestReturnsClosestInRangeSample) {
    Buffer<TestTransform, 8, 1> buffer;

    buffer.push(1000, TestTransform::from_translation(1.0, 0.0, 0.0));
    buffer.push(2000, TestTransform::from_translation(2.0, 0.0, 0.0));

    auto result = buffer.lookup(1400, nearest);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->timestamp, 1000);
    EXPECT_NEAR(result->value.translation().x(), 1.0, 1e-6);

    result = buffer.lookup(1600, nearest);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->timestamp, 2000);
    EXPECT_NEAR(result->value.translation().x(), 2.0, 1e-6);
}

TEST(BufferTest, NearestBreaksExactMidpointTowardOlderSample) {
    Buffer<TestTransform, 8, 1> buffer;

    buffer.push(1000, TestTransform::from_translation(1.0, 0.0, 0.0));
    buffer.push(2000, TestTransform::from_translation(2.0, 0.0, 0.0));

    const auto result = buffer.lookup(1500, nearest);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->timestamp, 1000);
    EXPECT_NEAR(result->value.translation().x(), 1.0, 1e-6);
}

TEST(BufferTest, InterpolateUsesAdjacentSamplesWithoutExtrapolation) {
    Buffer<TestTransform, 8, 1> buffer;

    buffer.push(1000, TestTransform::from_translation(1.0, 0.0, 0.0));
    buffer.push(2000, TestTransform::from_translation(3.0, 0.0, 0.0));

    auto result = buffer.lookup(1500, interpolate);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->timestamp, 1500);
    EXPECT_NEAR(result->value.translation().x(), 2.0, 1e-6);

    result = buffer.lookup(500, interpolate);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), TransformError::PastExtrapolationRequired);
}

TEST(BufferTest, ClampedReturnsBoundarySamplesOutsideRange) {
    Buffer<TestTransform, 8, 1> buffer;

    buffer.push(1000, TestTransform::from_translation(1.0, 0.0, 0.0));
    buffer.push(2000, TestTransform::from_translation(3.0, 0.0, 0.0));

    auto result = buffer.lookup(500, clamped);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->timestamp, 1000);
    EXPECT_NEAR(result->value.translation().x(), 1.0, 1e-6);

    result = buffer.lookup(2500, clamped);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->timestamp, 2000);
    EXPECT_NEAR(result->value.translation().x(), 3.0, 1e-6);

    result = buffer.lookup(1500, clamped);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->timestamp, 1500);
    EXPECT_NEAR(result->value.translation().x(), 2.0, 1e-6);
}

TEST(BufferTest, StaticModeReturnsSameValueForAnyQueryTime) {
    Buffer<TestTransform, 0, 1> buffer(TestTransform::from_translation(5.0, 0.0, 0.0));

    const auto result = buffer.lookup(4242, exact);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->timestamp, 4242);
    EXPECT_NEAR(result->value.translation().x(), 5.0, 1e-6);
}

TEST(BufferTest, PushRotateOnlyKeepsLatestTranslation) {
    Buffer<TestTransform, 8, 1> buffer;

    buffer.push(1000, TestTransform::from_translation(1.0, 2.0, 3.0));
    buffer.push_rotate_only(2000, math_fuxk::rpy(0.1, -0.2, 0.3));

    const auto result = buffer.lookup(2000, exact);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->timestamp, 2000);
    EXPECT_NEAR(result->value.translation().x(), 1.0, 1e-6);
    EXPECT_NEAR(result->value.translation().y(), 2.0, 1e-6);
    EXPECT_NEAR(result->value.translation().z(), 3.0, 1e-6);

    const auto euler = result->value.euler_rot();
    EXPECT_NEAR(euler.roll, 0.1, 1e-6);
    EXPECT_NEAR(euler.pitch, -0.2, 1e-6);
    EXPECT_NEAR(euler.yaw, 0.3, 1e-6);
}
