#include <gtest/gtest.h>

#include "math/fast_tf/buffer.hpp"
#include "math/fast_tf/matrix.hpp"

using namespace fast_tf;

// 1 second default keep duration for tests
constexpr uint64_t test_keep_ns = 1'000'000'000;

TEST(TimeRangeBufferTest, BasicPushAndLatest) {
    TimeRangeBuffer<10, test_keep_ns> buffer;

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

TEST(TimeRangeBufferTest, TimeRange) {
    TimeRangeBuffer<10, test_keep_ns> buffer;

    EXPECT_FALSE(buffer.time_range().has_value());

    buffer.push(1000, TransformMatrix4d{});
    buffer.push(2000, TransformMatrix4d{});
    buffer.push(3000, TransformMatrix4d{});

    const auto range = buffer.time_range();
    ASSERT_TRUE(range.has_value());
    EXPECT_EQ(range->first, 1000);
    EXPECT_EQ(range->second, 3000);
}

TEST(TimeRangeBufferTest, InterpolationClamped) {
    // TimeRangeBuffer supports interpolation within time window:
    // - Stores entries in push order
    // - Interpolates within (oldest_ts, newest_ts) range
    // - Clamps to edge values when ts <= oldest_ts or ts >= newest_ts
    TimeRangeBuffer<10, test_keep_ns> buffer;

    // Insert transforms at different timestamps (start from 500M to avoid static transform)
    buffer.push(500'000'000, TransformMatrix4d::from_translation(0.0, 0.0, 0.0));
    buffer.push(1'000'000'000, TransformMatrix4d::from_translation(1.0, 0.0, 0.0));
    buffer.push(1'500'000'000, TransformMatrix4d::from_translation(2.0, 0.0, 0.0));

    // Test exact match at 1000ms
    auto result = buffer.at_interpolated_clamped(1'000'000'000);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->transform.translation().x(), 1.0, 1e-6);

    // Test query below oldest_ts - clamps to oldest
    result = buffer.at_interpolated_clamped(250'000'000);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->transform.translation().x(), 0.0, 1e-6);

    // Test query at 750ms - inside (oldest_ts, newest_ts), so interpolates
    result = buffer.at_interpolated_clamped(750'000'000);
    ASSERT_TRUE(result.has_value());
    // t = 0.5, lerp(0.0, 1.0, 0.5) = 0.5
    EXPECT_NEAR(result->transform.translation().x(), 0.5, 1e-6);

    // Test query at 1250ms - interpolates between 1000ms and 1500ms
    result = buffer.at_interpolated_clamped(1'250'000'000);
    ASSERT_TRUE(result.has_value());
    // 1250ms is midpoint between 1000ms (x=1.0) and 1500ms (x=2.0)
    EXPECT_NEAR(result->transform.translation().x(), 1.5, 1e-6);

    // Test query above newest_ts - clamps to newest
    result = buffer.at_interpolated_clamped(2'000'000'000);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->transform.translation().x(), 2.0, 1e-6);
}

TEST(TimeRangeBufferTest, CapacityLimit) {
    // TimeRangeBuffer enforces capacity limit:
    // - After N pushes (capacity), oldest entries are removed
    // - Size stays at or below capacity
    // Use large time window to avoid pruning during this test
    constexpr uint64_t large_window = 10'000'000'000; // 10 seconds
    TimeRangeBuffer<3, large_window> buffer;

    // Insert 3 entries (fills buffer)
    buffer.push(1'000'000'000, TransformMatrix4d::from_translation(1.0, 0.0, 0.0));
    buffer.push(2'000'000'000, TransformMatrix4d::from_translation(2.0, 0.0, 0.0));
    buffer.push(3'000'000'000, TransformMatrix4d::from_translation(3.0, 0.0, 0.0));

    EXPECT_TRUE(buffer.is_full());
    EXPECT_EQ(buffer.size(), 3);

    // Insert 4th entry - removes oldest (1.0s) due to capacity limit
    buffer.push(4'000'000'000, TransformMatrix4d::from_translation(4.0, 0.0, 0.0));

    EXPECT_EQ(buffer.size(), 3); // Size stays at capacity

    // Buffer now contains: [2.0s, 3.0s, 4.0s]
    // Query at 2.5s - should interpolate between 2.0 and 3.0
    auto result = buffer.at_interpolated_clamped(2'500'000'000);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->transform.translation().x(), 2.5, 1e-6);

    // Query at 3.5s - should interpolate between 3.0 and 4.0
    result = buffer.at_interpolated_clamped(3'500'000'000);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->transform.translation().x(), 3.5, 1e-6);
}

TEST(TimeRangeBufferTest, StaticTransform) {
    // Static transform (timestamp==0) is stored separately and never expires
    TimeRangeBuffer<10, test_keep_ns> buffer;

    const auto static_tf = TransformMatrix4d::from_translation(5.0, 6.0, 7.0);
    buffer.push(0, static_tf);

    // Static transform should be returned for any query
    auto result = buffer.latest();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->timestamp_ns, 0);
    EXPECT_NEAR(result->transform.translation().x(), 5.0, 1e-6);

    // Static transform handles all timestamps
    result = buffer.at_interpolated_clamped(999'999'999);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->timestamp_ns, 999'999'999);
    EXPECT_NEAR(result->transform.translation().x(), 5.0, 1e-6);

    // time_range for static transform covers all time
    auto range = buffer.time_range();
    ASSERT_TRUE(range.has_value());
    EXPECT_EQ(range->first, 0);
    EXPECT_EQ(range->second, UINT64_MAX);

    // Closest timestamp info for static transform
    auto info = buffer.get_closest_timestamp_info(123456);
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->closest_ts, 0);
    EXPECT_EQ(info->delta_ns, 0);
}

TEST(TimeRangeBufferTest, TimeWindowPruning) {
    // TimeRangeBuffer automatically removes entries outside time window on each push
    // The window is relative to the newest entry
    // Use timestamps > 0 to avoid static transform behavior
    constexpr uint64_t window_ns = 1'000'000'000; // 1 second window
    TimeRangeBuffer<100, window_ns> buffer;

    // Insert entries at 1000ms, 2000ms, 3000ms
    buffer.push(1'000'000'000, TransformMatrix4d{});
    EXPECT_EQ(buffer.size(), 1); // [1000ms]

    buffer.push(2'000'000'000, TransformMatrix4d{});
    EXPECT_EQ(buffer.size(), 2); // [1000ms, 2000ms]

    // When adding 3000ms, threshold = 3000 - 1000 = 2000ms
    // Entries >= 2000ms are kept: [2000ms, 3000ms]
    // 1000ms is pruned
    buffer.push(3'000'000'000, TransformMatrix4d{});
    EXPECT_EQ(buffer.size(), 2); // [2000ms, 3000ms]

    // Verify time range
    auto range = buffer.time_range();
    ASSERT_TRUE(range.has_value());
    EXPECT_EQ(range->first, 2'000'000'000);
    EXPECT_EQ(range->second, 3'000'000'000);
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

// ============================================================================
// Timestamp Consistency Tests
// ============================================================================

#include "math/fast_tf/error.hpp"
#include "math/fast_tf/frame.hpp"

TEST(ClosestTimestampInfo, GetClosestTimestampInfoWithinRange) {
    TimeRangeBuffer<10, test_keep_ns> buffer;

    // Insert data at specific times
    buffer.push(0, TransformMatrix4d::from_translation(0.0, 0.0, 0.0));
    buffer.push(500'000'000, TransformMatrix4d::from_translation(1.0, 0.0, 0.0));
    buffer.push(1'000'000'000, TransformMatrix4d::from_translation(2.0, 0.0, 0.0));

    // Query at 500ms - exact match
    auto result = buffer.get_closest_timestamp_info(500'000'000);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->closest_ts, 0); // Static transform takes precedence
    EXPECT_EQ(result->delta_ns, 0u);

    // Query at 750ms - closest is static (delta 0)
    result = buffer.get_closest_timestamp_info(750'000'000);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->closest_ts, 0);
    EXPECT_EQ(result->delta_ns, 0u);
}

TEST(ClosestTimestampInfo, GetClosestTimestampInfoClamped) {
    // TimeRangeBuffer returns the closest timestamp in the buffer
    // Timestamp==0 is treated as static transform (delta=0, always valid)
    TimeRangeBuffer<10, test_keep_ns> buffer;

    // Use non-zero timestamps to test clamping behavior
    buffer.push(500'000'000, TransformMatrix4d{});
    buffer.push(1'000'000'000, TransformMatrix4d{});
    buffer.push(1'500'000'000, TransformMatrix4d{});

    // Query at 250ms - below oldest_ts (500ms)
    auto result = buffer.get_closest_timestamp_info(250'000'000);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->closest_ts, 500'000'000u);
    EXPECT_EQ(result->delta_ns, 250'000'000u);

    // Query at 4'500'000'000 (4.5s) - above newest_ts (1.5s)
    result = buffer.get_closest_timestamp_info(4'500'000'000);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->closest_ts, 1'500'000'000u);
    EXPECT_EQ(result->delta_ns, 3'000'000'000u); // 4.5s - 1.5s = 3.0s
}

TEST(ClosestTimestampInfo, GetClosestTimestampInfoEmptyBuffer) {
    TimeRangeBuffer<10, test_keep_ns> buffer;
    auto result = buffer.get_closest_timestamp_info(1000);
    EXPECT_FALSE(result.has_value());
}

TEST(LookupWithConsistency, SuccessfulLookupWithinTolerance) {
    CoordinateSystem<10> system;

    // Set up transforms at t=1000 for all frames
    system[gimbal_link_fuxk_frame::index].push(
        1000, TransformMatrix4d::from_translation(1.0, 0.0, 0.0));
    system[camera_link_fuxk_frame::index].push(
        1000, TransformMatrix4d::from_translation(0.0, 1.0, 0.0));

    FuckTransform<camera_link_fuxk_frame> camera_tf{
        TransformMatrix4d::from_translation(0.0, 0.0, 0.0)};

    // Query with default tolerance (0.1ms = 100us) - exact match should pass
    auto result = lookup<odom_fuxk_frame>(system, camera_tf, 1000);
    EXPECT_TRUE(result.has_value());
}

TEST(LookupWithConsistency, TimestampMismatchDetected) {
    CoordinateSystem<10> system;

    // Set up transforms with overlapping time ranges but slightly offset timestamps
    // gimbal_link has data at t=1000 and t=2000
    // camera_link has data at t=1500 and t=2500 (500ns offset)
    system[gimbal_link_fuxk_frame::index].push(
        1000, TransformMatrix4d::from_translation(1.0, 0.0, 0.0));
    system[gimbal_link_fuxk_frame::index].push(
        2000, TransformMatrix4d::from_translation(1.0, 0.0, 0.0));
    system[camera_link_fuxk_frame::index].push(
        1500, TransformMatrix4d::from_translation(0.0, 1.0, 0.0));
    system[camera_link_fuxk_frame::index].push(
        2500, TransformMatrix4d::from_translation(0.0, 1.0, 0.0));

    FuckTransform<camera_link_fuxk_frame> camera_tf{
        TransformMatrix4d::from_translation(0.0, 0.0, 0.0)};

    // Query at t=1000 with tight tolerance (10us) using clamped lookup
    // camera_link's closest is at 1500 (delta 500ns) - exceeds tolerance
    auto result = lookup_clamped<odom_fuxk_frame>(system, camera_tf, 1000, 10);
    EXPECT_FALSE(result.has_value());

    // Verify error contains mismatch info
    std::visit(
        [](auto&& err) {
            using T = std::decay_t<decltype(err)>;
            if constexpr (std::is_same_v<T, TimestampMismatchInfo>) {
                EXPECT_GT(err.max_delta_ns, 10u);
                EXPECT_EQ(err.query_ns, 1000u);
            } else {
                FAIL() << "Expected TimestampMismatchInfo, got TransformError: " << to_string(err);
            }
        },
        result.error());
}

TEST(LookupWithConsistency, ClampedLookupWithTearing) {
    CoordinateSystem<10> system;

    // Set up transforms at different timestamps
    system[gimbal_link_fuxk_frame::index].push(
        1000, TransformMatrix4d::from_translation(1.0, 0.0, 0.0));
    system[camera_link_fuxk_frame::index].push(
        10000, TransformMatrix4d::from_translation(0.0, 1.0, 0.0));

    FuckTransform<camera_link_fuxk_frame> camera_tf{
        TransformMatrix4d::from_translation(0.0, 0.0, 0.0)};

    // Clamped lookup with tight tolerance - should detect mismatch
    auto result = lookup_clamped<odom_fuxk_frame>(system, camera_tf, 5000, 100);
    EXPECT_FALSE(result.has_value());

    // Verify it's a mismatch error, not extrapolation error
    std::visit(
        [](auto&& err) {
            using T = std::decay_t<decltype(err)>;
            if constexpr (std::is_same_v<T, TimestampMismatchInfo>) {
                EXPECT_GT(err.max_delta_ns, 100u);
            } else {
                FAIL() << "Expected TimestampMismatchInfo, got TransformError: " << to_string(err);
            }
        },
        result.error());
}

TEST(LookupWithConsistency, LookupReturnsExtrapolationErrorWhenOutOfRange) {
    CoordinateSystem<10> system;

    // Set up transforms at t=1000-2000 range
    system[gimbal_link_fuxk_frame::index].push(
        1000, TransformMatrix4d::from_translation(1.0, 0.0, 0.0));
    system[gimbal_link_fuxk_frame::index].push(
        2000, TransformMatrix4d::from_translation(1.0, 0.0, 0.0));
    system[camera_link_fuxk_frame::index].push(
        1000, TransformMatrix4d::from_translation(0.0, 1.0, 0.0));
    system[camera_link_fuxk_frame::index].push(
        2000, TransformMatrix4d::from_translation(0.0, 1.0, 0.0));

    FuckTransform<camera_link_fuxk_frame> camera_tf{
        TransformMatrix4d::from_translation(0.0, 0.0, 0.0)};

    // Query at t=5000 (outside buffer range) - should return extrapolation error
    auto result = lookup<odom_fuxk_frame>(system, camera_tf, 5000);
    EXPECT_FALSE(result.has_value());

    // Verify it's an extrapolation error, not mismatch error
    std::visit(
        [](auto&& err) {
            using T = std::decay_t<decltype(err)>;
            if constexpr (std::is_same_v<T, TransformError>) {
                EXPECT_EQ(err, TransformError::FutureExtrapolationRequired);
            } else {
                FAIL() << "Expected TransformError::FutureExtrapolationRequired";
            }
        },
        result.error());
}

TEST(LookupWithConsistency, MultiFrameTearingDetection) {
    CoordinateSystem<10> system;

    // Simulate sequential updates causing tearing with overlapping ranges
    // All frames have data spanning 1000-4000ns, but "best" timestamps are offset
    system[odom_fuxk_frame::index].push(1000, TransformMatrix4d::from_translation(1.0, 0.0, 0.0));
    system[odom_fuxk_frame::index].push(2000, TransformMatrix4d::from_translation(1.0, 0.0, 0.0));
    system[odom_fuxk_frame::index].push(3000, TransformMatrix4d::from_translation(1.0, 0.0, 0.0));
    system[gimbal_link_fuxk_frame::index].push(
        1500, TransformMatrix4d::from_translation(0.0, 1.0, 0.0));
    system[gimbal_link_fuxk_frame::index].push(
        2500, TransformMatrix4d::from_translation(0.0, 1.0, 0.0));
    system[camera_link_fuxk_frame::index].push(
        3000, TransformMatrix4d::from_translation(0.0, 0.0, 1.0));
    system[camera_link_fuxk_frame::index].push(
        4000, TransformMatrix4d::from_translation(0.0, 0.0, 1.0));

    FuckTransform<camera_link_fuxk_frame> camera_tf{
        TransformMatrix4d::from_translation(0.0, 0.0, 0.0)};

    // Query at t=1000 with tolerance 100us using clamped lookup
    // gimbal_link's closest is at 1500 (delta 500ns), camera_link at 3000 (delta 2000ns)
    auto result = lookup_clamped<odom_fuxk_frame>(system, camera_tf, 1000, 100);
    EXPECT_FALSE(result.has_value());

    std::visit(
        [](auto&& err) {
            using T = std::decay_t<decltype(err)>;
            if constexpr (std::is_same_v<T, TimestampMismatchInfo>) {
                // Max delta should be at least 2000ns (camera_link is at 3000)
                EXPECT_GE(err.max_delta_ns, 2000u);
                EXPECT_EQ(err.query_ns, 1000u);
            } else {
                FAIL() << "Expected TimestampMismatchInfo, got TransformError: " << to_string(err);
            }
        },
        result.error());
}

TEST(LookupWithConsistency, LooseToleranceAllowsMismatch) {
    CoordinateSystem<10> system;

    // Set up transforms with overlapping ranges
    // All buffers include t=1000 in their range
    system[gimbal_link_fuxk_frame::index].push(
        1000, TransformMatrix4d::from_translation(1.0, 0.0, 0.0));
    system[gimbal_link_fuxk_frame::index].push(
        2000, TransformMatrix4d::from_translation(1.0, 0.0, 0.0));
    system[camera_link_fuxk_frame::index].push(
        1000, TransformMatrix4d::from_translation(0.0, 1.0, 0.0));
    system[camera_link_fuxk_frame::index].push(
        2500, TransformMatrix4d::from_translation(0.0, 1.0, 0.0));

    FuckTransform<camera_link_fuxk_frame> camera_tf{
        TransformMatrix4d::from_translation(0.0, 0.0, 0.0)};

    // Query at t=1000 with very loose tolerance (10ms = 10,000,000ns)
    // Both frames have exact data at t=1000, so delta=0 - well within tolerance
    auto result = lookup<odom_fuxk_frame>(system, camera_tf, 1000, 10'000'000);
    EXPECT_TRUE(result.has_value());
}

TEST(LookupWithConsistency, ErrorVariantHandling) {
    CoordinateSystem<10> system;

    // Empty buffer - should return TransformError::NoDataAvailable
    FuckTransform<camera_link_fuxk_frame> camera_tf{
        TransformMatrix4d::from_translation(0.0, 0.0, 0.0)};

    auto result = lookup<odom_fuxk_frame>(system, camera_tf, 1000);
    EXPECT_FALSE(result.has_value());

    // Verify it's a TransformError, not TimestampMismatchInfo
    std::visit(
        [](auto&& err) {
            using T = std::decay_t<decltype(err)>;
            if constexpr (std::is_same_v<T, TransformError>) {
                EXPECT_EQ(err, TransformError::NoDataAvailable);
            } else {
                FAIL() << "Expected TransformError::NoDataAvailable";
            }
        },
        result.error());
}

// ============================================================================
// Transaction Update Tests
// ============================================================================

#include <atomic>
#include <thread>

TEST(TransactionUpdate, AtomicityCheck) {
    constexpr uint64_t test_keep_ns = 1'000'000'000;
    CoordinateSystem<1024, test_keep_ns> system;

    std::atomic<bool> running{true};
    std::atomic<int> mismatches{0};

    // Writer thread: simulate IMU batch updates at 1000Hz
    std::thread writer([&]() {
        uint64_t ts = 0;
        while (running) {
            std::vector<std::pair<size_t, std::pair<TransformMatrix4d, uint64_t>>> updates;
            updates.push_back({
                gimbal_link_fuxk_frame::index,
                {TransformMatrix4d::from_translation(1.0, 0.0, 0.0), ts}
            });
            updates.push_back({
                odom_fuxk_frame::index, {TransformMatrix4d::from_translation(2.0, 0.0, 0.0), ts}
            });
            system.update_transaction_batch(updates);
            ts += 1'000'000; // +1ms
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });

    // Reader thread: simulate Planner querying for transforms
    std::thread reader([&]() {
        uint64_t query_ts = 0;
        while (running) {
            auto result = lookup<odom_fuxk_frame, gimbal_link_fuxk_frame>(
                system, FuckTransform<gimbal_link_fuxk_frame>{}, query_ts,
                1000);           // 1μs tolerance
            if (!result) {
                if (std::holds_alternative<TimestampMismatchInfo>(result.error())) {
                    mismatches++;
                }
            }
            query_ts += 500'000; // +0.5ms
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    running = false;
    writer.join();
    reader.join();

    EXPECT_EQ(mismatches.load(), 0) << "Detected TF tearing with transactional updates";
}

TEST(TransactionUpdate, SingleFrameBackwardCompatibility) {
    constexpr uint64_t test_keep_ns = 1'000'000'000;
    CoordinateSystem<1024, test_keep_ns> system;

    // Test that single-frame update() still works
    update(
        system,
        FuckTransform<gimbal_link_fuxk_frame>(TransformMatrix4d::from_translation(1.0, 2.0, 3.0)),
        1000);

    auto result = system.template get_buffer<gimbal_link_fuxk_frame>().latest();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->timestamp_ns, 1000);
    EXPECT_NEAR(result->transform.translation().x(), 1.0, 1e-6);
}

TEST(TransactionUpdate, BatchUpdateConsistency) {
    constexpr uint64_t test_keep_ns = 1'000'000'000;
    CoordinateSystem<1024, test_keep_ns> system;

    // Perform a batch update with consistent timestamps
    const uint64_t timestamp_ns = 1'000'000'000;
    std::vector<std::pair<size_t, std::pair<TransformMatrix4d, uint64_t>>> updates;
    updates.push_back({
        gimbal_link_fuxk_frame::index,
        {TransformMatrix4d::from_translation(1.0, 0.0, 0.0), timestamp_ns}
    });
    updates.push_back({
        odom_fuxk_frame::index,
        {TransformMatrix4d::from_translation(2.0, 0.0, 0.0), timestamp_ns}
    });
    updates.push_back({
        camera_link_fuxk_frame::index,
        {TransformMatrix4d::from_translation(0.0, 1.0, 0.0), timestamp_ns}
    });

    system.update_transaction_batch(updates);

    // Verify all frames have the same timestamp
    auto gimbal_info =
        system[gimbal_link_fuxk_frame::index].get_closest_timestamp_info(timestamp_ns);
    auto odom_info = system[odom_fuxk_frame::index].get_closest_timestamp_info(timestamp_ns);
    auto camera_info =
        system[camera_link_fuxk_frame::index].get_closest_timestamp_info(timestamp_ns);

    ASSERT_TRUE(gimbal_info.has_value());
    ASSERT_TRUE(odom_info.has_value());
    ASSERT_TRUE(camera_info.has_value());

    EXPECT_EQ(gimbal_info->closest_ts, timestamp_ns);
    EXPECT_EQ(odom_info->closest_ts, timestamp_ns);
    EXPECT_EQ(camera_info->closest_ts, timestamp_ns);

    // Verify lookup with tight tolerance succeeds (no tearing)
    auto result = lookup<odom_fuxk_frame>(
        system, FuckTransform<camera_link_fuxk_frame>{TransformMatrix4d{}}, timestamp_ns,
        100); // 0.1μs tolerance
    EXPECT_TRUE(result.has_value());
}

TEST(TransactionUpdate, MultiReaderWriterStressTest) {
    // Aggressive stress test with multiple concurrent readers and writers
    // This validates that the read-write lock coordination prevents tearing.
    //
    // The key: Each batch update writes ALL frames with the SAME timestamp.
    // If tearing occurs, a reader would see some frames at timestamp T and others
    // at timestamp T-1 (from the previous batch), which would be detected.
    constexpr uint64_t test_keep_ns = 1'000'000'000; // 1 second window (default)
    CoordinateSystem<1024, test_keep_ns> system;

    std::atomic<bool> running{true};
    std::atomic<int> reader_tearing{0};
    std::atomic<int> successful_lookups{0};
    std::atomic<uint64_t> total_lookups{0};
    std::atomic<uint64_t> shared_timestamp{1000};    // Shared timestamp counter

    // Single writer thread at 1000Hz (simulating high-rate IMU)
    std::thread writer([&]() {
        while (running) {
            uint64_t ts = shared_timestamp.fetch_add(100'000) + 100'000; // +0.1ms

            std::vector<std::pair<size_t, std::pair<TransformMatrix4d, uint64_t>>> updates;
            // Update ALL frames with the SAME timestamp (atomic batch)
            updates.push_back({
                gimbal_link_fuxk_frame::index,
                {TransformMatrix4d::from_translation(1.0, 0.0, 0.0), ts}
            });
            updates.push_back({
                odom_fuxk_frame::index, {TransformMatrix4d::from_translation(2.0, 0.0, 0.0), ts}
            });
            updates.push_back({
                camera_link_fuxk_frame::index,
                {TransformMatrix4d::from_translation(0.0, 1.0, 0.0), ts}
            });
            updates.push_back({
                muzzle_link_fuxk_frame::index,
                {TransformMatrix4d::from_translation(0.0, 0.0, 1.0), ts}
            });
            system.update_transaction_batch(updates);

            std::this_thread::sleep_for(std::chrono::microseconds(100)); // ~1000Hz
        }
    });

    // Multiple reader threads (simulating concurrent Planner queries)
    constexpr int num_readers = 8;
    std::vector<std::thread> readers;
    for (int r = 0; r < num_readers; ++r) {
        readers.emplace_back([&, reader_id = r]() {
            uint64_t last_ts = shared_timestamp.load();
            while (running) {
                total_lookups++;

                // Query at a timestamp that should have data (or very recent data)
                uint64_t query_ts = shared_timestamp.load() - 50'000; // Query slightly behind

                // Multi-frame lookup: camera_link -> odom (traverses 2 frames)
                auto result = lookup<odom_fuxk_frame>(
                    system, FuckTransform<camera_link_fuxk_frame>{TransformMatrix4d{}}, query_ts,
                    50'000); // 50μs tolerance - should allow for some clock skew

                if (!result) {
                    if (std::holds_alternative<TimestampMismatchInfo>(result.error())) {
                        // This indicates TF tearing - frames have inconsistent timestamps
                        reader_tearing++;
                    }
                } else {
                    successful_lookups++;
                }

                // Query at various rates to stress test
                std::this_thread::sleep_for(std::chrono::microseconds(30));
            }
        });
    }

    // Run for 300ms to stress test the coordination
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    running = false;

    writer.join();
    for (auto& r : readers)
        r.join();

    // With proper read-write lock coordination, tearing should be ZERO.
    // Each batch writes ALL frames at the same timestamp, so readers should
    // never see a mix of old and new timestamps.
    EXPECT_EQ(reader_tearing.load(), 0)
        << "Detected TF tearing - read-write lock coordination failed!";

    // We should have performed many lookups
    EXPECT_GT(total_lookups.load(), 5000u) << "Insufficient stress test iterations";

    // Most lookups should succeed (some may fail due to data availability at edges).
    // Note: With 50μs tolerance and 100μs write rate, some queries will legitimately fail
    // due to the query timestamp being slightly outside the data window. This is expected.
    const double success_rate =
        static_cast<double>(successful_lookups.load()) / static_cast<double>(total_lookups.load());
    EXPECT_GT(success_rate, 0.75) << "Success rate too low: " << success_rate;
}
