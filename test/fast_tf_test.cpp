#include <cmath>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

#include "fast_tf/mod.hpp"

using namespace fast_tf;

// ============================================================================
// TransformMatrix4d Tests
// ============================================================================

class TransformMatrix4dTest : public ::testing::Test {
protected:
    const double EPSILON = 1e-10;

    bool
        matrices_equal(const TransformMatrix4d& a, const TransformMatrix4d& b, double eps = 1e-10) {
        return (a.matrix() - b.matrix()).cwiseAbs().maxCoeff() < eps;
    }
};

TEST_F(TransformMatrix4dTest, IdentityTransform) {
    auto identity  = TransformMatrix4d();
    auto identity2 = TransformMatrix4d();
    EXPECT_TRUE(matrices_equal(identity, identity2));

    // Translation should be zero
    auto trans = identity.translation();
    EXPECT_NEAR(trans.x(), 0.0, EPSILON);
    EXPECT_NEAR(trans.y(), 0.0, EPSILON);
    EXPECT_NEAR(trans.z(), 0.0, EPSILON);
}

TEST_F(TransformMatrix4dTest, CreationFromTranslation) {
    // Create pure translation using from_rpy with zero rotation
    auto tf    = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 1.0, 2.0, 3.0);
    auto trans = tf.translation();

    EXPECT_NEAR(trans.x(), 1.0, EPSILON);
    EXPECT_NEAR(trans.y(), 2.0, EPSILON);
    EXPECT_NEAR(trans.z(), 3.0, EPSILON);
}

TEST_F(TransformMatrix4dTest, CreationFromRPY) {
    // Create from RPY: Roll=0.5, Pitch=0.3, Yaw=0.1 rad
    auto tf    = TransformMatrix4d::from_rpy(0.5, 0.3, 0.1, 1.0, 2.0, 3.0);
    auto trans = tf.translation();

    EXPECT_NEAR(trans.x(), 1.0, EPSILON);
    EXPECT_NEAR(trans.y(), 2.0, EPSILON);
    EXPECT_NEAR(trans.z(), 3.0, EPSILON);

    // Verify it's a valid rotation matrix (determinant = 1)
    auto rot   = tf.rotation();
    double det = rot.determinant();
    EXPECT_NEAR(det, 1.0, EPSILON);
}

TEST_F(TransformMatrix4dTest, TransformComposition) {
    // T1: translate by (1, 0, 0)
    auto tf1 = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    // T2: translate by (0, 1, 0)
    auto tf2 = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    // T_combined = T2 * T1
    auto combined = tf2 * tf1;
    auto trans    = combined.translation();

    EXPECT_NEAR(trans.x(), 1.0, EPSILON);
    EXPECT_NEAR(trans.y(), 1.0, EPSILON);
    EXPECT_NEAR(trans.z(), 0.0, EPSILON);
}

TEST_F(TransformMatrix4dTest, InverseTransform) {
    auto tf     = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 1.0, 2.0, 3.0);
    auto tf_inv = tf.inverse();

    auto identity = tf * tf_inv;
    EXPECT_TRUE(matrices_equal(identity, TransformMatrix4d()));
}

TEST_F(TransformMatrix4dTest, PointTransformation) {
    auto tf = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 1.0, 2.0, 3.0);
    Eigen::Vector3d point(0.0, 0.0, 0.0);
    Eigen::Vector4d point_homo(point.x(), point.y(), point.z(), 1.0);

    auto transformed_homo = tf.matrix() * point_homo;
    Eigen::Vector3d transformed(transformed_homo.x(), transformed_homo.y(), transformed_homo.z());

    EXPECT_NEAR(transformed.x(), 1.0, EPSILON);
    EXPECT_NEAR(transformed.y(), 2.0, EPSILON);
    EXPECT_NEAR(transformed.z(), 3.0, EPSILON);
}

// ============================================================================
// CircularTransformBuffer Tests
// ============================================================================

class CircularTransformBufferTest : public ::testing::Test {
protected:
    static constexpr int BUFFER_SIZE = 10;
    using TestBuffer                 = CircularTransformBuffer<BUFFER_SIZE>;

    const double EPSILON = 1e-10;
};

// Note: CircularTransformBuffer stores TransformEntry with timestamp
// Latest() returns the most recent entry correctly
TEST_F(CircularTransformBufferTest, LatestTransformTracking) {
    TestBuffer buffer;

    auto tf1 = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    auto tf2 = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 2.0, 0.0, 0.0);
    auto tf3 = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 3.0, 0.0, 0.0);

    buffer.push(1000, tf1);
    buffer.push(2000, tf2);
    buffer.push(3000, tf3);

    // Verify latest returns the most recent entry
    auto latest = buffer.latest();
    ASSERT_TRUE(latest.has_value());
    EXPECT_EQ(latest->timestamp_ns, 3000);
    EXPECT_NEAR(latest->transform.translation().x(), 3.0, EPSILON);
}

TEST_F(CircularTransformBufferTest, BufferWraparound) {
    TestBuffer buffer;

    // Fill buffer beyond capacity
    for (int i = 0; i < BUFFER_SIZE + 5; ++i) {
        auto tf = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, i, 0.0, 0.0);
        buffer.push(1000 + i * 100, tf);
    }

    // Latest should be the last one added
    auto latest = buffer.latest();
    ASSERT_TRUE(latest.has_value());
    EXPECT_NEAR(latest->transform.translation().x(), BUFFER_SIZE + 4, EPSILON);
    EXPECT_EQ(latest->timestamp_ns, 1000 + (BUFFER_SIZE + 4) * 100);
}

TEST_F(CircularTransformBufferTest, ThreadSafety) {
    TestBuffer buffer;

    // Writer thread
    std::thread writer([&buffer]() {
        for (int i = 0; i < 50; ++i) {
            auto tf = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, i, 0.0, 0.0);
            buffer.push(1000 + i * 10, tf);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    // Reader threads
    std::vector<std::thread> readers;
    for (int t = 0; t < 3; ++t) {
        readers.emplace_back([&buffer]() {
            for (int i = 0; i < 50; ++i) {
                (void)buffer.latest();
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        });
    }

    writer.join();
    for (auto& r : readers) {
        r.join();
    }

    SUCCEED(); // If we get here without deadlock, test passes
}

// ============================================================================
// Buffer std::expected Tests
// ============================================================================

class BufferExpectedTest : public ::testing::Test {
protected:
    static constexpr int BUFFER_SIZE = 10;
    using TestBuffer                 = CircularTransformBuffer<BUFFER_SIZE>;
    const double EPSILON             = 1e-10;
};

TEST_F(BufferExpectedTest, EmptyBufferReturnsError) {
    TestBuffer buffer;

    auto latest = buffer.latest();
    EXPECT_FALSE(latest.has_value());
    EXPECT_EQ(latest.error(), TransformError::NoDataAvailable);

    auto at_result = buffer.at(1000);
    EXPECT_FALSE(at_result.has_value());
    EXPECT_EQ(at_result.error(), TransformError::NoDataAvailable);

    auto interp = buffer.at_interpolated(1000);
    EXPECT_FALSE(interp.has_value());
    EXPECT_EQ(interp.error(), TransformError::NoDataAvailable);
}

TEST_F(BufferExpectedTest, LatestReturnsValue) {
    TestBuffer buffer;

    auto tf = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 1.0, 2.0, 3.0);
    buffer.push(1000, tf);

    auto latest = buffer.latest();
    ASSERT_TRUE(latest.has_value());
    EXPECT_EQ(latest->timestamp_ns, 1000);
    EXPECT_NEAR(latest->transform.translation().x(), 1.0, EPSILON);
}

TEST_F(BufferExpectedTest, AtReturnsExactMatch) {
    TestBuffer buffer;

    auto tf1 = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    auto tf2 = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 2.0, 0.0, 0.0);

    buffer.push(1000, tf1);
    buffer.push(2000, tf2);

    auto result = buffer.at(1000);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->timestamp_ns, 1000);
    EXPECT_NEAR(result->transform.translation().x(), 1.0, EPSILON);
}

TEST_F(BufferExpectedTest, AtReturnsExtrapolationErrorForMissing) {
    TestBuffer buffer;

    auto tf = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    buffer.push(1000, tf);

    auto result = buffer.at(2000);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), TransformError::ExtrapolationRequired);
}

TEST_F(BufferExpectedTest, AtInterpolatedWorks) {
    TestBuffer buffer;

    auto tf1 = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    auto tf2 = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 10.0, 0.0, 0.0);

    buffer.push(1000, tf1);
    buffer.push(2000, tf2);

    // Query at midpoint
    auto result = buffer.at_interpolated(1500);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->timestamp_ns, 1500);
    // Should interpolate to x=5.0
    EXPECT_NEAR(result->transform.translation().x(), 5.0, EPSILON);
}

TEST_F(BufferExpectedTest, AtInterpolatedExtrapolationError) {
    TestBuffer buffer;

    auto tf1 = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    auto tf2 = TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 10.0, 0.0, 0.0);

    buffer.push(1000, tf1);
    buffer.push(2000, tf2);

    // Query outside range
    auto before = buffer.at_interpolated(500);
    EXPECT_FALSE(before.has_value());
    EXPECT_EQ(before.error(), TransformError::ExtrapolationRequired);

    auto after = buffer.at_interpolated(3000);
    EXPECT_FALSE(after.has_value());
    EXPECT_EQ(after.error(), TransformError::ExtrapolationRequired);
}

// ============================================================================
// TransformBuffer Tests (New API)
// ============================================================================

class TransformBufferTest : public ::testing::Test {
protected:
    const double EPSILON = 1e-6;

    // Default Infantry config
    Eigen::Vector3d t_camera{0.04444, 0.0, -0.0363};
    Eigen::Matrix3d R_camera = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_muzzle{0.0874, 0.0, -0.0874};
};

TEST_F(TransformBufferTest, Construction) {
    TransformBuffer<100> buffer(t_camera, R_camera, t_muzzle);

    // Buffer should be empty initially
    EXPECT_TRUE(buffer.buffer_empty());
    EXPECT_EQ(buffer.buffer_size(), 0);
}

TEST_F(TransformBufferTest, UpdateGimbal) {
    TransformBuffer<100> buffer(t_camera, R_camera, t_muzzle);

    buffer.update_gimbal(0.1, 0.2, 0.3, 1000);
    buffer.update_gimbal(0.2, 0.3, 0.4, 2000);

    EXPECT_FALSE(buffer.buffer_empty());
    EXPECT_EQ(buffer.buffer_size(), 2);

    auto range = buffer.get_buffer_time_range();
    ASSERT_TRUE(range.has_value());
    EXPECT_EQ(range->first, 1000);
    EXPECT_EQ(range->second, 2000);
}

TEST_F(TransformBufferTest, TransformPointToWorld) {
    TransformBuffer<100> buffer(t_camera, R_camera, t_muzzle);

    // Set gimbal to identity
    buffer.update_gimbal(0.0, 0.0, 0.0, 1000);

    // Transform origin point
    Eigen::Vector3d p_camera(0.0, 0.0, 1.0); // 1m forward in camera frame
    auto result = buffer.transform_point_to_world(p_camera, 1000);

    ASSERT_TRUE(result.has_value());
    // Point should be transformed through optical → camera → gimbal → world
}

TEST_F(TransformBufferTest, GetGimbalTransform) {
    TransformBuffer<100> buffer(t_camera, R_camera, t_muzzle);

    buffer.update_gimbal(0.1, 0.2, 0.3, 1000);

    auto gimbal = buffer.get_gimbal_transform();
    // Verify it's not identity (has rotation)
    auto rot = gimbal.rotation();
    EXPECT_NE(rot(0, 0), 1.0);
}

TEST_F(TransformBufferTest, StaticTransformGetters) {
    TransformBuffer<100> buffer(t_camera, R_camera, t_muzzle);

    // Verify camera offset
    auto camera = buffer.get_camera_to_gimbal();
    auto trans  = camera.translation();
    EXPECT_NEAR(trans.x(), t_camera.x(), EPSILON);
    EXPECT_NEAR(trans.y(), t_camera.y(), EPSILON);
    EXPECT_NEAR(trans.z(), t_camera.z(), EPSILON);

    // Verify muzzle offset
    auto muzzle       = buffer.get_muzzle_to_gimbal();
    auto muzzle_trans = muzzle.translation();
    EXPECT_NEAR(muzzle_trans.x(), t_muzzle.x(), EPSILON);
    EXPECT_NEAR(muzzle_trans.y(), t_muzzle.y(), EPSILON);
    EXPECT_NEAR(muzzle_trans.z(), t_muzzle.z(), EPSILON);
}

TEST_F(TransformBufferTest, GetTransformTemplate) {
    TransformBuffer<100> buffer(t_camera, R_camera, t_muzzle);
    buffer.update_gimbal(0.1, 0.2, 0.3, 1000);

    // Test template-based transform query
    auto gimbal  = buffer.get_transform<gimbal_link_frame, odom_frame>();
    auto camera  = buffer.get_transform<camera_link_frame, gimbal_link_frame>();
    auto optical = buffer.get_transform<camera_optical_frame_frame, camera_link_frame>();
    auto muzzle  = buffer.get_transform<muzzle_link_frame, gimbal_link_frame>();
    auto odom    = buffer.get_transform<odom_frame, world_frame>();

    // odom → world should be identity
    auto odom_trans = odom.translation();
    EXPECT_NEAR(odom_trans.x(), 0.0, EPSILON);
    EXPECT_NEAR(odom_trans.y(), 0.0, EPSILON);
    EXPECT_NEAR(odom_trans.z(), 0.0, EPSILON);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST(ErrorTest, ErrorToStringWorks) {
    EXPECT_EQ(to_string(TransformError::FrameNotFound), "frame not found");
    EXPECT_EQ(to_string(TransformError::NoDataAvailable), "no data available");
    EXPECT_EQ(to_string(TransformError::ExtrapolationRequired), "extrapolation required");
    EXPECT_EQ(to_string(TransformError::Timeout), "timeout");
    EXPECT_EQ(to_string(TransformError::PathNotFound), "path not found");
}

// ============================================================================
// Frame Hierarchy Tests
// ============================================================================

TEST(FrameTest, FrameNamesAreCorrect) {
    EXPECT_EQ(world_frame::frame_name, "world");
    EXPECT_EQ(odom_frame::frame_name, "odom");
    EXPECT_EQ(gimbal_link_frame::frame_name, "gimbal_link");
    EXPECT_EQ(camera_link_frame::frame_name, "camera_link");
    EXPECT_EQ(camera_optical_frame_frame::frame_name, "camera_optical_frame");
    EXPECT_EQ(muzzle_link_frame::frame_name, "muzzle_link");
}

TEST(FrameTest, FrameParentRelations) {
    // Check parent types
    static_assert(std::is_same_v<odom_frame::parent, world_frame>);
    static_assert(std::is_same_v<gimbal_link_frame::parent, odom_frame>);
    static_assert(std::is_same_v<camera_link_frame::parent, gimbal_link_frame>);
    static_assert(std::is_same_v<camera_optical_frame_frame::parent, camera_link_frame>);
    static_assert(std::is_same_v<muzzle_link_frame::parent, gimbal_link_frame>);

    SUCCEED();
}

TEST(FrameTest, FrameValidation) {
    // These should compile (same tree)
    static_assert(validate_frames_same_tree<camera_optical_frame_frame, world_frame>::value);
    static_assert(validate_frames_same_tree<muzzle_link_frame, odom_frame>::value);
    static_assert(validate_frames_same_tree<gimbal_link_frame, camera_link_frame>::value);

    SUCCEED();
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
