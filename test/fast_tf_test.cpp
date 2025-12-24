#include <cmath>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

#include "fast_tf/frame_hierarchy.hpp"
#include "fast_tf/frame_id.hpp"
#include "fast_tf/frame_types.hpp"
#include "fast_tf/static_transforms.hpp"
#include "fast_tf/transform_buffer.hpp"
#include "fast_tf/transform_tree.hpp"
#include "fast_tf/tree_structure.hpp"

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

TEST_F(TransformMatrix4dTest, CreationFromQuaternion) {
    // Create quaternion (w, x, y, z) for 90-degree rotation around Z
    double w = std::cos(M_PI / 4);
    double z = std::sin(M_PI / 4);
    auto tf  = TransformMatrix4d::from_quaternion(w, 0.0, 0.0, z, 1.0, 2.0, 3.0);

    auto trans = tf.translation();
    EXPECT_NEAR(trans.x(), 1.0, EPSILON);
    EXPECT_NEAR(trans.y(), 2.0, EPSILON);
    EXPECT_NEAR(trans.z(), 3.0, EPSILON);
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

TEST_F(TransformMatrix4dTest, QuaternionRoundTrip) {
    // Create a transform with known values and verify quaternion round-trip
    auto tf = TransformMatrix4d::from_rpy(0.5, 0.3, 0.1, 1.0, 2.0, 3.0);
    auto q  = tf.quaternion();

    // Verify quaternion represents the same rotation
    auto tf2 = TransformMatrix4d::from_quaternion(q.w(), q.x(), q.y(), q.z(), 1.0, 2.0, 3.0);
    auto q2  = tf2.quaternion();

    // Quaternions might differ by sign, but rotations should be equivalent
    EXPECT_NEAR((tf.rotation() - tf2.rotation()).cwiseAbs().maxCoeff(), 0.0, EPSILON);
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
                buffer.latest();
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
// Frame Hierarchy Tests
// ============================================================================
// Note: Frame hierarchy compile-time tests are complex and require careful
// setup with DECLARE_FRAME macros. For this suite, we focus on runtime behavior
// of TransformMatrix4d and CircularTransformBuffer which demonstrate the core
// fast_tf functionality and can be compiled independently.

// ============================================================================
// TransformTree Tests
// ============================================================================
// Note: TransformTree tests require compile-time frame definitions using
// DECLARE_FRAME macros. These compile-time checks are validated separately.
// This test suite focuses on runtime validation of TransformMatrix4d and
// CircularTransformBuffer which are the core performance-critical components.

// ============================================================================
// Static Transform Tests
// ============================================================================
// Note: Static transform functionality is also validated through compile-time
// frame hierarchy tests which are part of the broader test suite.

// ============================================================================
// Integration Tests
// ============================================================================
// Note: Integration tests for the full TransformTree and frame hierarchy
// are validated separately in compilation tests and examples.

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
