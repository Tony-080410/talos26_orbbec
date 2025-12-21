/**
 * Unit Tests for Frame Types and Transform Data Structures
 *
 * Tests:
 * - RPYTransformData: validation and properties
 * - QuaternionTransformData: validation and quaternion operations
 * - TransformMatrix4d: matrix operations, composition, interpolation
 * - Transform conversions between formats
 */

#include <gtest/gtest.h>
#include <cmath>
#include "fast_tf/frame_types.hpp"

using namespace fast_tf;

// ============================================================================
// RPYTransformData Tests
// ============================================================================

class RPYTransformDataTest : public ::testing::Test {
protected:
    constexpr static double EPSILON = 1e-10;
};

TEST_F(RPYTransformDataTest, DefaultConstructor) {
    RPYTransformData data;
    EXPECT_EQ(data.roll, 0.0);
    EXPECT_EQ(data.pitch, 0.0);
    EXPECT_EQ(data.yaw, 0.0);
    EXPECT_EQ(data.x, 0.0);
    EXPECT_EQ(data.y, 0.0);
    EXPECT_EQ(data.z, 0.0);
}

TEST_F(RPYTransformDataTest, ParameterizedConstructor) {
    RPYTransformData data(0.1, 0.2, 0.3, 1.0, 2.0, 3.0);
    EXPECT_DOUBLE_EQ(data.roll, 0.1);
    EXPECT_DOUBLE_EQ(data.pitch, 0.2);
    EXPECT_DOUBLE_EQ(data.yaw, 0.3);
    EXPECT_DOUBLE_EQ(data.x, 1.0);
    EXPECT_DOUBLE_EQ(data.y, 2.0);
    EXPECT_DOUBLE_EQ(data.z, 3.0);
}

TEST_F(RPYTransformDataTest, IsValid) {
    RPYTransformData data(0.1, 0.2, 0.3, 1.0, 2.0, 3.0);
    EXPECT_TRUE(data.is_valid());
}

// ============================================================================
// QuaternionTransformData Tests
// ============================================================================

class QuaternionTransformDataTest : public ::testing::Test {
protected:
    constexpr static double EPSILON = 1e-6;
};

TEST_F(QuaternionTransformDataTest, DefaultConstructor) {
    QuaternionTransformData data;
    EXPECT_DOUBLE_EQ(data.w, 1.0);  // Default: identity quaternion
    EXPECT_EQ(data.x, 0.0);
    EXPECT_EQ(data.y, 0.0);
    EXPECT_EQ(data.z, 0.0);
    EXPECT_EQ(data.tx, 0.0);
    EXPECT_EQ(data.ty, 0.0);
    EXPECT_EQ(data.tz, 0.0);
}

TEST_F(QuaternionTransformDataTest, ParameterizedConstructor) {
    QuaternionTransformData data(1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0);
    EXPECT_DOUBLE_EQ(data.w, 1.0);
    EXPECT_DOUBLE_EQ(data.x, 0.0);
    EXPECT_DOUBLE_EQ(data.y, 0.0);
    EXPECT_DOUBLE_EQ(data.z, 0.0);
    EXPECT_DOUBLE_EQ(data.tx, 1.0);
    EXPECT_DOUBLE_EQ(data.ty, 2.0);
    EXPECT_DOUBLE_EQ(data.tz, 3.0);
}

TEST_F(QuaternionTransformDataTest, IsValidUnitQuaternion) {
    // Unit quaternion (identity rotation)
    QuaternionTransformData data(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    EXPECT_TRUE(data.is_valid());
}

TEST_F(QuaternionTransformDataTest, IsInvalidNonUnitQuaternion) {
    // Non-unit quaternion
    QuaternionTransformData data(2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    EXPECT_FALSE(data.is_valid());
}

TEST_F(QuaternionTransformDataTest, Normalize) {
    QuaternionTransformData data(2.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0);
    data.normalize();

    // Check that quaternion is now unit (norm = 1)
    double norm = data.w * data.w + data.x * data.x +
                  data.y * data.y + data.z * data.z;
    EXPECT_NEAR(norm, 1.0, EPSILON);
}

TEST_F(QuaternionTransformDataTest, NormalizeAlreadyUnit) {
    QuaternionTransformData data(1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0);
    data.normalize();

    // Should remain unchanged
    EXPECT_DOUBLE_EQ(data.w, 1.0);
    EXPECT_DOUBLE_EQ(data.x, 0.0);
    EXPECT_DOUBLE_EQ(data.y, 0.0);
    EXPECT_DOUBLE_EQ(data.z, 0.0);
}

// ============================================================================
// TransformMatrix4d Tests
// ============================================================================

class TransformMatrix4dTest : public ::testing::Test {
protected:
    constexpr static double EPSILON = 1e-10;
};

TEST_F(TransformMatrix4dTest, DefaultConstructor) {
    TransformMatrix4d tf;
    EXPECT_TRUE(tf.matrix().isIdentity());
}

TEST_F(TransformMatrix4dTest, MatrixConstructor) {
    Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
    m(0, 3) = 1.0;
    m(1, 3) = 2.0;
    m(2, 3) = 3.0;

    TransformMatrix4d tf(m);
    Eigen::Vector3d trans = tf.translation();
    EXPECT_DOUBLE_EQ(trans.x(), 1.0);
    EXPECT_DOUBLE_EQ(trans.y(), 2.0);
    EXPECT_DOUBLE_EQ(trans.z(), 3.0);
}

TEST_F(TransformMatrix4dTest, FromRPY) {
    // Zero rotation, pure translation
    TransformMatrix4d tf = TransformMatrix4d::from_rpy(0, 0, 0, 1.0, 2.0, 3.0);

    Eigen::Vector3d trans = tf.translation();
    EXPECT_NEAR(trans.x(), 1.0, EPSILON);
    EXPECT_NEAR(trans.y(), 2.0, EPSILON);
    EXPECT_NEAR(trans.z(), 3.0, EPSILON);

    // Check rotation is identity
    EXPECT_TRUE(tf.rotation().isUnitary());
}

TEST_F(TransformMatrix4dTest, FromQuaternion) {
    // Identity quaternion (no rotation), translation only
    TransformMatrix4d tf = TransformMatrix4d::from_quaternion(1, 0, 0, 0, 1.0, 2.0, 3.0);

    Eigen::Vector3d trans = tf.translation();
    EXPECT_NEAR(trans.x(), 1.0, EPSILON);
    EXPECT_NEAR(trans.y(), 2.0, EPSILON);
    EXPECT_NEAR(trans.z(), 3.0, EPSILON);
}

TEST_F(TransformMatrix4dTest, Composition) {
    TransformMatrix4d tf1 = TransformMatrix4d::from_rpy(0, 0, 0, 1.0, 0, 0);
    TransformMatrix4d tf2 = TransformMatrix4d::from_rpy(0, 0, 0, 0, 1.0, 0);

    TransformMatrix4d composed = tf1 * tf2;

    Eigen::Vector3d trans = composed.translation();
    // Composition should combine translations
    EXPECT_NEAR(trans.x(), 1.0, EPSILON);
    EXPECT_NEAR(trans.y(), 1.0, EPSILON);
    EXPECT_NEAR(trans.z(), 0.0, EPSILON);
}

TEST_F(TransformMatrix4dTest, Inverse) {
    TransformMatrix4d tf = TransformMatrix4d::from_rpy(0.1, 0.2, 0.3, 1.0, 2.0, 3.0);
    TransformMatrix4d inv = tf.inverse();

    // tf * inv should be identity
    TransformMatrix4d result = tf * inv;
    EXPECT_TRUE(result.matrix().isIdentity(EPSILON));
}

TEST_F(TransformMatrix4dTest, GetTranslation) {
    TransformMatrix4d tf = TransformMatrix4d::from_rpy(0, 0, 0, 1.5, 2.5, 3.5);
    Eigen::Vector3d trans = tf.translation();

    EXPECT_NEAR(trans.x(), 1.5, EPSILON);
    EXPECT_NEAR(trans.y(), 2.5, EPSILON);
    EXPECT_NEAR(trans.z(), 3.5, EPSILON);
}

TEST_F(TransformMatrix4dTest, GetQuaternion) {
    // Identity transform - should get identity quaternion
    TransformMatrix4d tf = TransformMatrix4d::from_rpy(0, 0, 0, 0, 0, 0);
    Eigen::Quaterniond quat = tf.quaternion();

    // Identity quaternion is (1, 0, 0, 0) in (w, x, y, z) format
    EXPECT_NEAR(std::abs(quat.w()), 1.0, EPSILON);
    EXPECT_NEAR(quat.x(), 0.0, EPSILON);
    EXPECT_NEAR(quat.y(), 0.0, EPSILON);
    EXPECT_NEAR(quat.z(), 0.0, EPSILON);
}

TEST_F(TransformMatrix4dTest, LinearInterpolation) {
    TransformMatrix4d tf1 = TransformMatrix4d::from_rpy(0, 0, 0, 0, 0, 0);
    TransformMatrix4d tf2 = TransformMatrix4d::from_rpy(0, 0, 0, 2.0, 0, 0);

    // Interpolate at t=0.5
    TransformMatrix4d lerped = TransformMatrix4d::lerp(tf1, tf2, 0.5);

    Eigen::Vector3d trans = lerped.translation();
    EXPECT_NEAR(trans.x(), 1.0, EPSILON);  // Should be halfway between
    EXPECT_NEAR(trans.y(), 0.0, EPSILON);
    EXPECT_NEAR(trans.z(), 0.0, EPSILON);
}

TEST_F(TransformMatrix4dTest, InterpolationBoundaries) {
    TransformMatrix4d tf1 = TransformMatrix4d::from_rpy(0, 0, 0, 0, 0, 0);
    TransformMatrix4d tf2 = TransformMatrix4d::from_rpy(0, 0, 0, 1.0, 1.0, 1.0);

    // At t=0, should be tf1
    TransformMatrix4d at_0 = TransformMatrix4d::lerp(tf1, tf2, 0.0);
    EXPECT_TRUE(at_0.matrix().isApprox(tf1.matrix(), EPSILON));

    // At t=1, should be tf2
    TransformMatrix4d at_1 = TransformMatrix4d::lerp(tf1, tf2, 1.0);
    EXPECT_TRUE(at_1.matrix().isApprox(tf2.matrix(), EPSILON));
}

// ============================================================================
// FrameRegistry Tests
// ============================================================================

class FrameRegistryTest : public ::testing::Test {
protected:
    void SetUp() override {
        FrameRegistry::clear();
    }

    void TearDown() override {
        FrameRegistry::clear();
    }
};

TEST_F(FrameRegistryTest, RegisterFrame) {
    FrameRegistry::register_frame("world", 1);
    EXPECT_EQ(FrameRegistry::get_frame_name(1), "world");
}

TEST_F(FrameRegistryTest, RegisterAlias) {
    FrameRegistry::register_frame("world", 1);
    FrameRegistry::register_alias("root", "world");

    EXPECT_EQ(FrameRegistry::resolve_alias("root"), "world");
}

TEST_F(FrameRegistryTest, GetFrameIdByName) {
    FrameRegistry::register_frame("base_link", 2);
    uint64_t id = FrameRegistry::get_frame_id("base_link");
    EXPECT_EQ(id, 2);
}

TEST_F(FrameRegistryTest, UnknownFrameName) {
    EXPECT_EQ(FrameRegistry::get_frame_name(9999), "unknown_frame");
}

TEST_F(FrameRegistryTest, ClearRegistry) {
    FrameRegistry::register_frame("world", 1);
    FrameRegistry::clear();
    EXPECT_EQ(FrameRegistry::get_frame_name(1), "unknown_frame");
}

// ============================================================================
// Conversion Tests
// ============================================================================

class ConversionTest : public ::testing::Test {
protected:
    constexpr static double EPSILON = 1e-6;
};

TEST_F(ConversionTest, RPYToMatrixAndBack) {
    double roll = 0.1, pitch = 0.2, yaw = 0.3;
    double x = 1.0, y = 2.0, z = 3.0;

    // Create transform from RPY
    TransformMatrix4d tf = TransformMatrix4d::from_rpy(roll, pitch, yaw, x, y, z);

    // Extract translation
    Eigen::Vector3d trans = tf.translation();
    EXPECT_NEAR(trans.x(), x, EPSILON);
    EXPECT_NEAR(trans.y(), y, EPSILON);
    EXPECT_NEAR(trans.z(), z, EPSILON);

    // Extract quaternion
    Eigen::Quaterniond quat = tf.quaternion();
    EXPECT_TRUE(quat.norm() >= 0.999);  // Should be unit quaternion
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
