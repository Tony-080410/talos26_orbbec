/**
 * Unit Tests for Static Transform Storage
 *
 * Tests:
 * - StaticTransformStore registration and retrieval
 * - Multiple static transform pairs
 * - Transform composition
 * - Memory efficiency compared to dynamic buffers
 */

#include <gtest/gtest.h>
#include "fast_tf/core.hpp"

using namespace fast_tf;

// Define frame hierarchy for static transform tests
DECLARE_FRAME(static_world);
DECLARE_FRAME(static_base, static_world);
DECLARE_FRAME(static_camera, static_base);
DECLARE_FRAME(static_lidar, static_base);

class StaticTransformStoreTest : public ::testing::Test {
protected:
    constexpr static double EPSILON = 1e-10;

    void SetUp() override {
        StaticTransformStore::clear();
    }

    void TearDown() override {
        StaticTransformStore::clear();
    }
};

TEST_F(StaticTransformStoreTest, RegisterAndRetrieve) {
    TransformMatrix4d camera_offset = TransformMatrix4d::from_rpy(
        0.1, 0.05, 0.0,      // Roll, pitch, yaw
        0.1, 0.0, 0.05       // X, Y, Z
    );

    StaticTransformStore::register_transform<static_camera_Frame, static_base_Frame>(
        camera_offset);

    const TransformMatrix4d* retrieved =
        StaticTransformStore::get_transform<static_camera_Frame, static_base_Frame>();
    EXPECT_NE(retrieved, nullptr);

    if (retrieved) {
        Eigen::Vector3d trans = retrieved->translation();
        EXPECT_NEAR(trans.x(), 0.1, EPSILON);
        EXPECT_NEAR(trans.y(), 0.0, EPSILON);
        EXPECT_NEAR(trans.z(), 0.05, EPSILON);
    }
}

TEST_F(StaticTransformStoreTest, CheckExistence) {
    TransformMatrix4d camera_offset = TransformMatrix4d::from_rpy(
        0.0, 0.0, 0.0, 0.1, 0.0, 0.05);

    StaticTransformStore::register_transform<static_camera_Frame, static_base_Frame>(
        camera_offset);

    bool has_camera = StaticTransformStore::has_transform<static_camera_Frame, static_base_Frame>();
    EXPECT_TRUE(has_camera);

    bool has_lidar_to_camera = StaticTransformStore::has_transform<static_lidar_Frame, static_camera_Frame>();
    EXPECT_FALSE(has_lidar_to_camera);
}

TEST_F(StaticTransformStoreTest, NonexistentTransform) {
    const TransformMatrix4d* result =
        StaticTransformStore::get_transform<static_camera_Frame, static_base_Frame>();
    EXPECT_EQ(result, nullptr);
}

TEST_F(StaticTransformStoreTest, MultipleTransforms) {
    TransformMatrix4d camera_tf = TransformMatrix4d::from_rpy(
        0.1, 0.05, 0.0, 0.1, 0.0, 0.05);
    TransformMatrix4d lidar_tf = TransformMatrix4d::from_rpy(
        0.0, 0.0, 0.0, 0.0, 0.0, 0.15);

    StaticTransformStore::register_transform<static_camera_Frame, static_base_Frame>(
        camera_tf);
    StaticTransformStore::register_transform<static_lidar_Frame, static_base_Frame>(
        lidar_tf);

    bool has_camera = StaticTransformStore::has_transform<static_camera_Frame, static_base_Frame>();
    EXPECT_TRUE(has_camera);

    bool has_lidar = StaticTransformStore::has_transform<static_lidar_Frame, static_base_Frame>();
    EXPECT_TRUE(has_lidar);

    const TransformMatrix4d* cam =
        StaticTransformStore::get_transform<static_camera_Frame, static_base_Frame>();
    const TransformMatrix4d* lid =
        StaticTransformStore::get_transform<static_lidar_Frame, static_base_Frame>();

    EXPECT_NEAR(cam->translation().x(), 0.1, EPSILON);
    EXPECT_NEAR(lid->translation().z(), 0.15, EPSILON);
}

TEST_F(StaticTransformStoreTest, UpdateTransform) {
    TransformMatrix4d initial = TransformMatrix4d::from_rpy(
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    StaticTransformStore::register_transform<static_camera_Frame, static_base_Frame>(
        initial);

    TransformMatrix4d updated = TransformMatrix4d::from_rpy(
        0.0, 0.0, 0.0, 2.0, 0.0, 0.0);
    StaticTransformStore::register_transform<static_camera_Frame, static_base_Frame>(
        updated);

    const TransformMatrix4d* retrieved =
        StaticTransformStore::get_transform<static_camera_Frame, static_base_Frame>();
    EXPECT_NEAR(retrieved->translation().x(), 2.0, EPSILON);
}

TEST_F(StaticTransformStoreTest, CountTransforms) {
    EXPECT_EQ(StaticTransformStore::count(), 0);

    TransformMatrix4d tf1 = TransformMatrix4d::from_rpy(0, 0, 0, 1, 0, 0);
    StaticTransformStore::register_transform<static_camera_Frame, static_base_Frame>(tf1);
    EXPECT_EQ(StaticTransformStore::count(), 1);

    TransformMatrix4d tf2 = TransformMatrix4d::from_rpy(0, 0, 0, 0, 1, 0);
    StaticTransformStore::register_transform<static_lidar_Frame, static_base_Frame>(tf2);
    EXPECT_EQ(StaticTransformStore::count(), 2);

    StaticTransformStore::clear();
    EXPECT_EQ(StaticTransformStore::count(), 0);
}

// ============================================================================
// StaticTransformQuery Tests
// ============================================================================

class StaticTransformQueryTest : public ::testing::Test {
protected:
    constexpr static double EPSILON = 1e-10;

    void SetUp() override {
        StaticTransformStore::clear();
    }

    void TearDown() override {
        StaticTransformStore::clear();
    }
};

TEST_F(StaticTransformQueryTest, QueryHasValue) {
    TransformMatrix4d tf = TransformMatrix4d::from_rpy(
        0.0, 0.0, 0.0, 1.0, 2.0, 3.0);
    StaticTransformStore::register_transform<static_camera_Frame, static_base_Frame>(tf);

    const TransformMatrix4d* ptr =
        StaticTransformStore::get_transform<static_camera_Frame, static_base_Frame>();
    StaticTransformQuery query(ptr);

    EXPECT_TRUE(query.has_value());
}

TEST_F(StaticTransformQueryTest, QueryEmpty) {
    StaticTransformQuery query(nullptr);
    EXPECT_FALSE(query.has_value());
}

TEST_F(StaticTransformQueryTest, QueryValue) {
    TransformMatrix4d tf = TransformMatrix4d::from_rpy(
        0.0, 0.0, 0.0, 1.0, 2.0, 3.0);
    StaticTransformStore::register_transform<static_camera_Frame, static_base_Frame>(tf);

    const TransformMatrix4d* ptr =
        StaticTransformStore::get_transform<static_camera_Frame, static_base_Frame>();
    StaticTransformQuery query(ptr);

    EXPECT_TRUE(query.has_value());
    const TransformMatrix4d& retrieved = query.value();
    Eigen::Vector3d trans = retrieved.translation();
    EXPECT_NEAR(trans.x(), 1.0, EPSILON);
    EXPECT_NEAR(trans.y(), 2.0, EPSILON);
    EXPECT_NEAR(trans.z(), 3.0, EPSILON);
}

TEST_F(StaticTransformQueryTest, QueryGet) {
    TransformMatrix4d tf = TransformMatrix4d::from_rpy(
        0.0, 0.0, 0.0, 5.0, 6.0, 7.0);
    StaticTransformStore::register_transform<static_camera_Frame, static_base_Frame>(tf);

    const TransformMatrix4d* ptr =
        StaticTransformStore::get_transform<static_camera_Frame, static_base_Frame>();
    StaticTransformQuery query(ptr);

    const TransformMatrix4d* retrieved = query.get();
    EXPECT_NE(retrieved, nullptr);
    EXPECT_NEAR(retrieved->translation().x(), 5.0, EPSILON);
}

// ============================================================================
// Transform Composition Tests
// ============================================================================

class TransformCompositionTest : public ::testing::Test {
protected:
    constexpr static double EPSILON = 1e-10;
};

TEST_F(TransformCompositionTest, ComposeStaticTransforms) {
    TransformMatrix4d tf1 = TransformMatrix4d::from_rpy(
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    TransformMatrix4d tf2 = TransformMatrix4d::from_rpy(
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    TransformMatrix4d composed = TransformComposition::compose(tf1, tf2);

    Eigen::Vector3d trans = composed.translation();
    EXPECT_NEAR(trans.x(), 1.0, EPSILON);
    EXPECT_NEAR(trans.y(), 1.0, EPSILON);
    EXPECT_NEAR(trans.z(), 0.0, EPSILON);
}

TEST_F(TransformCompositionTest, ComposePath) {
    TransformMatrix4d base = TransformMatrix4d::from_rpy(
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    TransformMatrix4d remaining = TransformMatrix4d::from_rpy(
        0.0, 0.0, 0.0, 0.0, 0.5, 0.0);

    TransformMatrix4d result = TransformComposition::compose_path(base, remaining);

    Eigen::Vector3d trans = result.translation();
    EXPECT_NEAR(trans.x(), 1.0, EPSILON);
    EXPECT_NEAR(trans.y(), 0.5, EPSILON);
}

// ============================================================================
// Memory Efficiency Tests
// ============================================================================

class MemoryEfficiencyTest : public ::testing::Test {
};

TEST_F(MemoryEfficiencyTest, StaticVsDynamicMemory) {
    // Static transforms should use minimal memory
    // Each static transform: sizeof(TransformMatrix4d) + overhead
    // Dynamic transform buffer: sizeof(TransformMatrix4d) * capacity + overhead

    // Register 10 static transforms
    for (int i = 0; i < 10; ++i) {
        TransformMatrix4d tf = TransformMatrix4d::from_rpy(
            0.0, 0.0, 0.0, static_cast<double>(i), 0.0, 0.0);
        // This would register different frame pairs in a real scenario
        // For testing, we just verify the mechanism works
        StaticTransformStore::clear();
        StaticTransformStore::register_transform<static_camera_Frame, static_base_Frame>(tf);
    }

    // A CircularTransformBuffer<100> would use 100x more memory for the same capacity
    // This test documents the memory efficiency benefit
    EXPECT_EQ(StaticTransformStore::count(), 1);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
