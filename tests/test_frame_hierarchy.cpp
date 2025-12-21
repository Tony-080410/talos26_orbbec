/**
 * Unit Tests for Frame Hierarchy System
 *
 * Tests:
 * - Frame type creation and hierarchy definition
 * - Compile-time hierarchy validation
 * - Parent-child relationships
 * - Frame buffer configuration
 */

#include <gtest/gtest.h>
#include "fast_tf/core.hpp"

using namespace fast_tf;

// ============================================================================
// Frame Definition Tests
// ============================================================================

// Define a simple frame hierarchy for testing
DECLARE_FRAME(test_world);
DECLARE_FRAME(test_base_link, test_world);
DECLARE_FRAME(test_camera, test_base_link);
DECLARE_FRAME(test_lidar, test_base_link);

class FrameHierarchyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Clear any previous configuration
    }
};

TEST_F(FrameHierarchyTest, FrameNameConstants) {
    EXPECT_STREQ(test_world_Frame::frame_name.data(), "test_world");
    EXPECT_STREQ(test_base_link_Frame::frame_name.data(), "test_base_link");
    EXPECT_STREQ(test_camera_Frame::frame_name.data(), "test_camera");
    EXPECT_STREQ(test_lidar_Frame::frame_name.data(), "test_lidar");
}

TEST_F(FrameHierarchyTest, FrameIdUniqueness) {
    // Each frame should have a unique ID
    uint64_t world_id = test_world_Frame::frame_id;
    uint64_t base_link_id = test_base_link_Frame::frame_id;
    uint64_t camera_id = test_camera_Frame::frame_id;
    uint64_t lidar_id = test_lidar_Frame::frame_id;

    // All IDs should be different
    EXPECT_NE(world_id, base_link_id);
    EXPECT_NE(world_id, camera_id);
    EXPECT_NE(world_id, lidar_id);
    EXPECT_NE(base_link_id, camera_id);
    EXPECT_NE(base_link_id, lidar_id);
    EXPECT_NE(camera_id, lidar_id);
}

TEST_F(FrameHierarchyTest, FrameIdConsistency) {
    // Frame IDs should be consistent across multiple accesses
    uint64_t id1 = test_world_Frame::frame_id;
    uint64_t id2 = test_world_Frame::frame_id;
    EXPECT_EQ(id1, id2);
}

TEST_F(FrameHierarchyTest, FrameIdNonZero) {
    // Frame IDs should be non-zero (hash functions should not collide with 0)
    EXPECT_NE(test_world_Frame::frame_id, 0);
    EXPECT_NE(test_base_link_Frame::frame_id, 0);
    EXPECT_NE(test_camera_Frame::frame_id, 0);
}

// ============================================================================
// Buffer Configuration Tests
// ============================================================================

class FrameBufferConfigTest : public ::testing::Test {
protected:
    void TearDown() override {
        // Clear configuration after each test
        FrameBufferConfig::set_buffer_size<test_world_Frame>(100);
        FrameBufferConfig::set_buffer_size<test_base_link_Frame>(100);
        FrameBufferConfig::set_buffer_size<test_camera_Frame>(100);
        FrameBufferConfig::set_buffer_size<test_lidar_Frame>(100);
    }
};

TEST_F(FrameBufferConfigTest, SetAndGetBufferSize) {
    FrameBufferConfig::set_buffer_size<test_world_Frame>(100);
    EXPECT_EQ(FrameBufferConfig::get_buffer_size<test_world_Frame>(), 100);
}

TEST_F(FrameBufferConfigTest, DifferentBufferSizes) {
    FrameBufferConfig::set_buffer_size<test_world_Frame>(100);
    FrameBufferConfig::set_buffer_size<test_camera_Frame>(50);
    FrameBufferConfig::set_buffer_size<test_lidar_Frame>(200);

    EXPECT_EQ(FrameBufferConfig::get_buffer_size<test_world_Frame>(), 100);
    EXPECT_EQ(FrameBufferConfig::get_buffer_size<test_camera_Frame>(), 50);
    EXPECT_EQ(FrameBufferConfig::get_buffer_size<test_lidar_Frame>(), 200);
}

TEST_F(FrameBufferConfigTest, DefaultBufferSize) {
    // When not explicitly configured, should return default (100)
    // Using a pre-declared frame
    EXPECT_EQ(FrameBufferConfig::get_buffer_size<test_world_Frame>(), 100);
}

TEST_F(FrameBufferConfigTest, UpdateBufferSize) {
    FrameBufferConfig::set_buffer_size<test_camera_Frame>(30);
    EXPECT_EQ(FrameBufferConfig::get_buffer_size<test_camera_Frame>(), 30);

    FrameBufferConfig::set_buffer_size<test_camera_Frame>(60);
    EXPECT_EQ(FrameBufferConfig::get_buffer_size<test_camera_Frame>(), 60);
}

// ============================================================================
// Frame Registry Tests
// ============================================================================

class FrameRegistryHierarchyTest : public ::testing::Test {
protected:
    void SetUp() override {
        FrameRegistry::clear();
    }

    void TearDown() override {
        FrameRegistry::clear();
    }
};

TEST_F(FrameRegistryHierarchyTest, RegisterMultipleFrames) {
    FrameRegistry::register_frame("world", 1);
    FrameRegistry::register_frame("base_link", 2);
    FrameRegistry::register_frame("camera", 3);

    EXPECT_EQ(FrameRegistry::get_frame_name(1), "world");
    EXPECT_EQ(FrameRegistry::get_frame_name(2), "base_link");
    EXPECT_EQ(FrameRegistry::get_frame_name(3), "camera");
}

TEST_F(FrameRegistryHierarchyTest, AliasResolution) {
    FrameRegistry::register_frame("base_link", 2);
    FrameRegistry::register_alias("robot", "base_link");
    FrameRegistry::register_alias("platform", "base_link");

    EXPECT_EQ(FrameRegistry::resolve_alias("robot"), "base_link");
    EXPECT_EQ(FrameRegistry::resolve_alias("platform"), "base_link");
    EXPECT_EQ(FrameRegistry::resolve_alias("base_link"), "base_link");
}

TEST_F(FrameRegistryHierarchyTest, UnresolvedAlias) {
    // Alias that doesn't exist should return the input as-is
    EXPECT_EQ(FrameRegistry::resolve_alias("unknown"), "unknown");
}

// ============================================================================
// Compile-Time Hierarchy Validation
// ============================================================================

// Note: Most compile-time validations would be caught during compilation
// We test the runtime aspects here

class HierarchyValidationTest : public ::testing::Test {
};

TEST_F(HierarchyValidationTest, FrameTypesAreDistinct) {
    // Each frame type should be a distinct type
    static_assert(!std::is_same_v<test_world_Frame, test_base_link_Frame>);
    static_assert(!std::is_same_v<test_world_Frame, test_camera_Frame>);
    static_assert(!std::is_same_v<test_base_link_Frame, test_camera_Frame>);
    static_assert(!std::is_same_v<test_camera_Frame, test_lidar_Frame>);
}

TEST_F(HierarchyValidationTest, FrameNameStringsAreCorrect) {
    // Frame names should match their declaration
    EXPECT_EQ(std::string(test_world_Frame::frame_name), "test_world");
    EXPECT_EQ(std::string(test_base_link_Frame::frame_name), "test_base_link");
    EXPECT_EQ(std::string(test_camera_Frame::frame_name), "test_camera");
}

// ============================================================================
// Multiple Hierarchy Tests
// ============================================================================

// Test that we can have multiple independent frame hierarchies
DECLARE_FRAME(robot1_world);
DECLARE_FRAME(robot1_base, robot1_world);

DECLARE_FRAME(robot2_world);
DECLARE_FRAME(robot2_base, robot2_world);

TEST_F(HierarchyValidationTest, MultipleHierarchies) {
    EXPECT_NE(robot1_world_Frame::frame_id, robot2_world_Frame::frame_id);
    EXPECT_NE(robot1_base_Frame::frame_id, robot2_base_Frame::frame_id);

    // Each world should have different IDs
    EXPECT_EQ(std::string(robot1_world_Frame::frame_name), "robot1_world");
    EXPECT_EQ(std::string(robot2_world_Frame::frame_name), "robot2_world");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
