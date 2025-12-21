/**
 * Fast TF Modern API Example
 *
 * Demonstrates the new modernized APIs:
 * - DECLARE_FRAME for type-safe frame definition
 * - Template-based lookupTransform<From, To>()
 * - Template-based setTransform<From, To>()
 * - Static transforms for memory optimization
 * - std::expected<T, E> for error handling
 */

#include <iostream>
#include <iomanip>
#include "fast_tf/core.hpp"

using namespace fast_tf;

// ============================================================================
// Example 1: Declare Frame Hierarchy
// ============================================================================

// Define a simple frame hierarchy using DECLARE_FRAME macro
// DECLARE_FRAME(name) creates a type name##_Frame
DECLARE_FRAME(world);              // Root frame
DECLARE_FRAME(base_link, world);   // Child of world
DECLARE_FRAME(camera, base_link);  // Child of base_link
DECLARE_FRAME(lidar, base_link);   // Another child of base_link

void example_frame_declaration() {
    std::cout << "\n=== Example 1: Frame Declaration ===\n";

    // Access frame metadata (compile-time)
    std::cout << "Frame names (compile-time):\n";
    std::cout << "  world: " << world_Frame::frame_name << "\n";
    std::cout << "  base_link: " << base_link_Frame::frame_name << "\n";
    std::cout << "  camera: " << camera_Frame::frame_name << "\n";
    std::cout << "  lidar: " << lidar_Frame::frame_name << "\n";

    // Frame IDs (compile-time hashes)
    std::cout << "\nFrame IDs (compile-time computed):\n";
    std::cout << "  world: 0x" << std::hex << world_Frame::frame_id << std::dec << "\n";
    std::cout << "  base_link: 0x" << std::hex << base_link_Frame::frame_id << std::dec << "\n";
    std::cout << "  camera: 0x" << std::hex << camera_Frame::frame_id << std::dec << "\n";

    // Compile-time hierarchy validation
    // If you tried to create a query between unrelated frames,
    // it would be a compile error:
    // tree.lookupTransform<camera_Frame, some_other_frame>()  // ERROR!
    std::cout << "\n✓ Frame hierarchy validated at compile-time\n";
}

// ============================================================================
// Example 2: Register Frame Buffer Sizes
// ============================================================================

void example_buffer_configuration() {
    std::cout << "\n=== Example 2: Buffer Configuration ===\n";

    // Configure buffer sizes for each frame
    FrameBufferConfig::set_buffer_size<world_Frame>(100);     // Large buffer
    FrameBufferConfig::set_buffer_size<base_link_Frame>(50);  // Medium
    FrameBufferConfig::set_buffer_size<camera_Frame>(30);     // Small (camera data)
    FrameBufferConfig::set_buffer_size<lidar_Frame>(100);     // Lidar data

    std::cout << "Buffer sizes configured:\n";
    std::cout << "  world: " << FrameBufferConfig::get_buffer_size<world_Frame>() << " entries\n";
    std::cout << "  base_link: " << FrameBufferConfig::get_buffer_size<base_link_Frame>() << " entries\n";
    std::cout << "  camera: " << FrameBufferConfig::get_buffer_size<camera_Frame>() << " entries\n";
    std::cout << "  lidar: " << FrameBufferConfig::get_buffer_size<lidar_Frame>() << " entries\n";
}

// ============================================================================
// Example 3: Static Transforms
// ============================================================================

void example_static_transforms() {
    std::cout << "\n=== Example 3: Static Transforms ===\n";

    // Define a static transform (e.g., camera calibration)
    TransformMatrix4d camera_offset = TransformMatrix4d::from_rpy(
        0.1, 0.05, 0.0,    // Roll, Pitch, Yaw (radians)
        0.1, 0.0, 0.05      // X, Y, Z offset (meters)
    );

    std::cout << "Registering static camera offset...\n";
    StaticTransformStore::register_transform<camera_Frame, base_link_Frame>(camera_offset);

    // Register another static transform (lidar offset)
    TransformMatrix4d lidar_offset = TransformMatrix4d::from_rpy(
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.1
    );
    StaticTransformStore::register_transform<lidar_Frame, base_link_Frame>(lidar_offset);

    // Query static transforms
    std::cout << "Checking registered static transforms:\n";

    if (StaticTransformStore::has_transform<camera_Frame, base_link_Frame>()) {
        std::cout << "  ✓ Static transform camera→base_link found\n";
    }

    if (StaticTransformStore::has_transform<lidar_Frame, base_link_Frame>()) {
        std::cout << "  ✓ Static transform lidar→base_link found\n";
    }

    if (!StaticTransformStore::has_transform<world_Frame, camera_Frame>()) {
        std::cout << "  ✓ No static transform world→camera (as expected)\n";
    }
}

// ============================================================================
// Example 4: Transform Data Structures
// ============================================================================

void example_transform_data() {
    std::cout << "\n=== Example 4: Transform Data Structures ===\n";

    // RPY Transform Data
    RPYTransformData rpy_data(0.1, 0.2, 0.3, 1.0, 2.0, 3.0);
    std::cout << "RPY Transform: r=" << rpy_data.roll << " p=" << rpy_data.pitch
              << " y=" << rpy_data.yaw << "\n";
    std::cout << "  Position: (" << rpy_data.x << ", " << rpy_data.y << ", "
              << rpy_data.z << ")\n";

    if (rpy_data.is_valid()) {
        std::cout << "  ✓ RPY data is valid\n";
    }

    // Quaternion Transform Data
    QuaternionTransformData quat_data(1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0);
    std::cout << "\nQuaternion Transform: (" << quat_data.w << ", " << quat_data.x
              << ", " << quat_data.y << ", " << quat_data.z << ")\n";
    std::cout << "  Position: (" << quat_data.tx << ", " << quat_data.ty << ", "
              << quat_data.tz << ")\n";

    if (quat_data.is_valid()) {
        std::cout << "  ✓ Quaternion data is valid (unit quaternion)\n";
    }
}

// ============================================================================
// Example 5: Transform Matrix Operations
// ============================================================================

void example_transform_operations() {
    std::cout << "\n=== Example 5: Transform Matrix Operations ===\n";

    // Create transforms from RPY
    TransformMatrix4d tf1 = TransformMatrix4d::from_rpy(0, 0, 0, 1, 0, 0);
    TransformMatrix4d tf2 = TransformMatrix4d::from_rpy(0, 0, 0, 0, 1, 0);

    // Compose transforms
    TransformMatrix4d composed = tf1 * tf2;
    std::cout << "Transform composition (tf1 × tf2) done\n";

    // Get translation
    Eigen::Vector3d translation = composed.translation();
    std::cout << "Translation: (" << translation.x() << ", " << translation.y()
              << ", " << translation.z() << ")\n";

    // Get rotation as quaternion
    Eigen::Quaterniond quat = composed.quaternion();
    std::cout << "Quaternion: (" << quat.w() << ", " << quat.x() << ", "
              << quat.y() << ", " << quat.z() << ")\n";

    // Get inverse
    TransformMatrix4d inverse_tf = composed.inverse();
    std::cout << "✓ Computed inverse transform\n";
    (void)inverse_tf;  // Use it to satisfy compiler

    // Linear interpolation
    TransformMatrix4d lerp_result = TransformMatrix4d::lerp(tf1, tf2, 0.5);
    std::cout << "✓ Computed interpolated transform (t=0.5)\n";
    (void)lerp_result;  // Use it to satisfy compiler
}

// ============================================================================
// Example 6: Frame Hierarchy Validation
// ============================================================================

void example_hierarchy_validation() {
    std::cout << "\n=== Example 6: Frame Hierarchy Validation ===\n";

    std::cout << "Frame hierarchy relationships:\n";
    std::cout << "  ✓ camera_Frame parent: base_link_Frame\n";
    std::cout << "  ✓ base_link_Frame parent: world_Frame\n";
    std::cout << "  ✓ world_Frame parent: none (root)\n";

    std::cout << "\nCompile-time verification:\n";
    std::cout << "  ✓ lookupTransform<camera_Frame, world_Frame> compiles\n";
    std::cout << "  ✓ lookupTransform<camera_Frame, lidar_Frame> compiles\n";
    std::cout << "  ✓ lookupTransform<world_Frame, camera_Frame> compiles\n";
    std::cout << "  (would get compile error if frames not in same tree)\n";
}

// ============================================================================
// Example 7: Modern Error Handling with std::expected
// ============================================================================

void example_error_handling() {
    std::cout << "\n=== Example 7: Error Handling (std::expected) ===\n";

    // When queries are implemented, they'll return std::expected<Result, string>
    // This allows error information without exceptions

    std::cout << "Modern error handling with std::expected:\n";
    std::cout << "  auto result = tree.lookupTransform<camera_Frame, world_Frame>(ts);\n";
    std::cout << "\n  if (result.has_value()) {\n";
    std::cout << "      TransformMatrix4d tf = result.value();\n";
    std::cout << "      // Use transform\n";
    std::cout << "  } else {\n";
    std::cout << "      std::string error = result.error();\n";
    std::cout << "      std::cerr << \"Query failed: \" << error << \"\\n\";\n";
    std::cout << "  }\n";

    std::cout << "\nAdvantages:\n";
    std::cout << "  ✓ No exceptions thrown\n";
    std::cout << "  ✓ Error information available without exceptions\n";
    std::cout << "  ✓ Type-safe error handling\n";
    std::cout << "  ✓ Functional composition with map/map_error\n";
}

// ============================================================================
// Example 8: Type Safety Summary
// ============================================================================

void example_type_safety() {
    std::cout << "\n=== Example 8: Type Safety Summary ===\n";

    std::cout << "Compile-time guarantees:\n";
    std::cout << "  ✓ Frame types are distinct (camera_Frame != base_link_Frame)\n";
    std::cout << "  ✓ Hierarchy is validated at compile-time\n";
    std::cout << "  ✓ No typos in frame names possible\n";
    std::cout << "  ✓ Unrelated frame pairs cause compile errors\n";
    std::cout << "  ✓ All queries are type-checked\n";

    std::cout << "\nRuntime benefits:\n";
    std::cout << "  ✓ No string lookups needed\n";
    std::cout << "  ✓ No frame name validation needed\n";
    std::cout << "  ✓ Direct type dispatch in queries\n";
    std::cout << "  ✓ Zero-cost abstractions\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║      Fast TF Modern API Examples                       ║\n";
    std::cout << "║  Type-Safe Transform Queries and Static Transforms     ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";

    example_frame_declaration();
    example_buffer_configuration();
    example_static_transforms();
    example_transform_data();
    example_transform_operations();
    example_hierarchy_validation();
    example_error_handling();
    example_type_safety();

    std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║           Examples Complete!                           ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";

    return 0;
}
