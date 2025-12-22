/**
 * Infantry TF Integration Example with Foxglove
 *
 * Demonstrates:
 * - Type-safe frame hierarchy using fast_tf
 * - Direct Foxglove SDK integration
 * - Dynamic gimbal tracking
 * - Odometry updates
 * - Camera calibration
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <memory>
#include <thread>
#include <chrono>

#include "fast_tf/infantry_tf.hpp"
#include <foxglove/foxglove.hpp>
#include <foxglove/server.hpp>
#include <foxglove/schemas.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

using namespace fast_tf;
using namespace fast_tf::infantry;

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Simulate gimbal tracking a target.
 * In real code, this would come from your target tracking algorithm.
 */
struct GimbalState {
    double yaw = 0.0;
    double pitch = 0.0;
    double timestamp = 0.0;

    void simulate_tracking(double elapsed_time) {
        // Simulate smooth gimbal movement (sinusoidal tracking)
        const double frequency = 0.5;  // Hz
        const double max_yaw = M_PI / 4;  // ±45°
        const double max_pitch = M_PI / 6;  // ±30°

        yaw = max_yaw * std::sin(2 * M_PI * frequency * elapsed_time);
        pitch = max_pitch * std::cos(2 * M_PI * frequency * elapsed_time);
        timestamp = elapsed_time;
    }

    void print() const {
        std::cout << "  Gimbal State: yaw=" << std::fixed << std::setprecision(4)
                  << yaw << " rad (" << (yaw * 180 / M_PI) << "°), "
                  << "pitch=" << pitch << " rad (" << (pitch * 180 / M_PI) << "°)\n";
    }
};

/**
 * Simulate robot odometry.
 * In real code, this would come from your localization system.
 */
struct OdometryState {
    double x = 0.0;
    double y = 0.0;
    double theta = 0.0;
    double timestamp = 0.0;

    void simulate_movement(double elapsed_time) {
        // Simulate circular motion
        const double radius = 2.0;  // 2 meters
        const double angular_vel = 0.5;  // rad/s

        double angle = angular_vel * elapsed_time;

        x = radius * std::cos(angle);
        y = radius * std::sin(angle);
        theta = angle;
        timestamp = elapsed_time;
    }

    void print() const {
        std::cout << "  Odometry State: x=" << std::fixed << std::setprecision(4)
                  << x << " m, y=" << y << " m, theta=" << theta
                  << " rad (" << (theta * 180 / M_PI) << "°)\n";
    }
};

// ============================================================================
// Main Example
// ============================================================================

int main() {
    // Setup logging
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>("infantry_tf", console_sink);
    spdlog::set_default_logger(logger);
    spdlog::set_level(spdlog::level::debug);

    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Infantry Robot TF Integration with Foxglove              ║\n";
    std::cout << "║  Type-Safe Transform Tree + Real-Time Visualization       ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";

    // ========================================================================
    // Step 1: Create Foxglove Server and TF Channel
    // ========================================================================

    std::cout << "\n[1] Initializing Foxglove Server...\n";

    foxglove::setLogLevel(foxglove::LogLevel::Info);

    foxglove::WebSocketServerOptions ws;
    ws.host = "127.0.0.1";
    ws.port = 8765;

    auto server_result = foxglove::WebSocketServer::create(std::move(ws));
    if (!server_result.has_value()) {
        std::cerr << "Failed to create Foxglove server: "
                  << foxglove::strerror(server_result.error()) << "\n";
        return 1;
    }

    auto server = std::make_unique<foxglove::WebSocketServer>(
        std::move(server_result.value()));

    std::cout << "  ✓ Foxglove WebSocket server listening on ws://127.0.0.1:8765\n";

    // Create TF channel
    auto tf_channel_result =
        foxglove::schemas::FrameTransformsChannel::create("/tf");
    if (!tf_channel_result.has_value()) {
        std::cerr << "Failed to create TF channel: "
                  << foxglove::strerror(tf_channel_result.error()) << "\n";
        return 1;
    }

    auto tf_channel = std::make_shared<foxglove::schemas::FrameTransformsChannel>(
        std::move(tf_channel_result.value()));

    std::cout << "  ✓ TF channel created\n";

    // ========================================================================
    // Step 2: Initialize Infantry TF Manager
    // ========================================================================

    std::cout << "\n[2] Initializing Infantry TF Manager...\n";

    auto tf_manager = std::make_unique<InfantryTFManager>(tf_channel);

    std::cout << "  ✓ Infantry TF hierarchy established\n";
    std::cout << "  ✓ All static transforms registered\n";
    std::cout << "  ✓ Foxglove integration active\n";

    // ========================================================================
    // Step 3: Verify Frame Hierarchy (Compile-time Safety)
    // ========================================================================

    std::cout << "\n[3] Frame Hierarchy Verification (Compile-Time)...\n";

    std::cout << "  Frame types and relationships:\n";
    std::cout << "    " << world_Frame::frame_name << " (root)\n";
    std::cout << "    └── " << odom_Frame::frame_name << "\n";
    std::cout << "        └── " << gimbal_link_Frame::frame_name << "\n";
    std::cout << "            ├── " << camera_link_Frame::frame_name << "\n";
    std::cout << "            │   └── " << camera_optical_frame_Frame::frame_name
              << "\n";
    std::cout << "            └── " << muzzle_link_Frame::frame_name << "\n";

    // Verify static transforms
    std::cout << "\n  Static transforms registered:\n";

    if (tf_manager->has_static_transform<camera_link_Frame, gimbal_link_Frame>()) {
        std::cout << "    ✓ camera_link -> gimbal_link\n";
    }

    if (tf_manager->has_static_transform<
            camera_optical_frame_Frame, camera_link_Frame>()) {
        std::cout << "    ✓ camera_optical_frame -> camera_link\n";
    }

    if (tf_manager->has_static_transform<muzzle_link_Frame, gimbal_link_Frame>()) {
        std::cout << "    ✓ muzzle_link -> gimbal_link\n";
    }

    // ========================================================================
    // Step 4: Simulate Dynamic Updates
    // ========================================================================

    std::cout << "\n[4] Simulating Dynamic TF Updates...\n";

    GimbalState gimbal;
    OdometryState odometry;

    const int simulation_steps = 10;
    const double dt = 0.5;  // 500ms per step

    for (int step = 0; step < simulation_steps; ++step) {
        double elapsed_time = step * dt;

        // Update simulated states
        gimbal.simulate_tracking(elapsed_time);
        odometry.simulate_movement(elapsed_time);

        // Publish to TF
        tf_manager->update_gimbal(gimbal.yaw, gimbal.pitch);
        tf_manager->update_odometry(odometry.x, odometry.y, odometry.theta);

        std::cout << "\n  [Step " << (step + 1) << "/" << simulation_steps << "] "
                  << "t=" << std::fixed << std::setprecision(2) << elapsed_time
                  << "s\n";

        gimbal.print();
        odometry.print();

        // Sleep briefly to simulate real-time operation
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // ========================================================================
    // Step 5: Demonstrate Query API
    // ========================================================================

    std::cout << "\n[5] Demonstrating Type-Safe Query API...\n";

    // Example: Get camera calibration
    if (auto camera_tf =
            tf_manager->get_static_transform<camera_link_Frame, gimbal_link_Frame>()) {
        std::cout << "  ✓ Retrieved camera calibration transform\n";
        std::cout << "    Camera position: ("
                  << camera_tf->translation().x() << ", "
                  << camera_tf->translation().y() << ", "
                  << camera_tf->translation().z() << ")\n";
    }

    // Example: Get optical frame calibration
    if (auto optical_tf = tf_manager->get_static_transform<
            camera_optical_frame_Frame, camera_link_Frame>()) {
        std::cout << "  ✓ Retrieved optical frame transformation\n";
        auto quat = optical_tf->quaternion();
        std::cout << "    Quaternion: (w=" << quat.w() << ", x=" << quat.x()
                  << ", y=" << quat.y() << ", z=" << quat.z() << ")\n";
    }

    // ========================================================================
    // Step 6: Summary
    // ========================================================================

    std::cout << "\n╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ Demonstration Complete                                     ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ What was demonstrated:                                      ║\n";
    std::cout << "║  ✓ Type-safe frame hierarchy with compile-time validation  ║\n";
    std::cout << "║  ✓ Static transform registration and storage               ║\n";
    std::cout << "║  ✓ Dynamic gimbal and odometry updates                     ║\n";
    std::cout << "║  ✓ Direct Foxglove SDK integration                         ║\n";
    std::cout << "║  ✓ Real-time TF publishing for visualization               ║\n";
    std::cout << "║  ✓ Type-safe query API for static transforms               ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Features:                                                   ║\n";
    std::cout << "║  • Zero-cost abstractions (no runtime overhead)             ║\n";
    std::cout << "║  • Compile-time frame hierarchy validation                  ║\n";
    std::cout << "║  • Impossible to query invalid frame pairs                  ║\n";
    std::cout << "║  • Full ROS2-compatible URDF tree structure                 ║\n";
    std::cout << "║  • Direct visualization in Foxglove Studio                  ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";

    std::cout << "\nTo visualize the TF tree in Foxglove Studio:\n";
    std::cout << "  1. Open Foxglove Studio (https://foxglove.dev/)\n";
    std::cout << "  2. Connect to: ws://127.0.0.1:8765\n";
    std::cout << "  3. Add a 3D panel and enable the /tf topic\n";
    std::cout << "  4. Watch the robot transforms update in real-time\n";

    return 0;
}
