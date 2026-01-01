#pragma once

#include "../core/channel_tags.hpp"
#include "../core/types.hpp"
#include "at_legacy_detector_backend.hpp"
#include "at_legacy_pnp_solver.hpp"

#include "math/fast_tf/infantry.hpp"

#include <chrono>
#include <memory>
#include <scheduler/bevy_ecs.hpp>

namespace fcs {

// ============================================================================
// AT Legacy Armor Detection Pipeline Systems
// ============================================================================

/// Register AT Legacy detection pipeline systems with the scheduler
///
/// Pipeline architecture:
///   Camera System (external) → Image [SPMC]
///         ↓
///   ArmorDetectorSystem (pool_compute) → Detection [SPMC]
///         ↓
///   PnPSolverSystem (pool_compute) → Measurement [SPMC]
///         ↓
///   TrackerSystem (external)
///
/// Usage:
///   auto app = talos::App::create();
///
///   // Register resources
///   app.insert_resource<AtLegacyDetectorBackend>(std::move(backend));
///   app.insert_resource<AtLegacyPnPSolver>(std::move(solver));
///
///   // Register channels
///   app.init_spmc_channel<AtLegacyImageFrame, AtLegacyImageChannelTag>();
///   app.init_spmc_channel<AtLegacyArmorDetectionBatch, AtLegacyDetectionChannelTag>();
///   app.init_spmc_channel<AtLegacyArmorMeasurementBatch, AtLegacyMeasurementChannelTag>();
///
///   // Register systems
///   register_at_legacy_detection_systems(app);

template <typename App>
void register_at_legacy_detection_systems(App& app) {
    // ArmorDetectorSystem: Image → Detection
    // Triggered by Image channel updates, runs in thread pool
    app.template add_system<talos::pool_compute>(
        "at_legacy_armor_detector",
        [](talos::spmc<AtLegacyImageFrame, AtLegacyImageChannelTag> image_in,
           talos::spmc_mut<AtLegacyArmorDetectionBatch, AtLegacyDetectionChannelTag> detection_out,
           talos::res_mut<std::shared_ptr<AtLegacyDetectorBackend>> backend_ptr) {
            // Check for new image
            if (!image_in.has_new())
                return;
            auto frame = image_in.read();
            if (!frame)
                return;

            auto& backend = *backend_ptr;
            if (!backend)
                return;

            // Run detection
            auto result = backend->detect(frame->image);
            if (!result) {
                // Detection failed, output empty batch
                detection_out.write(
                    AtLegacyArmorDetectionBatch{{}, frame->timestamp_ns, frame->frame_id});
                return;
            }

            // Output detection batch
            detection_out.write(
                AtLegacyArmorDetectionBatch{
                    std::move(*result), frame->timestamp_ns, frame->frame_id});
        });

    // PnPSolverSystem: Detection → Measurement (in odom frame)
    // Triggered by Detection channel updates, runs in thread pool
    app.template add_system<talos::pool_compute>(
        "at_legacy_pnp_solver",
        [](talos::spmc<AtLegacyArmorDetectionBatch, AtLegacyDetectionChannelTag> detection_in,
           talos::spmc_mut<AtLegacyArmorMeasurementBatch, AtLegacyMeasurementChannelTag>
               measurement_out,
           talos::res<std::shared_ptr<AtLegacyPnPSolver>> solver_ptr,
           talos::res<std::shared_ptr<fast_tf::CoordinateSystem<1024>>> tf_system) {
            // Check for new detection
            if (!detection_in.has_new())
                return;
            auto batch = detection_in.read();
            if (!batch || batch->empty()) {
                // No detections, output empty batch
                if (batch) {
                    measurement_out.write(
                        AtLegacyArmorMeasurementBatch{{}, batch->timestamp_ns, batch->frame_id});
                }
                return;
            }

            auto& solver = *solver_ptr;
            if (!solver)
                return;

            // Solve PnP for all detections (in camera_optical frame)
            auto measurements = solver->solve_batch(batch->detections, batch->timestamp_ns);
            /*
                        // Wait for TF to be available (similar to ROS2 MessageFilter)
                        // This ensures gimbal pose for this timestamp has arrived
                        constexpr int kMaxWaitMs = 10;
                        bool tf_ready            = false;
                        for (int i = 0; i < kMaxWaitMs; ++i) {
                            auto range =
                                (**tf_system)[fast_tf::gimbal_link_fuxk_frame::index].time_range();
                            if (range && range->second >= batch->timestamp_ns) {
                                tf_ready = true;
                                break;
                            }
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        }
                        if (!tf_ready) {
                            spdlog::warn(
                                "TF not available for timestamp {}, dropping frame",
               batch->timestamp_ns); return;
                        }
            */
            // Transform each measurement from camera_optical to odom frame
            for (auto& meas : measurements) {
                auto tf_result =
                    fast_tf::lookup<fast_tf::odom_fuxk_frame, fast_tf::camera_optical_fuxk_frame>(
                        **tf_system,
                        fast_tf::FuckTransform<fast_tf::camera_optical_fuxk_frame>(*meas.transform),
                        batch->timestamp_ns);

                if (tf_result) {
                    // Replace with odom frame transform (type-erased)
                    meas.transform = *tf_result;
                } else {
                    spdlog::error(
                        "TF lookup failed for timestamp {}: {}", batch->timestamp_ns,
                        fast_tf::to_string(tf_result.error()));
                }
            }

            // Output measurement batch
            measurement_out.write(
                AtLegacyArmorMeasurementBatch{
                    std::move(measurements), batch->timestamp_ns, batch->frame_id});
        });
}

// ============================================================================
// Configuration Helpers
// ============================================================================

/// Create PnP solver from camera info
/// @param config camera config
[[nodiscard]] inline AtLegacyPnPSolver create_pnp_solver(const CameraConfig& config) {
    AtLegacyPnPSolver solver;
    solver.init(config);
    return solver;
}

// Note: Use create_detector_backend(const AtLegacyConfig&) from at_legacy_detector_backend.hpp
// to create a detector backend with unified config management

} // namespace fcs
