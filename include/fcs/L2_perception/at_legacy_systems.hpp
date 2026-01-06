#pragma once

#include "../core/channel_tags.hpp"
#include "../core/types.hpp"
#include "at_legacy_detector_backend.hpp"
#include "at_legacy_pnp_solver.hpp"
#include "auto_aim/config.hpp"
#include "scheduler/spin.hpp"
#include <memory>
#include <scheduler/bevy_ecs.hpp>
#include <spdlog/spdlog.h>

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
        "at_legacy_armor_detector_pnp_solver",
        [](talos::spmc<AtLegacyImageFrame, AtLegacyImageChannelTag> image_in,
           talos::spmc_mut<AtLegacyArmorDetectionBatch, AtLegacyDetectionChannelTag> detection_out,
           talos::spmc_mut<AtLegacyArmorMeasurementBatch, AtLegacyMeasurementChannelTag>
               measurement_out,
           talos::res_mut<std::shared_ptr<AtLegacyDetectorBackend>> backend_ptr,
           talos::res<std::shared_ptr<AtLegacyPnPSolver>> solver_ptr,
           talos::res<std::shared_ptr<fast_tf::CoordinateSystem<1024>>> tf_system) {
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
            // Solve PnP for all detections (in camera_optical frame)
            auto tf_odom_camera = fast_tf::lookup<fast_tf::odom_fuxk_frame>(
                **tf_system, fast_tf::FuckTransform<fast_tf::camera_optical_fuxk_frame>(),
                frame->timestamp_ns);

            if (!tf_odom_camera) {
                spdlog::error(
                    "TF lookup failed for timestamp {}: {}", frame->timestamp_ns,
                    fast_tf::to_string(tf_odom_camera.error()));
                return;
            }

            auto detections = std::move(*result);

            // Output detection batch
            detection_out.write(
                AtLegacyArmorDetectionBatch{detections, frame->timestamp_ns, frame->frame_id});

            auto measurements = (*solver_ptr)
                                    ->solve_batch_with_ba(
                                        detections, tf_odom_camera.value()->rotation().matrix(),
                                        frame->timestamp_ns);
            // Transform each measurement from camera_optical to odom frame.
            // Match atvision behavior: if TF is not available, drop the whole batch (never feed a
            // wrong-frame measurement into the tracker).
            const auto& T_odom_camera = *tf_odom_camera.value();
            for (auto& meas : measurements) {
                meas.transform = fast_tf::FuckTransform<fast_tf::odom_fuxk_frame>{
                    T_odom_camera * (*meas.transform)};
            }
            // Output measurement batch
            measurement_out.write(
                AtLegacyArmorMeasurementBatch{
                    std::move(measurements), frame->timestamp_ns, frame->frame_id});
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
