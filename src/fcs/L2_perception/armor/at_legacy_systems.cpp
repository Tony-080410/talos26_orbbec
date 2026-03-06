#include "L2_perception/armor/at_legacy_systems.hpp"

#include "L2_perception/armor/at_legacy_config.hpp"
#include "L2_perception/armor/at_legacy_detector_backend.hpp"
#include "L2_perception/armor/at_legacy_pnp_solver.hpp"
#include "core/channel_topics.hpp"
#include "frame.hpp"
#include "scheduler/scheduler.hpp"

#include <algorithm>
#include <spdlog/spdlog.h>

namespace fcs::L2 {

void register_at_legacy_detection_systems(talos::Scheduler& scheduler, AtLegacyConfig&& config) {
    scheduler.world().insert_resource(std::move(config));
    scheduler.add_system<talos::fixed_rate<200>>(
        "at_legacy_armor_detector_pnp_solver",
        [](talos::spmc<AtLegacyImageFrame, AtLegacyImageChannelTopic> image_in,
           talos::spmc_mut<AtLegacyArmorDetectionBatch, AtLegacyDetectionChannelTopic>
               detection_out,
           talos::spmc_mut<AtLegacyArmorMeasurementBatch, AtLegacyMeasurementChannelTopic>
               measurement_out,
           talos::res_mut<std::shared_ptr<AtLegacyDetectorBackend>> backend_ptr,
           talos::res<std::shared_ptr<AtLegacyPnPSolver>> solver_ptr,
           talos::res<std::shared_ptr<fast_tf::CoordinateSystem<1024>>> tf_system) mutable {
            if (!image_in.has_new()) {
                return;
            }
            auto frame = image_in.read();
            if (!frame) {
                return;
            }

            auto& backend = *backend_ptr;
            if (!backend) {
                return;
            }

            const auto detect_color = config->detector.detect_color;
            // Re-apply detect color before each detection and enforce color before PnP.
            backend->set_detect_color(detect_color);

            auto result = backend->detect(frame->image);
            if (!result) {
                detection_out.write(
                    AtLegacyArmorDetectionBatch{
                        {}, frame->image, frame->timestamp_ns, frame->frame_id});
                return;
            }

            auto detections = std::move(*result);
            detections.erase(
                std::remove_if(
                    detections.begin(), detections.end(),
                    [detect_color](const AtLegacyArmorDetection& det) {
                        return det.color != detect_color;
                    }),
                detections.end());

            detection_out.write(
                AtLegacyArmorDetectionBatch{
                    detections, frame->image, frame->timestamp_ns, frame->frame_id});

            auto tf_odom_camera = fast_tf::lookup_clamped<fast_tf::odom>(
                **tf_system, fast_tf::FuckTransform<fast_tf::camera_optical>(),
                frame->timestamp_ns);

            if (!tf_odom_camera) {
                // SPDLOG_ERROR(
                //     "TF lookup failed for timestamp {}: {}", frame->timestamp_ns,
                //     tf_odom_camera.error());
                return;
            }

            auto measurements = (*solver_ptr)
                                    ->solve_batch_with_ba(
                                        detections, tf_odom_camera.value()->rotation().matrix(),
                                        frame->timestamp_ns);

            // Transform measurements from camera_optical frame to odom frame
            // T_odom_armor = T_odom_camera * T_camera_optical_armor
            const auto& T_odom_camera = *tf_odom_camera.value();
            for (auto& meas : measurements) {
                meas.transform =
                    fast_tf::FuckTransform<fast_tf::odom>{T_odom_camera * (*meas.transform)};
            }

            measurement_out.write(
                AtLegacyArmorMeasurementBatch{
                    std::move(measurements), frame->timestamp_ns, frame->frame_id});
        });
}

std::expected<DetectorBackendHandle, std::string>
    create_detector_backend_handle(const AtLegacyConfig& config) {
    auto backend_result = create_detector_backend(config);
    if (!backend_result) {
        return std::unexpected(std::string(detector_error_str(backend_result.error())));
    }

    auto backend_ptr  = std::make_shared<AtLegacyDetectorBackend>(std::move(*backend_result));
    auto backend_name = std::string(backend_ptr->name());
    return DetectorBackendHandle{std::move(backend_ptr), std::move(backend_name)};
}

std::shared_ptr<AtLegacyPnPSolver> create_pnp_solver(const CameraConfig& config) {
    auto solver = std::make_shared<AtLegacyPnPSolver>();
    solver->init(config);
    return solver;
}

} // namespace fcs::L2
