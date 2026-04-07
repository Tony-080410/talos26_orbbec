#include "L2_perception/armor/at_legacy_systems.hpp"

#include "L2_perception/armor/at_legacy_config.hpp"
#include "L2_perception/armor/at_legacy_detector_backend.hpp"
#include "L2_perception/armor/at_legacy_pnp_solver.hpp"
#include "L2_perception/armor/at_legacy_readback_roi.hpp"
#include "core/armor_types.hpp"
#include "core/channel_topics.hpp"
#include "core/runtime.hpp"
#include "frame.hpp"
#include "scheduler/scheduler.hpp"

#include <algorithm>
#include <iterator>
#include <spdlog/spdlog.h>

namespace fcs::L2 {

namespace {

[[nodiscard]] bool snapshot_is_fresh(
    const AtLegacyReadbackRoiSnapshot& snapshot, const AtLegacyReadbackRoiConfig& config,
    uint64_t image_timestamp_ns) noexcept {
    if (!snapshot.valid || snapshot.timestamp_ns == 0
        || image_timestamp_ns < snapshot.timestamp_ns) {
        return false;
    }

    const auto max_age_ns = static_cast<uint64_t>(std::max(0.0, config.stale_timeout_s) * 1e9);
    return image_timestamp_ns - snapshot.timestamp_ns <= max_age_ns;
}

void offset_detections(std::vector<AtLegacyArmorDetection>& detections, const cv::Point& offset) {
    if (offset.x == 0 && offset.y == 0) {
        return;
    }

    const cv::Point2f offset_f(static_cast<float>(offset.x), static_cast<float>(offset.y));
    for (auto& detection : detections) {
        for (auto& corner : detection.corners) {
            corner += offset_f;
        }
        detection.rect.x += static_cast<float>(offset.x);
        detection.rect.y += static_cast<float>(offset.y);
    }
}

[[nodiscard]] bool is_sub_frame_roi(const cv::Rect& roi, const cv::Size& frame_size) noexcept {
    return roi.x > 0 || roi.y > 0 || roi.width < frame_size.width || roi.height < frame_size.height;
}

} // namespace

void register_at_legacy_detection_systems(talos::Scheduler& scheduler) {
    auto& world = scheduler.world();
    if (!world.has_resource<AtLegacyReadbackRoiConfig>()) {
        world.insert_resource(AtLegacyReadbackRoiConfig{});
    }
    if (!world.has_resource<AtLegacyReadbackRoiCache>()) {
        world.insert_resource(AtLegacyReadbackRoiCache{});
    }

    scheduler.add_system<talos::fixed_rate<200>>(
        "at_legacy_armor_detector_pnp_solver",
        [](talos::spmc<AtLegacyImageFrame, AtLegacyImageChannelTopic> image_in,
           talos::spmc_mut<AtLegacyArmorDetectionBatch, AtLegacyDetectionChannelTopic>
               detection_out,
           talos::spmc_mut<AtLegacyArmorMeasurementBatch, AtLegacyMeasurementChannelTopic>
               measurement_out,
           talos::res_mut<std::shared_ptr<AtLegacyDetectorBackend>> backend,
           talos::res<std::shared_ptr<AtLegacyPnPSolver>> solver_ptr,
           talos::res<fast_tf::CoordinateSystem> tf_system,
           talos::res<AtLegacyReadbackRoiConfig> readback_roi_config,
           talos::res<AtLegacyReadbackRoiCache> readback_roi_cache,
           core::detecting_color detecting_color_) mutable {
            auto frame = image_in.read();
            if (!frame) {
                return;
            }

            ArmorColor detect_color = *detecting_color_;
            // Re-apply detect color before each detection and enforce color before PnP.
            backend->get()->set_detect_color(detect_color);

            cv::Rect detector_roi(0, 0, frame->image.cols, frame->image.rows);
            cv::Mat detector_input = frame->image;
            if (readback_roi_config->enabled && backend->get()->supports_readback_roi()) {
                const auto snapshot = readback_roi_cache->load();
                if (snapshot_is_fresh(snapshot, *readback_roi_config, frame->timestamp_ns)) {
                    const auto maybe_roi = resolve_readback_roi(
                        frame->image.size(), snapshot.raw_roi, *readback_roi_config,
                        backend->get()->input_resolution());
                    if (maybe_roi) {
                        detector_roi   = *maybe_roi;
                        detector_input = frame->image(detector_roi);
                    }
                }
            }
            const bool has_detector_roi = is_sub_frame_roi(detector_roi, frame->image.size());

            auto result = backend->get()->detect(detector_input);
            if (!result) {
                detection_out.write(
                    AtLegacyArmorDetectionBatch{
                        {},
                        frame->image,
                        frame->timestamp_ns,
                        frame->frame_id,
                        has_detector_roi,
                        detector_roi});
                return;
            }

            auto detections = std::move(*result);
            offset_detections(detections, detector_roi.tl());
            std::vector<AtLegacyArmorDetection> detections_for_pnp;
            detections_for_pnp.reserve(detections.size());
            std::copy_if(
                detections.begin(), detections.end(), std::back_inserter(detections_for_pnp),
                [detect_color](const AtLegacyArmorDetection& det) {
                    return det.color == detect_color;
                });

            detection_out.write(
                AtLegacyArmorDetectionBatch{
                    std::move(detections), frame->image, frame->timestamp_ns, frame->frame_id,
                    has_detector_roi, detector_roi});

            auto tf_camera_odom = fast_tf::lookup_clamped<fast_tf::odom, fast_tf::camera_optical>(
                *tf_system, frame->timestamp_ns);

            if (!tf_camera_odom) {
                // SPDLOG_ERROR(
                //     "TF lookup failed for timestamp {}: {}", frame->timestamp_ns,
                //     tf_camera_odom.error());
                return;
            }

            const auto& T_camera_odom = *tf_camera_odom;
            // BA expects R_imu_cam in the sense of imu <- camera (here: odom <- camera).
            const Eigen::Matrix3d R_odom_camera = T_camera_odom.rotation();
            auto measurements =
                (*solver_ptr)
                    ->solve_batch_with_ba(detections_for_pnp, R_odom_camera, frame->timestamp_ns);

            std::vector<AtLegacyArmorMeasurement> odom_measurements;
            odom_measurements.reserve(measurements.size());

            for (const auto& meas : measurements) {
                odom_measurements.push_back(meas.reframe(T_camera_odom));
            }

            measurement_out.write(
                AtLegacyArmorMeasurementBatch{
                    std::move(odom_measurements), frame->timestamp_ns, frame->frame_id});
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
