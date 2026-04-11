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
#include <optional>
#include <spdlog/spdlog.h>

#include <magic_enum/magic_enum.hpp>

namespace fcs::L2 {

namespace {

[[nodiscard]] std::optional<cv::Point2f> normalize_image_point(
    const cv::Point2f& image_point, const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs) noexcept {
    std::vector<cv::Point2f> norm_points;
    cv::undistortPoints(
        std::vector<cv::Point2f>{image_point}, norm_points, camera_matrix, dist_coeffs);
    if (norm_points.size() != 1) {
        return std::nullopt;
    }

    const auto& point = norm_points.front();
    if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
        return std::nullopt;
    }
    return point;
}

[[nodiscard]] bool tracker_snapshot_matches_detection(
    const AtLegacyTrackerReadbackSnapshot& snapshot,
    const AtLegacyArmorDetection& detection) noexcept {
    return snapshot.valid && snapshot.tracker.target_name == detection.name
        && snapshot.tracker.target_color == detection.color;
}

[[nodiscard]] std::vector<AtLegacyPnPSolver::PosePrior> make_pose_priors(
    const AtLegacyArmorDetection& detection, const AtLegacyTrackerReadbackSnapshot& snapshot,
    const fast_tf::TransformMatrixd<fast_tf::odom, fast_tf::camera_optical>& T_odom_camera,
    const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs) noexcept {
    if (!tracker_snapshot_matches_detection(snapshot, detection)) {
        return {};
    }

    const auto det_center_norm =
        normalize_image_point(detection.center(), camera_matrix, dist_coeffs);
    if (!det_center_norm.has_value()) {
        return {};
    }

    const auto armor_poses = tracker_armor_poses(snapshot.tracker);
    if (armor_poses.empty()) {
        return {};
    }

    const double armor_pitch_rad = armor_pitch_rad_for(detection.name);
    const double cp              = std::cos(armor_pitch_rad);
    const double sp              = std::sin(armor_pitch_rad);
    Eigen::Matrix3d R_pitch;
    R_pitch << cp, 0.0, sp, 0.0, 1.0, 0.0, -sp, 0.0, cp;

    const Eigen::Matrix3d R_odom_camera = T_odom_camera.rotation();
    const Eigen::Vector3d t_odom_camera = T_odom_camera.translation();

    std::vector<AtLegacyPnPSolver::PosePrior> priors;
    priors.reserve(armor_poses.size());

    for (size_t i = 0; i < armor_poses.size(); ++i) {
        const auto& armor_pose = armor_poses[i];
        const double armor_yaw = armor_pose[3];
        const double cy        = std::cos(armor_yaw);
        const double sy        = std::sin(armor_yaw);

        Eigen::Matrix3d R_yaw;
        R_yaw << cy, -sy, 0.0, sy, cy, 0.0, 0.0, 0.0, 1.0;

        const Eigen::Matrix3d R_odom_armor   = R_yaw * R_pitch;
        const Eigen::Matrix3d R_camera_armor = R_odom_camera * R_odom_armor;
        const Eigen::Vector3d t_camera_armor =
            R_odom_camera * Eigen::Vector3d(armor_pose[0], armor_pose[1], armor_pose[2])
            + t_odom_camera;

        if (!std::isfinite(t_camera_armor.x()) || !std::isfinite(t_camera_armor.y())
            || !std::isfinite(t_camera_armor.z()) || t_camera_armor.z() <= 1e-3) {
            continue;
        }

        const cv::Point2f prior_center_norm(
            static_cast<float>(t_camera_armor.x() / t_camera_armor.z()),
            static_cast<float>(t_camera_armor.y() / t_camera_armor.z()));
        if (!std::isfinite(prior_center_norm.x) || !std::isfinite(prior_center_norm.y)) {
            continue;
        }

        cv::Mat R_cv;
        cv::eigen2cv(R_camera_armor, R_cv);
        cv::Mat rvec;
        cv::Rodrigues(R_cv, rvec);

        const double dx  = static_cast<double>(prior_center_norm.x - det_center_norm->x);
        const double dy  = static_cast<double>(prior_center_norm.y - det_center_norm->y);
        double hint_cost = dx * dx + dy * dy;
        if (static_cast<int>(i) == snapshot.selected_armor_id) {
            hint_cost *= 0.85;
        } else if (static_cast<int>(i) == snapshot.rough_selected_armor_id) {
            hint_cost *= 0.92;
        }

        priors.push_back(
            AtLegacyPnPSolver::PosePrior{
                .rvec      = cv::Vec3d(rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2)),
                .tvec      = cv::Vec3d(t_camera_armor.x(), t_camera_armor.y(), t_camera_armor.z()),
                .hint_cost = hint_cost,
                .armor_id  = static_cast<int>(i),
            });
    }

    return priors;
}

void offset_detections(
    std::vector<AtLegacyArmorDetection>& detections, const cv::Point& offset) noexcept {
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

void register_at_legacy_detection_systems(talos::Scheduler& scheduler) noexcept {
    auto& world = scheduler.world();
    if (!world.has_resource<AtLegacyReadbackRoiConfig>()) {
        world.insert_resource(AtLegacyReadbackRoiConfig{});
    }
    if (!world.has_resource<AtLegacyTrackerReadbackCache>()) {
        world.insert_resource(AtLegacyTrackerReadbackCache{});
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
           talos::res<fast_tf::CoordinateSystem> tf_system, talos::res<CameraConfig> camera_config,
           talos::res<AtLegacyReadbackRoiConfig> readback_roi_config,
           talos::res<AtLegacyTrackerReadbackCache> readback_cache,
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
            cv::Mat camera_matrix_cv;
            cv::Mat dist_coeffs_cv;
            cv::eigen2cv(camera_config->camera_matrix, camera_matrix_cv);
            cv::eigen2cv(camera_config->distort_coefficient, dist_coeffs_cv);

            std::optional<fast_tf::TransformMatrixd<fast_tf::camera_optical, fast_tf::odom>>
                T_camera_odom;
            std::optional<fast_tf::TransformMatrixd<fast_tf::odom, fast_tf::camera_optical>>
                T_odom_camera;

            if (readback_roi_config->enabled) {
                const auto snapshot = readback_cache->load();
                if (tracker_snapshot_is_fresh(
                        snapshot, *readback_roi_config, frame->timestamp_ns)) {
                    if (!T_camera_odom.has_value()) {
                        auto tf_lookup =
                            fast_tf::lookup_clamped<fast_tf::odom, fast_tf::camera_optical>(
                                *tf_system, frame->timestamp_ns);
                        if (tf_lookup) {
                            T_camera_odom = *tf_lookup;
                            T_odom_camera = tf_lookup->inverse();
                        }
                    }
                }

                if (tracker_snapshot_is_fresh(snapshot, *readback_roi_config, frame->timestamp_ns)
                    && T_odom_camera.has_value()) {
                    const auto maybe_roi = resolve_readback_roi(
                        frame->image.size(),
                        project_tracker_box_to_image(
                            snapshot.tracker, *T_odom_camera, *camera_config,
                            readback_roi_config->box_size_m)
                            .value_or(cv::Rect2f{}),
                        *readback_roi_config, backend->get()->input_resolution());
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

            if (!T_camera_odom.has_value()) {
                auto tf_lookup = fast_tf::lookup_clamped<fast_tf::odom, fast_tf::camera_optical>(
                    *tf_system, frame->timestamp_ns);
                if (tf_lookup) {
                    T_camera_odom = *tf_lookup;
                    T_odom_camera = tf_lookup->inverse();
                }
            }

            if (!T_camera_odom.has_value()) {
                // SPDLOG_ERROR(
                //     "TF lookup failed for timestamp {}: {}", frame->timestamp_ns,
                //     tf_camera_odom.error());
                return;
            }

            const auto& T_camera_odom_ref = *T_camera_odom;
            if (!T_odom_camera.has_value()) {
                T_odom_camera = T_camera_odom_ref.inverse();
            }
            // BA expects R_imu_cam in the sense of imu <- camera (here: odom <- camera).
            const Eigen::Matrix3d R_odom_camera = T_camera_odom_ref.rotation();
            const auto tracker_snapshot         = readback_cache->load();
            std::vector<AtLegacyCameraArmorMeasurement> measurements;
            measurements.reserve(detections_for_pnp.size());
            for (const auto& detection : detections_for_pnp) {
                const auto priors = tracker_snapshot_is_fresh(
                                        tracker_snapshot, *readback_roi_config, frame->timestamp_ns)
                                      ? make_pose_priors(
                                            detection, tracker_snapshot, *T_odom_camera,
                                            camera_matrix_cv, dist_coeffs_cv)
                                      : std::vector<AtLegacyPnPSolver::PosePrior>{};
                auto result =
                    (*solver_ptr)
                        ->solve_with_ba(detection, R_odom_camera, frame->timestamp_ns, priors);
                if (result) {
                    measurements.push_back(std::move(*result));
                }
            }

            std::vector<AtLegacyArmorMeasurement> odom_measurements;
            odom_measurements.reserve(measurements.size());

            for (const auto& meas : measurements) {
                odom_measurements.push_back(meas.reframe(T_camera_odom_ref));
            }

            measurement_out.write(
                AtLegacyArmorMeasurementBatch{
                    std::move(odom_measurements), frame->timestamp_ns, frame->frame_id});
        });
}

std::expected<DetectorBackendHandle, std::string>
    create_detector_backend_handle(const AtLegacyConfig& config) noexcept {
    auto backend_result = create_detector_backend(config);
    if (!backend_result) {
        return std::unexpected(std::move(backend_result.error()));
    }

    auto backend_ptr  = std::make_shared<AtLegacyDetectorBackend>(std::move(*backend_result));
    auto backend_name = std::string(magic_enum::enum_name(config.backend_type));
    return DetectorBackendHandle{std::move(backend_ptr), std::move(backend_name)};
}

std::shared_ptr<AtLegacyPnPSolver> create_pnp_solver(const CameraConfig& config) noexcept {
    return std::make_shared<AtLegacyPnPSolver>(config);
}

} // namespace fcs::L2
