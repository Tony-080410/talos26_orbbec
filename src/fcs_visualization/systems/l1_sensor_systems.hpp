#pragma once

#include "base.hpp"
#include "foxglove_types.hpp"
#include "scene_builder.hpp"
#include "system_helpers.hpp"

#include "L2_perception/armor/at_legacy_config.hpp"
#include "camera_config.hpp"
#include "core/channel_topics.hpp"
#include "core/types.hpp"
#include "frame.hpp"

#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fcs::foxglove::systems {

/// @brief Register L1 sensor layer systems (image publishing)
///
/// This includes:
/// - foxglove_image_pub: Publishes camera images with detection boxes overlaid
/// - Also publishes TF and camera calibration
template <typename App>
void register_l1_sensor_systems(App& app) {

    // =========================================================================
    // IMAGE PUBLISHER (Camera calibration + TF)
    //
    // Synchronized rendering: image is bundled with detection result
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_l1_image_pub",
        [](talos::spmc<AtLegacyArmorDetectionBatch, AtLegacyDetectionChannelTopic> det_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server, talos::res<CameraConfig> cam,
           talos::res<L2::AtLegacyConfig> legacy_config,
           talos::res<std::shared_ptr<fast_tf::CoordinateSystem>> tf_buffer) {
            if (!detail::foxglove_ready(*server, det_in)) {
                return;
            }

            auto batch = det_in.read();
            if (!batch || batch->image.empty()) {
                return;
            }

            // Publish TF
            (*server)->publish_tf(**tf_buffer, batch->timestamp_ns);

            // Clone image and draw detection boxes
            cv::Mat img_bgr = batch->image.clone();

            // Determine ally color from config (opposite of detect_color)
            // detect_color is enemy color, so ally is the opposite
            const cv::Scalar ally_color =
                (legacy_config->detector.detect_color == ArmorColor::Red)
                    ? cv::Scalar(255, 0, 0)  // BGR: Blue (ally is blue when detecting red enemies)
                    : cv::Scalar(0, 0, 255); // BGR: Red (ally is red when detecting blue enemies)

            // Draw optical center (principal point) with ally color
            // cx, cy are at camera_matrix(0,2) and camera_matrix(1,2)
            const double cx      = cam->camera_matrix(0, 2);
            const double cy      = cam->camera_matrix(1, 2);
            const int cross_size = 15;
            cv::drawMarker(
                img_bgr, cv::Point(static_cast<int>(cx), static_cast<int>(cy)), ally_color,
                cv::MARKER_CROSS, cross_size, 2);

            // Box color is always green
            const cv::Scalar box_color(0, 255, 0);
            std::array<cv::Scalar, 4> colors = {
                cv::Scalar(0, 255, 0),   // 绿色 (LT)
                cv::Scalar(255, 255, 0), // 青色 (LB)
                cv::Scalar(255, 0, 255), // 品红 (RB)
                cv::Scalar(0, 255, 255)  // 黄色 (RT)
            };
            for (const auto& det : batch->detections) {
                // Draw detection box
                for (size_t j = 0; j < det.corners.size(); j++) {
                    auto pp1 = det.corners[j];
                    auto pp2 = det.corners[(j + 1) % 4];
                    cv::circle(img_bgr, pp1, 5, colors[(j + 1) % 4], 2);
                    cv::line(img_bgr, pp1, pp2, box_color, 2);
                }

                // Draw debug info below the box
                // Text position: bottom-left corner of the detection box
                cv::Point2f text_pos(det.rect.x, det.rect.y + det.rect.height + 20);

                // Line 1: name and type (e.g., "Sentry Small")
                std::string name_type = fmt::format(
                    "{} {}", magic_enum::enum_name(det.name), magic_enum::enum_name(det.type));
                cv::putText(
                    img_bgr, name_type, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1);

                // Line 2: confidence (e.g., "0.95")
                cv::Point2f conf_pos(det.rect.x, det.rect.y + det.rect.height + 40);
                std::string conf_str = fmt::format("conf: {:.2f}", det.confidence);
                cv::putText(
                    img_bgr, conf_str, conf_pos, cv::FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1);
            }

            // Compress to JPEG
            std::vector<uint8_t> compressed;
            std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, 85};
            if (!cv::imencode(".jpg", img_bgr, compressed, params)) {
                return;
            }

            // Construct image message
            ImageMessage msg;
            msg.img.timestamp = timestamp_from_ns(batch->timestamp_ns);
            msg.img.frame_id  = "camera_optical_frame";
            msg.img.format    = "jpeg";
            msg.img.data      = std::vector<std::byte>(
                reinterpret_cast<const std::byte*>(compressed.data()),
                reinterpret_cast<const std::byte*>(compressed.data() + compressed.size()));

            (*server)->enqueue_message(std::move(msg));

            // Publish camera calibration
            std::array<double, 9> camera_matrix_arr;
            std::copy_n(cam->camera_matrix.data(), 9, camera_matrix_arr.begin());
            std::vector<double> dist_coeffs(
                cam->distort_coefficient.data(),
                cam->distort_coefficient.data() + cam->distort_coefficient.size());

            (*server)->publish_camera_calibration(
                cam->width, cam->height, camera_matrix_arr, dist_coeffs, batch->timestamp_ns);
        });
}

} // namespace fcs::foxglove::systems
