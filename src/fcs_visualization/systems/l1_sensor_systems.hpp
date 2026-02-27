#pragma once

#include "base.hpp"
#include "foxglove_types.hpp"
#include "scene_builder.hpp"
#include "system_helpers.hpp"

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
           talos::res<std::shared_ptr<fast_tf::CoordinateSystem<1024>>> tf_buffer) {
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
            for (const auto& det : batch->detections) {
                cv::circle(img_bgr, det.center(), 5, cv::Scalar(0, 0, 255), 5);
                for (size_t j = 0; j < det.corners.size(); j++) {
                    auto pp1 = det.corners[j];
                    auto pp2 = det.corners[(j + 1) % 4];
                    cv::line(img_bgr, pp1, pp2, cv::Scalar(0, 255, 0), 2);
                }
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
