#pragma once

#include "base.hpp"
#include "fcs_visualization/foxglove_types.hpp"
#include "fcs_visualization/scene_builder.hpp"
#include "fcs_visualization/system_helpers.hpp"

#include "camera_config.hpp"
#include "fcs/core/channel_topics.hpp"
#include "fcs/core/types.hpp"
#include "math/fast_tf/frame.hpp"
#include "scheduler/ecs/channel_types.hpp"
#include "scheduler/ecs/execution_policy.hpp"

#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fcs::foxglove::systems {

/// @brief Register L1 sensor layer systems (image publishing)
///
/// This includes:
/// - foxglove_image_pub: Publishes camera images with detection boxes overlaid
/// - Also publishes TF and camera calibration at 30Hz
template <typename App>
void register_l1_sensor_systems(App& app) {

    // =========================================================================
    // IMAGE PUBLISHER (Camera calibration + TF)
    // =========================================================================

    app.template add_system<talos::pool_compute>(
        "foxglove_l1_image_pub",
        [](talos::spmc<AtLegacyImageFrame, AtLegacyImageChannelTopic> img_in,
           talos::res<std::shared_ptr<FoxgloveServer>> server, talos::res<CameraConfig> cam,
           talos::spmc<AtLegacyArmorDetectionBatch, AtLegacyDetectionChannelTopic> det_in,
           talos::res<std::shared_ptr<fast_tf::CoordinateSystem<1024>>> tf_buffer) {
            if (!detail::foxglove_ready(*server, img_in)) {
                return;
            }

            auto frame = img_in.read();
            if (!frame || frame->image.empty()) {
                return;
            }

            // Publish TF
            (*server)->publish_tf(**tf_buffer, frame->timestamp_ns);

            // Clone image and draw detection boxes
            cv::Mat img_bgr = frame->image.clone();
            auto batch      = det_in.read_current();
            if (batch) {
                for (size_t i = 0; i < batch->detections.size(); ++i) {
                    const auto& det = batch->detections[i];
                    cv::circle(img_bgr, det.center(), 5, cv::Scalar(0, 0, 255), 5);
                    for (auto i = 0; i < det.corners.size(); i++) {
                        auto pp1 = det.corners[i];
                        auto pp2 = det.corners[(i + 1) % 4];
                        cv::line(img_bgr, pp1, pp2, cv::Scalar(0, 255, 0), 4);
                    }
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
            msg.img.timestamp = timestamp_from_ns(frame->timestamp_ns);
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
                cam->width, cam->height, camera_matrix_arr, dist_coeffs, frame->timestamp_ns);
        });
}

} // namespace fcs::foxglove::systems
