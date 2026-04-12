#include "L2_perception/ldm/types.hpp"
#include "base.hpp"
#include "core/runtime.hpp"
#include "foxglove_config.hpp"
#include "foxglove_types.hpp"
#include "quanta/transcoder.hpp"

#include "camera_config.hpp"
#include "core/channel_topics.hpp"
#include "core/types.hpp"
#include "frame.hpp"
#include "quanta/stream_encoder.hpp"

#include <algorithm>
#include <cstddef>
#include <deque>
#include <memory>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>

namespace fcs::visualization::foxglove::systems {

namespace detail {

struct QuantaFrameStamp {
    int64_t pts{0};
    uint64_t timestamp_ns{0};
};

struct LdmOverlayState {
    std::optional<L2::ldm::LdmDetection> latest_detection{};
    std::optional<L2::ldm::LdmMeasurement> latest_measurement{};
};

struct QuantaPublisherState {
    std::optional<quanta::StreamEncoder> encoder{};
    std::deque<QuantaFrameStamp> pending_frames{};
    std::string last_error{};
    int src_width{0};
    int src_height{0};
    int framerate{0};
    int64_t next_pts{0};
};

inline void log_quanta_error_once(QuantaPublisherState& state, std::string message) noexcept {
    if (state.last_error == message)
        return;
    state.last_error = std::move(message);
    SPDLOG_WARN("Foxglove quanta encoder: {}", state.last_error);
}

[[nodiscard]] inline bool ensure_quanta_encoder(
    QuantaPublisherState& state, const quanta::EncodeParams& cfg, int src_width,
    int src_height) noexcept {
    if (state.encoder && state.src_width == src_width && state.src_height == src_height
        && state.framerate == cfg.framerate) {
        return true;
    }

    auto enc = quanta::StreamEncoder::create(cfg, src_width, src_height, cfg.framerate);
    if (!enc) {
        log_quanta_error_once(state, std::move(enc.error()));
        state.encoder.reset();
        return false;
    }
    state.encoder.emplace(std::move(*enc));
    state.pending_frames.clear();
    state.next_pts   = 0;
    state.src_width  = src_width;
    state.src_height = src_height;
    state.framerate  = cfg.framerate;

    state.last_error.clear();
    SPDLOG_INFO(
        "Foxglove quanta encoder initialized: {}x{} -> <= {}x{}, {} bps @ {} fps", src_width,
        src_height, cfg.max_width, cfg.max_height, cfg.target_bitrate, cfg.framerate);
    return true;
}

[[nodiscard]] inline uint64_t take_timestamp_for_pts(
    QuantaPublisherState& state, int64_t pts, uint64_t fallback_timestamp_ns) noexcept {
    for (auto it = state.pending_frames.begin(); it != state.pending_frames.end(); ++it) {
        if (it->pts != pts)
            continue;

        const uint64_t timestamp_ns = it->timestamp_ns;
        state.pending_frames.erase(it);
        return timestamp_ns;
    }
    return fallback_timestamp_ns;
}

inline void publish_quanta_video(
    FoxgloveServer& server, QuantaPublisherState& state, const quanta::EncodeParams& cfg,
    const cv::Mat& image_bgr, uint64_t timestamp_ns) noexcept {
    if (!ensure_quanta_encoder(state, cfg, image_bgr.cols, image_bgr.rows))
        return;

    const int64_t pts = state.next_pts++;
    auto push_result =
        state.encoder->push_frame(image_bgr.data, static_cast<int>(image_bgr.step[0]), pts);
    if (!push_result) {
        log_quanta_error_once(state, std::move(push_result.error()));
        return;
    }

    state.pending_frames.push_back(QuantaFrameStamp{.pts = pts, .timestamp_ns = timestamp_ns});

    while (auto packet = state.encoder->poll_packet()) {
        const uint64_t packet_timestamp_ns =
            take_timestamp_for_pts(state, packet->pts, timestamp_ns);

        VideoMessage msg;
        msg.payload.timestamp = timestamp_from_ns(packet_timestamp_ns);
        msg.payload.frame_id  = "camera_optical_frame";
        msg.payload.format    = "h265";
        msg.payload.data      = std::vector<std::byte>(
            reinterpret_cast<const std::byte*>(packet->data.get()),
            reinterpret_cast<const std::byte*>(packet->data.get() + packet->size));

        server.enqueue_message(std::move(msg));
    }
}

inline void publish_jpeg_image(
    FoxgloveServer& server, const cv::Mat& image_bgr, uint64_t timestamp_ns) noexcept {
    std::vector<uint8_t> compressed;
    std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, 45};
    if (!cv::imencode(".jpg", image_bgr, compressed, params)) {
        return;
    }

    ImageMessage msg;
    msg.payload.timestamp = timestamp_from_ns(timestamp_ns);
    msg.payload.frame_id  = "camera_optical_frame";
    msg.payload.format    = "jpeg";
    msg.payload.data      = std::vector<std::byte>(
        reinterpret_cast<const std::byte*>(compressed.data()),
        reinterpret_cast<const std::byte*>(compressed.data() + compressed.size()));

    server.enqueue_message(std::move(msg));
}

[[nodiscard]] inline bool frame_sync_ok(
    uint64_t producer_timestamp_ns, uint64_t consumer_timestamp_ns, uint64_t max_skew_ns) noexcept {
    const uint64_t skew = (producer_timestamp_ns > consumer_timestamp_ns)
                            ? (producer_timestamp_ns - consumer_timestamp_ns)
                            : (consumer_timestamp_ns - producer_timestamp_ns);
    return skew <= max_skew_ns;
}

} // namespace detail

/// @brief Register L1 sensor layer systems (image publishing)
///
/// This includes:
/// - foxglove_image_pub: Publishes camera images with detection boxes overlaid
/// - Also publishes TF and camera calibration
void register_l1_sensor_systems(talos::scheduler::Scheduler& app) {

    // =========================================================================
    // IMAGE PUBLISHER (Camera calibration + TF)
    //
    // Synchronized rendering: image is bundled with detection result
    // =========================================================================

    app.add_system<talos::pool_compute>(
        "foxglove_l1_image_pub",
        [video_state = detail::QuantaPublisherState{}, ldm_state = detail::LdmOverlayState{}](
            talos::spmc<ArmorDetectionBatch, DetectionChannelTopic> det_in,
            talos::spmc<L2::ldm::LdmDetection, LdmDetectionChannelTopic> ldm_in,
            talos::spmc<L2::ldm::LdmMeasurement, LdmMeasurementChannelTopic> ldm_meas_in,
            talos::res<std::shared_ptr<FoxgloveServer>> server, talos::res<CameraConfig> cam,
            talos::res<FoxgloveConfig> foxglove_cfg, core::detecting_color detecting_color_,
            talos::res<fast_tf::CoordinateSystem> tf_buffer) mutable {
            if (!foxglove_ready(*server, det_in)) {
                return;
            }

            auto batch = det_in.read();
            if (!batch || batch->image.empty()) {
                return;
            }

            // Publish TF
            (*server)->publish_tf(*tf_buffer, batch->timestamp_ns);

            // Clone image and draw detection boxes
            cv::Mat img_bgr = batch->image.clone();

            const cv::Scalar roi_color =
                batch->has_detector_roi ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 0, 255);
            cv::rectangle(img_bgr, batch->detector_roi, roi_color, 2, cv::LINE_AA);

            // Determine ally color from config (opposite of detect_color)
            // detect_color is enemy color, so ally is the opposite
            const cv::Scalar ally_color =
                (*detecting_color_ == ArmorColor::Red)
                    ? cv::Scalar(0, 0, 255)  // BGR: Red (ally is red when detecting red enemies)
                    : cv::Scalar(255, 0, 0); // BGR: Blue (ally is blue when detecting blue enemies)

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
                    "{} {} {}", magic_enum::enum_name(det.name), magic_enum::enum_name(det.type),
                    magic_enum::enum_name(det.color));
                cv::putText(
                    img_bgr, name_type, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1);

                // Line 2: confidence (e.g., "0.95")
                cv::Point2f conf_pos(det.rect.x, det.rect.y + det.rect.height + 40);
                std::string conf_str = fmt::format("conf: {:.2f}", det.confidence);
                cv::putText(
                    img_bgr, conf_str, conf_pos, cv::FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1);
            }
            constexpr uint64_t kMaxLdmOverlaySkewNs = 50'000'000;
            const auto maybe_ldm_det                = ldm_in.read_current();
            const auto maybe_ldm_meas               = ldm_meas_in.read_current();
            if (maybe_ldm_det) {
                ldm_state.latest_detection = *maybe_ldm_det;
            }
            if (maybe_ldm_meas) {
                ldm_state.latest_measurement = *maybe_ldm_meas;
            }

            const auto use_ldm_det = ldm_state.latest_detection.has_value()
                                  && ldm_state.latest_detection->valid
                                  && (ldm_state.latest_detection->frame_id == batch->frame_id
                                      || detail::frame_sync_ok(
                                          ldm_state.latest_detection->timestamp_ns,
                                          batch->timestamp_ns, kMaxLdmOverlaySkewNs));

            if (use_ldm_det) {
                const auto& ldm_det = *ldm_state.latest_detection;
                const cv::Scalar blob_color(0, 165, 255);
                const cv::Scalar pair_color(255, 0, 255);
                const cv::Scalar center_color(0, 255, 255);
                cv::Rect2f ldm_rect = ldm_det.rect;

                if (ldm_det.selected_candidate_idx.has_value()
                    && *ldm_det.selected_candidate_idx >= 0
                    && static_cast<size_t>(*ldm_det.selected_candidate_idx)
                           < ldm_det.mesh_candidates.size()) {
                    const auto& selected =
                        ldm_det
                            .mesh_candidates[static_cast<size_t>(*ldm_det.selected_candidate_idx)];
                    std::vector<cv::Point2f> candidate_points;
                    candidate_points.reserve(selected.pair_indices.size() * 2);
                    for (const int pair_idx : selected.pair_indices) {
                        if (pair_idx < 0 || static_cast<size_t>(pair_idx) >= ldm_det.pairs.size()) {
                            continue;
                        }
                        const auto& pair = ldm_det.pairs[static_cast<size_t>(pair_idx)];
                        candidate_points.push_back(pair.top_center_px);
                        candidate_points.push_back(pair.bottom_center_px);
                    }
                    if (!candidate_points.empty()) {
                        ldm_rect = cv::boundingRect(candidate_points);
                    }
                }

                if (ldm_rect.width > 0.0f && ldm_rect.height > 0.0f) {
                    cv::rectangle(img_bgr, ldm_rect, box_color, 2, cv::LINE_AA);
                    cv::putText(
                        img_bgr, fmt::format("LDM {}", ldm_det.pairs.size()),
                        cv::Point(
                            static_cast<int>(ldm_rect.x),
                            std::max(20, static_cast<int>(ldm_rect.y) - 8)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, cv::LINE_AA);
                }

                for (const auto& blob : ldm_det.blobs) {
                    cv::rectangle(img_bgr, blob.rect, blob_color, 1, cv::LINE_AA);
                }

                for (size_t i = 0; i < ldm_det.pairs.size(); ++i) {
                    const auto& pair = ldm_det.pairs[i];
                    cv::line(
                        img_bgr, pair.top_center_px, pair.bottom_center_px, pair_color, 2,
                        cv::LINE_AA);
                    cv::circle(img_bgr, pair.top_center_px, 4, colors[0], 2);
                    cv::circle(img_bgr, pair.bottom_center_px, 4, colors[2], 2);
                    cv::circle(img_bgr, pair.midpoint_px, 3, center_color, -1);
                    cv::putText(
                        img_bgr, fmt::format("p{}", i), pair.midpoint_px + cv::Point2f(4.0f, -4.0f),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, center_color, 1, cv::LINE_AA);
                }

                if (ldm_det.center_image_px.has_value()) {
                    cv::drawMarker(
                        img_bgr, *ldm_det.center_image_px, center_color, cv::MARKER_CROSS, 18, 2);
                }

                if (ldm_det.selected_candidate_idx.has_value()
                    && *ldm_det.selected_candidate_idx >= 0
                    && static_cast<size_t>(*ldm_det.selected_candidate_idx)
                           < ldm_det.mesh_candidates.size()) {
                    const auto& selected =
                        ldm_det
                            .mesh_candidates[static_cast<size_t>(*ldm_det.selected_candidate_idx)];
                    std::string faces_text;
                    for (size_t i = 0; i < selected.octagon_face_indices.size(); ++i) {
                        if (!faces_text.empty()) {
                            faces_text += ",";
                        }
                        faces_text += std::to_string(selected.octagon_face_indices[i]);
                    }
                    cv::putText(
                        img_bgr,
                        fmt::format(
                            "ldm faces={} rmse={:.2f}", faces_text, selected.reprojection_rmse_px),
                        cv::Point(
                            static_cast<int>(ldm_rect.x),
                            std::max(20, static_cast<int>(ldm_rect.y) - 28)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, center_color, 1, cv::LINE_AA);
                }
            }

            const auto use_ldm_meas = ldm_state.latest_measurement.has_value()
                                   && ldm_state.latest_measurement->valid
                                   && (ldm_state.latest_measurement->frame_id == batch->frame_id
                                       || detail::frame_sync_ok(
                                           ldm_state.latest_measurement->timestamp_ns,
                                           batch->timestamp_ns, kMaxLdmOverlaySkewNs));
            if (use_ldm_meas) {
                const auto& ldm_meas = *ldm_state.latest_measurement;
                cv::putText(
                    img_bgr,
                    fmt::format(
                        "LDM pairs={}/{} {} conf={:.2f}", ldm_meas.selected_pair_count,
                        ldm_meas.pair_count_total, ldm_meas.depth_quality, ldm_meas.confidence),
                    cv::Point(30, 32), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2,
                    cv::LINE_AA);

                if (ldm_meas.whole_center_cam.has_value()) {
                    const auto& center = *ldm_meas.whole_center_cam;
                    cv::putText(
                        img_bgr,
                        fmt::format(
                            "cam=[{:.2f},{:.2f},{:.2f}]m", center.x(), center.y(), center.z()),
                        cv::Point(30, 56), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255),
                        1, cv::LINE_AA);
                }
            }

            if (foxglove_cfg->transport == FoxgloveTransport::Mcap) {
                detail::publish_quanta_video(
                    *(*server), video_state, foxglove_cfg->quanta, img_bgr, batch->timestamp_ns);
            } else {
                detail::publish_jpeg_image(*(*server), img_bgr, batch->timestamp_ns);
            }

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

} // namespace fcs::visualization::foxglove::systems
