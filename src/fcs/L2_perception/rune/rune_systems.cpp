#include "L2_perception/rune/rune_systems.hpp"

#include "L2_perception/rune/rune_config.hpp"
#include "L2_perception/rune/rune_detector.hpp"
#include "L2_perception/rune/types.hpp"
#include "camera_config.hpp"
#include "core/channel_topics.hpp"
#include "core/types.hpp"
#include "frame.hpp"
#include "scheduler/scheduler.hpp"

#include <spdlog/spdlog.h>

#include <opencv2/core/eigen.hpp>
#include <opencv2/imgcodecs.hpp>

#include <memory>
#include <utility>

namespace fcs::rune {
namespace {

[[nodiscard]] auto to_rectf(const cv::Rect2f& r) noexcept -> RectF {
    return RectF{.x = r.x, .y = r.y, .w = r.width, .h = r.height};
}

[[nodiscard]] auto encode_jpeg(const cv::Mat& img) -> std::vector<uint8_t> {
    if (img.empty()) {
        return {};
    }

    std::vector<uint8_t> encoded;
    std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, 80};
    if (!cv::imencode(".jpg", img, encoded, params)) {
        return {};
    }
    return encoded;
}

[[nodiscard]] auto to_enemy_color(ArmorColor color) -> ::fyt::rune::EnemyColor {
    if (color == ArmorColor::Red) {
        return ::fyt::rune::EnemyColor::RED;
    }
    return ::fyt::rune::EnemyColor::BLUE;
}

[[nodiscard]] auto ns_to_sec(uint64_t ns) -> double { return static_cast<double>(ns) * 1e-9; }

} // namespace

void register_rune_detection_systems(talos::Scheduler& scheduler, RuneDetectorConfig&& config) {
    scheduler.world().insert_resource(config);

    scheduler.add_system<talos::pool_compute>(
        "l2_rune_detector",
        [detector = std::make_shared<::fyt::rune::RuneDetector>(
             to_enemy_color(config.detect_color), config.arrow_threshold, config.target_threshold,
             config.rcenter_threshold)](
            talos::spmc<AtLegacyImageFrame, AtLegacyImageChannelTopic> image_in,
            talos::spmc_mut<RuneObservation, RuneObservationChannelTopic> obs_out,
            talos::spmc_mut<RuneDebugFrame, RuneDebugFrameChannelTopic> debug_out,
            talos::res<RuneDetectorConfig> cfg, talos::res<CameraConfig> cam_cfg,
            talos::res<std::shared_ptr<fast_tf::CoordinateSystem<1024>>> tf_system) mutable {
            if (!image_in.has_new()) {
                return;
            }

            auto frame = image_in.read();
            if (!frame) {
                return;
            }

            RuneObservation out;
            out.timestamp_ns = frame->timestamp_ns;
            out.frame_id     = frame->frame_id;
            out.valid        = false;

            RuneDebugFrame dbg;
            dbg.timestamp_ns = frame->timestamp_ns;
            dbg.frame_id     = frame->frame_id;
            dbg.enabled      = cfg->enabled;

            if (!cfg->enabled) {
                dbg.status_code = 0;
                debug_out.write(std::move(dbg));
                obs_out.write(out);
                return;
            }

            if (cfg.is_changed()) {
                *detector = ::fyt::rune::RuneDetector(
                    to_enemy_color(cfg->detect_color), cfg->arrow_threshold, cfg->target_threshold,
                    cfg->rcenter_threshold);
            }

            const auto reversed =
                detector->detect(frame->image, frame->image.cols, frame->image.rows);
            ;

            if (!detector->targets.empty() && detector->rcenter.center != cv::Point2f(0, 0)) {
                detector->setKeyPoints();
            }

            dbg.detect_reversed = reversed;
            dbg.status_code     = static_cast<uint32_t>(detector->status);
            dbg.arrows_count    = static_cast<uint32_t>(detector->arrows.size());
            dbg.targets_count   = static_cast<uint32_t>(detector->targets.size());
            dbg.global_roi      = to_rectf(detector->globalRoi);
            dbg.center_roi      = to_rectf(detector->centerRoi);
            dbg.target_rois.reserve(detector->targetROIs.size());
            for (const auto& r : detector->targetROIs) {
                dbg.target_rois.emplace_back(to_rectf(r));
            }
            dbg.arrow_jpeg   = encode_jpeg(detector->arrowImg);
            dbg.target_jpeg  = encode_jpeg(detector->targetImg);
            dbg.rcenter_jpeg = encode_jpeg(detector->rCenterImg);

            // if (!dbg.detect_reversed) {
            // debug_out.write(std::move(dbg));
            // obs_out.write(out);
            //    return;
            //}

            auto tf_odom_camera = fast_tf::lookup_clamped<fast_tf::odom>(
                **tf_system, fast_tf::FuckTransform<fast_tf::camera_optical>(),
                frame->timestamp_ns);

            if (!tf_odom_camera) {
                SPDLOG_DEBUG(
                    "rune: TF lookup failed at {}: {}", frame->timestamp_ns,
                    tf_odom_camera.error());
                dbg.tf_ok = false;
                debug_out.write(std::move(dbg));
                obs_out.write(out);
                return;
            }
            dbg.tf_ok = true;

            cv::Mat camera_matrix;
            cv::Mat dist_coeffs;
            cv::eigen2cv(cam_cfg->camera_matrix, camera_matrix);
            cv::eigen2cv(cam_cfg->distort_coefficient, dist_coeffs);

            const Eigen::Matrix4d odom_T_cam = tf_odom_camera.value()->matrix();
            const double timestamp_sec       = ns_to_sec(frame->timestamp_ns);

            auto pose_result =
                detector->solve(camera_matrix, dist_coeffs, timestamp_sec, odom_T_cam);
            dbg.solve_ok = pose_result.has_value();
            if (!pose_result) {
                debug_out.write(std::move(dbg));
                obs_out.write(out);
                return;
            }

            const Eigen::Matrix4d& odom_T_rcenter = *pose_result;
            const Eigen::Matrix3d rot             = odom_T_rcenter.block<3, 3>(0, 0);

            out.valid         = true;
            out.r_center_odom = odom_T_rcenter.block<3, 1>(0, 3);

            const Eigen::Vector3d blade_center_local(0.0, 0.0, ::fyt::rune::RUNE_R2PANCENTER);
            out.target_positions_odom.emplace_back(rot * blade_center_local + out.r_center_odom);
            out.target_quats_odom.emplace_back(rot);

            dbg.observation_valid = true;
            debug_out.write(std::move(dbg));
            obs_out.write(out);
        });
}

} // namespace fcs::rune
