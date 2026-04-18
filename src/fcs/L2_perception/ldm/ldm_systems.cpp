#include "L2_perception/ldm/ldm_systems.hpp"

#include "L2_perception/ldm/ldm_config.hpp"
#include "L2_perception/ldm/ldm_detector.hpp"
#include "L2_perception/ldm/ldm_solver.hpp"
#include "camera_config.hpp"
#include "core/channel_topics.hpp"
#include "core/runtime.hpp"
#include "core/types.hpp"
#include "frame.hpp"
#include "scheduler/scheduler.hpp"

#include <spdlog/spdlog.h>

namespace fcs::L2::ldm {

namespace {} // namespace

void register_ldm_systems(
    talos::Scheduler& scheduler, LdmDetectorConfig&& config,
    const CameraConfig& camera_config) noexcept {
    scheduler.world().insert_resource(config);

    auto solver_ptr = std::make_shared<LdmSolver>(camera_config, config);

    scheduler.add_system<talos::fixed_rate<200>>(
        "ldm_detector",
        [detector = std::make_shared<LdmDetector>(config), solver_ptr](
            talos::spmc<ImageFrame, ImageChannelTopic> image_in,
            talos::spmc_mut<LdmDetection, LdmDetectionChannelTopic> detection_out,
            talos::spmc_mut<LdmMeasurement, LdmMeasurementChannelTopic> measurement_out,
            talos::res<LdmDetectorConfig> cfg_, core::detecting_color detecting_color_,
            talos::res<fast_tf::CoordinateSystem> tf_system) mutable {
            if (!cfg_->enabled) {
                return;
            }
            auto frame = image_in.read();
            if (!frame) {
                return;
            }
            // Re-apply detect color before each detection and enforce color before PnP.
            detector->set_target_color(*detecting_color_);

            auto det_result = detector->detect(frame->image);
            if (!det_result) {
                return;
            }

            LdmDetection detection;
            detection.timestamp_ns = frame->timestamp_ns;
            detection.frame_id     = frame->frame_id;
            detection.color        = *detecting_color_;

            if (det_result->has_value()) {
                detection              = std::move(**det_result);
                detection.timestamp_ns = frame->timestamp_ns;
                detection.frame_id     = frame->frame_id;
                detection.color        = *detecting_color_;
            }

            std::optional<fast_tf::TransformMatrixd<fast_tf::odom, fast_tf::camera_optical>>
                T_camera_odom;
            auto tf_lookup = fast_tf::lookup_clamped<
                fast_tf::odom_fuxk_frame, fast_tf::camera_optical_fuxk_frame>(
                *tf_system, frame->timestamp_ns);
            if (tf_lookup) {
                T_camera_odom = *tf_lookup;
            }

            if (detection.valid) {
                auto meas_result = solver_ptr->solve(detection, T_camera_odom);
                if (meas_result) {
                    detection.mesh_candidates        = meas_result->mesh_candidates;
                    detection.selected_candidate_idx = meas_result->selected_candidate_idx;
                    detection.center_image_px        = meas_result->center_image_px;
                    measurement_out.write(std::move(*meas_result));
                }
            }

            detection_out.write(std::move(detection));
        });
}

} // namespace fcs::L2::ldm
