#pragma once

#include "core/armor_types.hpp"
#include "frame.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>

namespace fcs {

struct at_legacy_armor_frame_tag {};

using at_legacy_armor_frame = at_legacy_armor_frame_tag;

// ============================================================================
// AT Legacy Frame Data Types (for Scheduler Channels)
// ============================================================================

/// Image frame with timestamp
struct AtLegacyImageFrame {
    cv::Mat image;
    uint64_t timestamp_ns = 0;
    uint64_t frame_id     = 0;

    AtLegacyImageFrame() = default;
    AtLegacyImageFrame(cv::Mat img, uint64_t ts, uint64_t fid)
        : image(std::move(img))
        , timestamp_ns(ts)
        , frame_id(fid) {}
};

/// Single armor detection result (2D, before PnP)
struct AtLegacyArmorDetection {
    /// 4 corners: TL, TR, BR, BL
    std::array<cv::Point2f, 4> corners{};
    cv::Rect2f rect{};
    ArmorName name   = ArmorName::Invalid;
    ArmorColor color = ArmorColor::Neutral;
    ArmorType type   = ArmorType::Invalid;
    float confidence = 0.0f;
    int area         = 0;

    AtLegacyArmorDetection() = default;

    /// Construct from NN output (armor_name, color as int)
    AtLegacyArmorDetection(
        std::array<cv::Point2f, 4> pts, ArmorName armor_name, ArmorColor color, float conf);

    /// Get center point
    [[nodiscard]] cv::Point2f center() const noexcept {
        return (corners[0] + corners[1] + corners[2] + corners[3]) * 0.25f;
    }

    /// Get image points (4 corners)
    [[nodiscard]] std::vector<cv::Point2f> image_points() const {
        return {corners[0], corners[1], corners[2], corners[3]};
    }
};

/// Batch of detections from a single frame
struct AtLegacyArmorDetectionBatch {
    std::vector<AtLegacyArmorDetection> detections;
    cv::Mat image; ///< Source image (shared reference for visualization)
    bool has_detector_roi = false;
    cv::Rect detector_roi{};
    uint64_t timestamp_ns = 0;
    uint64_t frame_id     = 0;

    AtLegacyArmorDetectionBatch() = default;
    AtLegacyArmorDetectionBatch(
        std::vector<AtLegacyArmorDetection> dets, cv::Mat img, uint64_t ts, uint64_t fid,
        bool has_roi = false, cv::Rect roi = {})
        : detections(std::move(dets))
        , image(std::move(img))
        , has_detector_roi(has_roi)
        , detector_roi(roi)
        , timestamp_ns(ts)
        , frame_id(fid) {}

    [[nodiscard]] bool empty() const noexcept { return detections.empty(); }
    [[nodiscard]] size_t size() const noexcept { return detections.size(); }
};

/// Single armor measurement (3D pose in odom frame after TF transform)
template <fast_tf::frame Frame>
struct AtLegacyArmorMeasurementT {
    using Transform = fast_tf::TransformMatrixd<at_legacy_armor_frame, Frame>;

    Transform transform{};
    ArmorName name                 = ArmorName::Invalid;
    ArmorColor color               = ArmorColor::Neutral;
    ArmorType type                 = ArmorType::Small;
    float confidence               = 0.0f;
    float distance_to_image_center = 0.0f; // Pixel distance from armor center to image center
    uint64_t timestamp_ns          = 0;

    AtLegacyArmorMeasurementT() = default;

    /// Get distance to camera
    [[nodiscard]] double distance() const noexcept { return transform.translation().norm(); }

    /// Get yaw angle (in radians)
    [[nodiscard]] double yaw() const noexcept { return transform.euler_rot().yaw; }

    /// Get pitch angle (in radians)
    [[nodiscard]] double pitch() const noexcept { return transform.euler_rot().pitch; }

    template <fast_tf::frame ToFrame>
    [[nodiscard]] AtLegacyArmorMeasurementT<ToFrame>
        reframe(const fast_tf::FrameTransform<Frame, ToFrame>& frame_transform) const {
        AtLegacyArmorMeasurementT<ToFrame> reframed;
        reframed.transform                = transform * frame_transform;
        reframed.name                     = name;
        reframed.color                    = color;
        reframed.type                     = type;
        reframed.confidence               = confidence;
        reframed.distance_to_image_center = distance_to_image_center;
        reframed.timestamp_ns             = timestamp_ns;
        return reframed;
    }
};

using AtLegacyArmorMeasurement       = AtLegacyArmorMeasurementT<fast_tf::odom>;
using AtLegacyCameraArmorMeasurement = AtLegacyArmorMeasurementT<fast_tf::camera_optical>;

/// Batch of measurements from a single frame
template <fast_tf::frame Frame>
struct AtLegacyArmorMeasurementBatchT {
    std::vector<AtLegacyArmorMeasurementT<Frame>> measurements;
    uint64_t timestamp_ns = 0;
    uint64_t frame_id     = 0;

    AtLegacyArmorMeasurementBatchT() = default;
    AtLegacyArmorMeasurementBatchT(
        std::vector<AtLegacyArmorMeasurementT<Frame>> meas, uint64_t ts, uint64_t fid)
        : measurements(std::move(meas))
        , timestamp_ns(ts)
        , frame_id(fid) {}

    [[nodiscard]] bool empty() const noexcept { return measurements.empty(); }
    [[nodiscard]] size_t size() const noexcept { return measurements.size(); }
};

using AtLegacyArmorMeasurementBatch = AtLegacyArmorMeasurementBatchT<fast_tf::odom>;

} // namespace fcs
