#pragma once

#include "core/armor_types.hpp"
#include "frame.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace fcs {

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
        std::array<cv::Point2f, 4> pts, ArmorName armor_name, ArmorColor color, float conf)
        : corners(pts)
        , name(armor_name)
        , color(color)
        , type(at_legacy_cls_to_armor_type(armor_name))
        , confidence(conf) {
        // Calculate bounding rect
        std::vector<cv::Point2f> pts_vec(corners.begin(), corners.end());
        rect = cv::boundingRect(pts_vec);
        area = static_cast<int>(cv::contourArea(pts_vec));
    }

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
    uint64_t timestamp_ns = 0;
    uint64_t frame_id     = 0;

    AtLegacyArmorDetectionBatch() = default;
    AtLegacyArmorDetectionBatch(
        std::vector<AtLegacyArmorDetection> dets, cv::Mat img, uint64_t ts, uint64_t fid)
        : detections(std::move(dets))
        , image(std::move(img))
        , timestamp_ns(ts)
        , frame_id(fid) {}

    [[nodiscard]] bool empty() const noexcept { return detections.empty(); }
    [[nodiscard]] size_t size() const noexcept { return detections.size(); }
};

/// Single armor measurement (3D pose in odom frame after TF transform)
struct AtLegacyArmorMeasurement {
    fast_tf::FuckTransform<fast_tf::odom_fuxk_frame> transform;
    ArmorName name                 = ArmorName::Invalid;
    ArmorColor color               = ArmorColor::Neutral;
    ArmorType type                 = ArmorType::Small;
    float confidence               = 0.0f;
    float distance_to_image_center = 0.0f; // Pixel distance from armor center to image center
    uint64_t timestamp_ns          = 0;

    AtLegacyArmorMeasurement() = default;

    /// Construct from Detection + PnP result
    ///
    /// NOTE: from_pnp returns the pose in camera_optical frame (OpenCV convention):
    /// - X axis: right
    /// - Y axis: down
    /// - Z axis: forward (pointing into the scene)
    ///
    /// The transform here is temporarily stored as odom frame for type consistency,
    /// but the actual data is in camera_optical frame. It will be properly
    /// transformed to odom frame by the detection system in at_legacy_systems.cpp.
    AtLegacyArmorMeasurement(
        const AtLegacyArmorDetection& det, const cv::Mat& rvec, const cv::Mat& tvec, uint64_t ts,
        float dist_to_center = 0.0f)
        : name(det.name)
        , color(det.color)
        , type(det.type)
        , confidence(det.confidence)
        , distance_to_image_center(dist_to_center)
        , timestamp_ns(ts) {
        // Store PnP result (camera_optical frame) in odom-typed transform.
        // This will be corrected by the TF transform in at_legacy_systems.cpp.
        transform = fast_tf::FuckTransform<fast_tf::odom_fuxk_frame>(
            fast_tf::TransformMatrix4d::from_pnp(rvec, tvec));
    }

    /// Get distance to camera
    [[nodiscard]] double distance() const noexcept { return transform->translation().norm(); }

    /// Get yaw angle (in radians)
    [[nodiscard]] double yaw() const noexcept { return transform->euler_rot().yaw; }

    /// Get pitch angle (in radians)
    [[nodiscard]] double pitch() const noexcept { return transform->euler_rot().pitch; }
};

/// Batch of measurements from a single frame
struct AtLegacyArmorMeasurementBatch {
    std::vector<AtLegacyArmorMeasurement> measurements;
    uint64_t timestamp_ns = 0;
    uint64_t frame_id     = 0;

    AtLegacyArmorMeasurementBatch() = default;
    AtLegacyArmorMeasurementBatch(
        std::vector<AtLegacyArmorMeasurement> meas, uint64_t ts, uint64_t fid)
        : measurements(std::move(meas))
        , timestamp_ns(ts)
        , frame_id(fid) {}

    [[nodiscard]] bool empty() const noexcept { return measurements.empty(); }
    [[nodiscard]] size_t size() const noexcept { return measurements.size(); }
};

} // namespace fcs
