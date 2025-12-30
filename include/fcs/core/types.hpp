#pragma once

#include <array>
#include <cstdint>
#include <optional>
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
// Armor Enums
// ============================================================================

enum class ArmorType : uint8_t {
    Small   = 0,
    Large   = 1,
    Invalid = 2,
};

const std::array<std::string, 2> ARMOR_TYPES = {"big", "small"};
constexpr std::string_view armor_type(const ArmorType c) noexcept {
    return ARMOR_TYPES[static_cast<uint8_t>(c)];
}

enum class ArmorColor : uint8_t {
    Red     = 0,
    Blue    = 1,
    Neutral = 2,
    Purple  = 3,
};

const std::array<std::string, 4> COLORS = {"red", "blue", "neutral", "purple"};
constexpr std::string_view armor_color(const ArmorColor c) noexcept {
    return COLORS[static_cast<std::underlying_type_t<decltype(c)>>(c)];
}

enum class ArmorName : uint8_t { One = 0, Two, Three, Four, Five, Sentry, Outpost, Base, Invalid };

const std::array<std::string, 9> ARMOR_NAMES = {"one",    "two",     "three", "four",   "five",
                                                "sentry", "outpost", "base",  "invalid"};
constexpr std::string_view armor_name(const ArmorName c) noexcept {
    return ARMOR_NAMES[static_cast<std::underlying_type_t<decltype(c)>>(c)];
}

enum class TrackStatus : uint8_t {
    Invalid   = 0, // 无效
    Tentative = 1, // 正在确认
    Confirmed = 2, // 已确认
    Coasting  = 3, // 暂时丢失量测， 使用预测值
};

const std::array<std::string, 4> TRACKER_STATE_NAMES = {
    "INVALID", "TENTATIVE", "CONFIRMED", "COASTING"};
constexpr std::string_view track_status(const TrackStatus c) noexcept {
    return TRACKER_STATE_NAMES[static_cast<std::underlying_type_t<decltype(c)>>(c)];
}

// ============================================================================
// AT Legacy Armor Constants (meters)
// ============================================================================

inline constexpr double AT_LEGACY_SMALL_ARMOR_WIDTH  = 135.0 / 1000.0;
inline constexpr double AT_LEGACY_SMALL_ARMOR_HEIGHT = 55.0 / 1000.0;
inline constexpr double AT_LEGACY_LARGE_ARMOR_WIDTH  = 230.0 / 1000.0;
inline constexpr double AT_LEGACY_LARGE_ARMOR_HEIGHT = 55.0 / 1000.0;

// ============================================================================
// AT Legacy cls/color Mapping (from NN output)
// ============================================================================

/// Map AT legacy NN cls output to ArmorName
/// cls: 0=sentry, 1-5=numbers, 6=outpost, 7=base_small, 8=base_large
[[nodiscard]] constexpr ArmorName at_legacy_cls_to_armor_name(int cls) noexcept {
    switch (cls) {
    case 0: return ArmorName::Sentry;
    case 1: return ArmorName::One;
    case 2: return ArmorName::Two;
    case 3: return ArmorName::Three;
    case 4: return ArmorName::Four;
    case 5: return ArmorName::Five;
    case 6: return ArmorName::Outpost;
    case 7:
    case 8: return ArmorName::Base;
    default: return ArmorName::Invalid;
    }
}

/// Map AT legacy NN color output to ArmorColor
/// color: 0=blue, 1=red, 2=gray, 3=purple
[[nodiscard]] constexpr ArmorColor at_legacy_color_to_armor_color(int color) noexcept {
    switch (color) {
    case 0: return ArmorColor::Blue;
    case 1: return ArmorColor::Red;
    case 2: return ArmorColor::Neutral;
    case 3: return ArmorColor::Purple;
    default: return ArmorColor::Neutral;
    }
}

/// Determine ArmorType from AT legacy cls
/// cls=1 (Hero) and cls=8 (base_large) are Large armor
[[nodiscard]] constexpr ArmorType at_legacy_cls_to_armor_type(int cls) noexcept {
    return (cls == 1 || cls == 8) ? ArmorType::Large : ArmorType::Small;
}

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
    std::array<cv::Point2f, 4> corners{}; // 4 corners: TL, TR, BR, BL
    cv::Rect2f rect{};
    ArmorName name   = ArmorName::Invalid;
    ArmorColor color = ArmorColor::Neutral;
    ArmorType type   = ArmorType::Invalid;
    float confidence = 0.0f;
    int area         = 0;

    AtLegacyArmorDetection() = default;

    /// Construct from NN output (cls, color as int)
    AtLegacyArmorDetection(std::array<cv::Point2f, 4> pts, int cls, int clr, float conf)
        : corners(pts)
        , name(at_legacy_cls_to_armor_name(cls))
        , color(at_legacy_color_to_armor_color(clr))
        , type(at_legacy_cls_to_armor_type(cls))
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

    /// Get 3D object points for PnP (armor coordinate: X forward, Y left, Z up)
    [[nodiscard]] std::vector<cv::Point3f> object_points() const {
        const double w =
            (type == ArmorType::Large) ? AT_LEGACY_LARGE_ARMOR_WIDTH : AT_LEGACY_SMALL_ARMOR_WIDTH;
        const double h = (type == ArmorType::Large) ? AT_LEGACY_LARGE_ARMOR_HEIGHT
                                                    : AT_LEGACY_SMALL_ARMOR_HEIGHT;
        const auto hw  = static_cast<float>(w / 2.0);
        const auto hh  = static_cast<float>(h / 2.0);
        // Order: TL, TR, BR, BL
        return {
            cv::Point3f(0, hw, hh), cv::Point3f(0, -hw, hh), cv::Point3f(0, -hw, -hh),
            cv::Point3f(0, hw, -hh)};
    }

    /// Get image points (4 corners)
    [[nodiscard]] std::vector<cv::Point2f> image_points() const {
        return {corners[0], corners[1], corners[2], corners[3]};
    }
};

/// Batch of detections from a single frame
struct AtLegacyArmorDetectionBatch {
    std::vector<AtLegacyArmorDetection> detections;
    uint64_t timestamp_ns = 0;
    uint64_t frame_id     = 0;

    AtLegacyArmorDetectionBatch() = default;
    AtLegacyArmorDetectionBatch(std::vector<AtLegacyArmorDetection> dets, uint64_t ts, uint64_t fid)
        : detections(std::move(dets))
        , timestamp_ns(ts)
        , frame_id(fid) {}

    [[nodiscard]] bool empty() const noexcept { return detections.empty(); }
    [[nodiscard]] size_t size() const noexcept { return detections.size(); }
};

/// Single armor measurement (3D pose in camera_optical frame)
struct AtLegacyArmorMeasurement {
    Eigen::Vector3d position{0, 0, 0};          // Position in camera_optical frame
    Eigen::Quaterniond orientation{1, 0, 0, 0}; // Orientation
    ArmorName name        = ArmorName::Invalid;
    ArmorColor color      = ArmorColor::Neutral;
    ArmorType type        = ArmorType::Small;
    float confidence      = 0.0f;
    uint64_t timestamp_ns = 0;

    AtLegacyArmorMeasurement() = default;

    /// Construct from Detection + PnP result
    AtLegacyArmorMeasurement(
        const AtLegacyArmorDetection& det, const cv::Mat& rvec, const cv::Mat& tvec, uint64_t ts)
        : name(det.name)
        , color(det.color)
        , type(det.type)
        , confidence(det.confidence)
        , timestamp_ns(ts) {
        // Convert tvec to position
        position = Eigen::Vector3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

        // Convert rvec to quaternion
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        Eigen::Matrix3d eigen_R;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                eigen_R(i, j) = R.at<double>(i, j);
            }
        }
        orientation = Eigen::Quaterniond(eigen_R);
    }

    /// Get distance to camera
    [[nodiscard]] double distance() const noexcept { return position.norm(); }

    /// Get yaw angle (in radians)
    [[nodiscard]] double yaw() const noexcept { return std::atan2(position.y(), position.x()); }

    /// Get pitch angle (in radians)
    [[nodiscard]] double pitch() const noexcept {
        return std::atan2(
            -position.z(), std::sqrt(position.x() * position.x() + position.y() * position.y()));
    }
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
