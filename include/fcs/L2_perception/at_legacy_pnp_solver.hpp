#pragma once

#include "../core/types.hpp"

#include <array>
#include <expected>
#include <optional>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

namespace fcs {

// ============================================================================
// PnP Solver Error Types
// ============================================================================

enum class PnPError {
    NotInitialized,
    InvalidDetection,
    SolveFailed,
};

[[nodiscard]] constexpr std::string_view pnp_error_str(PnPError e) noexcept {
    switch (e) {
    case PnPError::NotInitialized: return "Not initialized";
    case PnPError::InvalidDetection: return "Invalid detection";
    case PnPError::SolveFailed: return "Solve failed";
    }
    return "Unknown error";
}

// ============================================================================
// AT Legacy PnP Solver
// ============================================================================

/// PnP solver for armor pose estimation
class AtLegacyPnPSolver {
public:
    using PoseResult = std::expected<AtLegacyArmorMeasurement, PnPError>;

    AtLegacyPnPSolver() = default;

    /// Initialize camera parameters
    /// @param config camera config
    bool init(const CameraConfig& config) {
        cv::eigen2cv(config.camera_matrix, camera_matrix_);
        cv::eigen2cv(config.distort_coefficient, dist_coeffs_);

        // Pre-compute image center for distance calculation
        image_center_ = cv::Point2f(
            config.width / 2.0, // cx
            config.height / 2.0 // cy
        );

        build_model_points();
        initialized_ = true;
        return true;
    }

    /// Solve PnP for a single detection
    /// @param detection Armor detection with 4 corners
    /// @param timestamp_ns Timestamp for the measurement
    /// @return Measurement with 3D pose, or error
    [[nodiscard]] PoseResult
        solve(const AtLegacyArmorDetection& detection, uint64_t timestamp_ns) const {
        if (!initialized_) {
            return std::unexpected(PnPError::NotInitialized);
        }

        // Get object and image points
        const auto& obj_points =
            (detection.type == ArmorType::Large) ? large_armor_points_ : small_armor_points_;
        const auto img_points = detection.image_points();

        if (img_points.size() != 4) {
            return std::unexpected(PnPError::InvalidDetection);
        }

        // Solve PnP using IPPE (Infinitesimal Plane-based Pose Estimation)
        cv::Mat rvec, tvec;
        bool success = cv::solvePnP(
            obj_points, img_points, camera_matrix_, dist_coeffs_, rvec, tvec,
            false, // useExtrinsicGuess
            cv::SOLVEPNP_IPPE);

        if (!success) {
            return std::unexpected(PnPError::SolveFailed);
        }

        return AtLegacyArmorMeasurement(detection, rvec, tvec, timestamp_ns);
    }

    /// Solve PnP for multiple detections
    [[nodiscard]] std::vector<AtLegacyArmorMeasurement> solve_batch(
        const std::vector<AtLegacyArmorDetection>& detections, uint64_t timestamp_ns) const {
        std::vector<AtLegacyArmorMeasurement> measurements;
        measurements.reserve(detections.size());

        for (const auto& det : detections) {
            auto result = solve(det, timestamp_ns);
            if (result) {
                measurements.push_back(std::move(*result));
            }
        }

        return measurements;
    }

    /// Calculate distance from armor center to image center
    [[nodiscard]] float distance_to_center(const AtLegacyArmorDetection& detection) const noexcept {
        return cv::norm(detection.center() - image_center_);
    }

    [[nodiscard]] bool is_initialized() const noexcept { return initialized_; }

    /// Get camera matrix
    [[nodiscard]] const cv::Mat& camera_matrix() const noexcept { return camera_matrix_; }

    /// Get distortion coefficients
    [[nodiscard]] const cv::Mat& dist_coeffs() const noexcept { return dist_coeffs_; }

private:
    void build_model_points() {
        // Small armor: 135mm x 55mm
        const float sw      = static_cast<float>(AT_LEGACY_SMALL_ARMOR_WIDTH / 2.0);
        const float sh      = static_cast<float>(AT_LEGACY_SMALL_ARMOR_HEIGHT / 2.0);
        small_armor_points_ = {
            cv::Point3f(0, sw, sh),   // Top-left
            cv::Point3f(0, -sw, sh),  // Top-right
            cv::Point3f(0, -sw, -sh), // Bottom-right
            cv::Point3f(0, sw, -sh)   // Bottom-left
        };

        // Large armor: 230mm x 55mm
        const float lw      = static_cast<float>(AT_LEGACY_LARGE_ARMOR_WIDTH / 2.0);
        const float lh      = static_cast<float>(AT_LEGACY_LARGE_ARMOR_HEIGHT / 2.0);
        large_armor_points_ = {
            cv::Point3f(0, lw, lh),   // Top-left
            cv::Point3f(0, -lw, lh),  // Top-right
            cv::Point3f(0, -lw, -lh), // Bottom-right
            cv::Point3f(0, lw, -lh)   // Bottom-left
        };
    }

private:
    bool initialized_ = false;
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    cv::Point2f image_center_;

    std::vector<cv::Point3f> small_armor_points_;
    std::vector<cv::Point3f> large_armor_points_;
};

} // namespace fcs
