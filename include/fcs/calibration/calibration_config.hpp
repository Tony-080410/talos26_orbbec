#pragma once

#include "calibration_types.hpp"
#include "toml_helper.hpp"

#include <expected>
#include <string>

namespace fcs::calibration {

/// @brief Calibration board configuration
struct BoardConfig {
    BoardType type{BoardType::Chessboard};
    uint32_t width{11};       ///< Number of inner corners (width)
    uint32_t height{8};       ///< Number of inner corners (height)
    double square_size{0.02}; ///< Square size in meters

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ(tbl, type);
        READ(tbl, width);
        READ(tbl, height);
        READ(tbl, square_size);
        return {};
    }
};

/// @brief ChArUco-specific configuration
struct CharucoConfig {
    double marker_size{0.015}; ///< ArUco marker size in meters
    ArucoDictionary dictionary{ArucoDictionary::DICT_6X6_250};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ_OPT(tbl, marker_size);
        READ_OPT(tbl, dictionary);
        return {};
    }
};

/// @brief Sample capture configuration
struct CaptureConfig {
    uint32_t min_samples{30};          ///< Minimum samples required for calibration
    uint32_t max_samples{100};         ///< Maximum samples to collect
    double min_angle_diff{15.0};       ///< Minimum angle difference between samples (degrees)
    double min_translation_diff{0.05}; ///< Minimum translation difference (meters)
    bool auto_capture{false};          ///< Automatic capture mode
    uint32_t capture_interval_ms{500}; ///< Auto capture interval (milliseconds)

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ_OPT(tbl, min_samples);
        READ_OPT(tbl, max_samples);
        READ_OPT(tbl, min_angle_diff);
        READ_OPT(tbl, min_translation_diff);
        READ_OPT(tbl, auto_capture);
        READ_OPT(tbl, capture_interval_ms);
        return {};
    }
};

/// @brief Intrinsic calibration flags
struct IntrinsicConfig {
    bool fix_aspect_ratio{false};    ///< Fix fx/fy ratio to 1
    bool fix_principal_point{false}; ///< Fix principal point to image center
    bool zero_tangent_dist{false};   ///< Set tangential distortion to zero

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ_OPT(tbl, fix_aspect_ratio);
        READ_OPT(tbl, fix_principal_point);
        READ_OPT(tbl, zero_tangent_dist);
        return {};
    }

    /// @brief Convert to OpenCV calibration flags
    [[nodiscard]] int to_opencv_flags() const noexcept {
        int flags = 0;
        if (fix_aspect_ratio)
            flags |= cv::CALIB_FIX_ASPECT_RATIO;
        if (fix_principal_point)
            flags |= cv::CALIB_FIX_PRINCIPAL_POINT;
        if (zero_tangent_dist)
            flags |= cv::CALIB_ZERO_TANGENT_DIST;
        return flags;
    }
};

/// @brief Hand-eye calibration configuration
struct HandeyeConfig {
    HandEyeMethod method{HandEyeMethod::Tsai};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ_OPT(tbl, method);
        return {};
    }
};

/// @brief Foxglove visualization configuration
struct VisualizationConfig {
    bool show_corners{true};      ///< Show detected corners overlay
    bool show_reprojection{true}; ///< Show reprojection error visualization
    bool show_pose{true};         ///< Show board pose axes

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ_OPT(tbl, show_corners);
        READ_OPT(tbl, show_reprojection);
        READ_OPT(tbl, show_pose);
        return {};
    }
};

/// @brief Output path configuration
struct OutputConfig {
    std::string intrinsic_path{"camera_intrinsic.toml"};
    std::string extrinsic_path{"camera_extrinsic.toml"};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ_OPT(tbl, intrinsic_path);
        READ_OPT(tbl, extrinsic_path);
        return {};
    }
};

/// @brief Foxglove server configuration
struct FoxgloveCalibConfig {
    bool enabled{true};
    std::string host{"0.0.0.0"};
    uint16_t port{8766}; // Different from main program (8765)

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ_OPT(tbl, enabled);
        READ_OPT(tbl, host);
        READ_OPT(tbl, port);
        return {};
    }
};

/// @brief Input source configuration (hardware vs simulator)
struct InputConfig {
    bool daedalus{false}; ///< Use daedalus simulator instead of real hardware

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ_OPT(tbl, daedalus);
        return {};
    }
};

/// @brief Main calibration configuration
struct CalibrationConfig {
    CalibrationMode mode{CalibrationMode::Intrinsic};
    BoardConfig board{};
    CharucoConfig charuco{};
    CaptureConfig capture{};
    IntrinsicConfig intrinsic{};
    HandeyeConfig handeye{};
    VisualizationConfig visualization{};
    OutputConfig output{};
    FoxgloveCalibConfig foxglove{};
    InputConfig input{}; ///< Input source (hardware/daedalus)

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ(tbl, mode);
        READ(tbl, board);
        READ_OPT(tbl, charuco);
        READ_OPT(tbl, capture);
        READ_OPT(tbl, intrinsic);
        READ_OPT(tbl, handeye);
        READ_OPT(tbl, visualization);
        READ_OPT(tbl, output);
        READ_OPT(tbl, foxglove);
        READ_OPT(tbl, input);
        return {};
    }

    /// @brief Load configuration from TOML file
    [[nodiscard]] static std::expected<CalibrationConfig, std::string>
        load_from_file(const std::string& path) noexcept {
        auto result = toml::parse_file(path);
        if (!result) {
            return std::unexpected(
                fmt::format("Failed to parse {}: {}", path, result.error().description()));
        }

        const auto& tbl = result.table();
        if (auto calib_tbl = tbl["calibration"].as_table()) {
            CalibrationConfig config;
            if (auto res = config.read_from(*calib_tbl); !res) {
                return std::unexpected(res.error());
            }
            return config;
        }

        return std::unexpected("Missing [calibration] section in config file");
    }
};

} // namespace fcs::calibration
