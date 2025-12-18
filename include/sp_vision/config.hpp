#pragma once
#include "scheduler/scheduler.hpp"

#include <Eigen/Eigen>
#include <expected>

#include "detector_state.hpp"
#include "spdlog_hook.hpp"
#include "toml++/toml.hpp"
#include "toml_helper.hpp"

constexpr std::expected<void, std::string> read_runtime_config(scheduler::RuntimeConfig *config,
                                                               const toml::table &tbl) {
    if (const auto _t = tbl["runtime"].as_table()) {
        auto t = *_t;
        READ(t, "detector_mode", config->detector_mode);
        READ(t, "detector_threads", config->detector_threads);
        READ(t, "max_frame_skip", config->max_frame_skip);
        return {};
    }
    return std::unexpected("runtime is not a table");
}

struct SpVisionDetectorConfig {
    double threshold;
    double max_angle_error;
    double min_lightbar_ratio;
    double max_lightbar_ratio;
    double min_lightbar_length;
    double min_armor_ratio;
    double max_armor_ratio;
    double max_side_ratio;
    double min_confidence;
    double max_rectangular_error;

    std::expected<void, std::string> read_from(const toml::table &tbl) {
        if (const auto _t = tbl["detector"].as_table()) {
            auto t = *_t;
            READ_KEY(t, threshold);
            READ_KEY(t, max_angle_error);
            READ_KEY(t, min_lightbar_ratio);
            READ_KEY(t, max_lightbar_ratio);
            READ_KEY(t, min_lightbar_length);
            READ_KEY(t, min_armor_ratio);
            READ_KEY(t, max_armor_ratio);
            READ_KEY(t, max_side_ratio);
            READ_KEY(t, min_confidence);
            READ_KEY(t, max_rectangular_error);
            return {};
        }
        return std::unexpected("detector is not a table");
    }
};

enum DeviceType : int {
    Auto = 0,
    Cpu = 1,
    Gpu = 2,
};

/**
 * @brief 分类器配置
 */
struct SpVisionClassifierConfig {
    std::string model_path; // ONNX 模型路径
    bool use_openvino{true}; // 是否使用 OpenVINO (否则用 OpenCV DNN)
    DeviceType device{Auto}; // OpenVINO 设备: "AUTO", "CPU", "GPU"

    [[nodiscard]] constexpr std::string device_type() const {
        if (device == Auto) {
            return "AUTO";
        }
        if (device == Cpu) {
            return "CPU";
        }
        if (device == Gpu) {
            return "GPU";
        }
        return {};
    }

    std::expected<void, std::string> read_from(const toml::table &tbl) {
        if (const auto _t = tbl["classifier"].as_table()) {
            auto t = *_t;
            READ_KEY(t, model_path);
            READ_KEY(t, use_openvino);
            const std::optional<std::string> device_type = t["device"].value<std::string>();
            if (!device_type) {
                return std::unexpected("invalid device type");
            }
            if (const auto &dev = *device_type; dev == "AUTO") {
                device = Auto;
            } else if (dev == "CPU") {
                device = Cpu;
            } else if (dev == "GPU") {
                device = Gpu;
            } else {
                return std::unexpected("invalid device type: " + dev);
            }
            return {};
        }
        return std::unexpected("classifier is not a table");
    }
};

/**
 * @brief PnP解算器配置
 */
struct SpVisionSolverConfig {
    // 相机内参
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix; // 3x3 row-major
    Eigen::Matrix<double, 1, 5> distort_coeffs; // 1x5

    // 外参：相机到云台、云台到IMU
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R_camera2gimbal; // 3x3 row-major
    Eigen::Matrix<double, 3, 1> t_camera2gimbal; // 3x1
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R_gimbal2imubody; // 3x3 row-major

    std::expected<void, std::string> read_from(const toml::table &tbl) {
        if (const auto _t = tbl["solver"].as_table()) {
            auto t = *_t;
            PARSE_DOUBLE_ARRAY(c, t, "camera_matrix", camera_matrix, 9);
            PARSE_DOUBLE_ARRAY(d, t, "distort_coeffs", distort_coeffs, 5);
            PARSE_DOUBLE_ARRAY(r, t, "R_camera2gimbal", R_camera2gimbal, 9);
            PARSE_DOUBLE_ARRAY(tt, t, "t_camera2gimbal", t_camera2gimbal, 3);
            PARSE_DOUBLE_ARRAY(rr, t, "R_gimbal2imubody", R_gimbal2imubody, 9);
            return {};
        }
        return std::unexpected("solver section not found");
    }
};

/**
 * @brief 跟踪器配置
 */
struct SpVisionTrackerConfig {
    spike::Color enemy_color{spike::Color::red}; // 0=red, 1=blue
    uint16_t min_detect_count{3}; // 确认跟踪所需的最小检测次数
    uint16_t max_temp_lost_count{30}; // 临时丢失的最大帧数
    uint16_t outpost_max_temp_lost_count{100}; // 前哨站临时丢失的最大帧数

    std::expected<void, std::string> read_from(const toml::table &tbl) {
        if (const auto _t = tbl["tracker"].as_table()) {
            auto t = *_t;

            READ_KEY(t, enemy_color);
            READ_KEY(t, min_detect_count);
            READ_KEY(t, max_temp_lost_count);
            READ_KEY(t, outpost_max_temp_lost_count);

            return {};
        }
        return std::unexpected("tracker section not found");
    }
};

#define SUB(val, tbl)                                                                                                  \
    if (auto read = (val).read_from(tbl); !read) {                                                                     \
        return read;                                                                                                   \
    }

struct SpVisionConfig : scheduler::RuntimeConfig {
    SpVisionClassifierConfig classifier;
    SpVisionDetectorConfig detector;
    SpVisionSolverConfig solver;
    SpVisionTrackerConfig tracker;

    std::expected<void, std::string> read_from(const toml::table &tbl) {
        if (auto read = read_runtime_config(this, tbl); !read) {
            return read;
        }
        SUB(classifier, tbl);
        SUB(detector, tbl);
        SUB(solver, tbl);
        SUB(tracker, tbl);
        return {};
    }
};
