#pragma once
#include "scheduler/scheduler.hpp"

#include <expected>
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
            READ(t, "threshold", threshold);
            READ(t, "max_angle_error", max_angle_error);
            READ(t, "min_lightbar_ratio", min_lightbar_ratio);
            READ(t, "max_lightbar_ratio", max_lightbar_ratio);
            READ(t, "min_lightbar_length", min_lightbar_length);
            READ(t, "min_armor_ratio", min_armor_ratio);
            READ(t, "max_armor_ratio", max_armor_ratio);
            READ(t, "max_side_ratio", max_side_ratio);
            READ(t, "min_confidence", min_confidence);
            READ(t, "max_rectangular_error", max_rectangular_error);
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
            READ(t, "model_path", model_path);
            READ(t, "use_openvino", use_openvino);
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

struct SpVisionConfig : scheduler::RuntimeConfig {
    SpVisionClassifierConfig classifier{};
    SpVisionDetectorConfig detector{};

    constexpr std::expected<void, std::string> read_from(const toml::table &tbl) {
        if (auto read = read_runtime_config(this, tbl); !read) {
            return read;
        }
        if (auto read = classifier.read_from(tbl); !read) {
            return read;
        }
        if (auto read = detector.read_from(tbl); !read) {
            return read;
        }
        return {};
    }
};
