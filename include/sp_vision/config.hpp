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

/**
 * @brief PnP解算器配置
 */
struct SpVisionSolverConfig {
    // 相机内参
    std::vector<double> camera_matrix;  // 3x3 row-major
    std::vector<double> distort_coeffs; // 1x5

    // 外参：相机到云台、云台到IMU
    std::vector<double> R_camera2gimbal;  // 3x3 row-major
    std::vector<double> t_camera2gimbal;  // 3x1
    std::vector<double> R_gimbal2imubody; // 3x3 row-major

    std::expected<void, std::string> read_from(const toml::table &tbl) {
        if (const auto _t = tbl["solver"].as_table()) {
            auto t = *_t;

            // 读取相机内参
            if (const auto arr = t["camera_matrix"].as_array()) {
                camera_matrix.clear();
                for (const auto &v : *arr) {
                    if (const auto val = v.value<double>()) {
                        camera_matrix.push_back(*val);
                    }
                }
                if (camera_matrix.size() != 9) {
                    return std::unexpected("camera_matrix must have 9 elements");
                }
            } else {
                return std::unexpected("camera_matrix not found");
            }

            // 读取畸变系数
            if (const auto arr = t["distort_coeffs"].as_array()) {
                distort_coeffs.clear();
                for (const auto &v : *arr) {
                    if (const auto val = v.value<double>()) {
                        distort_coeffs.push_back(*val);
                    }
                }
                if (distort_coeffs.size() != 5) {
                    return std::unexpected("distort_coeffs must have 5 elements");
                }
            } else {
                return std::unexpected("distort_coeffs not found");
            }

            // 读取外参
            if (const auto arr = t["R_camera2gimbal"].as_array()) {
                R_camera2gimbal.clear();
                for (const auto &v : *arr) {
                    if (const auto val = v.value<double>()) {
                        R_camera2gimbal.push_back(*val);
                    }
                }
                if (R_camera2gimbal.size() != 9) {
                    return std::unexpected("R_camera2gimbal must have 9 elements");
                }
            } else {
                return std::unexpected("R_camera2gimbal not found");
            }

            if (const auto arr = t["t_camera2gimbal"].as_array()) {
                t_camera2gimbal.clear();
                for (const auto &v : *arr) {
                    if (const auto val = v.value<double>()) {
                        t_camera2gimbal.push_back(*val);
                    }
                }
                if (t_camera2gimbal.size() != 3) {
                    return std::unexpected("t_camera2gimbal must have 3 elements");
                }
            } else {
                return std::unexpected("t_camera2gimbal not found");
            }

            if (const auto arr = t["R_gimbal2imubody"].as_array()) {
                R_gimbal2imubody.clear();
                for (const auto &v : *arr) {
                    if (const auto val = v.value<double>()) {
                        R_gimbal2imubody.push_back(*val);
                    }
                }
                if (R_gimbal2imubody.size() != 9) {
                    return std::unexpected("R_gimbal2imubody must have 9 elements");
                }
            } else {
                return std::unexpected("R_gimbal2imubody not found");
            }

            return {};
        }
        return std::unexpected("solver section not found");
    }
};

/**
 * @brief 跟踪器配置
 */
struct SpVisionTrackerConfig {
    int enemy_color{0};              // 0=red, 1=blue
    int min_detect_count{3};         // 确认跟踪所需的最小检测次数
    int max_temp_lost_count{30};     // 临时丢失的最大帧数
    int outpost_max_temp_lost_count{100}; // 前哨站临时丢失的最大帧数

    std::expected<void, std::string> read_from(const toml::table &tbl) {
        if (const auto _t = tbl["tracker"].as_table()) {
            auto t = *_t;

            if (const auto val = t["enemy_color"].value<std::string>()) {
                enemy_color = (*val == "red") ? 0 : 1;
            }
            if (const auto val = t["min_detect_count"].value<int64_t>()) {
                min_detect_count = static_cast<int>(*val);
            }
            if (const auto val = t["max_temp_lost_count"].value<int64_t>()) {
                max_temp_lost_count = static_cast<int>(*val);
            }
            if (const auto val = t["outpost_max_temp_lost_count"].value<int64_t>()) {
                outpost_max_temp_lost_count = static_cast<int>(*val);
            }

            return {};
        }
        return std::unexpected("tracker section not found");
    }
};

struct SpVisionConfig : scheduler::RuntimeConfig {
    SpVisionClassifierConfig classifier{};
    SpVisionDetectorConfig detector{};
    SpVisionSolverConfig solver{};
    SpVisionTrackerConfig tracker{};

    std::expected<void, std::string> read_from(const toml::table &tbl) {
        if (auto read = read_runtime_config(this, tbl); !read) {
            return read;
        }
        if (auto read = classifier.read_from(tbl); !read) {
            return read;
        }
        if (auto read = detector.read_from(tbl); !read) {
            return read;
        }
        if (auto read = solver.read_from(tbl); !read) {
            return read;
        }
        if (auto read = tracker.read_from(tbl); !read) {
            return read;
        }
        return {};
    }
};
