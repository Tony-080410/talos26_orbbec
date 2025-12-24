#pragma once
#include "scheduler/scheduler.hpp"

#include <Eigen/Eigen>
#include <expected>
#include <optional>

#include "detector_state.hpp"
#include "spdlog_hook.hpp"
#include "toml_helper.hpp"
#include <toml++/toml.hpp>

constexpr std::expected<void, std::string>
    read_runtime_config(scheduler::RuntimeConfig* config, const toml::table& tbl) noexcept {
    if (const auto _t = tbl["runtime"].as_table()) {
        auto t = *_t;
        READ(t, "detector_mode", config->detector_mode);
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

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
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
    Cpu  = 1,
    Gpu  = 2,
};

/**
 * @brief 分类器配置
 */
struct SpVisionClassifierConfig {
    std::string model_path;  // ONNX 模型路径
    bool use_openvino{true}; // 是否使用 OpenVINO (否则用 OpenCV DNN)
    DeviceType device{Auto}; // OpenVINO 设备: "AUTO", "CPU", "GPU"

    [[nodiscard]] constexpr std::string device_type() const noexcept {
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

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        if (const auto _t = tbl["classifier"].as_table()) {
            auto t = *_t;
            READ_KEY(t, model_path);
            READ_KEY(t, use_openvino);
            const std::optional<std::string> device_type = t["device"].value<std::string>();
            if (!device_type) {
                return std::unexpected("invalid device type");
            }
            if (const auto& dev = *device_type; dev == "AUTO") {
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
    Eigen::Matrix<double, 1, 5> distort_coefficient;            // 1x5

    // 外参：相机到云台、云台到IMU
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> extrinsic_camera_gimbal; // 3x3 row-major
    Eigen::Matrix<double, 3, 1> extrinsic_camera_gimbal_tf;               // 3x1
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> extrinsic_gimbal_imu;    // 3x3 row-major

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        if (const auto _t = tbl["solver"].as_table()) {
            auto t = *_t;
            PARSE_DOUBLE_MATRIX(c, t, camera_matrix);
            PARSE_DOUBLE_MATRIX(d, t, distort_coefficient);
            PARSE_DOUBLE_MATRIX(r, t, extrinsic_camera_gimbal);
            PARSE_DOUBLE_MATRIX(tt, t, extrinsic_camera_gimbal_tf);
            PARSE_DOUBLE_MATRIX(rr, t, extrinsic_gimbal_imu);
            return {};
        }
        return std::unexpected("solver section not found");
    }
};

/**
 * @brief 跟踪器配置
 */
struct SpVisionTrackerConfig {
    talos::Color enemy_color{talos::Color::red}; // 0=red, 1=blue
    uint16_t min_detect_count{3};                // 确认跟踪所需的最小检测次数
    uint16_t max_temp_lost_count{30};            // 临时丢失的最大帧数
    uint16_t outpost_max_temp_lost_count{100};   // 前哨站临时丢失的最大帧数

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
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

/**
 * @brief 瞄准器配置
 */
struct SpVisionAimerConfig {
    double yaw_offset{0.0};                   // yaw 偏移 (rad)
    double pitch_offset{0.0};                 // pitch 偏移 (rad)
    double comming_angle{70.0 / 57.3};        // 小陀螺来袭角度阈值 (rad)
    double leaving_angle{30.0 / 57.3};        // 小陀螺离开角度阈值 (rad)
    double high_speed_delay_time{0.1};        // 高速延时 (s)
    double low_speed_delay_time{0.05};        // 低速延时 (s)
    double decision_speed{2.0};               // 速度阈值 (rad/s)
    std::optional<double> left_yaw_offset{};  // 左枪管偏移 (rad)
    std::optional<double> right_yaw_offset{}; // 右枪管偏移 (rad)

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        if (const auto _t = tbl["aimer"].as_table()) {
            auto t = *_t;

            // 角度转弧度
            if (const auto val = t["yaw_offset"].value<double>()) {
                yaw_offset = *val / 57.3;
            }
            if (const auto val = t["pitch_offset"].value<double>()) {
                pitch_offset = *val / 57.3;
            }
            if (const auto val = t["comming_angle"].value<double>()) {
                comming_angle = *val / 57.3;
            }
            if (const auto val = t["leaving_angle"].value<double>()) {
                leaving_angle = *val / 57.3;
            }

            READ_KEY(t, high_speed_delay_time);
            READ_KEY(t, low_speed_delay_time);
            READ_KEY(t, decision_speed);

            // 可选项
            if (const auto val = t["left_yaw_offset"].value<double>()) {
                left_yaw_offset = *val / 57.3;
            }
            if (const auto val = t["right_yaw_offset"].value<double>()) {
                right_yaw_offset = *val / 57.3;
            }

            return {};
        }
        return std::unexpected("aimer section not found");
    }
};

/**
 * @brief 发射器配置
 */
struct SpVisionShooterConfig {
    double judge_distance{5.0};     // 距离判断阈值 (m)
    double first_tolerance{0.087};  // 近距离容差 (rad, 约5度)
    double second_tolerance{0.052}; // 远距离容差 (rad, 约3度)
    bool auto_fire{true};           // 是否启用自动发射

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        if (const auto _t = tbl["shooter"].as_table()) {
            auto t = *_t;

            READ_KEY(t, judge_distance);

            // 角度转弧度
            if (const auto val = t["first_tolerance"].value<double>()) {
                first_tolerance = *val / 57.3;
            }
            if (const auto val = t["second_tolerance"].value<double>()) {
                second_tolerance = *val / 57.3;
            }

            READ_KEY(t, auto_fire);

            return {};
        }
        return std::unexpected("shooter section not found");
    }
};

#define SUB(val, tbl)                              \
    if (auto read = (val).read_from(tbl); !read) { \
        return read;                               \
    }

struct SpVisionConfig : scheduler::RuntimeConfig {
    SpVisionClassifierConfig classifier{};
    SpVisionDetectorConfig detector{};
    SpVisionSolverConfig solver{};
    SpVisionTrackerConfig tracker{};
    SpVisionAimerConfig aimer{};
    SpVisionShooterConfig shooter{};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        if (auto read = read_runtime_config(this, tbl); !read) {
            return read;
        }
        SUB(classifier, tbl);
        SUB(detector, tbl);
        SUB(solver, tbl);
        SUB(tracker, tbl);
        SUB(aimer, tbl);
        SUB(shooter, tbl);
        return {};
    }
};
