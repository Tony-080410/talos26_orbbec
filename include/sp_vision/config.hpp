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


struct SpVisionConfig : scheduler::RuntimeConfig {
    SpVisionDetectorConfig detector{};

    constexpr std::expected<void, std::string> read_from(const toml::table &tbl) {
        if (auto read = read_runtime_config(this, tbl); !read) {
            return read;
        }
        if (auto read = detector.read_from(tbl); !read) {
            return read;
        }
        return {};
    }
};


constexpr auto convert(const SpVisionDetectorConfig &config) {
    return auto_aim::DetectorConfig{config.threshold,           config.max_angle_error,
                                    config.min_lightbar_ratio,  config.max_lightbar_ratio,
                                    config.min_lightbar_length, config.min_armor_ratio,
                                    config.max_armor_ratio,     config.max_side_ratio,
                                    config.min_confidence,      config.max_rectangular_error};
}
