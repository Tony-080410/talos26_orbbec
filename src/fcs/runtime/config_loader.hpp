#pragma once

#include "config.hpp"
#include "runtime/capturer.hpp"

#include <expected>
#include <optional>
#include <string>
#include <string_view>

namespace fcs {

// Entry-point configuration parsed from at_vision.toml
struct LaunchConfig {
    bool daedalus{false};
    toml_helper::required<std::string> robot;
    toml_helper::required<std::string> vision;
    FoxgloveConfig foxglove;
    CapturerConfig capturer;
};

// Fully-resolved runtime configuration after merging base + override TOML
struct RuntimeConfig {
    RobotType robot_type;
    std::optional<HardwareConfig> hardware;
    FoxgloveConfig foxglove;
    CapturerConfig capturer;
    VisionConfig vision;
    runtime::CapturerLaunchContext launch;
};

/// Load and merge all configuration files from the given entry path.
/// Returns fully-resolved RuntimeConfig or a formatted error string.
[[nodiscard]] std::expected<RuntimeConfig, std::string> load_config(std::string_view path);

} // namespace fcs
