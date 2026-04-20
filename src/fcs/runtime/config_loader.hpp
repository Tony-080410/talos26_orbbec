#pragma once

#include "config.hpp"
#include "foxglove_config.hpp"
#include "runtime/capturer.hpp"

#include <expected>
#include <map>
#include <optional>
#include <string>
#include <string_view>

namespace fcs {

/// Per-hostname preset entry in [preset] table.
struct PresetEntry {
    bool daedalus{true};
    toml_helper::required<std::string> robot;
    toml_helper::required<std::string> vision;
};

// Entry-point configuration parsed from at_vision.toml
struct LaunchConfig {
    bool daedalus{false};
    toml_helper::required<std::string> robot;
    toml_helper::required<std::string> vision;
    std::map<std::string, PresetEntry> preset;
    FoxgloveConfig foxglove;
    CapturerConfig capturer;
};

// Fully-resolved runtime configuration after merging base + override TOML
struct RuntimeConfig {
    std::optional<HardwareConfig> hardware;
    FoxgloveConfig foxglove;
    CapturerConfig capturer;
    VisionConfig vision;
    runtime::CapturerLaunchContext launch;
};

/// Load and merge all configuration files from the given entry path.
/// When [preset] is present in the entry config, uses the system hostname
/// to auto-select robot and vision presets. Falls back to explicit fields otherwise.
/// Returns fully-resolved RuntimeConfig or a formatted error string.
[[nodiscard]] std::expected<RuntimeConfig, std::string> load_config(std::string_view path);

} // namespace fcs
