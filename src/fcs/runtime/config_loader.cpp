#include "runtime/config_loader.hpp"

#include "config.hpp"
#include "toml_helper.hpp"
#include "toml_helper_eigen.hpp"

#include <fmt/core.h>
#include <primitive/system_info.hpp>
#include <toml++/toml.hpp>

namespace fcs {

namespace {

auto format_parse_error(const toml::parse_error& e) -> std::string {
    const auto& src = e.source();
    return fmt::format(
        "{} at {}:{}:{}", e.description(), src.path->data(), src.begin.line, src.begin.column);
}

/// Apply hostname-based preset override to a fully-parsed LaunchConfig.
/// When [preset] exists and hostname matches, the matched entry overrides
/// robot, vision, and daedalus. Otherwise the explicit fields are used as-is.
auto parse_preset(LaunchConfig& launch) -> std::expected<PresetEntry, std::string> {
    const auto hostname_result = talos::primitive::SystemInfo::get_hostname();
    if (!hostname_result) {
        return std::unexpected(
            fmt::format(
                "entry config: [preset] table present but hostname lookup failed: {}",
                hostname_result.error()));
    }
    const std::string_view hostname = *hostname_result;

    const auto it = launch.preset.find(std::string{hostname});
    if (it == launch.preset.end()) {
        // Hostname not in preset — fall through to explicit fields
        SPDLOG_INFO("hostname '{}' not in [preset], using explicit robot/vision fields", hostname);
        return {};
    }

    const auto& entry = it->second;

    SPDLOG_INFO(
        "hostname: '{}' -> robot='{}', vision='{}', backend={}", hostname, entry.robot.get(),
        entry.vision.get(), entry.backend);

    return entry;
}

} // namespace

[[nodiscard]] std::expected<RuntimeConfig, std::string> load_config(std::string_view path) {
    // Parse entry config
    auto entry_tbl = toml::parse_file(path);
    if (entry_tbl.failed()) {
        return std::unexpected(format_parse_error(entry_tbl.error()));
    }

    auto launch_config = toml_helper::from_table<LaunchConfig>(entry_tbl.table());
    if (!launch_config) {
        return std::unexpected(fmt::format("entry config: {}", launch_config.error()));
    }
    auto launch = std::move(*launch_config);

    // Apply hostname-based preset override
    auto preset_result = parse_preset(launch);
    if (!preset_result) {
        return std::unexpected(preset_result.error());
    }
    auto preset = preset_result.value();

    if (launch.capturer.enabled && launch.capturer.output_dir.empty()) {
        return std::unexpected(
            "entry config: capturer.output_dir is required when capturer.enabled=true");
    }

    auto vision_override_path = fmt::format("config/vision/{}.toml", preset.vision);
    // Parse base and override configs
    auto base_tbl = toml::parse_file("config/vision_base.toml");
    if (base_tbl.failed()) {
        return std::unexpected(
            fmt::format("config/vision_base.toml: {}", format_parse_error(base_tbl.error())));
    }

    auto override_tbl = toml::parse_file(vision_override_path);
    if (override_tbl.failed()) {
        return std::unexpected(
            fmt::format("{}: {}", vision_override_path, format_parse_error(override_tbl.error())));
    }

    // Merge configs using monadic operations
    auto merged_vision = toml_helper::merge_configs(base_tbl.table(), override_tbl.table())
                             .transform_error([](const auto& err) {
                                 return fmt::format("merge vision config: {}", err);
                             });
    if (!merged_vision) {
        return std::unexpected(merged_vision.error());
    }

    auto vision_config = toml_helper::from_table<VisionConfig>(*merged_vision);
    if (!vision_config) {
        return std::unexpected(fmt::format("vision config: {}", vision_config.error()));
    }

    hardware::HardwareBackendConfig cfg;
    switch (preset.backend) {
    case Chiral:
    case CameraOnly:
    case Direct: {
        bool camera_only = preset.backend == CameraOnly;
        // Load hardware config
        auto hardware_path = fmt::format("config/robot/{}.toml", preset.robot);
        auto hardware_tbl  = toml::parse_file(hardware_path);
        if (hardware_tbl.failed()) {
            return std::unexpected(
                fmt::format("{}: {}", hardware_path, format_parse_error(hardware_tbl.error())));
        }
        auto hardware_config = toml_helper::from_table<HardwareConfig>(hardware_tbl.table());
        if (!hardware_config) {
            return std::unexpected(fmt::format("hardware config: {}", hardware_config.error()));
        }
        cfg = hardware::DirectConfig{
            .camera_only = camera_only,
            .transport   = preset.backend != Chiral ? hardware::Transport::Direct
                                                    : hardware::Transport::Chiral,
            .hardware    = std::move(hardware_config.value())};
        break;
    };
    case Daedalus: {
        cfg = launch.daedalus;
        break;
    }
    }

    return RuntimeConfig{
        .backend  = std::move(cfg),
        .foxglove = std::move(launch.foxglove),
        .capturer = std::move(launch.capturer),
        .vision   = std::move(*vision_config),
        .launch =
            runtime::CapturerLaunchContext{
                                           .backend = preset.backend,
                                           .robot   = preset.robot.get(),
                                           .vision  = preset.vision.get(),
                                           },
        .scheduler = launch.scheduler
    };
}

} // namespace fcs
