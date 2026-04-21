#include "runtime/config_loader.hpp"

#include "toml_helper.hpp"

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

auto validate_foxglove_config(const FoxgloveConfig& config) -> std::optional<std::string> {
    if (config.transport == FoxgloveTransport::Mcap && config.mcap_path.empty()) {
        return "entry config: foxglove.mcap_path is required when foxglove.transport=\"Mcap\"";
    }
    if (config.quanta.target_bitrate <= 0) {
        return "entry config: foxglove.quanta.target_bitrate must be positive";
    }
    if (config.quanta.max_width <= 0 || config.quanta.max_height <= 0) {
        return "entry config: foxglove.quanta.max_width and max_height must be positive";
    }
    if (config.quanta.gop_size <= 0) {
        return "entry config: foxglove.quanta.gop_size must be positive";
    }
    if (config.quanta.framerate <= 0) {
        return "entry config: foxglove.quanta.framerate must be positive";
    }
    if (config.quanta.lookahead < 0) {
        return "entry config: foxglove.quanta.lookahead must be non-negative";
    }
    return std::nullopt;
}

/// Apply hostname-based preset override to a fully-parsed LaunchConfig.
/// When [preset] exists and hostname matches, the matched entry overrides
/// robot, vision, and daedalus. Otherwise the explicit fields are used as-is.
auto apply_preset(LaunchConfig& launch) -> std::expected<void, std::string> {
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
    launch.daedalus   = entry.daedalus;
    launch.robot      = entry.robot;
    launch.vision     = entry.vision;
    launch.mock_mcu   = entry.mock_mcu;

    SPDLOG_INFO(
        "hostname: '{}' -> robot='{}', vision='{}', mock_mcu={}, daedalus={}", hostname,
        entry.robot.get(), entry.vision.get(), entry.mock_mcu, entry.daedalus);

    return {};
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
    if (auto r = apply_preset(launch); !r) {
        return std::unexpected(r.error());
    }

    if (launch.capturer.enabled && launch.capturer.output_dir.empty()) {
        return std::unexpected(
            "entry config: capturer.output_dir is required when capturer.enabled=true");
    }
    if (auto error = validate_foxglove_config(launch.foxglove)) {
        return std::unexpected(std::move(*error));
    }

    std::string vision_override_path = fmt::format("config/vision/{}.toml", launch.vision);

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

    // Load hardware config
    std::string hardware_path = fmt::format("config/robot/{}.toml", launch.robot);
    auto hardware_tbl         = toml::parse_file(hardware_path);
    if (hardware_tbl.failed()) {
        return std::unexpected(
            fmt::format("{}: {}", hardware_path, format_parse_error(hardware_tbl.error())));
    }

    auto hardware_config = toml_helper::from_table<HardwareConfig>(hardware_tbl.table());
    if (!hardware_config) {
        return std::unexpected(fmt::format("hardware config: {}", hardware_config.error()));
    }

    return RuntimeConfig{
        .hardware = std::optional<HardwareConfig>{std::move(*hardware_config)},
        .foxglove = std::move(launch.foxglove),
        .capturer = std::move(launch.capturer),
        .vision   = std::move(*vision_config),
        .launch =
            runtime::CapturerLaunchContext{
                                                  .daedalus = launch.daedalus,
                                                  .robot    = launch.robot.get(),
                                                  .vision   = launch.vision.get(),
                                                  },
        .scheduler = launch.scheduler
    };
}

} // namespace fcs
