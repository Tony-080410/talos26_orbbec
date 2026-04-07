#include "runtime/config_loader.hpp"

#include "toml_helper.hpp"

#include <fmt/core.h>
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
    return std::nullopt;
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
        .robot_type = hardware_config->type,
        .hardware   = std::optional<HardwareConfig>{std::move(*hardware_config)},
        .foxglove   = std::move(launch.foxglove),
        .capturer   = std::move(launch.capturer),
        .vision     = std::move(*vision_config),
        .launch =
            runtime::CapturerLaunchContext{
                                                    .daedalus = launch.daedalus,
                                                    .robot    = launch.robot.get(),
                                                    .vision   = launch.vision.get(),
                                                    },
    };
}

} // namespace fcs
