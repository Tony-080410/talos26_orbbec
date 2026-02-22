#include "fcs/L2_perception/armor/at_legacy_systems.hpp"
#include "fcs/L2_perception/rune/rune_systems.hpp"
#include "fcs/L3_estimation/energy_meter/energy_meter_systems.hpp"
#include "fcs/L3_estimation/tracker_systems.hpp"
#include "fcs/L4_planning/planning_systems.hpp"
#include "fcs/L5_weapon/enhanced/weapon_systems.hpp"
#include "fcs/config.hpp"
#include "fcs/core/trajectory/resource.hpp"
#include "fcs/runtime/l1_l2_setup.hpp"
#include "fcs_visualization/foxglove_systems.hpp"
#include "magic_enum.hpp"
#include "robot_factory.hpp"
#include "scheduler/error_formatter.hpp"
#include "scheduler/scheduler.hpp"
#include "spdlog_hook.hpp"
#include "toml_helper.hpp"
#include <chrono>
#include <expected>
#include <fmt/core.h>
#include <fmt/std.h>
#include <memory>
#include <optional>
#include <toml++/impl/parse_error.hpp>
#include <tuple>
#include <utility>

struct ImuData;

template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

using read_config_error = std::variant<toml::parse_error, std::string>;

constexpr auto to_string(read_config_error err) -> std::string {
    return std::visit(
        overloaded{
            [](const toml::parse_error& e) {
                const auto& src = e.source();
                return fmt::format(
                    "parse toml: {} at {}:{}:{}", e.description(), src.path->data(), src.begin.line,
                    src.begin.column);
            },
            [](const std::string& s) { return fmt::format("read toml: {}", s); }},
        err);
}

using namespace std::chrono_literals;

struct LaunchConfig {
    bool daedalus;
    fcs::RobotType robot;
    std::string vision;
    fcs::FoxgloveConfig foxglove;
    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ(tbl, daedalus);
        READ(tbl, robot);
        READ(tbl, vision);
        READ(tbl, foxglove);
        return {};
    }
};

std::string strtolower(std::string_view str) {
    std::locale loc;
    std::string result;
    result.reserve(str.size());

    std::transform(str.begin(), str.end(), std::back_inserter(result), [&loc](char c) {
        return std::tolower(c, loc);
    });

    return result;
}

std::expected<
    std::tuple<
        fcs::RobotType, bool, std::optional<fcs::HardwareConfig>, fcs::FoxgloveConfig,
        fcs::VisionConfig>,
    read_config_error>
    read_config() {
    // Parse entry config
    auto entry_tbl = toml::parse_file("at_vision.toml");
    if (entry_tbl.failed()) {
        return std::unexpected(entry_tbl.error());
    }

    LaunchConfig launch_config;
    if (auto result = launch_config.read_from(entry_tbl.table()); !result) {
        return std::unexpected(fmt::format("entry config: {}", result.error()));
    }

    const auto robot_name            = strtolower(magic_enum::enum_name(launch_config.robot));
    std::string vision_override_path = fmt::format("config/vision/{}.toml", launch_config.vision);

    // Parse base and override configs
    auto base_tbl = toml::parse_file("config/vision_base.toml");
    if (base_tbl.failed()) {
        return std::unexpected(
            fmt::format("config/vision_base.toml: {}", to_string(base_tbl.error())));
    }

    auto override_tbl = toml::parse_file(vision_override_path);
    if (override_tbl.failed()) {
        return std::unexpected(
            fmt::format("{}: {}", vision_override_path, to_string(override_tbl.error())));
    }

    // Merge configs using monadic operations
    auto merged_vision = toml_helper::merge_configs(base_tbl.table(), override_tbl.table())
                             .transform_error([](const auto& err) {
                                 return fmt::format("merge vision config: {}", err);
                             });
    if (!merged_vision) {
        return std::unexpected(merged_vision.error());
    }

    fcs::VisionConfig vision_config;
    if (auto result = vision_config.read_from(*merged_vision); !result) {
        return std::unexpected(fmt::format("vision config: {}", result.error()));
    }

    // Early return for daedalus mode
    if (launch_config.daedalus) {
        return std::tuple(
            launch_config.robot, true, std::optional<fcs::HardwareConfig>{},
            std::move(launch_config.foxglove), std::move(vision_config));
    }

    // Load hardware config
    std::string hardware_path = fmt::format("config/robot/{}.toml", robot_name);
    auto hardware_tbl         = toml::parse_file(hardware_path);
    if (hardware_tbl.failed()) {
        return std::unexpected(
            fmt::format("{}: {}", hardware_path, to_string(hardware_tbl.error())));
    }

    fcs::HardwareConfig hardware_config;
    if (auto result = hardware_config.read_from(hardware_tbl.table()); !result) {
        return std::unexpected(fmt::format("hardware config: {}", result.error()));
    }

    return std::tuple(
        launch_config.robot, false, std::optional<fcs::HardwareConfig>{std::move(hardware_config)},
        std::move(launch_config.foxglove), std::move(vision_config));
}

int main() {
    init_logger();
    SPDLOG_INFO("talos is initializing...");
    const auto start = std::chrono::system_clock::now();

    auto config_result = read_config();
    if (!config_result.has_value()) {
        SPDLOG_CRITICAL("load config: {}", to_string(config_result.error()));
        return 1;
    }

    auto [robot, daedalus, hardware, foxglove, vision] = config_result.value();

    SPDLOG_INFO(
        "init done. ({}ms)", std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::system_clock::now() - start)
                                 .count());

    talos::World world;
    talos::Scheduler scheduler(world);

    // Create coordinate system for fast_tf lookups (required by IMU and planning systems)
    switch (robot) {
    case fcs::RobotType::Infantry: world.insert_resource(fast_tf::create_infantry_system()); break;
    case fcs::RobotType::Hero: world.insert_resource(fast_tf::create_hero_system()); break;
    case fcs::RobotType::Sentry: world.insert_resource(fast_tf::create_sentry_system()); break;
    }

    // Insert config as shared resource (for foxglove and other systems)
    world.insert_resource(robot);
    world.insert_resource(std::make_shared<fcs::VisionConfig>(vision));

    // Save config values for logging before move
    const double bullet_speed = vision.trajectory.model.bullet_speed;

    auto setup_result = fcs::runtime::setup_l1_l2(
        world, scheduler, vision, daedalus, hardware ? &*hardware : nullptr);
    if (!setup_result) {
        SPDLOG_CRITICAL("setup l1/l2: {}", setup_result.error());
        return 3;
    }

    // Create Tracker using config from toml
    world.insert_resource(fcs::L3::create_tracker(vision.l3.tracker));
    SPDLOG_INFO("Tracker initialized");

    // Register trajectory solver resource (used by L4/L5)
    fcs::core::trajectory::register_resource(scheduler, std::move(vision.trajectory));
    SPDLOG_INFO("Trajectory solver registered");

    SPDLOG_INFO("Trajectory resources configured (bullet_speed={:.1f}m/s)", bullet_speed);

    // Register AT Legacy detection pipeline systems
    fcs::L2::register_at_legacy_detection_systems(scheduler, std::move(vision.at_legacy));
    fcs::rune::register_rune_detection_systems(scheduler, std::move(vision.rune_detector));

    // Register L3 Tracker pipeline (Measurement -> TrackerOutput)
    fcs::L3::register_tracker_systems(scheduler, std::move(vision.l3.tracker));
    fcs::energy_meter::register_energy_meter_systems(scheduler, std::move(vision.energy_meter));

    // Register L4 Planning pipeline (Aimer + Armor Plan Adapter)
    fcs::L4::register_l4_planning_systems(scheduler, std::move(vision.l4));

    // Register L5 MPC Weapon system (consumes GimbalPlan from L4)
    fcs::L5::register_enhanced_weapon_system(scheduler, std::move(vision.l5.weapon));

    if (foxglove.enabled) {
        auto foxglove_server = fcs::try_create_foxglove_server(foxglove.port, foxglove.host);
        if (foxglove_server) {
            world.insert_resource(std::move(foxglove_server));
            fcs::register_foxglove_systems(scheduler, &scheduler);
            SPDLOG_INFO("Foxglove server enabled on ws://{}:{}", foxglove.host, foxglove.port);
        } else {
            SPDLOG_WARN("Foxglove server failed to initialize, visualization disabled");
        }
    } else {
        SPDLOG_INFO("Foxglove visualization disabled by config");
    }

    // Build
    if (auto build_result = scheduler.build(); !build_result) {
        SPDLOG_CRITICAL("build scheduler: {}", build_result.error());
        return 1;
    }

    if (const auto result = scheduler.run(); !result) {
        SPDLOG_CRITICAL("run scheduler: {}", result.error());
    }
}
