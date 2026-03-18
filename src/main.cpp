#include "fcs/L2_perception/armor/at_legacy_systems.hpp"
#include "fcs/L2_perception/rune/rune_systems.hpp"
#include "fcs/L3_estimation/energy_meter/energy_meter_systems.hpp"
#include "fcs/L3_estimation/tracker_systems.hpp"
#include "fcs/L4_planning/planning_systems.hpp"
#include "fcs/L5_weapon/enhanced/weapon_systems.hpp"
#include "fcs/chiral/chiral_collector_system.hpp"
#include "fcs/config.hpp"
#include "fcs/core/trajectory/resource.hpp"
#include "fcs/runtime/l1_l2_setup.hpp"
#include "fcs_visualization/foxglove_systems.hpp"
#include "frame.hpp"
#include "magic_enum.hpp"
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
    toml_helper::required<std::string> robot;
    toml_helper::required<std::string> vision;
    fcs::FoxgloveConfig foxglove;
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

inline constexpr auto to_rad(double x) { return (x / 180.0) * CV_PI; }

inline fast_tf::CoordinateSystem coord_new(fcs::RobotExtrinsicConfig config) {
    auto system       = fast_tf::CoordinateSystem();
    constexpr fp_t pi = std::numbers::pi;

    update<fast_tf::odom>(system, {}, 0);
    update<fast_tf::gimbal_yaw_fuxk_frame>(system, {}, 0);
    auto a = config.gimbal_yaw.gimbal_pitch.translation;
    update<fast_tf::gimbal_pitch_fuxk_frame>(
        system,
        fast_tf::EdgeTransform<fast_tf::gimbal_pitch_fuxk_frame>::from_translation(
            a.x(), a.y(), a.z()),
        0);

    auto b = config.gimbal_yaw.gimbal_pitch.camera_link.translation;
    auto c = config.gimbal_yaw.gimbal_pitch.camera_link.rotation();

    update<fast_tf::camera_link_fuxk_frame>(
        system,
        fast_tf::EdgeTransform<fast_tf::camera_link_fuxk_frame>::from_rpy(
            to_rad(c.roll), to_rad(c.pitch), to_rad(c.yaw), b.x(), b.y(), b.z()),
        0);

    update<fast_tf::camera_optical_fuxk_frame>(
        system,
        fast_tf::EdgeTransform<fast_tf::camera_optical_fuxk_frame>::from_rpy(
            -pi / 2.0, 0.0, -pi / 2.0, 0.0, 0.0, 0.0),
        0);

    auto d = config.gimbal_yaw.gimbal_pitch.muzzle_link.translation;
    update<fast_tf::muzzle_link_fuxk_frame>(
        system,
        fast_tf::EdgeTransform<fast_tf::muzzle_link_fuxk_frame>::from_translation(
            d.x(), d.y(), d.z()),
        0);
    return system;
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

    auto launch_config = toml_helper::from_table<LaunchConfig>(entry_tbl.table());
    if (!launch_config) {
        return std::unexpected(fmt::format("entry config: {}", launch_config.error()));
    }

    std::string vision_override_path = fmt::format("config/vision/{}.toml", launch_config->vision);

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

    auto vision_config = toml_helper::from_table<fcs::VisionConfig>(*merged_vision);
    if (!vision_config) {
        return std::unexpected(fmt::format("vision config: {}", vision_config.error()));
    }

    // Load hardware config
    std::string hardware_path = fmt::format("config/robot/{}.toml", launch_config->robot);
    auto hardware_tbl         = toml::parse_file(hardware_path);
    if (hardware_tbl.failed()) {
        return std::unexpected(
            fmt::format("{}: {}", hardware_path, to_string(hardware_tbl.error())));
    }

    auto hardware_config = toml_helper::from_table<fcs::HardwareConfig>(hardware_tbl.table());
    if (!hardware_config) {
        return std::unexpected(fmt::format("hardware config: {}", hardware_config.error()));
    }

    if (launch_config->daedalus) {
        return std::tuple(
            hardware_config->type, true,
            std::optional<fcs::HardwareConfig>{std::move(*hardware_config)},
            std::move(launch_config->foxglove), std::move(*vision_config));
    }

    return std::tuple(
        hardware_config->type, false,
        std::optional<fcs::HardwareConfig>{std::move(*hardware_config)},
        std::move(launch_config->foxglove), std::move(*vision_config));
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
    talos::World world;
    talos::Scheduler scheduler(world);

    // Create coordinate system for fast_tf lookups (required by IMU and planning systems)
    world.emplace_resource<fast_tf::CoordinateSystem>(coord_new(hardware->extrinsic));

    // Insert config as shared resource (for foxglove and other systems)
    world.insert_resource(robot);

    // Save config values for logging before move
    const double bullet_speed = hardware->mcu->bullet_speed_default;

    world.insert_resource(fcs::core::trajectory::bullet_speed_data{.bullet_speed = bullet_speed});
    world.insert_resource(vision.at_legacy->default_detect_color);
    auto setup_result =
        fcs::runtime::setup_l1(world, scheduler, daedalus, hardware ? &*hardware : nullptr);
    if (!setup_result) {
        SPDLOG_CRITICAL("setup l1: {}", setup_result.error());
        return 3;
    }

    auto backend_handle_result = fcs::L2::create_detector_backend_handle(vision.at_legacy.get());
    if (!backend_handle_result) {
        SPDLOG_CRITICAL("create detector backend: {}", backend_handle_result.error());
        return 4;
    }

    SPDLOG_INFO(
        "detector backend: {} (color={})", backend_handle_result->name,
        vision.at_legacy->default_detect_color);

    world.insert_resource(std::move(backend_handle_result->backend));
    world.insert_resource(fcs::L2::create_pnp_solver(setup_result.value().camera_config));

    // Register trajectory solver resource (used by L4/L5)
    fcs::core::trajectory::register_resource(scheduler, std::move(vision.trajectory));
    SPDLOG_INFO("Trajectory resources configured (bullet_speed={:.1f}m/s)", bullet_speed);

    // Register AT Legacy detection pipeline systems
    fcs::L2::register_at_legacy_detection_systems(scheduler);
    fcs::rune::register_rune_detection_systems(scheduler, std::move(vision.rune_detector));

    // Register L3 Tracker pipeline (Measurement -> TrackerOutput)
    fcs::L3::register_tracker_systems(scheduler, std::move(vision.l3->tracker));
    fcs::energy_meter::register_energy_meter_systems(scheduler, std::move(vision.energy_meter));

    // Register L4 Planning pipeline (Aimer + Armor Plan Adapter)
    fcs::L4::register_l4_planning_systems(scheduler, std::move(vision.l4.get()));

    // Register L5 MPC Weapon system (consumes GimbalPlan from L4)
    fcs::L5::register_enhanced_weapon_system(scheduler, std::move(vision.l5->mpc_weapon));

    // Register Chiral data collection system (publishes TrackerOutput to shared memory)
    fcs::chiral::register_chiral_collector_system(scheduler);
    SPDLOG_INFO("Chiral data collection system registered");

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

    SPDLOG_INFO(
        "init done. ({}ms)", std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::system_clock::now() - start)
                                 .count());

    if (const auto result = scheduler.run(); !result) {
        SPDLOG_CRITICAL("run scheduler: {}", result.error());
    }
}
