#include "runtime/boot.hpp"

#include "L2_perception/armor/readback_roi.hpp"
#include "L2_perception/armor/systems.hpp"
#include "L2_perception/ldm/ldm_systems.hpp"
#include "L2_perception/rune/rune_systems.hpp"
#include "L3_estimation/energy_meter/energy_meter_systems.hpp"
#include "L3_estimation/tracker_systems.hpp"
#include "L4_planning/planning_systems.hpp"
#include "L5_weapon/enhanced/weapon_systems.hpp"
#include "chiral/chiral_collector_system.hpp"
#include "core/trajectory/resource.hpp"
#include "frame.hpp"
#include "runtime/capturer.hpp"
#include "runtime/l1_l2_setup.hpp"
#include "scheduler/error_formatter.hpp"

#include <fmt/core.h>
#include <numbers>
#include <spdlog/spdlog.h>

namespace fcs {

namespace {

inline constexpr auto to_rad(double x) noexcept { return (x / 180.0) * std::numbers::pi_v<double>; }

void init_coordinate_system(fast_tf::CoordinateSystem& system, const RobotExtrinsicConfig& config) {
    constexpr fp_t pi = std::numbers::pi;

    fast_tf::update<fast_tf::odom>(system, {}, 0);
    fast_tf::update<fast_tf::gimbal_yaw_fuxk_frame>(system, {}, 0);

    auto a = config.gimbal_yaw.gimbal_pitch.translation;
    fast_tf::update<fast_tf::gimbal_pitch_fuxk_frame>(
        system,
        fast_tf::EdgeTransform<fast_tf::gimbal_pitch_fuxk_frame>::from_translation(
            a.x(), a.y(), a.z()),
        0);

    auto b = config.gimbal_yaw.gimbal_pitch.camera_link.translation;
    auto c = config.gimbal_yaw.gimbal_pitch.camera_link.rotation();

    fast_tf::update<fast_tf::camera_link_fuxk_frame>(
        system,
        fast_tf::EdgeTransform<fast_tf::camera_link_fuxk_frame>::from_rpy(
            to_rad(c.roll), to_rad(c.pitch), to_rad(c.yaw), b.x(), b.y(), b.z()),
        0);

    fast_tf::update<fast_tf::camera_optical_fuxk_frame>(
        system,
        fast_tf::EdgeTransform<fast_tf::camera_optical_fuxk_frame>::from_rpy(
            -pi / 2.0, 0.0, -pi / 2.0, 0.0, 0.0, 0.0),
        0);

    auto d = config.gimbal_yaw.gimbal_pitch.muzzle_link.translation;
    fast_tf::update<fast_tf::muzzle_link_fuxk_frame>(
        system,
        fast_tf::EdgeTransform<fast_tf::muzzle_link_fuxk_frame>::from_translation(
            d.x(), d.y(), d.z()),
        0);
}

} // namespace

[[nodiscard]] std::expected<void, std::string>
    boot(talos::World& world, talos::Scheduler& scheduler, RuntimeConfig&& config) {
    if (!config.hardware) {
        return std::unexpected("runtime config missing hardware section");
    }

    // Build the TF buffer in place to avoid moving the large tuple of buffers during startup.
    auto& coordinate_system = world.emplace_resource<fast_tf::CoordinateSystem>();
    init_coordinate_system(coordinate_system, config.hardware->extrinsic);

    // Insert config as shared resource (for foxglove and other systems)
    world.insert_resource(config.foxglove);

    // Save config values for logging before move
    const double bullet_speed = config.hardware->mcu->bullet_speed_default;

    world.insert_resource(core::trajectory::bullet_speed_data{.bullet_speed = bullet_speed});
    world.insert_resource(config.vision.at_legacy->default_detect_color);
    world.insert_resource(config.vision.at_legacy->readback_roi);
    world.insert_resource(L2::TrackerReadbackCache{});

    // Setup L1 sensor systems
    auto setup_result = runtime::setup_l1(
        world, scheduler, config.launch.daedalus, config.hardware ? &*config.hardware : nullptr);
    if (!setup_result) {
        return std::unexpected(fmt::format("setup l1: {}", setup_result.error()));
    }

    // Create detector backend
    auto backend_handle_result = L2::create_detector_backend_handle(config.vision.at_legacy.get());
    if (!backend_handle_result) {
        return std::unexpected(
            fmt::format("create detector backend: {}", backend_handle_result.error()));
    }

    SPDLOG_INFO(
        "detector backend: {} (color={})", backend_handle_result->name,
        config.vision.at_legacy->default_detect_color);

    world.insert_resource(setup_result.value().camera_config);
    world.insert_resource(std::move(backend_handle_result->backend));
    world.insert_resource(L2::create_pnp_solver(setup_result.value().camera_config));

    // Register trajectory solver resource (used by L4/L5)
    core::trajectory::register_resource(scheduler, std::move(config.vision.trajectory));
    SPDLOG_INFO("Trajectory resources configured (bullet_speed={:.1f}m/s)", bullet_speed);

    // Register AT Legacy detection pipeline systems
    L2::register_detection_systems(scheduler);
    L2::ldm::register_ldm_systems(
        scheduler, std::move(config.vision.ldm), setup_result.value().camera_config);
    rune::register_rune_detection_systems(scheduler, std::move(config.vision.rune_detector));

    // Register L3 Tracker pipeline (Measurement -> TrackerOutput)
    L3::register_tracker_systems(scheduler, std::move(config.vision.l3->tracker));
    energy_meter::register_energy_meter_systems(scheduler, std::move(config.vision.energy_meter));

    // Register L4 Planning pipeline (Aimer + Armor Plan Adapter)
    L4::register_l4_planning_systems(scheduler, std::move(config.vision.l4.get()));

    // Register L5 MPC Weapon system (consumes GimbalPlan from L4)
    L5::register_enhanced_weapon_system(scheduler, std::move(config.vision.l5.get()));

    // Register Chiral data collection system (publishes TrackerOutput to shared memory)
    if (config.hardware->chiral) {
        chiral::register_chiral_collector_system(scheduler);
        SPDLOG_INFO("chiral system registered");
    }

    runtime::register_runtime_capturer_system(
        scheduler, config.capturer, config.launch, setup_result.value().camera_config);

    // Build scheduler (freezes resource structure)
    if (auto build_result = scheduler.build(); !build_result) {
        return std::unexpected(fmt::format("build scheduler: {}", build_result.error()));
    }

    return {};
}

} // namespace fcs
