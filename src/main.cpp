#include "L2_perception/at_legacy_systems.hpp"
#include "auto_aim/config.hpp"
#include "core/time.hpp"
#include "fcs/L1_sensor/input_interface.hpp"
#include "fcs/L1_sensor/output_interface.hpp"
#include "fcs/L3_estimation/L3_estimation.hpp"
#include "fcs/L4_planning/gimbal_planner/trajectory_compensator.hpp"
#include "fcs/L4_planning/planning_systems.hpp"
#include "fcs/L5_weapon/enhanced/weapon_systems.hpp"
#include "fcs/visualization/foxglove_systems.hpp"
#include "math/fast_tf/robot_factory.hpp"
#include "scheduler/scheduler.hpp"
#include "spdlog_hook.hpp"
#include "talos_gimbal/stm32.hpp"
#include "toml_helper.hpp"

#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <optional>
#include <tbb/parallel_pipeline.h>
#include <toml++/impl/parse_error.hpp>

// ============================================================================
// Config (使用 fcs::TalosConfig)
// ============================================================================

struct ImuData;
using TalosConfig = fcs::TalosConfig;

template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

using read_config_error = std::variant<toml::parse_error, std::string>;
using namespace std::chrono_literals;

inline std::expected<TalosConfig, read_config_error> read_config() {
    TalosConfig config;
    auto tbl = toml::parse_file("at_vision.toml");
    if (tbl.failed()) {
        return std::unexpected(tbl.error());
    }
    if (auto result = config.read_from(tbl.table()); !result) {
        return std::unexpected(result.error());
    }

    // Read robot-specific configuration
    auto robots_tbl = toml::parse_file("robots.toml");
    if (robots_tbl.failed()) {
        SPDLOG_CRITICAL("parse robots.toml: {}", robots_tbl.error().description());
        return config; // Return config with defaults
    }

    // Get robot-specific camera configuration based on robot type
    std::string robot_section;
    switch (config.robot) {
    case fcs::RobotType::Infantry: robot_section = "robot_infantry"; break;
    case fcs::RobotType::Hero: robot_section = "robot_hero"; break;
    case fcs::RobotType::Sentry: robot_section = "robot_sentry"; break;
    }

    if (auto robot_tbl = robots_tbl.table()[robot_section]; robot_tbl) {
        if (auto camera_tbl = robot_tbl.as_table()->operator[]("camera").as_table()) {
            fcs::CameraConfig robot_camera_config;
            if (auto camera_result = robot_camera_config.read_from(*camera_tbl); camera_result) {
                config.camera = robot_camera_config;
                SPDLOG_INFO("Loaded robot-specific camera configuration for {}", robot_section);
            } else {
                SPDLOG_ERROR("read camera config for {}: {}", robot_section, camera_result.error());
            }
        }
    } else {
        SPDLOG_ERROR("Robot section {} not found in robots.toml", robot_section);
    }

    return config;
}

int main() {
    init_logger();
    SPDLOG_INFO("talos is initializing...");
    const auto start = std::chrono::system_clock::now();

    auto config_result = read_config();
    if (!config_result.has_value()) {
        std::visit(
            overloaded{
                [](const toml::parse_error& e) {
                    const auto& src = e.source();
                    SPDLOG_CRITICAL(
                        "parse toml: {} at {}:{}:{}", e.description(), src.path->data(),
                        src.begin.line, src.begin.column);
                },
                [](const std::string& s) { SPDLOG_CRITICAL("read toml: {}", s); }},
            config_result.error());
        return 1;
    }
    auto& config             = config_result.value();
    const bool daedalus      = config.daedalus;
    const auto imu_vendor_id = config.imu_vendor_id;
    const std::optional<decltype(config.imu_product_id)> imu_product_id =
        config.imu_product_id == 0
            ? std::nullopt
            : std::optional<decltype(config.imu_product_id)>{config.imu_product_id};
    const bool untrust_roll = config.untrust_roll;

    std::expected<fcs::L1::InputInterface, fcs::L1::InputError> input_result;
    std::unique_ptr<fcs::L1::OutputInterface> output_interface;

    // USB 设备和 IMU
    talos_gimbal::ReceiveImuData imu_data{};
    talos_gimbal::Stm32Parser::latest_imu = &imu_data;
    auto usb_device = std::make_shared<talos_gimbal::Stm32Impl<talos_gimbal::Stm32Parser>>();
    std::shared_ptr<ipc::ShmClient> ipc_device;

    if (daedalus) {
        auto ipc_ = ipc::ShmClient::connect();
        if (!ipc_) {
            SPDLOG_CRITICAL("connect ipc: {}", ipc::to_string(ipc_.error()));
            return 1;
        }
        ipc_device   = std::make_shared<ipc::ShmClient>(std::move(*ipc_));
        input_result = fcs::L1::InputInterface::create_ipc(ipc_device);
        if (auto ipc_output = fcs::L1::OutputInterface::create_ipc(ipc_device)) {
            output_interface = std::make_unique<fcs::L1::OutputInterface>(std::move(*ipc_output));
            SPDLOG_INFO("endpoint: daedalus");
        } else {
            SPDLOG_CRITICAL("init daedalus endpoint: {}", ipc_output.error());
            return 2;
        }
    } else {
        if (imu_product_id) {
            SPDLOG_INFO("connecting to USB {:#x}:{:#x}", imu_vendor_id, *imu_product_id);
        } else {
            SPDLOG_INFO("connecting to USB {:#x} (product_id unspecified)", imu_vendor_id);
        }
        input_result = fcs::L1::InputInterface::create_hik(config.camera);
        if (auto result = usb_device->connect(imu_vendor_id, imu_product_id); result) {
            output_interface =
                std::make_unique<fcs::L1::OutputInterface>(fcs::L1::UsbOutput{usb_device});
            SPDLOG_INFO("endpoint: USB (STM32)");
        } else {
            SPDLOG_CRITICAL(
                "init USB (STM32) endpoint: {}", talos_gimbal::to_string(result.error()));
            return 2;
        }
    }
    if (!input_result) {
        std::visit(
            overloaded{
                [](fcs::L1::Timeout) { SPDLOG_CRITICAL("input: timeout"); },
                [](fcs::L1::Disconnected) { SPDLOG_CRITICAL("input: disconnected"); },
                [](fcs::L1::ProducerNotAvailable) {
                    SPDLOG_CRITICAL("input: producer not available");
                },
                [](const hikcamera::CameraError& e) { SPDLOG_CRITICAL("input: {}", e.what()); },
                [](ipc::ShmError e) { SPDLOG_CRITICAL("input: {}", ipc::to_string(e)); },
            },
            input_result.error());
        return 2;
    }
    auto input = std::move(*input_result);

    // IPC 模式下更新相机内参
    if (daedalus) {
        const auto& info                  = input.camera_info();
        config.camera.camera_matrix       = info.camera_matrix;
        config.camera.distort_coefficient = info.distort_coefficient;
        config.camera                     = info;
    }

    SPDLOG_INFO(
        "init done. ({}ms)", std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::system_clock::now() - start)
                                 .count());

    std::atomic<uint64_t> last_seq{0};

    talos::World world;
    talos::Scheduler scheduler(world);

    struct ImageRawChannel {};

    struct ImuSnapshot {
        timestamp_ns_t timestamp;
        double roll_deg, pitch_deg, yaw_deg, bullet_speed;
    };

    // ============================================================================
    // AT Legacy Detection Pipeline Setup
    // ============================================================================

    // Create detector backend from config
    auto backend_result = fcs::create_detector_backend(config.at_legacy);
    if (!backend_result) {
        SPDLOG_CRITICAL(
            "create detector backend: {}", fcs::detector_error_str(backend_result.error()));
        return 3;
    }
    auto detector_backend = std::move(*backend_result);
    SPDLOG_INFO(
        "detector backend: {} (color={})", detector_backend.name(),
        fcs::to_string(config.at_legacy.detector.detect_color));

    auto pnp_solver = fcs::create_pnp_solver(config.camera);
    SPDLOG_INFO("PnP solver initialized");

    // Insert resources (wrapped in shared_ptr for std::any compatibility)
    world.insert_resource(
        std::make_shared<fcs::AtLegacyDetectorBackend>(std::move(detector_backend)));
    world.insert_resource(std::make_shared<fcs::AtLegacyPnPSolver>(std::move(pnp_solver)));

    // Note: SPMC channels are auto-created when systems access them

    // ============================================================================
    // End AT Legacy Setup
    // ============================================================================

    // ============================================================================
    // L3 Tracker and L4 Planner Setup
    // ============================================================================

    // Create Tracker using config from toml
    world.insert_resource(std::make_shared<fcs::L3::Tracker>(config.tracker()));
    SPDLOG_INFO("Tracker initialized");

    // Create GimbalPlanner using config from toml (L4 Planning Layer)
    world.insert_resource(std::make_shared<fcs::L4::GimbalPlanner>(config.planner()));
    SPDLOG_INFO(
        "GimbalPlanner initialized (bullet_speed={:.1f}m/s)",
        config.planner().trajectory.bullet_speed);

    // Create coordinate system for fast_tf lookups
    switch (config.robot) {
    case fcs::RobotType::Infantry: world.insert_resource(fast_tf::create_infantry_system()); break;
    case fcs::RobotType::Hero: world.insert_resource(fast_tf::create_hero_system()); break;
    case fcs::RobotType::Sentry: world.insert_resource(fast_tf::create_sentry_system()); break;
    }

    // Insert config as shared resource (for foxglove and other systems)
    world.insert_resource(std::make_shared<fcs::TalosConfig>(config));

    // ============================================================================
    // End L3 Tracker and L4 Planner Setup
    // ============================================================================

    // ============================================================================
    // Foxglove Visualization Setup (Optional)
    // ============================================================================

    if (config.foxglove.enabled) {
        auto foxglove_server =
            std::make_shared<fcs::FoxgloveServer>(config.foxglove.port, config.foxglove.host);
        if (foxglove_server->is_initialized()) {
            world.insert_resource(foxglove_server);
            fcs::register_foxglove_systems(scheduler, &scheduler);
            SPDLOG_INFO(
                "Foxglove server enabled on ws://{}:{}", config.foxglove.host,
                config.foxglove.port);
        } else {
            SPDLOG_WARN("Foxglove server failed to initialize, visualization disabled");
        }
    } else {
        SPDLOG_INFO("Foxglove visualization disabled by config");
    }

    // ============================================================================
    // End Foxglove Setup
    // ============================================================================

    world.insert_resource(ImuSnapshot{});
    // IMU: 100Hz, silent update (no notify)
    scheduler.add_system<talos::exclusive_silent<1000, 1>>(
        "imu_reader",
        [untrust_roll, daedalus, &ipc_device, &usb_device, &imu_data](
            talos::res_mut<std::shared_ptr<fast_tf::CoordinateSystem<1024>>> tf_buffer,
            talos::res_mut<ImuSnapshot> imu_out) {
            if (daedalus) {
                if (const auto pose = ipc_device->recv_pose(ipc::POSE_GIMBAL)) {
                    const Eigen::Quaterniond q(pose->qw, pose->qx, pose->qy, pose->qz);
                    const auto [roll, pitch, yaw] = math_fuxk::rpy(q).rpy();
                    imu_out->timestamp            = pose->timestamp_ns;
                    imu_out->roll_deg             = roll;
                    imu_out->pitch_deg            = pitch;
                    imu_out->yaw_deg              = yaw;
                    // Update gimbal_link TF (critical for camera_link lookup)
                    fast_tf::update(
                        *tf_buffer->get(),
                        fast_tf::FuckTransform<fast_tf::gimbal_link_fuxk_frame>(
                            fast_tf::TransformMatrix4d::from_rpy(
                                roll, pitch, yaw, pose->x, pose->y, pose->z)),
                        pose->timestamp_ns);
                }
                if (const auto pose = ipc_device->recv_pose(ipc::POSE_ODOM)) {
                    // strategy.set_bullet_speed(imu_data.data.bullet_speed);
                    fast_tf::update(
                        *tf_buffer->get(),
                        fast_tf::FuckTransform<fast_tf::odom_fuxk_frame>(
                            fast_tf::TransformMatrix4d::from_translation(
                                pose->x, pose->y, pose->z)),
                        pose->timestamp_ns);
                }
                if (const auto pose = ipc_device->recv_pose(ipc::POSE_CAMERA)) {
                    const Eigen::Quaterniond q(pose->qw, pose->qx, pose->qy, pose->qz);
                    auto [roll, pitch, yaw] = math_fuxk::rpy(q).rpy();
                    if (untrust_roll) {
                        roll = 0.0;
                    }
                    // strategy.set_bullet_speed(imu_data.data.bullet_speed);
                    fast_tf::update(
                        *tf_buffer->get(),
                        fast_tf::FuckTransform<fast_tf::camera_link_fuxk_frame>(
                            fast_tf::TransformMatrix4d::from_rpy(
                                roll, pitch, yaw, pose->x, pose->y, pose->z)),
                        pose->timestamp_ns);
                }
                if (const auto pose = ipc_device->recv_pose(ipc::POSE_MUZZLE)) {
                    const Eigen::Quaterniond q(pose->qw, pose->qx, pose->qy, pose->qz);
                    const auto [roll, pitch, yaw] = math_fuxk::rpy(q).rpy();
                    // strategy.set_bullet_speed(imu_data.data.bullet_speed);
                    fast_tf::update(
                        *tf_buffer->get(),
                        fast_tf::FuckTransform<fast_tf::muzzle_link_fuxk_frame>(
                            fast_tf::TransformMatrix4d::from_rpy(
                                roll, pitch, yaw, pose->x, pose->y, pose->z)),
                        pose->timestamp_ns);
                }
            } else {
                usb_device->handle_events();
                imu_out->timestamp = fcs::clock::now_ns();
                imu_out->roll_deg  = imu_data.data.roll;
                imu_out->pitch_deg = imu_data.data.pitch;
                imu_out->yaw_deg   = imu_data.data.yaw;
                // strategy.set_bullet_speed(imu_data.data.bullet_speed);
                fast_tf::update(
                    *tf_buffer->get(),
                    fast_tf::FuckTransform<fast_tf::gimbal_link_fuxk_frame>(
                        fast_tf::TransformMatrix4d::from_rpy(
                            imu_out->roll_deg, imu_out->pitch_deg, imu_out->yaw_deg)),
                    imu_out->timestamp);
            }
        });

    // Camera: 160Hz, notify (triggers compute)
    scheduler.add_system<talos::exclusive<250, 1>>(
        "camera_reader",
        [daedalus, &ipc_device, &input, &last_seq](
            talos::spmc_mut<fcs::AtLegacyImageFrame, fcs::AtLegacyImageChannelTag> cam_out) {
            const auto frame = input.try_recv();
            if (!frame) {
                return;
            }
            const auto seq = frame->seq;
            if (seq != last_seq + 1 && last_seq > 0) {
                SPDLOG_INFO("skip {} frame", seq - last_seq - 1);
            }
            last_seq = frame->seq;

            auto img = frame->image;
            cam_out.write(fcs::AtLegacyImageFrame{std::move(img), frame->timestamp_ns, seq});
        });

    // Register AT Legacy detection pipeline systems
    fcs::register_at_legacy_detection_systems(scheduler);

    // Register L3 Tracker pipeline (Measurement -> TrackerOutput)
    fcs::L3::register_tracker_systems(scheduler);

    // Register L4 Planning pipeline (TrackerOutput -> GimbalPlan + FireCommand)
    fcs::L4::register_planning_systems_with_config(scheduler, config.planner());

    // Register L5 Enhanced Weapon System (GimbalPlan + FireCommand -> WeaponCommand)
    fcs::L5::register_enhanced_weapon_system_with_config_and_compensator(
        scheduler, config.weapon(), fcs::L4::create_compensator(config.planner().compensator_type));

    // L1 Output System: Send WeaponCommand to device
    scheduler.add_system<talos::exclusive<250, 1>>(
        "weapon_output",
        [&output_interface](
            talos::spmc<fcs::L5::WeaponCommand, fcs::WeaponCommandChannelTag> cmd_in) {
            auto cmd = cmd_in.read();
            if (!cmd) {
                return;
            }

            // Send to device (USB or IPC)
            output_interface->send(*cmd);
        });

    // Build
    if (auto build_result = scheduler.build(); !build_result) {
        SPDLOG_CRITICAL(
            "build scheduler: {}", talos::format_build_error(build_result.error()).c_str());
        return 1;
    }

    if (const auto result = scheduler.run(); !result) {
        SPDLOG_CRITICAL("run scheduler: {}", to_string(result.error()));
    }
}
