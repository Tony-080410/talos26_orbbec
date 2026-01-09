#include "L2_perception/at_legacy_systems.hpp"
#include "auto_aim/config.hpp"
#include "camera_config.hpp"
#include "core/time.hpp"
#include "fcs/L1_sensor/camera_interface.hpp"
#include "fcs/L1_sensor/output_interface.hpp"
#include "fcs/L3_estimation/L3_estimation.hpp"
#include "fcs/L4_planning/gimbal_planner/trajectory_compensator.hpp"
#include "fcs/L4_planning/planning_systems.hpp"
#include "fcs/L5_weapon/enhanced/weapon_systems.hpp"
#include "fcs/visualization/foxglove_systems.hpp"
#include "magic_enum.hpp"
#include "math/fast_tf/robot_factory.hpp"
#include "scheduler/scheduler.hpp"
#include "spdlog_hook.hpp"
#include "talos_gimbal/stm32.hpp"
#include "toml_helper.hpp"
#include "usb.hpp"

#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <expected>
#include <memory>
#include <optional>
#include <toml++/impl/parse_error.hpp>
#include <tuple>
#include <type_traits>
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

struct ImuSnapshot {
    timestamp_ns_t timestamp;
    double roll_deg, pitch_deg, yaw_deg, bullet_speed;
};

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

struct Daedalus {};

using hardware_config = std::variant<Daedalus, fcs::HardwareConfig>;

struct DaedalusImu {
    std::shared_ptr<ipc::ShmClient> ipc_device;

    auto system(
        talos::res_mut<std::shared_ptr<fast_tf::CoordinateSystem<1024>>> tf_buffer,
        talos::res_mut<ImuSnapshot> imu_out) {
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
            fast_tf::update(
                *tf_buffer->get(),
                fast_tf::FuckTransform<fast_tf::odom_fuxk_frame>(
                    fast_tf::TransformMatrix4d::from_translation(pose->x, pose->y, pose->z)),
                pose->timestamp_ns);
        }
        if (const auto pose = ipc_device->recv_pose(ipc::POSE_CAMERA)) {
            const Eigen::Quaterniond q(pose->qw, pose->qx, pose->qy, pose->qz);
            auto [roll, pitch, yaw] = math_fuxk::rpy(q).rpy();
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
            fast_tf::update(
                *tf_buffer->get(),
                fast_tf::FuckTransform<fast_tf::muzzle_link_fuxk_frame>(
                    fast_tf::TransformMatrix4d::from_rpy(
                        roll, pitch, yaw, pose->x, pose->y, pose->z)),
                pose->timestamp_ns);
        }
    };
};

template <talos_gimbal::Device Device>
struct UsbImu {
    std::shared_ptr<Device> device;
    talos_gimbal::ReceiveImuData* _Nonnull imu_data;

    auto system(
        talos::res_mut<std::shared_ptr<fast_tf::CoordinateSystem<1024>>> tf_buffer,
        talos::res_mut<ImuSnapshot> imu_out) {
        device->handle_events();
        imu_out->timestamp = fcs::clock::now_ns();
        imu_out->roll_deg  = imu_data->data.roll;
        imu_out->pitch_deg = imu_data->data.pitch;
        imu_out->yaw_deg   = imu_data->data.yaw;
        // TODO: whether we need bullet speed from IMU
        // we do not have odometry for robots, so we skip it for now
        fast_tf::update(
            *tf_buffer->get(),
            fast_tf::FuckTransform<fast_tf::gimbal_link_fuxk_frame>(
                fast_tf::TransformMatrix4d::from_rpy(
                    imu_out->roll_deg, imu_out->pitch_deg, imu_out->yaw_deg)),
            imu_out->timestamp);
    };
};

struct Imu {
    using UsbImu = UsbImu<talos_gimbal::Stm32Impl<talos_gimbal::Stm32Parser>>;
    std::variant<DaedalusImu, UsbImu> imu_system;

    template <typename... Args>
    constexpr auto system(Args&&... args) {
        std::visit(
            [&](auto&& hw) -> void {
                using inner_type = std::decay_t<decltype(hw)>;
                if constexpr (std::is_same_v<inner_type, DaedalusImu>) {
                    return hw.system(std::forward<Args>(args)...);
                }
                if constexpr (std::is_same_v<inner_type, UsbImu>) {
                    return hw.system(std::forward<Args>(args)...);
                }
                std::unreachable();
            },
            imu_system);
    }
};

talos_gimbal::ReceiveImuData imu_data{};
using make_interface_result_type = std::expected<
    std::tuple<
        std::unique_ptr<Imu>, std::unique_ptr<fcs::L1::CameraInterface>,
        std::unique_ptr<fcs::L1::OutputInterface>>,
    std::string>;
make_interface_result_type make_interface(hardware_config&& cfg) {
    return std::visit(
        [](auto&& hw) -> make_interface_result_type {
            using inner_type = std::decay_t<decltype(hw)>;
            if constexpr (std::is_same_v<inner_type, Daedalus>) {
                std::unique_ptr<fcs::L1::OutputInterface> output_interface;
                std::expected<fcs::L1::CameraInterface, fcs::L1::InputError> camera_interface;

                SPDLOG_INFO("connecting to daedalus ipc...");
                auto ipc_ = ipc::ShmClient::connect();
                if (!ipc_) {
                    return std::unexpected(
                        fmt::format("connect ipc: {}", ipc::to_string(ipc_.error())));
                }
                auto ipc_device = std::make_shared<ipc::ShmClient>(std::move(*ipc_));

                SPDLOG_INFO("initializing daedalus input interface...");
                auto input_result = fcs::L1::CameraInterface::create_ipc(ipc_device);
                if (!input_result) {
                    return std::unexpected(
                        fmt::format(
                            "init daedalus input: {}", fcs::L1::to_string(input_result.error())));
                }

                SPDLOG_INFO("initializing daedalus output interface...");
                auto out_result = fcs::L1::OutputInterface::create_ipc(ipc_device);
                if (!out_result) {
                    return std::unexpected(
                        fmt::format("init daedalus output: {}", out_result.error()));
                }

                return std::tuple(
                    std::make_unique<Imu>(DaedalusImu{ipc_device}),
                    std::make_unique<fcs::L1::CameraInterface>(std::move(*input_result)),
                    std::make_unique<fcs::L1::OutputInterface>(std::move(*out_result)));
            }
            if constexpr (std::is_same_v<inner_type, fcs::HardwareConfig>) {
                talos_gimbal::Stm32Parser::latest_imu = &imu_data;

                const auto& config = hw;

                auto usb_device =
                    std::make_shared<talos_gimbal::Stm32Impl<talos_gimbal::Stm32Parser>>();
                if (config.imu_product_id_opt()) {
                    SPDLOG_INFO(
                        "connecting to USB {:#x}:{:#x}", config.imu_vendor_id,
                        config.imu_product_id);
                } else {
                    SPDLOG_INFO(
                        "connecting to USB {:#x} (product_id unspecified)", config.imu_vendor_id);
                }

                SPDLOG_INFO("initializing hik input interface...");
                auto input_result = fcs::L1::CameraInterface::create_hik(config.camera);
                if (!input_result) {
                    return std::unexpected(
                        fmt::format(
                            "init hik input: {}", fcs::L1::to_string(input_result.error())));
                }

                SPDLOG_INFO("connecting to USB device...");
                auto connect_result =
                    usb_device->connect(config.imu_vendor_id, config.imu_product_id_opt());
                if (!connect_result) {
                    return std::unexpected(
                        fmt::format(
                            "connect USB device: {}",
                            talos_gimbal::to_string(connect_result.error())));
                }

                return std::tuple(
                    std::make_unique<Imu>(UsbImu{
                        .device   = usb_device,
                        .imu_data = &imu_data,
                    }),
                    std::make_unique<fcs::L1::CameraInterface>(std::move(*input_result)), nullptr);
            }
            return std::unexpected(fmt::format("unsupported hardware, this should never happen"));
        },
        std::move(cfg));
}

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
    std::tuple<fcs::RobotType, hardware_config, fcs::FoxgloveConfig, fcs::VisionConfig>,
    read_config_error>
    read_config() {
    auto entry_tbl = toml::parse_file("at_vision.toml");
    if (entry_tbl.failed()) {
        return std::unexpected(entry_tbl.error());
    }
    LaunchConfig launch_config;
    if (auto result = launch_config.read_from(entry_tbl.table()); !result) {
        return std::unexpected(fmt::format("entry config: {}", result.error()));
    }
    auto base_tbl = toml::parse_file("config/vision_base.toml");
    if (base_tbl.failed()) {
        return std::unexpected(
            fmt::format("config/vision_base.toml: {}", to_string(base_tbl.error())));
    }
    const auto robot_name            = strtolower(magic_enum::enum_name(launch_config.robot));
    std::string vision_override_path = fmt::format("config/vision/{}.toml", launch_config.vision);
    auto vision_override_tbl         = toml::parse_file(vision_override_path);
    if (vision_override_tbl.failed()) {
        return std::unexpected(
            fmt::format("{}: {}", vision_override_path, to_string(vision_override_tbl.error())));
    }

    auto merged_vision = toml_helper::merge_configs(base_tbl.table(), vision_override_tbl.table());
    if (!merged_vision) {
        return std::unexpected(fmt::format("merge vision config: {}", merged_vision.error()));
    }

    // Step 3: Load robot config
    std::string hardware_path = fmt::format("config/robot/{}.toml", robot_name);
    auto hardware_tbl         = toml::parse_file(hardware_path);
    if (hardware_tbl.failed()) {
        return std::unexpected(
            fmt::format("{}: {}", hardware_path, to_string(hardware_tbl.error())));
    }
    fcs::VisionConfig vision_config;
    if (auto result = vision_config.read_from(*merged_vision); !result) {
        return std::unexpected(fmt::format("vision config: {}", result.error()));
    }
    if (launch_config.daedalus) {
        return std::tuple(
            launch_config.robot, Daedalus{}, std::move(launch_config.foxglove),
            std::move(vision_config));
    }
    fcs::HardwareConfig hardware_config;
    if (auto result = hardware_config.read_from(hardware_tbl.table()); !result) {
        return std::unexpected(fmt::format("hardware config: {}", result.error()));
    }
    return std::tuple(
        launch_config.robot, std::move(hardware_config), std::move(launch_config.foxglove),
        std::move(vision_config));
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

    auto [robot, hardware, foxglove, vision] = config_result.value();

    auto interface_result = make_interface(std::move(hardware));
    if (!interface_result) {
        SPDLOG_CRITICAL("make interface: {}", interface_result.error());
        return 1;
    }

    const auto [imu, camera, output] = std::move(*interface_result);

    SPDLOG_INFO(
        "init done. ({}ms)", std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::system_clock::now() - start)
                                 .count());

    std::atomic<uint64_t> last_seq{0};

    talos::World world;
    talos::Scheduler scheduler(world);

    struct ImageRawChannel {};

    auto backend_result = fcs::create_detector_backend(vision.at_legacy);
    if (!backend_result) {
        SPDLOG_CRITICAL(
            "create detector backend: {}", fcs::detector_error_str(backend_result.error()));
        return 3;
    }
    auto detector_backend = std::move(*backend_result);

    SPDLOG_INFO(
        "detector backend: {} (color={})", detector_backend.name(),
        fcs::to_string(vision.at_legacy.detector.detect_color));

    auto pnp_solver = fcs::create_pnp_solver(camera->camera_info());
    SPDLOG_INFO("PnP solver initialized");

    world.insert_resource(
        std::make_shared<fcs::AtLegacyDetectorBackend>(std::move(detector_backend)));
    world.insert_resource(std::make_shared<fcs::AtLegacyPnPSolver>(std::move(pnp_solver)));

    // Create Tracker using config from toml
    world.insert_resource(std::make_shared<fcs::L3::Tracker>(vision.tracker()));
    SPDLOG_INFO("Tracker initialized");

    // Create GimbalPlanner using config from toml (L4 Planning Layer)
    world.insert_resource(std::make_shared<fcs::L4::GimbalPlanner>(vision.planner()));
    SPDLOG_INFO(
        "GimbalPlanner initialized (bullet_speed={:.1f}m/s)",
        vision.planner().trajectory.bullet_speed);

    // Create coordinate system for fast_tf lookups
    switch (robot) {
    case fcs::RobotType::Infantry: world.insert_resource(fast_tf::create_infantry_system()); break;
    case fcs::RobotType::Hero: world.insert_resource(fast_tf::create_hero_system()); break;
    case fcs::RobotType::Sentry: world.insert_resource(fast_tf::create_sentry_system()); break;
    }

    // Insert config as shared resource (for foxglove and other systems)
    world.insert_resource(robot);
    world.insert_resource(std::make_shared<fcs::VisionConfig>(vision));

    auto& info = camera->camera_info();

    fcs::CameraConfig cam;

    cam.camera_matrix       = info.camera_matrix;
    cam.distort_coefficient = info.distort_coefficient;
    cam.height              = info.height;
    cam.width               = info.width;
    cam.profile             = info.profile;

    world.insert_resource(cam);

    world.insert_resource(ImuSnapshot{});
    // IMU: 100Hz, silent update (no notify)
    scheduler.add_system<talos::exclusive_silent<1000, 1>>(
        "imu_reader",
        [&imu](
            talos::res_mut<std::shared_ptr<fast_tf::CoordinateSystem<1024>>> tf_buffer,
            talos::res_mut<ImuSnapshot> imu_out) { imu->system(tf_buffer, imu_out); });

    // Camera: 160Hz, notify (triggers compute)
    scheduler.add_system<talos::exclusive<250, 1>>(
        "camera_reader",
        [&camera, &last_seq](
            talos::spmc_mut<fcs::AtLegacyImageFrame, fcs::AtLegacyImageChannelTag> cam_out) {
            const auto frame = camera->try_recv();
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
    fcs::L4::register_planning_systems_with_config(scheduler, vision.planner());

    // Register L5 Enhanced Weapon System (GimbalPlan + FireCommand -> WeaponCommand)
    fcs::L5::register_enhanced_weapon_system_with_config_and_compensator(
        scheduler, vision.weapon(), fcs::L4::create_compensator(vision.planner().compensator_type));

    // L1 Output System: Send WeaponCommand to device
    scheduler.add_system<talos::exclusive<250, 1>>(
        "weapon_output",
        [&output](talos::spmc<fcs::L5::WeaponCommand, fcs::WeaponCommandChannelTag> cmd_in) {
            auto cmd = cmd_in.read();
            if (!cmd) {
                return;
            }
            // Send to device (USB or IPC)
            output->send(*cmd);
        });

    if (foxglove.enabled) {
        auto foxglove_server = std::make_shared<fcs::FoxgloveServer>(foxglove.port, foxglove.host);
        if (foxglove_server->is_initialized()) {
            world.insert_resource(foxglove_server);
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
        SPDLOG_CRITICAL(
            "build scheduler: {}", talos::format_build_error(build_result.error()).c_str());
        return 1;
    }

    if (const auto result = scheduler.run(); !result) {
        SPDLOG_CRITICAL("run scheduler: {}", to_string(result.error()));
    }
}
