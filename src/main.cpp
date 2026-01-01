#include "auto_aim/config.hpp"
#include "core/time.hpp"
#include "ecs/app.hpp"
#include "fcs/L1_sensor/input_interface.hpp"
#include "fcs/L1_sensor/output_interface.hpp"
#include "fcs/L2_perception/at_legacy.hpp"
#include "fcs/L3_estimation/armor_solver.hpp"
#include "fcs/L5_weapon/weapon_systems.hpp"
#include "fcs/visualization/foxglove_systems.hpp"
#include "math/fast_tf/infantry.hpp"
#include "scheduler/scheduler.hpp"
#include "spdlog_hook.hpp"
#include "talos_gimbal/stm32.hpp"

#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <tbb/parallel_pipeline.h>

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
    toml::table tbl;
    try {
        tbl = toml::parse_file("at_vision.toml");
    } catch (const toml::parse_error& err) {
        return std::unexpected(err);
    }
    if (auto result = config.read_from(tbl); !result) {
        return std::unexpected(result.error());
    }
    return config;
}

int main() {
    init_logger();
    spdlog::info("talos is initializing...");
    const auto start = std::chrono::system_clock::now();

    auto config_result = read_config();
    if (!config_result.has_value()) {
        std::visit(
            overloaded{
                [](const toml::parse_error& e) {
                    const auto& src = e.source();
                    spdlog::critical(
                        "parse toml: {} at {}:{}:{}", e.description(), src.path->data(),
                        src.begin.line, src.begin.column);
                },
                [](const std::string& s) { spdlog::critical("read toml: {}", s); }},
            config_result.error());
        return 1;
    }
    auto& config        = config_result.value();
    const bool daedalus = config.daedalus;

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
            spdlog::critical("connect ipc: {}", ipc::to_string(ipc_.error()));
            return 1;
        }
        ipc_device   = std::make_shared<ipc::ShmClient>(std::move(*ipc_));
        input_result = fcs::L1::InputInterface::create_ipc(ipc_device);
        if (auto ipc_output = fcs::L1::OutputInterface::create_ipc(ipc_device)) {
            output_interface = std::make_unique<fcs::L1::OutputInterface>(std::move(*ipc_output));
            spdlog::info("endpoint: daedalus");
        } else {
            spdlog::critical("init daedalus endpoint: {}", ipc_output.error());
            return 2;
        }
    } else {
        input_result = fcs::L1::InputInterface::create_hik({}, config.camera.camera_info);
        if (auto result = usb_device->connect(0x0483, 0x5740); result) {
            output_interface =
                std::make_unique<fcs::L1::OutputInterface>(fcs::L1::UsbOutput{usb_device});
            spdlog::info("endpoint: USB (STM32)");
        } else {
            spdlog::critical(
                "init USB (STM32) endpoint: {}", talos_gimbal::to_string(result.error()));
            return 2;
        }
    }
    if (!input_result) {
        std::visit(
            overloaded{
                [](fcs::L1::Timeout) { spdlog::critical("input: timeout"); },
                [](fcs::L1::Disconnected) { spdlog::critical("input: disconnected"); },
                [](fcs::L1::ProducerNotAvailable) {
                    spdlog::critical("input: producer not available");
                },
                [](const hikcamera::CameraError& e) { spdlog::critical("input: {}", e.what()); },
                [](ipc::ShmError e) { spdlog::critical("input: {}", ipc::to_string(e)); },
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
        config.camera.camera_info         = info;
    }

    spdlog::info(
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
        spdlog::critical(
            "create detector backend: {}", fcs::detector_error_str(backend_result.error()));
        return 3;
    }
    auto detector_backend = std::move(*backend_result);
    spdlog::info(
        "detector backend: {} (color={})", detector_backend.name(),
        config.at_legacy.detector.detect_color == 0 ? "BLUE" : "RED");

    auto pnp_solver = fcs::create_pnp_solver(config.camera);
    spdlog::info("PnP solver initialized");

    // Insert resources (wrapped in shared_ptr for std::any compatibility)
    world.insert_resource(
        std::make_shared<fcs::AtLegacyDetectorBackend>(std::move(detector_backend)));
    world.insert_resource(std::make_shared<fcs::AtLegacyPnPSolver>(std::move(pnp_solver)));

    // Note: SPMC channels are auto-created when systems access them

    // ============================================================================
    // End AT Legacy Setup
    // ============================================================================

    // ============================================================================
    // L3 Armor Solver Setup
    // ============================================================================

    // Create Tracker using config from toml
    world.insert_resource(std::make_shared<fcs::L3::Tracker>(config.tracker()));
    spdlog::info(
        "Tracker initialized (lost_thres={:.1f}s, tracking_thres={})", config.tracker().lost_thres,
        config.tracker().tracking_thres);

    // Create Solver using config from toml
    world.insert_resource(std::make_shared<fcs::L3::Solver>(config.solver()));
    spdlog::info(
        "Solver initialized (bullet_speed={:.1f}m/s)", config.solver().trajectory.bullet_speed);

    // Create coordinate system for fast_tf lookups
    world.insert_resource(fast_tf::infantry_new<1024>());

    // Insert config as shared resource (for foxglove and other systems)
    world.insert_resource(std::make_shared<fcs::TalosConfig>(config));

    // ============================================================================
    // End L3 Armor Solver Setup
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
            spdlog::info(
                "Foxglove server enabled on ws://{}:{}", config.foxglove.host,
                config.foxglove.port);
        } else {
            spdlog::warn("Foxglove server failed to initialize, visualization disabled");
        }
    } else {
        spdlog::info("Foxglove visualization disabled by config");
    }

    // ============================================================================
    // End Foxglove Setup
    // ============================================================================

    world.insert_resource(ImuSnapshot{});
    // IMU: 100Hz, silent update (no notify)
    scheduler.add_system<talos::exclusive_silent<1000>>(
        "imu_reader",
        [daedalus, &ipc_device, &usb_device, &imu_data](
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
                }
            } else {
                usb_device->handle_events();
                imu_out->timestamp = fcs::clock::now_ns();
                imu_out->roll_deg  = imu_data.data.roll;
                imu_out->pitch_deg = imu_data.data.pitch;
                imu_out->yaw_deg   = imu_data.data.yaw;
            }
            // strategy.set_bullet_speed(imu_data.data.bullet_speed);
            fast_tf::update(
                *tf_buffer->get(),
                fast_tf::FuckTransform<fast_tf::gimbal_link_fuxk_frame>(
                    fast_tf::TransformMatrix4d::from_rpy(
                        imu_out->roll_deg, imu_out->pitch_deg, imu_out->yaw_deg)),
                imu_out->timestamp);
        });

    // Camera: 160Hz, notify (triggers compute)
    scheduler.add_system<talos::exclusive<160>>(
        "camera_reader",
        [daedalus, &ipc_device, &input, &last_seq](
            talos::spmc_mut<fcs::AtLegacyImageFrame, fcs::AtLegacyImageChannelTag> cam_out) {
            const auto frame = input.try_recv();
            if (!frame) {
                return;
            }
            const auto seq = frame->seq;
            if (seq != last_seq + 1 && last_seq > 0) {
                spdlog::info("skip {} frame", seq - last_seq - 1);
            }
            last_seq = frame->seq;

            auto img = frame->image;
            cam_out.write(fcs::AtLegacyImageFrame{std::move(img), frame->timestamp_ns, seq});
        });

    // Register AT Legacy detection pipeline systems
    fcs::register_at_legacy_detection_systems(scheduler);

    // Register L3 Armor Solver pipeline (Measurement -> TrackerOutput -> GimbalCommand)
    fcs::L3::register_armor_solver_systems(scheduler);

    // Register L5 Weapon System (GimbalCommand -> WeaponCommand)
    fcs::L5::register_weapon_system(scheduler);

    // L1 Output System: Send WeaponCommand to device
    scheduler.add_system<talos::exclusive<250>>(
        "weapon_output",
        [&output_interface](
            talos::spmc<fcs::L5::WeaponCommand, fcs::WeaponCommandChannelTag> cmd_in) {
            auto cmd = cmd_in.read();
            if (!cmd || !cmd->valid) {
                return;
            }

            // Send to device (USB or IPC)
            output_interface->send(*cmd);
        });
    scheduler.print_systems();

    // Build
    if (auto build_result = scheduler.build(); !build_result) {
        spdlog::critical(
            "build scheduler: {}", talos::format_build_error(build_result.error()).c_str());
        return 1;
    }
    if (const auto result = scheduler.run(); !result) {
        spdlog::critical("run scheduler: {}", to_string(result.error()));
    }
}
