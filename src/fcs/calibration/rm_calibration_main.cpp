/**
 * @file rm_calibration_main.cpp
 * @brief RM Calibration Tool - Camera intrinsic and hand-eye calibration
 *
 * Usage:
 *   ./rm_calibration --config calibration.toml
 *   ./rm_calibration --mode Intrinsic
 *   ./rm_calibration --mode Handeye --intrinsic camera_intrinsic.toml
 *   ./rm_calibration --daedalus  # Use daedalus simulator
 */

#include "fcs/calibration/calibration_board.hpp"
#include "fcs/calibration/calibration_config.hpp"
#include "fcs/calibration/calibration_systems.hpp"
#include "fcs/calibration/chessboard_detector.hpp"
#include "fcs/calibration/handeye_calibrator.hpp"
#include "fcs/calibration/intrinsic_calibrator.hpp"
#include "shm_region.hpp"
#include "spdlog_hook.hpp"

#include "auto_aim/config.hpp"
#include "fcs/L1_sensor/camera_interface.hpp"
#include "fcs/L1_sensor/parcel.hpp"
#include "fcs/core/channel_topics.hpp"
#include "fcs/core/time.hpp"
#include "fcs/visualization/foxglove_server.hpp"
#include "ipc/shm_client.hpp"
#include "math/euler.hpp"
#include "math/fast_tf/infantry.hpp"
#include "scheduler/scheduler/error_formatter.hpp"
#include "scheduler/scheduler/scheduler.hpp"
#include "talos_gimbal/packet.hpp"
#include "talos_gimbal/stm32.hpp"
#include "toml_helper.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <spdlog/spdlog.h>
#include <thread>

using namespace fcs::calibration;

// Global signal handler
std::atomic<bool> g_running{true};

void signal_handler(int) { g_running = false; }

void print_usage() {
    std::cout << R"(
RM Calibration Tool - Camera Intrinsic and Hand-Eye Calibration

Usage:
  rm_calibration [options]

Options:
  --config <path>       Configuration file path (default: calibration.toml)
  --mode <mode>         Override calibration mode: Intrinsic, Handeye, Full
  --intrinsic <path>    Load existing intrinsic calibration (required for Handeye mode)
  --daedalus            Use daedalus simulator instead of real hardware
  --help                Show this help message

Examples:
  # Run intrinsic calibration with real hardware
  rm_calibration --mode Intrinsic

  # Run intrinsic calibration with daedalus simulator
  rm_calibration --mode Intrinsic --daedalus

  # Run hand-eye calibration with existing intrinsics
  rm_calibration --mode Handeye --intrinsic camera_intrinsic.toml

  # Run full calibration
  rm_calibration --mode Full

Press 'Space' to capture sample, 'c' to run calibration, 'q' to quit.
)";
}

int main(int argc, char* argv[]) {
    init_logger();
    // Parse command line arguments
    std::string config_path = "calibration.toml";
    std::string intrinsic_path;
    std::optional<CalibrationMode> mode_override;
    std::optional<bool> daedalus_override;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        } else if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--mode" && i + 1 < argc) {
            std::string mode_str = argv[++i];
            auto mode            = magic_enum::enum_cast<CalibrationMode>(mode_str);
            if (mode) {
                mode_override = *mode;
            } else {
                SPDLOG_ERROR("Invalid mode: {}", mode_str);
                return 1;
            }
        } else if (arg == "--intrinsic" && i + 1 < argc) {
            intrinsic_path = argv[++i];
        } else if (arg == "--daedalus") {
            daedalus_override = true;
        }
    }

    // Setup signal handler
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    SPDLOG_INFO("RM Calibration Tool starting...");
    SPDLOG_INFO("Config file: {}", config_path);

    // Load configuration
    auto config_result = CalibrationConfig::load_from_file(config_path);
    if (!config_result) {
        SPDLOG_ERROR("Failed to load config: {}", config_result.error());
        return 1;
    }
    auto config = std::make_shared<CalibrationConfig>(std::move(*config_result));

    // Apply mode override
    if (mode_override) {
        config->mode = *mode_override;
    }

    // Apply daedalus override
    if (daedalus_override) {
        config->input.daedalus = *daedalus_override;
    }

    const bool use_daedalus = config->input.daedalus;

    SPDLOG_INFO("Calibration mode: {}", magic_enum::enum_name(config->mode));
    SPDLOG_INFO("Input source: {}", use_daedalus ? "daedalus simulator" : "real hardware");
    SPDLOG_INFO("Board type: {}", magic_enum::enum_name(config->board.type));
    SPDLOG_INFO(
        "Board size: {}x{}, square: {}mm", config->board.width, config->board.height,
        config->board.square_size * 1000);

    // Create calibration board detector
    auto board_result = create_board(config->board, config->charuco);
    if (!board_result) {
        SPDLOG_ERROR("Failed to create board detector: {}", board_result.error());
        return 1;
    }
    auto board = std::shared_ptr<CalibrationBoard>(std::move(*board_result));

    // Create calibrators
    auto intrinsic_calibrator =
        std::make_shared<IntrinsicCalibrator>(config->capture, config->intrinsic);
    auto handeye_calibrator = std::make_shared<HandEyeCalibrator>(config->capture, config->handeye);

    // Load existing intrinsic if provided or required
    auto intrinsic_result = std::make_shared<IntrinsicResult>();
    if (!intrinsic_path.empty()) {
        auto loaded = toml::parse_file(intrinsic_path);
        if (!loaded) {
            SPDLOG_ERROR("Failed to load intrinsic file: {}", intrinsic_path);
            return 1;
        }

        if (auto cam = loaded.table()["camera"].as_table()) {
            fcs::CameraConfig cam_cfg;
            if (auto res = cam_cfg.read_from(*cam); res) {
                intrinsic_result->camera_matrix       = cam_cfg.camera_matrix;
                intrinsic_result->distort_coefficient = cam_cfg.distort_coefficient;
                intrinsic_result->width               = cam_cfg.width;
                intrinsic_result->height              = cam_cfg.height;
                SPDLOG_INFO("Loaded intrinsic calibration from {}", intrinsic_path);
            }
        }
    }

    // Validate requirements
    if (config->mode == CalibrationMode::Handeye && intrinsic_result->width == 0) {
        SPDLOG_ERROR("Handeye mode requires intrinsic calibration. Use --intrinsic <path>");
        return 1;
    }

    // Create calibration status
    auto status            = std::make_shared<CalibrationStatus>();
    status->state          = CalibrationState::Idle;
    status->target_samples = config->capture.min_samples;

    // Initialize input source (camera + IMU)
    std::shared_ptr<ipc::ShmClient> ipc_device;
    fcs::L1::CameraInterface input;
    bool imu_available = false;

    if (use_daedalus) {
        // Connect to daedalus simulator via shared memory
        SPDLOG_INFO("Connecting to daedalus simulator...");
        auto ipc_result = ipc::ShmClient::connect();
        if (!ipc_result) {
            SPDLOG_ERROR("Failed to connect to daedalus: {}", ipc::to_string(ipc_result.error()));
            return 1;
        }
        ipc_device = std::make_shared<ipc::ShmClient>(std::move(*ipc_result));

        // Create IPC input interface
        auto input_result = fcs::L1::CameraInterface::create_ipc(ipc_device);
        if (!input_result) {
            SPDLOG_ERROR("create IPC input: {}", fcs::L1::to_string(input_result.error()));
            return 1;
        }
        input = std::move(*input_result);

        // Update camera config from daedalus
        const auto& cam_info                  = input.camera_info();
        intrinsic_result->camera_matrix       = cam_info.camera_matrix;
        intrinsic_result->distort_coefficient = cam_info.distort_coefficient;
        intrinsic_result->width               = cam_info.width;
        intrinsic_result->height              = cam_info.height;

        SPDLOG_INFO("Daedalus connected - camera: {}x{}", cam_info.width, cam_info.height);
        imu_available = true; // Daedalus always provides pose data
    } else {
        // Initialize real hardware
        SPDLOG_INFO("Initializing camera...");
        fcs::CameraConfig camera_config;
        camera_config.width   = config->width;
        camera_config.height  = config->height;
        camera_config.profile = config->profile;
        auto input_result     = fcs::L1::CameraInterface::create_hik(camera_config);
        if (!input_result) {
            SPDLOG_ERROR("create camera input: {}", fcs::L1::to_string(input_result.error()));
            return 1;
        }
        input = std::move(*input_result);
    }

    // Initialize STM32 for IMU (only for real hardware in hand-eye mode)
    talos_gimbal::ReceiveImuData imu_data{};
    std::shared_ptr<talos_gimbal::Stm32Impl<talos_gimbal::Stm32Parser>> stm32;
    bool stm32_connected = false;

    if (!use_daedalus
        && (config->mode == CalibrationMode::Handeye || config->mode == CalibrationMode::Full)) {
        talos_gimbal::Stm32Parser::latest_imu = &imu_data;
        stm32 = std::make_shared<talos_gimbal::Stm32Impl<talos_gimbal::Stm32Parser>>();

        SPDLOG_INFO("Initializing STM32 for IMU...");
        if (auto result = stm32->connect(); result) {
            stm32_connected = true;
            imu_available   = true;
            SPDLOG_INFO("STM32 connected successfully");
        } else {
            SPDLOG_WARN("STM32 not available - hand-eye calibration may not work");
        }
    }

    // Create TF buffer
    auto tf_buffer = fast_tf::infantry_new<1024>();

    // Initialize Foxglove server
    std::shared_ptr<fcs::FoxgloveServer> foxglove_server;
    if (config->foxglove.enabled) {
        foxglove_server =
            std::make_shared<fcs::FoxgloveServer>(config->foxglove.port, config->foxglove.host);
        if (foxglove_server->is_initialized()) {
            SPDLOG_INFO(
                "Foxglove server started on {}:{}", config->foxglove.host, config->foxglove.port);
        } else {
            SPDLOG_WARN("Foxglove server failed to initialize");
            foxglove_server.reset();
        }
    }

    // Setup scheduler
    talos::World world;
    talos::Scheduler scheduler(world);

    // Insert resources
    world.insert_resource(config);
    world.insert_resource(status);
    world.insert_resource(tf_buffer);
    world.insert_resource(intrinsic_result);
    if (foxglove_server) {
        world.insert_resource(foxglove_server);
    }

    // Camera reader system
    scheduler.add_system<talos::fixed_rate<30>>(
        "camera_reader",
        [&input](talos::spmc_mut<fcs::AtLegacyImageFrame, fcs::AtLegacyImageChannelTopic> cam_out) {
            auto frame = input.try_recv();
            if (!frame)
                return;

            fcs::AtLegacyImageFrame out_frame;
            out_frame.image        = std::move(frame->image);
            out_frame.timestamp_ns = frame->timestamp_ns;
            cam_out.write(std::move(out_frame));
        });

    // IMU reader system - different implementation for daedalus vs real hardware
    if (use_daedalus && ipc_device) {
        // Daedalus: read pose from shared memory
        scheduler.add_system<talos::fixed_rate_silent<100>>(
            "imu_reader", [ipc_device, tf_buffer]() {
                if (const auto pose = ipc_device->recv_pose(ipc::POSE_GIMBAL)) {
                    const Eigen::Quaterniond q(pose->qw, pose->qx, pose->qy, pose->qz);
                    const auto euler = math_fuxk::rpy(q);

                    // Update gimbal_link transform
                    fast_tf::update(
                        *tf_buffer,
                        fast_tf::FuckTransform<fast_tf::gimbal_link_fuxk_frame>(
                            fast_tf::TransformMatrix4d::from_rpy(
                                euler.roll, euler.pitch, euler.yaw)),
                        pose->timestamp_ns);
                }
            });
    } else if (stm32_connected && stm32) {
        // Real hardware: read from STM32
        scheduler.add_system<talos::fixed_rate_silent<100>>(
            "imu_reader", [stm32, tf_buffer, &imu_data]() {
                // Handle USB events (this updates latest_imu via callback)
                stm32->handle_events();

                // Read the latest IMU data
                if (talos_gimbal::Stm32Parser::latest_imu) {
                    auto& imu = *talos_gimbal::Stm32Parser::latest_imu;
                    auto ts   = fcs::clock::now_ns();

                    // Update gimbal_link transform
                    fast_tf::update(
                        *tf_buffer,
                        fast_tf::FuckTransform<fast_tf::gimbal_link_fuxk_frame>(
                            fast_tf::TransformMatrix4d::from_rpy(
                                imu.data.roll, imu.data.pitch, imu.data.yaw)),
                        ts);
                }
            });
    }

    // Register calibration systems
    register_calibration_systems(
        scheduler, config, board, intrinsic_calibrator, handeye_calibrator, intrinsic_result);

    // Build scheduler
    if (auto build_result = scheduler.build(); !build_result) {
        SPDLOG_CRITICAL("Failed to build scheduler: {}", build_result.error());
        return 1;
    }

    SPDLOG_INFO("Calibration system built");
    SPDLOG_INFO("Starting scheduler... (Ctrl+C to quit)");

    // Set initial state to capturing
    status->state = CalibrationState::Capturing;

    // Add a system to handle auto-calibration logic
    // Note: For interactive calibration, consider running scheduler in a thread
    // and using stdin for commands

    // Run scheduler in a background thread
    std::thread scheduler_thread([&scheduler]() {
        if (const auto result = scheduler.run(); !result) {
            SPDLOG_ERROR("Scheduler error: {}", talos::to_string(result.error()));
        }
    });

    // Main loop - handle keyboard input and check progress
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Check if we have enough samples
        uint32_t current_samples = 0;
        if (config->mode == CalibrationMode::Intrinsic) {
            current_samples = intrinsic_calibrator->sample_count();
        } else if (config->mode == CalibrationMode::Handeye) {
            current_samples = handeye_calibrator->sample_count();
        }

        // Auto-calibrate when target reached (for automatic mode)
        if (config->capture.auto_capture && current_samples >= config->capture.min_samples
            && status->state == CalibrationState::Capturing) {

            SPDLOG_INFO("Target samples reached, running calibration...");
            status->state = CalibrationState::Calibrating;

            if (config->mode == CalibrationMode::Intrinsic) {
                auto result =
                    intrinsic_calibrator->calibrate(cv::Size(config->width, config->height));
                if (result) {
                    *intrinsic_result = *result;
                    save_intrinsic_result(*result, config->output.intrinsic_path);
                    SPDLOG_INFO("Intrinsic calibration saved to {}", config->output.intrinsic_path);
                    SPDLOG_INFO("RMS error: {:.4f} pixels", result->rms_error);
                    status->state = CalibrationState::Completed;
                } else {
                    SPDLOG_ERROR("Calibration failed: {}", result.error());
                    status->state = CalibrationState::Failed;
                }
            } else if (config->mode == CalibrationMode::Handeye) {
                auto result = handeye_calibrator->calibrate();
                if (result) {
                    save_handeye_result(
                        *result, config->output.extrinsic_path, config->handeye.method);
                    print_handeye_result(*result);
                    SPDLOG_INFO("Hand-eye calibration saved to {}", config->output.extrinsic_path);
                    status->state = CalibrationState::Completed;
                } else {
                    SPDLOG_ERROR("Calibration failed: {}", result.error());
                    status->state = CalibrationState::Failed;
                }
            }
            if (status->state == CalibrationState::Completed) {
                // Stop after completing calibration
                g_running = false;
            }
        }
    }

    // Cleanup
    SPDLOG_INFO("Shutting down...");
    scheduler.stop();
    if (scheduler_thread.joinable()) {
        scheduler_thread.join();
    }

    return 0;
}
