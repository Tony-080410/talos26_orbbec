#include "auto_aim/config.hpp"
#include "ecs/app.hpp"
#include "fcs/L1_sensor/input_interface.hpp"
#include "fcs/L1_sensor/output_interface.hpp"
#include "math/fast_tf/infantry.hpp"
#include "scheduler/scheduler.hpp"
#include "spdlog_hook.hpp"
#include "talos_gimbal/stm32.hpp"

#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <tbb/parallel_pipeline.h>

// ============================================================================
// Config
// ============================================================================

struct ImuData;
struct TalosConfig : SpVisionConfig {
    bool daedalus;
    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        if (auto read = SpVisionConfig::read_from(tbl); !read) {
            return read;
        }
        if (const auto _t = tbl["runtime"].as_table()) {
            auto t = *_t;
            READ_KEY(t, daedalus);
            return {};
        }
        return std::unexpected("runtime section not found");
    }
};

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
        tbl = toml::parse_file("sp_vision.toml");
    } catch (const toml::parse_error& err) {
        return std::unexpected(err);
    }
    if (auto result = config.read_from(tbl); !result) {
        return std::unexpected(result.error());
    }
    return config;
}

static uint64_t now_ns() {
    const auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
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
        input_result = fcs::L1::InputInterface::create_hik({}, config.solver.camera_info);
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
        config.solver.camera_matrix       = info.camera_matrix;
        config.solver.distort_coefficient = info.distort_coefficient;
        config.solver.camera_info         = info;
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

    world.insert_resource(ImuSnapshot{});
    // IMU: 100Hz, silent update (no notify)
    scheduler.add_system<talos::exclusive_silent<1000>>(
        "imu_reader",
        [daedalus, &ipc_device, &usb_device, &imu_data](talos::res_mut<ImuSnapshot> imu_out) {
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
                imu_out->timestamp = imu_data.time_stamp;
                imu_out->roll_deg  = imu_data.data.roll;
                imu_out->pitch_deg = imu_data.data.pitch;
                imu_out->yaw_deg   = imu_data.data.yaw;
            }
        });

    // Camera: 30Hz, notify (triggers compute)
    scheduler.add_system<talos::exclusive<160>>(
        "camera_reader", [daedalus, &ipc_device, &usb_device, &input,
                          &last_seq](talos::spmc_mut<fcs::L1::Frame, ImageRawChannel> cam_out) {
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
            if (daedalus) {
                img = cv::Mat();
                frame->image.copyTo(img);
                cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
                // v.image is &'loop ImageData,
                // but we need to pass a &'pipeline ImageData
            }
            cam_out.write(fcs::L1::Frame{seq, frame->timestamp_ns, img});
        });

    scheduler.add_system("preprocessor", [](talos::spmc<fcs::L1::Frame, ImageRawChannel> cam_in) {
        if (!cam_in.has_new()) {
            return;
        }
        auto frame = cam_in.read();
        if (!frame) {
            return;
        }
        spdlog::info("camera in {}", frame->seq);
        const auto image = frame->image;
        spdlog::info("camera in {}", image.size.dims());
    });

    scheduler.print_systems();

    // Build
    if (auto build_result = scheduler.build(); !build_result) {
        spdlog::critical(
            "build scheduler: %s", talos::format_build_error(build_result.error()).c_str());
        return 1;
    }
    if (const auto result = scheduler.run(); !result) {
        spdlog::critical("run scheduler: {}", to_string(result.error()));
    }
}