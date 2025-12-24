#include "auto_aim/config.hpp"
#include "auto_aim/detector.hpp"
#include "camera/hik_camera.hpp"
#include "fast_tf/infantry_tf.hpp"
#include "ipc/shm_client.hpp"
#include "spdlog_hook.hpp"
#include "strategy_wrapper.hpp"
#include "talos_gimbal/stm32.hpp"
#include "tracker.hpp"
#include <chrono>
#include <Eigen/Geometry>

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

// using RecvImageError = std::variant<hikcamera::CameraError, ipc::ShmError>;
// template<typename T>
// concept Camera = requires(T &camera) {
//     typename T::CameraProfile;
//     { camera.recv_image() } -> std::convertible_to<std::expected<ipc::ShmClient::ImageFrame,
//     RecvImageError>>;
// };

// hikcamera::ImageCapturer::recv_image() -> std::expected<ipc::ShmClient::ImageFrame,
// RecvImageError> {
//     return read(100ms);
// }

/**
 * @brief Get current time in nanoseconds
 * Uses std::chrono::high_resolution_clock for portability
 */
static uint64_t now_ns() {
    const auto now      = std::chrono::high_resolution_clock::now();
    const auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
}

int main() {
    init_logger();
    // spdlog::set_level(spdlog::level::debug);
    spdlog::info("talos is initializing...");
    const auto start   = std::chrono::system_clock::now();
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
    auto& config = config_result.value();

    std::optional<ipc::ShmClient> client;
    std::unique_ptr<hikcamera::ImageCapturer> camera_;
    auto daedalus = config.daedalus;
    if (daedalus) {
        while (!client) {
            if (auto result = ipc::ShmClient::connect()) {
                // 连接成功，但需要确认是活跃的共享内存（防止连接到旧文件）
                if (result->wait_for_producer(std::chrono::seconds(2))) {
                    client = std::move(*result);
                    spdlog::info("daedalus client connected");

                    const auto info = client->camera_info();
                    config.solver.camera_matrix << info.fx, 0.0, info.cx, 0.0, info.fy, info.cy,
                        0.0, 0.0, 1.0;
                    config.solver.distort_coefficient << info.distortion[0], info.distortion[1],
                        info.distortion[2], info.distortion[3], info.distortion[4];
                } else {
                    // heartbeat 未更新，可能是旧文件，等待 Rust 端重新创建
                    std::cout << "x" << std::flush;
                }
            } else {
                std::cout << "." << std::flush;
                std::this_thread::sleep_for(500ms);
            }
        }
    } else {
        hikcamera::ImageCapturer::CameraProfile profile;
        auto x = hikcamera::ImageCapturer::create(profile);
        if (!x) {
            spdlog::critical("hikcamera: {}", x.error().what());
            return 2;
        }
        camera_ = std::move(*x);
    }

    TalosStrategy strategy(false, config, 6, 1080, 1440);

    // USB 设备和 IMU 数据存储
    talos_gimbal::ReceiveImuData imu_data{};
    talos_gimbal::Stm32Parser::latest_imu = &imu_data;
    talos_gimbal::Stm32Impl<talos_gimbal::Stm32Parser> usb_device;

    // 初始化输出接口
    if (daedalus) {
        // IPC 模式: 通过共享内存发送到 Rust 模拟器
        if (auto ipc_output = auto_aim::OutputInterface::create_ipc()) {
            strategy.set_output_interface(std::move(*ipc_output));
            spdlog::info("endpoint: daedalus");
        } else {
            spdlog::critical("init daedalus endpoint: {}", ipc_output.error());
            return 2;
        }
    } else {
        // USB 模式: 通过串口发送到 STM32
        if (auto result = usb_device.connect(0x0483, 0x5740); result) {
            strategy.set_output_interface(
                auto_aim::OutputInterface(auto_aim::UsbOutput{&usb_device}));
            spdlog::info("endpoint: USB (STM32)");
        } else {
            spdlog::critical(
                "init USB (STM32) endpoint: {}", talos_gimbal::to_string(result.error()));
            return 2;
        }
    }

    FoxgloveServer foxglove_server(8765);
    FoxgloveWrapper wrapper(strategy, foxglove_server);
    scheduler::Scheduler<decltype(wrapper)> sched;
    sched.init(config, &wrapper);
    sched.start();

    spdlog::info(
        "done. ({}ms)", std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now() - start)
                            .count());

    uint64_t last_seq = 0;
    while (true) {
        // 处理 USB 事件 (非阻塞)
        if (!daedalus && usb_device.is_connected()) {
            usb_device.handle_events();
        }

        std::optional<ipc::ShmClient::ImageFrame> frame;
        if (daedalus) {
            frame = client->recv_image();
            auto pose = client->recv_pose(ipc::POSE_CAMERA);
            if (pose) {
                Eigen::Quaterniond q(pose->qw, pose->qx, pose->qy, pose->qz);
                strategy.solver.set_R_gimbal2world(q);

                // Extract yaw from quaternion for gimbal tracking
                auto euler = q.toRotationMatrix().eulerAngles(1,0,2); // ZYX order
                imu_data.data.yaw = euler[0];
                imu_data.data.pitch = -euler[1] ;
                imu_data.data.roll = euler[2] ;
            }
        } else {
            auto img_result = camera_->read(100ms);
            if (img_result) {
                static uint64_t seq = 0;
                auto img            = img_result.value();
                frame               = ipc::ShmClient::ImageFrame{
                                  .image = std::move(img), .seq = seq++, .timestamp_ns = now_ns()};
            } else {
                spdlog::critical("read hikcamera image: {}", img_result.error().what());
                return 2;
            }
        }

        if (!frame.has_value()) {
            // 没有新图像
            continue;
        }

        // 从 IMU 获取姿态角 (USB 模式)
        double roll  = imu_data.data.roll;
        double pitch = imu_data.data.pitch;
        double yaw   = imu_data.data.yaw;

        // 更新 strategy 的弹速和云台姿态
        strategy.set_bullet_speed(imu_data.data.bullet_speed);
        strategy.set_gimbal_pose(roll, pitch, yaw);

        Eigen::AngleAxisd rx(roll, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd ry(pitch, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd rz(yaw, Eigen::Vector3d::UnitZ());
        strategy.solver.set_R_gimbal2world(rx * ry * rz);

        // 跳帧检测
        auto [image, seq, timestamp_ns] = *frame;
        if (seq != last_seq + 1 && last_seq > 0) {
            spdlog::info("skip {} frame", seq - last_seq - 1);
        }

        last_seq = seq;
        // spdlog::info("处理 {} 帧", last_seq);
        auto img = image;
        if (daedalus) {
            img = cv::Mat();
            image.copyTo(img);
            cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
            // v.image is &'loop ImageData,
            // but we need to pass a &'pipeline ImageData
            // we cannot assure 'loop is longer than 'pipeline so we must copy the data.
        }
        sched.push_frame(
            scheduler::Frame{
                seq, timestamp_ns, scheduler::ImageBuffer::from_mat(std::move(img))});
        sched.tracker_event_.wait_for(1);
        // std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
}
