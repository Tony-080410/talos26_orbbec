#include <chrono>
#include "auto_aim/config.hpp"
#include "auto_aim/detector.hpp"
#include "camera/hik_camera.hpp"
#include "ipc/shm_client.hpp"
#include "spdlog_hook.hpp"
#include "strategy.hpp"
#include "talos_gimbal/stm32.hpp"
#include "tracker.hpp"

struct TalosConfig : SpVisionConfig {
    bool daedalus;
    std::expected<void, std::string> read_from(const toml::table &tbl) noexcept {
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

template<class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
template<class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

using read_config_error = std::variant<toml::parse_error, std::string>;

using namespace std::chrono_literals;

inline std::expected<TalosConfig, read_config_error> read_config() {
    TalosConfig config;
    toml::table tbl;
    try {
        tbl = toml::parse_file("sp_vision.toml");
    } catch (const toml::parse_error &err) {
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
//     { camera.recv_image() } -> std::convertible_to<std::expected<ipc::ShmClient::ImageFrame, RecvImageError>>;
// };

// hikcamera::ImageCapturer::recv_image() -> std::expected<ipc::ShmClient::ImageFrame, RecvImageError> {
//     return read(100ms);
// }

int main() {
    init_logger();
    spdlog::info("talos is initializing...");
    const auto start = std::chrono::system_clock::now();
    auto gimbal = talos_gimbal::Stm32Impl<talos_gimbal::Stm32Parser>{};
    if (const auto result = gimbal.connect(0x0483, 0x5740); !result) {
        spdlog::critical("gimbal: {}", talos_gimbal::to_string(result.error()));
        return 1;
    }
    const auto config_result = read_config();
    if (!config_result.has_value()) {
        std::visit(overloaded{[](const toml::parse_error &e) {
                                  const auto src = e.source();
                                  spdlog::critical("parse toml: {} at {}:{}:{}", e.description(), src.path->data(),
                                                   src.begin.line, src.begin.column);
                              },
                              [](const std::string &s) { spdlog::critical("read toml: {}", s); }},
                   config_result.error());
        return 1;
    }
    auto config = config_result.value();

    std::optional<ipc::ShmClient> client;
    std::unique_ptr<hikcamera::ImageCapturer> camera_;
    auto daedalus_or_camera = config.daedalus;
    if (daedalus_or_camera) {
        while (!client) {
            if (auto result = ipc::ShmClient::connect()) {
                client = std::move(*result);
                std::cout << "已连接到共享内存!\n";
            } else {
                std::cout << "." << std::flush;
                std::this_thread::sleep_for(500ms);
            }
        }
    } else {
        hikcamera::ImageCapturer::CameraProfile profile;
        camera_ = hikcamera::ImageCapturer::create(profile).value();
    }

    const auto info = client->camera_info();
    config.solver.camera_matrix << info.fx, 0.0, info.cx, 0.0, info.fy, info.cy, 0.0, 0.0, 1.0;
    config.solver.distort_coefficient << info.distortion[0], info.distortion[1], info.distortion[2], info.distortion[3],
            info.distortion[4];

    TalosStrategy strategy(false, config, 6, info.height, info.width);
    scheduler::Scheduler<TalosStrategy> sched;
    sched.init(config, &strategy);
    sched.start();

    spdlog::info(
            "done. ({}ms)",
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count());

    uint64_t last_seq = 0;
    while (true) {
        gimbal.handle_events();
        std::optional<ipc::ShmClient::ImageFrame> frame;
        if (daedalus_or_camera) {
            frame = client->recv_image();
        } else {
            auto img_result = camera_->read(100ms);
            if (img_result) {
                static uint64_t seq = 0;
                auto img = img_result.value();
                frame = ipc::ShmClient::ImageFrame{.image = std::move(img), .seq = seq++, .timestamp_ns = 0};
            } else {
                spdlog::critical("{}", img_result.error().what());
                return 2;
            }
        }

        if (!frame.has_value()) {
            // 没有新图像
            continue;
        }
        if (const auto pose_ = client->recv_pose(ipc::POSE_GIMBAL)) {
            const auto pose = *pose_;
            strategy.solver.set_R_gimbal2world(Eigen::Quaterniond{pose.qw, pose.qx, pose.qy, pose.qz});
        }
        // 跳帧检测
        auto v = *frame;
        cv::cvtColor(v.image, v.image, cv::COLOR_RGB2BGR);
        if (v.seq != last_seq + 1 && last_seq > 0) {
            spdlog::info("跳 {} 帧", v.seq - last_seq - 1);
        }
        last_seq = v.seq;
        spdlog::info("处理 {} 帧", last_seq);
        sched.push_frame(scheduler::Frame{v.seq, v.timestamp_ns, scheduler::ImageBuffer::from_mat(std::move(v.image))});
        // std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
}
