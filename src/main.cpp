#include <chrono>
#include "auto_aim/config.hpp"
#include "detector.hpp"
#include "shm_client.hpp"
#include "spdlog_hook.hpp"
#include "strategy.hpp"
#include "talos_gimbal/stm32.hpp"
#include "tracker.hpp"

template<class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
template<class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

using read_config_error = std::variant<toml::parse_error, std::string>;

using namespace std::chrono_literals;

inline std::expected<SpVisionConfig, read_config_error> read_config() {
    SpVisionConfig config;
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
    while (!client) {
        if (auto result = ipc::ShmClient::connect()) {
            client = std::move(*result);
            std::cout << "已连接到共享内存!\n";
        } else {
            std::cout << "." << std::flush;
            std::this_thread::sleep_for(500ms);
        }
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
        auto frame = client->recv_image();
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
