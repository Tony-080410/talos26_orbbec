#include "../include/math/euler.hpp"
#include "../include/scheduler/parcel/tracker.hpp"
#include "auto_aim/config.hpp"
#include "auto_aim/detector.hpp"
#include "fcs/L1_sensor/input_interface.hpp"
#include "math/fast_tf/infantry.hpp"
#include "spdlog_hook.hpp"
#include "talos_gimbal/stm32.hpp"

#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <tbb/parallel_pipeline.h>

// ============================================================================
// Config
// ============================================================================

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

// ============================================================================
// Pipeline Token - 在各阶段间传递的数据包
// ============================================================================

struct PipelineToken {
    // 输入数据
    cv::Mat image;
    uint64_t seq          = 0;
    uint64_t timestamp_ns = 0;

    // IMU 数据快照
    double roll         = 0;
    double pitch        = 0;
    double yaw          = 0;
    double bullet_speed = 0;

    // 预处理结果
    cv::Mat gray;
    cv::Mat binary;

    // 感知结果
    std::vector<Detection> detections;
    std::vector<Measurement> measurements;

    // 跟踪结果
    bool has_track = false;
    TrackState track;
    GimbalCommand command;

    bool valid = true;
};

// ============================================================================
// Main
// ============================================================================

int main() {
    init_logger();
    spdlog::info("talos is initializing...");
    const auto start = std::chrono::system_clock::now();

    // ------------------------------------------------------------------------
    // 1. 读取配置
    // ------------------------------------------------------------------------
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

    // ------------------------------------------------------------------------
    // 2. 初始化相机
    // ------------------------------------------------------------------------
    auto input_result = daedalus ? fcs::InputInterface::create_ipc()
                                 : fcs::InputInterface::create_hik({}, config.solver.camera_info);

    if (!input_result) {
        std::visit(
            overloaded{
                [](fcs::Timeout) { spdlog::critical("input: timeout"); },
                [](fcs::Disconnected) { spdlog::critical("input: disconnected"); },
                [](fcs::ProducerNotAvailable) {
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

    // ------------------------------------------------------------------------
    // 3. 初始化各组件
    // ------------------------------------------------------------------------
    FoxgloveServer foxglove_server(8765);
    auto& tf_buffer = foxglove_server.get_tf_buffer();

    // 处理组件（部分需要线程安全）
    Detector detector(config.detector);    // 并行阶段使用，需线程安全
    Solver solver(config.solver);          // 并行阶段使用，需线程安全
    Estimator estimator(config.estimator); // 串行阶段使用
    FireControl fire_control(config.fire_control);

    // USB 设备和 IMU
    talos_gimbal::ReceiveImuData imu_data{};
    talos_gimbal::Stm32Parser::latest_imu = &imu_data;
    talos_gimbal::Stm32Impl<talos_gimbal::Stm32Parser> usb_device;

    // 输出接口
    std::unique_ptr<auto_aim::OutputInterface> output_interface;
    if (daedalus) {
        if (auto ipc_output = auto_aim::OutputInterface::create_ipc()) {
            output_interface = std::make_unique<auto_aim::OutputInterface>(std::move(*ipc_output));
            spdlog::info("endpoint: daedalus");
        } else {
            spdlog::critical("init daedalus endpoint: {}", ipc_output.error());
            return 2;
        }
    } else {
        if (auto result = usb_device.connect(0x0483, 0x5740); result) {
            output_interface =
                std::make_unique<auto_aim::OutputInterface>(auto_aim::UsbOutput{&usb_device});
            spdlog::info("endpoint: USB (STM32)");
        } else {
            spdlog::critical(
                "init USB (STM32) endpoint: {}", talos_gimbal::to_string(result.error()));
            return 2;
        }
    }

    spdlog::info(
        "init done. ({}ms)", std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::system_clock::now() - start)
                                 .count());

    // ------------------------------------------------------------------------
    // 4. 流水线状态
    // ------------------------------------------------------------------------
    std::atomic<bool> running{true};
    std::atomic<uint64_t> last_seq{0};

    // 共享的 IMU 数据（输入阶段读取，估计阶段使用）
    struct ImuSnapshot {
        double roll, pitch, yaw, bullet_speed;
    };

    // ------------------------------------------------------------------------
    // 5. 启动流水线
    // ------------------------------------------------------------------------
    constexpr size_t MAX_TOKENS = 4; // 同时处理的帧数

    tbb::parallel_pipeline(
        MAX_TOKENS,

        // ====================================================================
        // Stage 1: 输入 (串行) - 获取图像和 IMU 数据
        // ====================================================================
        tbb::make_filter<void, std::shared_ptr<PipelineToken>>(
            tbb::filter_mode::serial_in_order,
            [&](tbb::flow_control& fc) -> std::shared_ptr<PipelineToken> {
                if (!running.load(std::memory_order_acquire)) {
                    fc.stop();
                    return nullptr;
                }

                // 处理 USB 事件
                if (!daedalus && usb_device.is_connected()) {
                    usb_device.handle_events();
                }

                // 获取图像帧
                auto frame = input.try_recv();
                if (!frame) {
                    return nullptr;
                }

                // IPC 模式下获取位姿数据
                if (auto* ipc_input = input.as_ipc()) {
                    if (auto pose = ipc_input->client().recv_pose(ipc::POSE_GIMBAL)) {
                        Eigen::Quaterniond q(pose->qw, pose->qx, pose->qy, pose->qz);
                        const auto [r, p, y] = math_fuxk::rpy(q).rpy();
                        imu_data.data.roll   = r;
                        imu_data.data.pitch  = p;
                        imu_data.data.yaw    = y;
                    }
                }

                auto [seq, timestamp_ns, image] = *frame;

                // 跳帧检测
                uint64_t prev_seq = last_seq.load();
                if (seq != prev_seq + 1 && prev_seq > 0) {
                    spdlog::warn("skip {} frames", seq - prev_seq - 1);
                }
                last_seq.store(seq);

                // 创建 token
                auto token          = std::make_shared<PipelineToken>();
                token->seq          = seq;
                token->timestamp_ns = timestamp_ns;

                // 拷贝图像（daedalus 模式需要转换颜色空间）
                if (daedalus) {
                    cv::cvtColor(image, token->image, cv::COLOR_RGB2BGR);
                } else {
                    token->image = std::move(image);
                }

                // 快照 IMU 数据
                token->roll         = imu_data.data.roll;
                token->pitch        = imu_data.data.pitch;
                token->yaw          = imu_data.data.yaw;
                token->bullet_speed = imu_data.data.bullet_speed;

                // 更新 TF
                fast_tf::update(
                    tf_buffer,
                    fast_tf::FuckTransform<fast_tf::gimbal_link_fuxk_frame>(
                        fast_tf::TransformMatrix4d::from_rpy(
                            token->roll, token->pitch, token->yaw)),
                    timestamp_ns);

                return token;
            })
            &

            // ====================================================================
            // Stage 2: 预处理 (并行) - 灰度化 + 二值化
            // ====================================================================
            tbb::make_filter<std::shared_ptr<PipelineToken>, std::shared_ptr<PipelineToken>>(
                tbb::filter_mode::parallel,
                [&](std::shared_ptr<PipelineToken> token) -> std::shared_ptr<PipelineToken> {
                    if (!token || token->image.empty()) {
                        return token;
                    }

                    cv::cvtColor(token->image, token->gray, cv::COLOR_BGR2GRAY);
                    cv::threshold(
                        token->gray, token->binary, config.detector.binary_threshold, 255,
                        cv::THRESH_BINARY);

                    return token;
                })
            &

            // ====================================================================
            // Stage 3: 感知 (并行) - 检测 + PnP 解算
            // ====================================================================
            tbb::make_filter<std::shared_ptr<PipelineToken>, std::shared_ptr<PipelineToken>>(
                tbb::filter_mode::parallel,
                [&](std::shared_ptr<PipelineToken> token) -> std::shared_ptr<PipelineToken> {
                    if (!token || token->image.empty()) {
                        return token;
                    }

                    // 检测装甲板
                    auto detections = detector.detect(
                        token->image, token->binary, token->timestamp_ns, token->seq);

                    // PnP 解算
                    for (const auto& det : detections) {
                        if (auto meas = solver.solve(det)) {
                            token->detections.push_back(det);
                            token->measurements.push_back(*meas);
                        }
                    }

                    spdlog::debug(
                        "[Pipeline] frame {} detected {} armors", token->seq,
                        token->measurements.size());

                    return token;
                })
            &

            // ====================================================================
            // Stage 4: 估计 (串行) - 跟踪 + 火控
            // ====================================================================
            tbb::make_filter<std::shared_ptr<PipelineToken>, std::shared_ptr<PipelineToken>>(
                tbb::filter_mode::serial_in_order,
                [&](std::shared_ptr<PipelineToken> token) -> std::shared_ptr<PipelineToken> {
                    if (!token)
                        return token;

                    // 预测
                    if (estimator.has_track()) {
                        estimator.predict(token->timestamp_ns);
                    }

                    // 寻找最佳量测
                    std::optional<size_t> best_idx;
                    double min_distance = 1e10;

                    for (size_t i = 0; i < token->measurements.size(); ++i) {
                        const auto& meas = token->measurements[i];

                        // 坐标变换到世界系
                        auto world_result = fast_tf::lookup<fast_tf::world_fuxk_frame>(
                            tf_buffer,
                            fast_tf::FuckTransform<fast_tf::camera_optical_fuxk_frame>(
                                fast_tf::TransformMatrix4d::from_translation(
                                    meas.position[0], meas.position[1], meas.position[2])),
                            meas.timestamp_ns);

                        if (!world_result)
                            continue;

                        const auto world_xyz = (*world_result)->translation();

                        if (estimator.has_track()) {
                            const auto& state = estimator.state();
                            Eigen::Vector3d track_pos(state.pos_x(), state.pos_y(), state.pos_z());
                            double dist = (world_xyz - track_pos).norm();
                            if (dist < min_distance) {
                                min_distance = dist;
                                best_idx     = i;
                            }
                        } else {
                            if (meas.distance < min_distance) {
                                min_distance = meas.distance;
                                best_idx     = i;
                            }
                        }
                    }

                    // 更新或初始化航迹
                    if (best_idx) {
                        const auto& meas = token->measurements[*best_idx];

                        auto world_result = fast_tf::lookup<fast_tf::world_fuxk_frame>(
                            tf_buffer,
                            fast_tf::FuckTransform<fast_tf::camera_optical_fuxk_frame>(
                                fast_tf::TransformMatrix4d::from_translation(
                                    meas.position[0], meas.position[1], meas.position[2])),
                            meas.timestamp_ns);

                        if (world_result) {
                            const auto world_xyz          = (*world_result)->translation();
                            const auto [roll, pitch, yaw] = (*world_result)->euler_rot().rpy();

                            int armor_num = 4;
                            if (meas.name == ArmorName::Outpost || meas.name == ArmorName::Base) {
                                armor_num = 3;
                            } else if (meas.type == ArmorType::Big) {
                                armor_num = 2;
                            }

                            if (estimator.has_track()) {
                                estimator.update(meas, world_xyz, yaw);
                            } else {
                                estimator.init(meas, world_xyz, yaw, armor_num);
                            }
                        }
                    }

                    // 输出
                    if (estimator.has_track() && estimator.converged()) {
                        token->has_track = true;
                        token->track     = estimator.state();
                        token->command   = fire_control.solve(
                            token->track, token->bullet_speed,
                            token->yaw, // gimbal_yaw
                            token->timestamp_ns);
                    }

                    return token;
                })
            &

            // ====================================================================
            // Stage 5: 输出 (串行) - 发送控制指令
            // ====================================================================
            tbb::make_filter<std::shared_ptr<PipelineToken>, void>(
                tbb::filter_mode::serial_in_order, [&](std::shared_ptr<PipelineToken> token) {
                    if (!token)
                        return;

                    if (token->has_track && token->command.valid) {
                        output_interface->send(token->command);

                        spdlog::debug(
                            "[Pipeline] frame {} output: yaw={:.3f} pitch={:.3f} fire={}",
                            token->seq, token->command.yaw, token->command.pitch,
                            token->command.fire);
                    }

                    // 可在此处发布 Foxglove 可视化数据
                    // foxglove_server.publish(...);
                }));

    spdlog::info("pipeline stopped");
    return 0;
}