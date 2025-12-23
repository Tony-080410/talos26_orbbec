#include "strategy_wrapper.hpp"
#include <spdlog/spdlog.h>

// FoxgloveServer 的构造函数实现
FoxgloveServer::FoxgloveServer(const uint16_t port, std::string host) noexcept {
    foxglove::setLogLevel(foxglove::LogLevel::Info);

    foxglove::WebSocketServerOptions ws;
    ws.host = host;
    ws.port = port;

    auto server_r = foxglove::WebSocketServer::create(std::move(ws));
    if (!server_r.has_value()) {
        spdlog::error("WebSocketServer create failed: {}", foxglove::strerror(server_r.error()));
        return;
    }
    server_.emplace(std::move(server_r.value()));
    spdlog::info("Foxglove server listening on ws://{}:{}", host, server_->port());

    auto tf_r = foxglove::schemas::FrameTransformsChannel::create("/tf");
    if (!tf_r.has_value()) {
        spdlog::error("FrameTransformsChannel create failed: {}", foxglove::strerror(tf_r.error()));
        return;
    }
    tf_ch_.emplace(std::move(tf_r.value()));

    auto scene_r = foxglove::schemas::SceneUpdateChannel::create("/scene");
    if (!scene_r.has_value()) {
        spdlog::error("SceneUpdateChannel create failed: {}", foxglove::strerror(scene_r.error()));
        return;
    }
    scene_ch_.emplace(std::move(scene_r.value()));

    auto img_r = foxglove::schemas::CompressedImageChannel::create("/image");
    if (!img_r.has_value()) {
        spdlog::error(
            "CompressedImageChannel create failed: {}", foxglove::strerror(img_r.error()));
        return;
    }
    img_ch_.emplace(std::move(img_r.value()));

    auto armor_r = foxglove::RawChannel::create("/armor_data", "json");
    if (!armor_r.has_value()) {
        spdlog::error("RawChannel create failed: {}", foxglove::strerror(armor_r.error()));
        return;
    }
    armor_ch_.emplace(std::move(armor_r.value()));

    // 新增: 调试图像 channels
    auto binary_img_r = foxglove::schemas::CompressedImageChannel::create("/debug/binary_img");
    if (!binary_img_r.has_value()) {
        spdlog::error("binary_img channel create failed: {}", foxglove::strerror(binary_img_r.error()));
        return;
    }
    binary_img_ch_.emplace(std::move(binary_img_r.value()));

    auto pattern_img_r = foxglove::schemas::CompressedImageChannel::create("/debug/number_img");
    if (!pattern_img_r.has_value()) {
        spdlog::error("pattern_img channel create failed: {}", foxglove::strerror(pattern_img_r.error()));
        return;
    }
    pattern_img_ch_.emplace(std::move(pattern_img_r.value()));

    // 新增: 调试数据 channels (JSON)
    auto debug_lights_r = foxglove::RawChannel::create("/debug/lights", "json");
    if (!debug_lights_r.has_value()) {
        spdlog::error("debug_lights channel create failed: {}", foxglove::strerror(debug_lights_r.error()));
        return;
    }
    debug_lights_ch_.emplace(std::move(debug_lights_r.value()));

    auto debug_armors_r = foxglove::RawChannel::create("/debug/armors", "json");
    if (!debug_armors_r.has_value()) {
        spdlog::error("debug_armors channel create failed: {}", foxglove::strerror(debug_armors_r.error()));
        return;
    }
    debug_armors_ch_.emplace(std::move(debug_armors_r.value()));

    // 新增: 解算/追踪 channels (JSON)
    auto measurement_r = foxglove::RawChannel::create("/solver/measurement", "json");
    if (!measurement_r.has_value()) {
        spdlog::error("measurement channel create failed: {}", foxglove::strerror(measurement_r.error()));
        return;
    }
    measurement_ch_.emplace(std::move(measurement_r.value()));

    auto target_r = foxglove::RawChannel::create("/solver/target", "json");
    if (!target_r.has_value()) {
        spdlog::error("target channel create failed: {}", foxglove::strerror(target_r.error()));
        return;
    }
    target_ch_.emplace(std::move(target_r.value()));

    auto cmd_gimbal_r = foxglove::RawChannel::create("/solver/cmd_gimbal", "json");
    if (!cmd_gimbal_r.has_value()) {
        spdlog::error("cmd_gimbal channel create failed: {}", foxglove::strerror(cmd_gimbal_r.error()));
        return;
    }
    cmd_gimbal_ch_.emplace(std::move(cmd_gimbal_r.value()));

    // 创建 CameraCalibration channel
    auto camera_calib_r = foxglove::schemas::CameraCalibrationChannel::create("/camera_info");
    if (!camera_calib_r.has_value()) {
        spdlog::error("camera_calib channel create failed: {}", foxglove::strerror(camera_calib_r.error()));
        return;
    }
    camera_calib_ch_.emplace(std::move(camera_calib_r.value()));

    // 创建 TF Manager (将 tf_ch_ 移动给它)
    tf_manager_ = std::make_unique<fast_tf::infantry::InfantryTFManager>(std::move(tf_ch_));
    tf_ch_.reset();  // 清空，因为已经移动给 tf_manager_

    // 发布初始 TF (建立完整的 TF 树)
    uint64_t init_ts = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    publish_tf(0.0, 0.0, 0.0, init_ts);

    server_initialized_ = true;

    // 启动消息发送线程
    sender_thread_ = std::thread(&FoxgloveServer::message_sender_thread, this);
}

FoxgloveServer::~FoxgloveServer() noexcept {
    // 关闭发送线程
    { should_stop_ = true; }
    if (sender_thread_.joinable()) {
        sender_thread_.join();
    }
}

void FoxgloveServer::message_sender_thread() {
    while (true) {
        if (should_stop_) [[unlikely]] {
            break;
        }
        FoxgloveMessage msg;
        if (const auto success = message_queue_.try_pop(msg); !success) {
            continue;
        }

        // 处理消息 - 在锁外执行以避免阻塞
        std::visit(
            [this](auto& m) {
                if constexpr (std::is_same_v<decltype(m), ImageMessage&>) {
                    if (img_ch_) {
                        img_ch_->log(m.img);
                    }
                } else if constexpr (std::is_same_v<decltype(m), SceneMessage&>) {
                    if (scene_ch_) {
                        scene_ch_->log(m.scene);
                    }
                } else if constexpr (std::is_same_v<decltype(m), TfMessage&>) {
                    if (tf_ch_) {
                        tf_ch_->log(m.tf);
                    }
                } else if constexpr (std::is_same_v<decltype(m), ArmorDataMessage&>) {
                    if (armor_ch_) {
                        armor_ch_->log(
                            reinterpret_cast<const std::byte*>(m.data.data()), m.data.size());
                    }
                } else if constexpr (std::is_same_v<decltype(m), BinaryImageMessage&>) {
                    if (binary_img_ch_) {
                        binary_img_ch_->log(m.img);
                    }
                } else if constexpr (std::is_same_v<decltype(m), PatternImageMessage&>) {
                    if (pattern_img_ch_) {
                        pattern_img_ch_->log(m.img);
                    }
                } else if constexpr (std::is_same_v<decltype(m), DebugLightsMessage&>) {
                    if (debug_lights_ch_) {
                        debug_lights_ch_->log(
                            reinterpret_cast<const std::byte*>(m.data.data()), m.data.size());
                    }
                } else if constexpr (std::is_same_v<decltype(m), DebugArmorsMessage&>) {
                    if (debug_armors_ch_) {
                        debug_armors_ch_->log(
                            reinterpret_cast<const std::byte*>(m.data.data()), m.data.size());
                    }
                } else if constexpr (std::is_same_v<decltype(m), MeasurementMessage&>) {
                    if (measurement_ch_) {
                        measurement_ch_->log(
                            reinterpret_cast<const std::byte*>(m.data.data()), m.data.size());
                    }
                } else if constexpr (std::is_same_v<decltype(m), TargetMessage&>) {
                    if (target_ch_) {
                        target_ch_->log(
                            reinterpret_cast<const std::byte*>(m.data.data()), m.data.size());
                    }
                } else if constexpr (std::is_same_v<decltype(m), GimbalCmdMessage&>) {
                    if (cmd_gimbal_ch_) {
                        cmd_gimbal_ch_->log(
                            reinterpret_cast<const std::byte*>(m.data.data()), m.data.size());
                    }
                }
            },
            msg);
    }
    spdlog::info("Foxglove message sender thread stopped");
}

bool FoxgloveServer::is_initialized() const noexcept { return server_initialized_; }

void FoxgloveServer::enqueue_message(FoxgloveMessage msg) noexcept {
    message_queue_.push(std::move(msg));
}

void FoxgloveServer::publish_tf(double gimbal_roll, double gimbal_pitch, double gimbal_yaw, uint64_t timestamp_ns) noexcept {
    if (!tf_manager_) {
        return;
    }

    auto& publisher = tf_manager_->get_publisher();

    // 1. world -> odom (identity)
    fast_tf::TransformMatrix4d odom_tf = fast_tf::TransformMatrix4d::from_rpy(
        0.0, 0.0, 0.0,  // roll, pitch, yaw
        0.0, 0.0, 0.0   // x, y, z
    );
    publisher.publish_transform<fast_tf::odom_Frame, fast_tf::world_Frame>(
        odom_tf, timestamp_ns);

    // 2. odom -> gimbal_link (云台姿态 - 使用实际 IMU 读数)
    // 注意: solver 使用 rx * ry * rz 顺序 (即 ZYX: yaw先, pitch次, roll最后)
    // 需要手动构建四元数来匹配这个顺序
    Eigen::AngleAxisd rx(0.0, Eigen::Vector3d::UnitX());  // roll = 0
    Eigen::AngleAxisd ry(gimbal_pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd rz(gimbal_yaw, Eigen::Vector3d::UnitZ());
    Eigen::Quaterniond q_gimbal = rx * ry * rz;  // 匹配 solver 的顺序

    fast_tf::TransformMatrix4d gimbal_tf = fast_tf::TransformMatrix4d::from_quaternion(
        q_gimbal.w(), q_gimbal.x(), q_gimbal.y(), q_gimbal.z(),
        0.0, 0.0, 0.0  // x, y, z translation
    );
    publisher.publish_transform<fast_tf::gimbal_link_Frame, fast_tf::odom_Frame>(
        gimbal_tf, timestamp_ns);

    // 3. gimbal_link -> camera_link (静态: x=0.04444)
    fast_tf::TransformMatrix4d camera_tf = fast_tf::TransformMatrix4d::from_rpy(
        0.0, 0.0, 0.0,      // roll, pitch, yaw
        0.04444, 0.0, 0.0   // x, y, z
    );
    publisher.publish_transform<fast_tf::camera_link_Frame, fast_tf::gimbal_link_Frame>(
        camera_tf, timestamp_ns);

    // 4. camera_link -> camera_optical_frame (静态: ROS 相机光学坐标系)
    const double pi_2 = M_PI / 2.0;
    fast_tf::TransformMatrix4d optical_tf = fast_tf::TransformMatrix4d::from_rpy(
        -pi_2, 0.0, -pi_2,  // roll, pitch, yaw
        0.0, 0.0, 0.0       // x, y, z
    );
    publisher.publish_transform<fast_tf::camera_optical_frame_Frame, fast_tf::camera_link_Frame>(
        optical_tf, timestamp_ns);
}

void FoxgloveServer::publish_camera_calibration(
    uint32_t width, uint32_t height,
    const std::array<double, 9>& camera_matrix,
    const std::vector<double>& distortion,
    uint64_t timestamp_ns) noexcept {
    if (!camera_calib_ch_) {
        return;
    }

    foxglove::schemas::CameraCalibration calib;

    // 时间戳 (optional)
    foxglove::schemas::Timestamp ts;
    ts.sec = static_cast<uint32_t>(timestamp_ns / 1'000'000'000);
    ts.nsec = static_cast<uint32_t>(timestamp_ns % 1'000'000'000);
    calib.timestamp = ts;

    // 帧ID
    calib.frame_id = "camera_optical_frame";

    // 图像尺寸
    calib.width = width;
    calib.height = height;

    // 畸变模型 (使用 plumb_bob 5参数模型)
    calib.distortion_model = "plumb_bob";

    // 畸变系数 d [k1, k2, p1, p2, k3]
    calib.d.reserve(distortion.size());
    for (const auto& coeff : distortion) {
        calib.d.push_back(coeff);
    }

    // 内参矩阵 k (row-major 3x3)
    // camera_matrix: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    calib.k = camera_matrix;

    // 矫正矩阵 r (单位矩阵，单目相机)
    calib.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

    // 投影矩阵 p (3x4, 单目相机: [fx 0 cx 0; 0 fy cy 0; 0 0 1 0])
    calib.p = {
        camera_matrix[0], 0.0, camera_matrix[2], 0.0,  // fx, 0, cx, 0
        0.0, camera_matrix[4], camera_matrix[5], 0.0,  // 0, fy, cy, 0
        0.0, 0.0, 1.0, 0.0                              // 0, 0, 1, 0
    };

    camera_calib_ch_->log(calib);
}
