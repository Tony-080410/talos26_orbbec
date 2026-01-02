# ROS2 → Talos 转换示例：相机+检测

## 原始 ROS2 代码

```cpp
// detection_node.cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

class DetectionNode : public rclcpp::Node {
public:
    DetectionNode() : Node("detection_node") {
        // 参数
        declare_parameter("detection_threshold", 0.5);
        threshold_ = get_parameter("detection_threshold").as_double();

        // TF2
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // 订阅
        image_sub_ = create_subscription<sensor_msgs::msg::Image>(
            "camera/image_raw", 10,
            std::bind(&DetectionNode::image_callback, this, std::placeholders::_1));

        imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
            "imu/data", 10,
            std::bind(&DetectionNode::imu_callback, this, std::placeholders::_1));

        // 发布
        pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("target/pose", 10);

        // Timer
        timer_ = create_wall_timer(
            std::chrono::milliseconds(6),  // ~160Hz
            std::bind(&DetectionNode::timer_callback, this));
    }

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        latest_image_ = msg;
    }

    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        latest_imu_ = msg;
    }

    void timer_callback() {
        if (!latest_image_) return;

        // 获取 TF
        geometry_msgs::msg::TransformStamped tf;
        try {
            tf = tf_buffer_->lookupTransform("world", "camera_link", tf2::TimePointZero);
        } catch (tf2::TransformException& ex) {
            RCLCPP_WARN(get_logger(), "TF error: %s", ex.what());
            return;
        }

        // 检测
        auto detections = detect(latest_image_, threshold_);

        // 发布结果
        for (const auto& det : detections) {
            geometry_msgs::msg::PoseStamped pose;
            pose.header.stamp = now();
            pose.header.frame_id = "world";
            pose.pose = transform_to_world(det, tf);
            pose_pub_->publish(pose);
        }
    }

    // 成员变量
    double threshold_;
    sensor_msgs::msg::Image::SharedPtr latest_image_;
    sensor_msgs::msg::Imu::SharedPtr latest_imu_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DetectionNode>());
    rclcpp::shutdown();
    return 0;
}
```

---

## 转换后的 Talos 代码

```cpp
// detection_main.cpp
#include "scheduler/scheduler.hpp"
#include "math/fast_tf/infantry.hpp"
#include <spdlog/spdlog.h>

// ============================================================================
// 1. 消息类型定义（替代 ROS msg）
// ============================================================================

struct Frame {
    uint64_t seq;
    timestamp_ns_t timestamp_ns;
    cv::Mat image;
};

struct ImuSnapshot {
    timestamp_ns_t timestamp;
    double roll_deg, pitch_deg, yaw_deg;
    Eigen::Quaterniond orientation;
};

struct Detection {
    timestamp_ns_t timestamp;
    Eigen::Vector3d position_camera;  // 相机坐标系
    double confidence;
};

struct TargetPose {
    timestamp_ns_t timestamp;
    Eigen::Vector3d position_world;   // 世界坐标系
};

// ============================================================================
// 2. Channel Tags（替代 topic 名称）
// ============================================================================

struct ImageChannel {};
struct DetectionChannel {};
struct TargetChannel {};

// ============================================================================
// 3. 配置（替代 ROS Parameter）
// ============================================================================

struct DetectionConfig {
    double threshold = 0.5;
    // 从 TOML 加载...
};

// ============================================================================
// 4. Main
// ============================================================================

int main() {
    init_logger();

    // 初始化硬件
    auto camera = Camera::create();
    auto imu_device = Imu::create();

    // === 创建调度器 ===
    talos::World world;
    talos::Scheduler scheduler(world);

    // === 插入资源 ===
    world.insert_resource(DetectionConfig{.threshold = 0.5});
    world.insert_resource(ImuSnapshot{});

    // TF: 编译期类型安全的坐标系统
    fast_tf::CoordinateSystem<100> coord_system;

    // =========================================================================
    // Systems
    // =========================================================================

    // 1. IMU Reader: 1000Hz 静默更新（不触发下游）
    //    替代: imu_sub_ + imu_callback
    scheduler.add_system<talos::exclusive_silent<1000>>(
        "imu_reader",
        [&imu_device](talos::res_mut<ImuSnapshot> imu_out) {
            auto data = imu_device.read();
            imu_out->timestamp   = data.timestamp_ns;
            imu_out->roll_deg    = data.roll;
            imu_out->pitch_deg   = data.pitch;
            imu_out->yaw_deg     = data.yaw;
            imu_out->orientation = data.quaternion;
        });

    // 2. Camera Reader: 160Hz 触发处理链
    //    替代: timer_callback 中的图像获取部分
    scheduler.add_system<talos::exclusive<160>>(
        "camera_reader",
        [&camera](talos::spmc_mut<Frame, ImageChannel> cam_out) {
            auto frame = camera.try_recv();
            if (!frame) return;
            cam_out.write(Frame{
                .seq = frame->seq,
                .timestamp_ns = frame->timestamp_ns,
                .image = frame->image
            });
        });

    // 3. Detector: 被 camera_reader 触发
    //    替代: timer_callback 中的检测部分
    scheduler.add_system(
        "detector",
        [](talos::spmc<Frame, ImageChannel> img_in,
           talos::res<DetectionConfig> config,
           talos::spmc_mut<Detection, DetectionChannel> det_out) {

            if (!img_in.has_new()) return;
            auto frame = img_in.read();
            if (!frame) return;

            // 检测
            auto detections = detect(frame->image, config->threshold);

            for (const auto& det : detections) {
                det_out.write(Detection{
                    .timestamp = frame->timestamp_ns,
                    .position_camera = det.position,
                    .confidence = det.confidence
                });
            }
        });

    // 4. Pose Solver: 坐标变换
    //    替代: timer_callback 中的 TF 查询 + 发布
    scheduler.add_system(
        "pose_solver",
        [&coord_system](
            talos::spmc<Detection, DetectionChannel> det_in,
            talos::res<ImuSnapshot> imu,
            talos::spmc_mut<TargetPose, TargetChannel> pose_out) {

            if (!det_in.has_new()) return;
            auto det = det_in.read();
            if (!det) return;

            // fast_tf: 编译期类型安全的坐标变换
            // 替代: tf_buffer_->lookupTransform("world", "camera_link", ...)
            auto source_tf = fast_tf::FuckTransform<camera_link_fuxk_frame>{
                fast_tf::TransformMatrix4d::from_translation(det->position_camera)
            };

            auto result = fast_tf::lookup<world_fuxk_frame, camera_link_fuxk_frame>(
                coord_system, source_tf, det->timestamp);

            if (!result) {
                spdlog::warn("TF lookup failed: {}", to_string(result.error()));
                return;
            }

            pose_out.write(TargetPose{
                .timestamp = det->timestamp,
                .position_world = result->transform.translation()
            });
        });

    // 5. Output: 输出结果
    //    替代: pose_pub_->publish()
    scheduler.add_system(
        "output",
        [](talos::spmc<TargetPose, TargetChannel> pose_in) {
            if (!pose_in.has_new()) return;
            auto pose = pose_in.read();
            if (!pose) return;

            spdlog::info("Target at ({:.2f}, {:.2f}, {:.2f})",
                pose->position_world.x(),
                pose->position_world.y(),
                pose->position_world.z());

            // 发送到控制器...
        });

    // =========================================================================
    // 构建运行
    // =========================================================================

    scheduler.print_systems();

    if (auto result = scheduler.build(); !result) {
        spdlog::critical("build: {}", talos::format_build_error(result.error()));
        return 1;
    }

    if (auto result = scheduler.run(); !result) {
        spdlog::critical("run: {}", talos::to_string(result.error()));
        return 1;
    }

    return 0;
}
```

---

## 转换对照表

| ROS2 原代码 | Talos 转换后 |
|-------------|--------------|
| `class DetectionNode : public Node` | `main()` + World + Scheduler |
| `image_sub_ = create_subscription<Image>(...)` | `scheduler.add_system("detector", [](spmc<Frame, ImageChannel> img_in) {...})` |
| `imu_sub_ = create_subscription<Imu>(...)` | `scheduler.add_system<exclusive_silent<1000>>("imu_reader", [](res_mut<ImuSnapshot>) {...})` |
| `timer_ = create_wall_timer(6ms, ...)` | `scheduler.add_system<exclusive<160>>("camera_reader", ...)` |
| `pose_pub_->publish(pose)` | `spmc_mut<TargetPose, TargetChannel> pose_out; pose_out.write(...)` |
| `tf_buffer_->lookupTransform(...)` | `fast_tf::lookup<world_fuxk_frame, camera_link_fuxk_frame>(...)` |
| `declare_parameter(...)` | `world.insert_resource(Config{...})` |
| `latest_image_` 成员变量 | Channel 自动缓存，`img_in.read()` 获取 |

---

## 数据流对比

```
ROS2:
  imu_callback ─────────────────────┐
                                    ▼
  timer_callback ──► detect ──► lookupTransform ──► publish
       ▲
  image_callback

Talos:
  imu_reader ──────► res<ImuSnapshot> ───────────────┐
  (exclusive_silent<1000>)                           │
                                                     ▼
  camera_reader ──► ImageChannel ──► detector ──► DetectionChannel ──► pose_solver ──► TargetChannel ──► output
  (exclusive<160>)                   (pool)                             (pool)                           (pool)
```
