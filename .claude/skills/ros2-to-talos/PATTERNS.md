# ROS2 → Talos 转换模式详解

参考 `src/main.cpp` 的实际用法。

## 1. 基本结构

```cpp
// ============== ROS2 ==============
class MyNode : public rclcpp::Node {
    MyNode() : Node("my_node") { ... }
};
int main() {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MyNode>());
}

// ============== Talos ==============
int main() {
    // 1. 创建 World 和 Scheduler
    talos::World world;
    talos::Scheduler scheduler(world);

    // 2. 定义 Channel Tags
    struct ImageRawChannel {};

    // 3. 插入资源
    world.insert_resource(Config{});

    // 4. 添加 Systems
    scheduler.add_system<talos::exclusive<160>>("camera", ...);
    scheduler.add_system("processor", ...);

    // 5. 构建并运行
    scheduler.build();
    scheduler.run();
}
```

## 2. 消息类型

```cpp
// ROS2: sensor_msgs/msg/Imu.msg
// Talos: 普通 struct
struct ImuSnapshot {
    timestamp_ns_t timestamp;
    double roll_deg, pitch_deg, yaw_deg, bullet_speed;
};
```

## 3. Timer → exclusive policy

### 数据源 Timer（触发处理链）

```cpp
// ============== ROS2 ==============
timer_ = create_wall_timer(6ms, [this]() {  // ~160Hz
    auto frame = camera_.read();
    publisher_->publish(frame);
});

// ============== Talos ==============
// exclusive<Hz>: 独占线程 + 通知调度器触发下游 system
scheduler.add_system<talos::exclusive<160>>(
    "camera_reader",
    [&input](talos::spmc_mut<Frame, ImageRawChannel> cam_out) {
        const auto frame = input.try_recv();
        if (!frame) return;
        cam_out.write(Frame{frame->seq, frame->timestamp_ns, frame->image});
    });
```

### 高频静默 Timer（不触发下游）

```cpp
// ============== ROS2 ==============
timer_ = create_wall_timer(1ms, [this]() {  // 1000Hz
    imu_data_ = read_imu();
});

// ============== Talos ==============
// exclusive_silent<Hz>: 独占线程，不通知调度器
world.insert_resource(ImuSnapshot{});

scheduler.add_system<talos::exclusive_silent<1000>>(
    "imu_reader",
    [&usb_device, &imu_data](talos::res_mut<ImuSnapshot> imu_out) {
        usb_device->handle_events();
        imu_out->timestamp  = imu_data.time_stamp;
        imu_out->roll_deg   = imu_data.data.roll;
        imu_out->pitch_deg  = imu_data.data.pitch;
        imu_out->yaw_deg    = imu_data.data.yaw;
    });
```

## 4. Subscription → spmc reader

```cpp
// ============== ROS2 ==============
sub_ = create_subscription<Image>("camera/image_raw", 10,
    [this](const Image::SharedPtr msg) {
        process_image(msg);
    });

// ============== Talos ==============
// pool_compute (默认): 由 exclusive system 触发执行
scheduler.add_system(
    "preprocessor",
    [](talos::spmc<fcs::L1::Frame, ImageRawChannel> cam_in) {
        if (!cam_in.has_new()) return;

        auto frame = cam_in.read();
        if (!frame) return;

        spdlog::info("camera in {}", frame->seq);
        // process frame...
    });
```

## 5. Publisher → spmc_mut writer

```cpp
// ============== ROS2 ==============
pub_ = create_publisher<Twist>("cmd_vel", 10);
pub_->publish(twist_msg);

// ============== Talos ==============
struct CmdVelChannel {};

scheduler.add_system(
    "controller",
    [](talos::spmc_mut<Twist, CmdVelChannel> cmd_out) {
        Twist twist{.linear_x = 1.0};
        cmd_out.write(std::move(twist));
    });
```

## 6. Lambda 捕获外部资源

```cpp
// Talos 支持在 lambda 中捕获外部设备/状态
std::atomic<uint64_t> last_seq{0};
auto usb_device = std::make_shared<Stm32>();

scheduler.add_system<talos::exclusive<160>>(
    "camera_reader",
    [daedalus, &ipc_device, &usb_device, &input, &last_seq]
    (talos::spmc_mut<Frame, ImageRawChannel> cam_out) {
        // 使用捕获的变量
        const auto frame = input.try_recv();
        last_seq = frame->seq;
        // ...
    });
```

## 7. Policy 选择

| Policy | 用途 | 示例 |
|--------|------|------|
| `exclusive<Hz>` | 数据源，触发处理链 | 相机 160Hz |
| `exclusive_silent<Hz>` | 高频更新，不触发 | IMU 1000Hz |
| `pool_compute` (默认) | 数据处理 | 图像预处理、推理 |

## 8. tf2 → fast_tf

```cpp
// ============== ROS2 ==============
auto tf = buffer.lookupTransform("world", "camera", tf2::TimePointZero);

// ============== fast_tf ==============
// 编译期类型安全，零堆分配
#include "math/fast_tf/infantry.hpp"

fast_tf::CoordinateSystem<100> coord_sys;

// 更新变换
fast_tf::update<camera_link_fuxk_frame>(coord_sys, tf, timestamp_ns);

// 查询变换
auto result = fast_tf::lookup<world_fuxk_frame, camera_link_fuxk_frame>(
    coord_sys, source_tf, timestamp_ns);
```

## 9. 完整模板

```cpp
#include "scheduler/scheduler.hpp"

int main() {
    // === 初始化设备 ===
    auto camera = Camera::create();
    auto imu = Imu::create();

    // === 创建调度器 ===
    talos::World world;
    talos::Scheduler scheduler(world);

    // === Channel Tags ===
    struct ImageChannel {};
    struct DetectionChannel {};

    // === 资源 ===
    struct ImuSnapshot { ... };
    world.insert_resource(ImuSnapshot{});

    // === Systems ===

    // 1. IMU: 1000Hz 静默更新
    scheduler.add_system<talos::exclusive_silent<1000>>(
        "imu_reader",
        [&imu](talos::res_mut<ImuSnapshot> out) {
            auto data = imu.read();
            out->timestamp = data.timestamp;
            out->orientation = data.orientation;
        });

    // 2. Camera: 160Hz 触发处理链
    scheduler.add_system<talos::exclusive<160>>(
        "camera_reader",
        [&camera](talos::spmc_mut<Frame, ImageChannel> out) {
            if (auto frame = camera.try_recv()) {
                out.write(std::move(*frame));
            }
        });

    // 3. Detector: 被 camera 触发
    scheduler.add_system(
        "detector",
        [](talos::spmc<Frame, ImageChannel> img_in,
           talos::res<ImuSnapshot> imu,  // 读取 IMU 快照
           talos::spmc_mut<Detection, DetectionChannel> det_out) {
            if (!img_in.has_new()) return;
            auto frame = img_in.read();
            if (!frame) return;

            auto det = detect(*frame, *imu);
            det_out.write(std::move(det));
        });

    // === 构建运行 ===
    if (auto r = scheduler.build(); !r) {
        spdlog::critical("build: {}", format_build_error(r.error()));
        return 1;
    }
    scheduler.run();
}
```
