---
name: ros2-to-talos
description: |
  将 ROS2 node 代码转换为 talos ECS 框架。Convert ROS2 rclcpp code to talos.
  在迁移 ROS2 代码、转换 subscriber/publisher/timer/callback 到 talos 的
  spmc/spsc/exclusive 模式时使用。tf2 转换为 fast_tf。
  关键词：ros2, talos, rclcpp, node, tf2, fast_tf, 转换, 迁移, 框架
---

# ROS2 → Talos 转换指南

## 核心映射表

| ROS2 (rclcpp)              | Talos                              |
|----------------------------|------------------------------------|
| `rclcpp::Node`             | `talos::App` + `talos::Scheduler`  |
| `rclcpp::Subscription<T>`  | `spmc<T, Tag>` (reader)            |
| `rclcpp::Publisher<T>`     | `spmc_mut<T, Tag>` (writer)        |
| `create_wall_timer(10Hz)`  | `exclusive<10>` policy             |
| `rclcpp::Parameter`        | `res<T>` / `res_mut<T>`            |
| callback function          | system function                    |
| `tf2_ros::Buffer`          | `fast_tf::CoordinateSystem<N>`     |
| `tf2::lookupTransform()`   | `fast_tf::lookup<To, From>()`      |

## 快速转换

### Subscriber → spmc reader

```cpp
// ROS2:
sub_ = create_subscription<Msg>("topic", 10, callback);

// Talos:
struct TopicChannel {};
scheduler.add_system("processor", [](spmc<Msg, TopicChannel> in) {
    if (!in.has_new()) return;
    auto msg = in.read();
    // process...
});
```

### Publisher → spmc_mut writer

```cpp
// ROS2:
pub_->publish(msg);

// Talos:
scheduler.add_system("publisher", [](spmc_mut<Msg, Channel> out) {
    out.write(std::move(msg));
});
```

### Timer → exclusive policy

```cpp
// ROS2:
timer_ = create_wall_timer(100ms, callback);

// Talos: exclusive<FreqHz> 表示独占线程 + 触发调度
scheduler.add_system<exclusive<10>>("timer", [](auto out) { ... });

// 高频静默更新用 exclusive_silent
scheduler.add_system<exclusive_silent<1000>>("imu", ...);
```

### tf2 → fast_tf

```cpp
// ROS2:
auto tf = buffer->lookupTransform("world", "camera", tf2::TimePointZero);

// Talos (编译期类型安全):
auto result = fast_tf::lookup<world_fuxk_frame, camera_link_fuxk_frame>(
    coordinate_system, source_tf, timestamp_ns);
```

## 详细参考

- 完整转换模式见 [PATTERNS.md](PATTERNS.md)
- 实际转换示例见 [EXAMPLES.md](EXAMPLES.md)
