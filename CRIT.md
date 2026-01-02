# CRIT.md - at_vision → talos_vision 迁移规范

## 概述

本文档定义从 ROS2 at_vision 框架迁移到 talos-cpp ECS 架构的标准化流程。

**推荐使用的 Skills:**

- `/scheduler` - 生成 Channel Tags、数据类型、System 定义
- `/fast-tf` - 生成坐标系框架、类型安全坐标包装器
- `/ros2-to-talos` - ROS2 代码转换辅助

---

## 1. 架构映射

### 1.1 层级对应

| ROS2 at_vision         | talos-cpp fcs                        | 说明         |
| ---------------------- | ------------------------------------ | ------------ |
| camera node            | `L1_sensor/input_interface`          | 相机输入抽象 |
| serial node            | `L1_sensor/output_interface`         | 云台输出抽象 |
| armor_detector         | `L2_perception/at_legacy*`           | 检测 + PnP   |
| armor_solver (tracker) | `L3_estimation/armor_solver/tracker` | 状态估计     |
| rm_gimbal (solver)     | `L3_estimation/armor_solver/solver`  | 弹道补偿     |
| rm_interfaces          | `fcs/core/types.hpp`                 | 数据类型     |

### 1.2 通信模式

| ROS2                    | talos-cpp                      | 转换规则     |
| ----------------------- | ------------------------------ | ------------ |
| `ros2::Publisher<T>`    | `spmc_mut<T, Tag>`             | 多订阅者广播 |
| `ros2::Subscriber<T>`   | `spmc<T, Tag>`                 | 数据消费     |
| `message_filters::sync` | 系统参数依赖                   | 自动同步     |
| `tf2::Buffer`           | `fast_tf::CoordinateSystem<N>` | 坐标系查询   |
| `ros2::Parameter`       | `TalosConfig` TOML             | 运行时配置   |

---

## 2. 迁移清单

### 2.1 数据类型迁移

**ROS2 Message → C++ Struct:**

```cpp
// ROS2: rm_interfaces/msg/Armor.msg
// string number
// string type
// geometry_msgs/Pose pose

// talos-cpp: include/fcs/core/types.hpp
struct AtLegacyArmorDetection {
    ArmorName name;                    // string → enum
    ArmorType type;                    // string → enum
    ArmorColor color;                  // 新增: 颜色信息
    std::array<cv::Point2f, 4> corners;
    float confidence;

    [[nodiscard]] cv::Point2f center() const noexcept;
};
```

**必须包含的字段:**

- `timestamp_ns_t timestamp_ns` - 纳秒时间戳
- `uint64_t frame_id` (可选) - 帧序号

**使用 `/scheduler` skill 生成数据类型模板**

### 2.2 Channel Tag 定义

**位置:** `include/fcs/core/channel_tags.hpp`

```cpp
// 命名规则: ${Module}${DataType}ChannelTag
struct AtLegacyImageChannelTag {};      // 图像数据
struct AtLegacyDetectionChannelTag {};  // 检测结果
struct AtLegacyMeasurementChannelTag {}; // PnP测量值
struct TrackerOutputChannelTag {};       // 跟踪输出
struct SolverOutputChannelTag {};        // 解算指令
```

**使用 `/scheduler` skill 生成 Channel Tag**

### 2.3 System 转换

**ROS2 Node → ECS System:**

| ROS2 模式              | talos 执行策略         | 适用场景         |
| ---------------------- | ---------------------- | ---------------- |
| `TimerCallback(1/Hz)`  | `exclusive<Hz>`        | 定周期采样       |
| `TimerCallback` (高频) | `exclusive_silent<Hz>` | IMU 等高频传感器 |
| `SubscriptionCallback` | `pool_compute`         | 数据驱动处理     |

**转换示例:**

```cpp
// ROS2:
void ArmorDetectorNode::imageCallback(const Image::SharedPtr msg) {
    auto armors = detector_->detect(msg);
    armors_pub_->publish(armors);
}

// talos-cpp:
app.add_system<talos::pool_compute>(
    "armor_detector",
    [](talos::spmc<AtLegacyImageFrame, AtLegacyImageChannelTag> img_in,
       talos::spmc_mut<AtLegacyArmorDetectionBatch, AtLegacyDetectionChannelTag> det_out,
       talos::res<std::shared_ptr<AtLegacyDetectorBackend>> detector) {
        if (!img_in.has_new()) return;
        auto frame = img_in.read();
        if (!frame) return;

        auto detections = (*detector)->detect(frame->image);
        det_out.write(AtLegacyArmorDetectionBatch{
            .timestamp_ns = frame->timestamp_ns,
            .detections = std::move(detections)
        });
    });
```

### 2.4 坐标系迁移

**TF2 → fast_tf:**

```cpp
// ROS2: tf2 lookup
geometry_msgs::TransformStamped t =
    tf_buffer_->lookupTransform("odom", "camera_optical_frame", stamp);

// talos-cpp: fast_tf lookup
auto result = fast_tf::lookup<odom_fuxk_frame, camera_optical_fuxk_frame>(
    *tf_buffer,
    FuckTransform<camera_optical_fuxk_frame>{armor_pose},
    timestamp_ns
);
```

**Frame 定义 (使用 `/fast-tf` skill):**

```cpp
// include/math/fast_tf/frame.hpp
DECL_ROOT(world);
DECL(odom, world);
DECL(gimbal_link, odom);
DECL(camera_link, gimbal_link);
DECL(camera_optical, camera_link);  // 相机光学坐标系
DECL(muzzle_link, gimbal_link);     // 枪口坐标系
```

### 2.5 配置迁移

**ROS2 Parameter → TOML Config:**

```yaml
# ROS2: config.yaml
/armor_detector:
  ros__parameters:
    binary_thres: 50
    detect_color: "RED"
    classifier_threshold: 0.5
```

```toml
# talos-cpp: at_vision.toml
[at_legacy.detector]
binary_thres = 50
detect_color = 1  # 0=BLUE, 1=RED
classifier_threshold = 0.5
```

**Config 结构:**

```cpp
// include/fcs/L2_perception/at_legacy_config.hpp
struct AtLegacyDetectorConfig {
    int binary_thres{50};
    int detect_color{1};
    float classifier_threshold{0.5f};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};
```

---

## 3. 文件组织规范

### 3.1 目录结构

```
include/fcs/
├── L1_sensor/           # 硬件抽象层
│   ├── input_interface.hpp
│   └── output_interface.hpp
├── L2_perception/       # 感知层
│   ├── ${module}_config.hpp    # 配置定义
│   ├── ${module}_types.hpp     # 数据类型 (可合并到 core/types.hpp)
│   ├── ${module}_backend.hpp   # 后端接口
│   └── ${module}_systems.hpp   # ECS系统注册
├── L3_estimation/       # 估计层
│   └── ${module}/
│       ├── config.hpp
│       ├── types.hpp
│       ├── tracker.hpp
│       ├── solver.hpp
│       └── ...
├── core/
│   ├── channel_tags.hpp        # 所有 Channel Tag
│   └── types.hpp               # 共享数据类型
└── visualization/
    └── foxglove_*.hpp
```

### 3.2 命名约定

| 类型        | 命名规则                     | 示例                           |
| ----------- | ---------------------------- | ------------------------------ |
| Channel Tag | `${Module}${Data}ChannelTag` | `AtLegacyImageChannelTag`      |
| 数据结构    | `${Module}${Data}`           | `AtLegacyArmorDetection`       |
| Batch 类型  | `${Module}${Data}Batch`      | `AtLegacyArmorDetectionBatch`  |
| 系统名      | `snake_case`                 | `armor_detector`, `pnp_solver` |
| 配置类      | `${Module}Config`            | `AtLegacyDetectorConfig`       |

---

## 4. 迁移步骤

### Step 1: 分析 ROS2 代码

1. 识别 Node 的输入/输出 Topic
2. 列出所有 Message 类型
3. 分析 TF2 使用情况
4. 记录 Parameter 定义

### Step 2: 定义数据类型

使用 `/scheduler` skill:

```
/scheduler
选择: Data Type
输入字段定义
```

### Step 3: 创建 Channel Tags

使用 `/scheduler` skill:

```
/scheduler
选择: Channel Tag
输入 Tag 名称和描述
```

### Step 4: 实现后端逻辑

1. 复制核心算法代码
2. 移除 ROS2 依赖
3. 替换 `RCLCPP_*` → `spdlog::*`
4. 使用 `std::expected` 替代异常

### Step 5: 注册 ECS System

使用 `/scheduler` skill:

```
/scheduler
选择: System
输入执行策略、输入输出 Channel
```

### Step 6: 坐标系集成

使用 `/fast-tf` skill:

```
/fast-tf
选择: Frame 或 Lookup 模式
```

### Step 7: 配置集成

1. 添加 Config struct 到 `*_config.hpp`
2. 添加到 `TalosConfig` 聚合结构
3. 更新 `at_vision.toml`

---

## 5. 已迁移模块参考

### armor_detector (L2_perception/at_legacy)

| ROS2 文件                  | talos-cpp 文件                   |
| -------------------------- | -------------------------------- |
| `armor_detector.hpp/cpp`   | `at_legacy_detector_backend.hpp` |
| `number_classifier.hpp`    | `at_legacy_nn_backend.hpp`       |
| `armor_pose_estimator.hpp` | `at_legacy_pnp.hpp`              |
| `types.hpp`                | `fcs/core/types.hpp`             |
| `armor_detector_node.hpp`  | `at_legacy_systems.hpp`          |

### armor_solver (L3_estimation/armor_solver)

| ROS2 文件                    | talos-cpp 文件                            |
| ---------------------------- | ----------------------------------------- |
| `tracker.hpp/cpp`            | `armor_solver/tracker.hpp`                |
| `motion_model.hpp`           | `armor_solver/motion_model.hpp`           |
| `isrckf.hpp`                 | `armor_solver/isrckf.hpp`                 |
| `data_associator.hpp`        | `armor_solver/data_associator.hpp`        |
| `armor_solver.hpp`           | `armor_solver/solver.hpp`                 |
| `trajectory_compensator.hpp` | `armor_solver/trajectory_compensator.hpp` |

---

## 6. 常见问题

### Q1: 如何处理 ROS2 message_filters 同步?

talos 调度器自动处理。`pool_compute` 系统按依赖顺序执行，无需显式同步。

### Q2: 如何替代 rclcpp::Parameter 动态参数?

使用 `AtomicConfig<T>` 包装器:

```cpp
world.insert_resource(std::make_shared<AtomicTalosConfig>(config));
```

### Q3: 如何处理 tf2 时间插值?

`fast_tf::lookup` 支持时间戳插值:

```cpp
auto result = fast_tf::lookup<To, From>(system, transform, timestamp_ns);
```

### Q4: 如何调试 Channel 数据流?

添加 `*_logger` 系统:

```cpp
app.add_system<pool_compute>("debug_logger",
    [](spmc<T, Tag> in) {
        if (in.has_new()) {
            auto data = in.read();
            spdlog::debug("received: {}", ...);
        }
    });
```

---

## 7. 下一步待迁移模块

1. **rm_gimbal/solver** - 云台控制解算
2. **buff_detector** - 能量机关检测
3. **rune_solver** - 能量机关解算
4. **sentry_behavior** - 哨兵决策

---

## 附录: Skill 使用示例

### 创建新 Channel

```
用户: /scheduler
Claude: 选择组件类型?
用户: Channel Tag
Claude: Tag 名称?
用户: BuffDetectionChannelTag
Claude: 描述?
用户: 能量机关检测结果
[生成代码]
```

### 创建坐标系 Frame

```
用户: /fast-tf
Claude: 选择组件类型?
用户: Frame
Claude: Frame 名称?
用户: buff_center
Claude: 父 Frame?
用户: camera_optical
[生成代码]
```

### ROS2 代码转换

```
用户: /ros2-to-talos
Claude: 粘贴 ROS2 代码或提供文件路径
用户: [粘贴 ROS2 Node 代码]
[分析并生成 talos System 代码]
```
