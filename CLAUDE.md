# Talos C++ 项目代码指南

> **代码定位**: 基于自定义调度器的机器人视觉框架，采用 5 级流水线（FCS - Fire Control System）架构
>
> **核心设计**: 零 ROS 依赖、类型安全数据流、DAG 依赖分析杜绝数据竞争

---

## 代码目录结构

```
talos-cpp/
├── crates/                      # 核心库
│   ├── primitive/               # 原语库（通道、lazy、spin）
│   │   └── src/primitive/
│   │       ├── channel.hpp              # 统一通道抽象
│   │       ├── spsc_triple_buffer.hpp   # SPSC 三重缓冲
│   │       ├── spmc_triple_buffer.hpp   # SPMC 三重缓冲
│   │       ├── lazy.hpp                 # 延迟构造
│   │       ├── spin.hpp                 # 自旋等待原语
│   │       └── performance_probe.hpp    # 性能探针
│   ├── scheduler/               # 调度器
│   │   └── src/scheduler/
│   │       ├── world.hpp                # World: 资源和通道容器
│   │       ├── scheduler.hpp/.cpp       # Scheduler: 调度器核心
│   │       ├── system/
│   │       │   ├── components.hpp       # 通道类型定义
│   │       │   ├── system.hpp           # System: 系统执行模型
│   │       │   └── execution_policy.hpp # 执行策略
│   │       └── rclcompat/               # ROS2 兼容层
│   ├── fast_tf/                 # 类型安全坐标变换
│   ├── math/                    # 数学工具（SO2、Euler）
│   ├── toml/                    # TOML 配置解析
│   ├── log/                     # 日志工具
│   └── hardware/                # 硬件驱动
│       ├── at_gimbal/           # 云台控制
│       └── hik_camera_driver/   # HIK 相机驱动（x86_64）
├── src/
│   ├── fcs/                     # 火控系统（5 级流水线）
│   │   ├── core/                # 共享类型、通道 topics
│   │   │   └── trajectory/      # 弹道模型
│   │   ├── L1_sensor/           # 采集
│   │   ├── L2_perception/       # 检测
│   │   ├── L3_estimation/       # 追踪
│   │   ├── L4_planning/         # 规划
│   │   ├── L5_weapon/           # 控制
│   │   └── runtime/             # 系统注册函数
│   ├── fcs_visualization/       # Foxglove 可视化
│   └── main.cpp
├── config/                      # 配置文件
├── 3dparty/                     # 第三方库
└── cmake/                       # CMake 模块
```

---

## 库结构

| 库名              | 类型      | 说明                                 |
| ----------------- | --------- | ------------------------------------ |
| toml              | INTERFACE | TOML 配置解析辅助                    |
| log               | INTERFACE | 日志工具封装                         |
| math              | INTERFACE | 数学工具（SO2、Euler）               |
| primitive         | SHARED    | 原语库（通道、lazy、spin、性能探针） |
| scheduler         | SHARED    | 调度器                               |
| fast_tf           | SHARED    | 类型安全坐标变换                     |
| at_gimbal         | INTERFACE | 云台控制                             |
| hik_camera_driver | SHARED    | HIK 相机驱动（仅 Linux x86_64）      |
| hardware_daedalus | SHARED    | 硬件抽象层（共享内存）               |
| fcs               | SHARED    | 火控系统（使用 PCH 加速 50-70%）     |
| fcs_visualization | SHARED    | Foxglove 可视化                      |

---

## 调度器编程模型

### System 定义

```cpp
// System 通过通道 (Channel) 和资源 (Resource) 访问数据
struct MySystem {
  // 通道访问
  spsc<Input, TagInput> reader;       // SPSC 只读
  spmc_mut<Output, TagOutput> writer; // SPMC 只写

  // 资源访问
  res<Config> config;                 // 只读资源（版本追踪）
  res_mut<State> state;               // 可写资源（写后版本号自增）
  local<Temp> temp;                   // 系统本地变量（不参与依赖分析）

  // bind(): 预创建通道（初始化期调用）
  void bind(World& world);

  // run(): 执行系统逻辑
  void run(World& world);
};
```

### 执行策略

| 策略                                     | 说明                         | 用途                   |
| ---------------------------------------- | ---------------------------- | ---------------------- |
| `fixed_rate<Freq, CPU, Priority>`        | 独占线程，定频触发，通知下游 | 数据源（如相机）       |
| `fixed_rate_silent<Freq, CPU, Priority>` | 独占线程，定频触发，不通知   | 高频静默更新（如 IMU） |
| `pool_compute`                           | TBB 线程池，依赖触发（默认） | 数据处理系统           |
| `pool_visualization`                     | 可视化系统专用               | TODO                   |

### 通道组件

| 组件               | 说明                           |
| ------------------ | ------------------------------ |
| `spsc<T, Tag>`     | SPSC 只读（点对点通信）        |
| `spsc_mut<T, Tag>` | SPSC 只写                      |
| `spmc<T, Tag>`     | SPMC 只读（可拷贝）            |
| `spmc_mut<T, Tag>` | SPMC 只写                      |
| `res<T>`           | 只读资源（版本追踪）           |
| `res_mut<T>`       | 可写资源（写后版本号自增）     |
| `local<T>`         | 系统本地变量（不参与依赖分析） |

### 自适应 Backoff

Compute loop 空闲时使用三阶段 backoff 策略：

| 阶段    | 次数     | 延迟       | 方法                        |
| ------- | -------- | ---------- | --------------------------- |
| Phase 1 | 0-100    | ~10 cycles | `SPIN_HINT()`               |
| Phase 2 | 100-1000 | ~1-10μs    | `std::this_thread::yield()` |
| Phase 3 | 1000+    | 10μs       | `sleep`                     |

---

## ROS2 兼容层 (rclcompat)

```cpp
// ROS2 风格 Node API
node.create_publisher<Message>("topic", qos);
node.create_subscription<Message>("topic", qos, callback);
node.create_wall_timer(period, callback);
node.create_resource<Type>();
node.finalize(); // 完成注册
```

---

## FCS 流水线代码组织

### 5 级流水线数据流

```
L1 (Sensor)     → ImageFrame [SPMC]
                     ↓
L2 (Perception) → ArmorDetectionBatch [SPMC]
                 → ArmorMeasurementBatch [SPMC]
                     ↓
L3 (Estimation) → TrackerOutput [SPMC]
                     ↓
L4 (Planning)   → GimbalPlan [SPMC]
                     ↓
L5 (Weapon)     → WeaponCommand [SPMC]
                     ↓
                 Hardware
```

### 目录结构

```
src/fcs/
├── core/              # 共享类型、通道 topics
│   └── trajectory/    # 弹道模型（ideal、linear_drag、quadratic_drag）
├── L1_sensor/         # 相机接口、输出
├── L2_perception/     # 装甲板检测
│   ├── armor/         # 装甲检测系统
│   └── rune/          # 能量机关检测
├── L3_estimation/     # 追踪、能量机关
│   ├── tracker/       # EKF + Ceres 自动微分、马氏距离门控
│   └── energy_meter/  # ISRCKF 能量机关追踪
├── L4_planning/       # 规划、瞄准器
│   ├── aimer/         # RobotAimer、OutpostAimer、RuneAimer
│   └── gimbal_planner/ # 云台规划器
├── L5_weapon/         # 武器控制
├── calibration/       # 标定
└── runtime/           # 系统注册函数
```

---

## Fast_TF 坐标系

类型安全的坐标变换（替代 tf2），使用类型包装防止误用：

```
crates/fast_tf/src/
├── frame.hpp           # 坐标系定义
├── types.hpp           # 类型包装
├── hero.hpp/.cpp       # 英雄机器人坐标系
├── infantry.hpp/.cpp   # 步兵机器人坐标系
├── sentry.hpp/.cpp     # 哨兵机器人坐标系
├── robot_factory.hpp   # 机器人工厂
├── validation.hpp      # 验证工具
├── matrix.hpp          # 矩阵运算
└── buffer.hpp          # 变换缓冲
```

---

## C++ 开发规范

### 元原则

1. **RAII 是信仰**，禁止手动 new/delete
2. **类型安全 > 灵活性**，用类型系统防止误用
3. **编译期检查 > 运行时检查**
4. **错误必须显式处理**，禁止吞掉异常
5. **Parse, Don't Validate**，边界数据进入核心域前必须解析为强类型

### DO / DON'T

| DO                          | DON'T               |
| --------------------------- | ------------------- |
| 智能指针表达所有权          | 裸指针表达所有权    |
| `std::optional` 表达值缺失  | `cstring`/`carray`  |
| `std::span` 代替裸指针+大小 | exception 传递错误  |
| `std::expected` 表达错误    | bool 表达多义性错误 |
| `enum class` 穷尽状态       | 嵌套 `bool`         |

### 错误处理

- 使用 `std::expected` 表达错误，禁止吞掉异常
- 错误消息必须包含完整上下文（`[模块::方法] 操作失败: 细节`），禁止 `"unknown error"` 之类的模糊描述
- 使用 `magic_enum` 统一转换枚举为小写
- **异常是系统边界**，不允许在系统内部传播。所有内部错误通过 `std::expected` 传播
- 返回 `std::expected` 的方法必须 `noexcept`，内部用 `try/catch(const std::exception&)` 捕获并带完整上下文转为 `std::unexpected`
- **禁止 `catch(...)`** — 只允许 `catch(const std::exception&)`，必须携带 `e.what()` 上下文

### 控制流

- 枚举 switch 开启 `-Wswitch-enum`，禁止 `default` 兜底
- 类型分发使用 `std::variant` + `std::visit` 穷尽匹配

### 类型安全

- 单参数构造函数强制 `explicit`
- 使用强类型包装业务概念
- 优先使用模板，而非 `std::function`
- 禁止使用 `std::any`（除 World 内部实现外）

### 构造与接口设计

1. **禁止两阶段构造** — 对象构造完成后必须处于可用状态，不允许 `init()` / `is_initialized()` 模式
2. **构造即初始化** — 如果初始化可能失败，使用 `static create() noexcept` 工厂方法返回 `std::expected<T, Error>`，构造函数 `private`
3. **接口用 C++20 concepts 强制约束** — CRTP 基类、策略模式等类型参数必须用 concept 约束，编译期检查 `noexcept`、返回类型、工厂签名
4. **`std::abort()` 仅用于不可恢复的逻辑错误** — "文件未找到"、"模型加载失败" 等可恢复错误必须通过 `std::expected` 返回
5. **不允许向后兼容废代码** — 删除废弃接口而非保留，`name()` / `supports_xxx()` 等运行时能力查询属于设计缺陷

### 其他规范

1. **禁止运行时 RTTI**（`dynamic_cast`、`typeid`），允许编译期类型检查（`if constexpr`、`concepts`）
2. **全局可变状态谨慎使用**，需注释说明必要性
3. **递归必须满足**之一：编译期深度限制、运行时深度参数 + 超限处理、可证明深度有界

---

## 常见模式和反模式

| 模式            | 正确                   | 错误                   |
| --------------- | ---------------------- | ---------------------- |
| 传递非拷贝组件  | `std::move(component)` | `my_system(component)` |
| Lambda 捕获组件 | `[&reader] mutable`    | `[reader]`             |

---

## 依赖库版本

| 库名     | 版本     | 说明                   |
| -------- | -------- | ---------------------- |
| CMake    | >= 3.14  | 构建系统               |
| TBB      | 2021.9+  | 线程池                 |
| Eigen3   | 3.4+     | 线性代数               |
| OpenCV   | 4.x      | 图像处理               |
| Ceres    | 2.x      | 非线性优化（自动微分） |
| OpenVINO | optional | 推理后端               |
| TensorRT | optional | 推理后端               |

---

## C++ 标准使用

- **C++23**: 主要代码
- **C++20**: fast_tf, 部分算法
- **C++17**: 依赖库兼容层
