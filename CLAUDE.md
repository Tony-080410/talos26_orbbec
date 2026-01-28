# 项目编译

> if sandboxed, append `-DENABLE_SCCACHE=OFF`

```bash
cmake -G Ninja -B build_agent -S .
cmake --build build_agent -- talos -j64
```

# 运行测试

```bash
# 运行全部测试
cd build && ctest -j11

# 运行单个测试
cd build && ./bin/scheduler_ecs_test
cd build && ./bin/fast_tf_test
cd build && ./bin/triple_buffer_test

# 启用 AddressSanitizer
cmake -G Ninja -B build -S . -DENABLE_ASAN=ON
cmake --build build -- talos -j64
```

# 项目架构

## 概览

Talos 是一个基于自定义 ECS 调度器的机器人视觉框架，采用 5 级流水线（FCS - Fire Control System）架构。

**核心设计理念：**
- 零 ROS 依赖，单二进制部署
- 类型安全的数据流（编译期 tag 类型）
- 锁无关的三重缓冲通道（SPSC/SPMC）
- 基于 Intel TBB 的多线程调度

## ECS 调度器框架

### 核心组件

1. **World** (`include/scheduler/ecs/world.hpp`)
   - 资源和通道的中心容器
   - 类型擦除存储（`std::any`）+ 调用点编译期类型检查
   - 版本化资源追踪

2. **通道类型** (`include/scheduler/ecs/channel_types.hpp`)
   ```cpp
   spsc<T, Tag>       // SPSC 读取端
   spsc_mut<T, Tag>   // SPSC 写入端
   spmc<T, Tag>       // SPMC 读取端
   spmc_mut<T, Tag>   // SPMC 写入端
   res<T>             // 不可变资源引用
   res_mut<T>         // 可变资源引用（写后版本号自增）
   atomic_ref<T, Tag> // 原子变量引用
   ```

3. **System 执行模型** (`include/scheduler/ecs/system.hpp`)
   - `bind()`: 单线程预创建所有通道/资源
   - `run()`: 执行系统，返回是否写入数据

4. **调度器** (`include/scheduler/scheduler/scheduler.hpp`)
   - **fixed_rate<FreqHz, Affinity, Priority>**: 独占线程，定频触发
   - **pool_compute**: TBB 线程池，依赖触发
   - 编译期验证：单写入者、无循环依赖、孤儿检测

### 创建新系统

```cpp
// 1. 在 fcs/core/channel_tags.hpp 定义通道 tag
struct MyInputChannelTag {};
struct MyOutputChannelTag {};

// 2. 在 fcs/LX_XXX_systems.hpp 声明注册函数
namespace fcs::LX {
void register_my_systems(talos::Scheduler& scheduler, MyConfig&& config);
}

// 3. 在 .cpp 实现
void register_my_systems(talos::Scheduler& scheduler, MyConfig&& config) {
    scheduler.add_system<talos::pool_compute>(
        "my_system",
        [config = std::move(config)](
            spmc<InputType, MyInputChannelTag> input,
            spsc_mut<OutputType, MyOutputChannelTag> output) {
            if (auto data = input.read()) {
                output.write(process(*data));
            }
        });
}

// 4. 在 main.cpp 调用
fcs::LX::register_my_systems(scheduler, config);
```

## FCS 流水线架构

```
L1 (Sensor)     → AtLegacyImageFrame [SPMC]
L2 (Perception) → AtLegacyArmorDetectionBatch [SPMC]
                 → AtLegacyArmorMeasurementBatch [SPMC]
L3 (Estimation) → TrackerOutput [SPMC]
L4 (Planning)   → GimbalPlan [SPMC]
L5 (Weapon)     → WeaponCommand [SPMC]
                → Hardware
```

### 目录结构

```
include/fcs/
├── core/              # 共享类型、通道 tag
├── L1_sensor/         # 相机接口、输出
├── L2_perception/     # 装甲板检测（NN/传统/TUP）
├── L3_estimation/     # 追踪（JPDA/InEKF）
├── L4_planning/       # MPC 规划、轨迹补偿
├── L5_weapon/         # 武器控制
├── visualization/     # Foxglove 集成
├── calibration/       # 标定
└── runtime/           # 系统注册函数
```

## Fast_TF 坐标系

**类型安全的坐标变换**（替代 tf2）：

```cpp
// 声明坐标系（include/math/fast_tf/frames.hpp）
DECL_ROOT(world);
DECL(odom, world);
DECL(gimbal_link, odom);
DECL(camera_link, gimbal_link);

// 使用类型包装防止误用
struct Pose {
    fast_tf::FuckTransform<fast_tf::odom_fuxk_frame> transform;
    double timestamp;
};
```

## 配置系统

**TOML 配置合并**：

```
at_vision.toml              # 入口配置
config/vision_base.toml     # 基础配置
config/vision/<name>.toml   # 算法特定配置
config/robot/<name>.toml    # 硬件配置
```

使用 `toml_helper::merge_configs()` 合并，覆盖语义。

**READ 宏**（`include/toml_helper.hpp`）提供声明式配置解析。

## 库结构

| 库名 | 类型 | 说明 |
|------|------|------|
| talos_std | INTERFACE | 性能探针、TOML 辅助 |
| scheduler | SHARED | ECS 调度器 |
| fcs | SHARED | 火控系统（使用 PCH 加速 50-70%） |
| fast_tf | SHARED | 类型安全坐标变换 |
| foxglove_cpp | SHARED | Foxglove WebSocket 服务器 |
| hikcamera | SHARED | HIK 相机驱动（仅 Linux x86_64） |
| talos_gimbal | INTERFACE | 云台控制 |

# C++ 开发规范

## 元原则

1. 能用 RAII 的，绝不手动管理资源
2. 能用类型系统的，绝不用运行时检查
3. 能用编译时检查的，绝不用文档约定
4. 能用成熟库的，绝不自造轮子

## 类型与内存

1. 禁止运行时 RTTI（dynamic_cast、typeid），允许编译期类型检查（if constexpr、concepts）
2. 禁止 NULL，优先 std::optional 表达值缺失，智能指针表达所有权
3. 禁止裸 new/delete，使用 std::make_unique / std::make_shared
4. 指针语义：
    - 拥有：unique_ptr（独占）、shared_ptr（共享）
    - 非拥有：引用（保证有效）、weak_ptr（需检测失效）、裸指针（生命周期明确时，需注释说明）
5. 禁止 cstring / carray，使用 std::string / std::array
6. 禁止 T* + size_t，使用 std::span<T>

## 错误处理

1. 禁止 exception，使用 std::expected
2. 构造函数必须 noexcept，复杂构造使用工厂函数
3. 第三方库异常在模块边界 try-catch 转换为 expected
4. 禁止 bool 表达多义性错误

## 错误消息规范

**核心原则：错误消息必须包含足够的诊断信息，避免歧义。**

### 详细错误消息要求

1. **禁止模糊错误** ❌
   ```cpp
   // 错误：不知道具体哪个字段有问题
   return std::unexpected("Invalid transform: NaN or Inf detected");

   // 错误：知道是 RPY 但不知道具体哪个值
   return std::unexpected("RPY is NaN");
   ```

2. **必须包含完整上下文** ✅
   ```cpp
   // 正确：明确指出哪个字段有问题、具体值是什么
   return std::unexpected(fmt::format(
       "StaticTransformDesc validation failed: [pitch is Inf]. "
       "Full values: roll=0°, pitch=inf°, yaw=1.35°, x=0.044m, y=0.010m, z=-0.053m"));
   ```

### 错误消息模板

| 场景 | 格式 | 示例 |
|------|------|------|
| **数值验证失败** | `"[字段名 is 错误类型]. Full values: 完整值列表"` | `"pitch is NaN. Full values: r=0, p=nan, y=1.35"` |
| **文件/资源错误** | `"操作 failed on 资源: 具体原因 (路径: xxx)"` | `"Failed to open config: Permission denied (path: /etc/robot.toml)"` |
| **类型转换错误** | `"类型转换失败: FromType → ToType, 值=xxx"` | `"Enum conversion failed: 5 → RobotType, value=5 is not valid"` |
| **约束违反** | `"约束违反: 约束名称 (期望值 vs 实际值)"` | `"Buffer overflow: capacity=1024 (attempted to write 1025)"` |
| **依赖缺失** | `"缺失依赖: dependency_name (需要版本)"` | `"Missing library: libtorch (required >= 2.0)"` |

### 枚举到字符串转换

使用 **magic_enum** 进行枚举到字符串转换，然后统一转换为小写：

```cpp
#include <magic_enum.hpp>
#include <algorithm>
#include <string>

enum class RobotType { Infantry, Hero, Sentry };

std::string robot_type_to_string(RobotType type) {
    std::string_view sv = magic_enum::enum_name(type);
    std::string result(sv);
    // 转换为小写
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}
// RobotType::Hero → "hero"
```

## 控制流

1. 枚举 switch：开启 -Wswitch-enum，禁止 default 兜底
2. 类型分发：强制 std::variant + std::visit 穷尽匹配
3. 递归必须满足之一：
    - 编译期深度限制
    - 运行时深度参数 + 超限处理
    - 可证明深度有界

## 类型安全

1. 单参数构造函数强制 explicit
2. 强类型参数优于弱类型（Position 而非 double x, y, z）
3. 限制 std::function，优先模板；必须使用时注释说明原因
4. 禁止 std::any

## 其他

1. 全局可变状态谨慎使用，需注释说明必要性
