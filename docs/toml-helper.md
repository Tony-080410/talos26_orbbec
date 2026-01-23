# toml-helper 快速参考

TOML 配置解析 - `READ` 宏, `merge_configs`, `read_from`

## 30秒速查

| 我想...              | 代码                                             |
| -------------------- | ------------------------------------------------ |
| **读取必需字段**     | `READ(tbl, variable);`                           |
| **读取可选字段**     | `READ_OPT(tbl, variable);`                       |
| **平铺读取嵌套结构** | `READ_FLATTEN(tbl, nested_struct);`              |
| **检查是否有未读键** | `ensure_all_read(tbl, "context");`               |
| **合并两个配置文件** | `merge_configs(base.table(), override.table());` |
| **解析 TOML 文件**   | `auto tbl = toml::parse_file("path.toml");`      |
| **读取嵌套表**       | `READ(tbl, inner_struct);` (自动递归)            |
| **读取枚举**         | `READ(tbl, enum_var);` (支持字符串/数字)         |

## 头文件

```cpp
#include "toml_helper.hpp"
```

---

## 完整配置系统示例

```cpp
#include "toml_helper.hpp"
#include <optional>
#include <vector>

// 场景: 装甲板追踪器配置系统
// - 支持多层级嵌套配置
// - 支持可选参数和默认值
// - 支持多文件合并 (base + robot + 算法)
// - 支持枚举类型 (字符串/数字)

enum class TrackerType {
    JPDA,        // 联合概率数据关联
    NearestNeighbor,
    MultipleHypothesis
};

enum class MotionModel {
    CV,          // 恒速模型
    CA,          // 恒加速模型
    CT           // 恒转速率模型
};

struct MotionModelConfig {
    double process_noise_pos = 0.1;      // 位置过程噪声
    double process_noise_vel = 0.5;      // 速度过程噪声
    double measurement_noise = 0.05;     // 测量噪声

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        // 读取必需字段 (缺失则报错)
        READ(tbl, process_noise_pos);
        READ(tbl, process_noise_vel);
        READ(tbl, measurement_noise);

        return {};
    }
};

// 中层配置: 单个目标类型的追踪参数
struct TargetConfig {
    // 追踪器参数
    double lost_threshold = 0.5;         // 丢失阈值
    double tracking_threshold = 2.0;     // 追踪阈值
    double matcher_gate = 1.0;           // 关联门限

    // 嵌套的运动模型配置
    MotionModelConfig model;

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        // 读取必需字段
        READ(tbl, lost_threshold);
        READ(tbl, tracking_threshold);
        READ(tbl, matcher_gate);
        // 读取嵌套表 (自动递归解析)
        READ(tbl, model);
        return {};
    }
};

// 外层配置: 完整追踪器配置
struct TrackerConfig {
    // 不同目标的配置
    TargetConfig robot;                  // 机器人目标
    TargetConfig outpost;                // 前哨站目标
    TargetConfig hero;                   // 英雄目标

    // 全局开关
    bool inekf_enabled = false;          // 是否启用 InEKF
    bool enable_visualization = false;   // 是否可视化

    // 追踪器类型 (枚举)
    TrackerType tracker_type = TrackerType::JPDA;

    // 可选的调试参数
    std::optional<std::string> log_file;      // 日志文件路径
    std::optional<int> debug_level;           // 调试级别

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        // 读取嵌套配置 (自动递归)
        READ(tbl, robot);          // 读取 [robot] 表
        READ(tbl, outpost);        // 读取 [outpost] 表
        READ(tbl, hero);           // 读取 [hero] 表

        // 读取全局开关
        READ(tbl, inekf_enabled);
        READ(tbl, enable_visualization);

        // 读取枚举 (支持字符串或数字)
        READ(tbl, tracker_type);

        // 读取可选字段 (缺失不影响)
        READ_OPT(tbl, log_file);
        READ_OPT(tbl, debug_level);

        return {};
    }
};

// ============================================================================
// 辅助配置: 相机内参 (平铺结构)
// ============================================================================

struct CameraIntrinsics {
    int width = 1920;
    int height = 1080;
    double fx = 1000.0;
    double fy = 1000.0;
    double cx = 960.0;
    double cy = 540.0;

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        // 平铺读取: 所有字段在同一个表中,无需 [camera] 子表
        READ(tbl, width);
        READ(tbl, height);
        READ(tbl, fx);
        READ(tbl, fy);
        READ(tbl, cx);
        READ(tbl, cy);

        return {};
    }
};

// ============================================================================
// 平铺配置包装: 在外层表中使用平铺内参
// ============================================================================

struct VisionConfig {
    std::string camera_model = "HIK-2030";

    // 平铺嵌入: camera 参数直接在 vision 表中
    CameraIntrinsics camera;

    // 其他配置
    double exposure_time = 5000.0;
    double gain = 10.0;

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ(tbl, camera_model);
        READ_FLATTEN(tbl, camera);      // 平铺读取 camera 配置
        READ(tbl, exposure_time);
        READ(tbl, gain);
        return {};
    }
};

// ============================================================================
// 配置加载函数: 单文件
// ============================================================================

std::expected<TrackerConfig, std::string>
    load_tracker_config(const std::string& path) {

    // ------------------------------------------------------------
    // 解析 TOML 文件
    // ------------------------------------------------------------
    auto tbl_result = toml::parse_file(path);
    if (!tbl_result) {
        return std::unexpected("Failed to parse TOML: " + tbl_result.error());
    }

    // ------------------------------------------------------------
    // 读取配置
    // ------------------------------------------------------------
    TrackerConfig config;
    auto result = config.read_from(tbl_result.table());

    if (!result) {
        return std::unexpected(result.error());
    }

    // ------------------------------------------------------------
    // 检查是否有未读的键 (防止配置拼写错误)
    // ------------------------------------------------------------
    auto check = toml_helper::ensure_all_read(tbl_result.table(), "tracker");
    if (!check) {
        return std::unexpected(check.error());
    }

    return config;
}

// ============================================================================
// 配置加载函数: 多文件合并
// ============================================================================

std::expected<TrackerConfig, std::string>
    load_merged_tracker_config(
        const std::string& base_path,
        const std::string& robot_path,
        const std::string& algorithm_path
    ) {
    // ------------------------------------------------------------
    // 解析基础配置 (通用参数)
    // ------------------------------------------------------------
    auto base_tbl = toml::parse_file(base_path);
    if (!base_tbl) {
        return std::unexpected("Failed to parse base config");
    }

    // ------------------------------------------------------------
    // 解析机器人配置 (硬件相关)
    // ------------------------------------------------------------
    auto robot_tbl = toml::parse_file(robot_path);
    if (!robot_tbl) {
        return std::unexpected("Failed to parse robot config");
    }

    // ------------------------------------------------------------
    // 解析算法配置 (算法相关)
    // ------------------------------------------------------------
    auto algo_tbl = toml::parse_file(algorithm_path);
    if (!algo_tbl) {
        return std::unexpected("Failed to parse algorithm config");
    }

    // ------------------------------------------------------------
    // 合并配置: base + robot + algorithm
    // 后面的覆盖前面的
    // ------------------------------------------------------------
    auto merged = toml_helper::merge_configs(
        base_tbl.table(),
        robot_tbl.table()
    );

    if (!merged) {
        return std::unexpected(merged.error());
    }

    auto merged_again = toml_helper::merge_configs(
        *merged,
        algo_tbl.table()
    );

    if (!merged_again) {
        return std::unexpected(merged_again.error());
    }

    // ------------------------------------------------------------
    // 读取合并后的配置
    // ------------------------------------------------------------
    TrackerConfig config;
    auto result = config.read_from(*merged_again);

    if (!result) {
        return std::unexpected(result.error());
    }

    // ------------------------------------------------------------
    // 检查未读键
    // ------------------------------------------------------------
    auto check = toml_helper::ensure_all_read(*merged_again, "merged_tracker");
    if (!check) {
        return std::unexpected(check.error());
    }

    return config;
}

// ============================================================================
// 配置数组类型
// ============================================================================

struct FilterConfig {
    std::vector<int> kernel_size{3, 3};
    std::vector<double> weights;

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ(tbl, kernel_size);
        READ(tbl, weights);
        return {};
    }
};

struct PreprocessConfig {
    FilterConfig blur_filter;
    FilterConfig edge_filter;

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ(tbl, blur_filter);
        READ(tbl, edge_filter);
        return {};
    }
};
```

---

## 对应的 TOML 配置文件

```toml
# ============================================================================
# config/tracker_base.toml (基础配置)
# ============================================================================

[robot]
lost_threshold = 0.5
tracking_threshold = 2.0
matcher_gate = 1.0

[robot.model]
process_noise_pos = 0.1
process_noise_vel = 0.5
measurement_noise = 0.05

[outpost]
lost_threshold = 1.0
tracking_threshold = 3.0
matcher_gate = 1.5

[outpost.model]
process_noise_pos = 0.05
process_noise_vel = 0.3
measurement_noise = 0.02

[hero]
lost_threshold = 0.3
tracking_threshold = 1.5
matcher_gate = 0.8

[hero.model]
process_noise_pos = 0.08
process_noise_vel = 0.4
measurement_noise = 0.03

inekf_enabled = true
enable_visualization = true

# 枚举支持字符串 (推荐)
tracker_type = "JPDA"

# 或数字 (等价)
# tracker_type = 0
```

```toml
# ============================================================================
# config/robot/hero.toml (机器人配置,覆盖基础配置)
# ============================================================================

# 覆盖 hero 的 matcher_gate
[hero]
matcher_gate = 0.5

# 新增调试配置
log_file = "/var/log/tracker.log"
debug_level = 2
```

```toml
# ============================================================================
# config/vision.toml (相机配置,平铺结构)
# ============================================================================

[camera]
model = "HIK-2030"
width = 1920
height = 1080
fx = 1200.0
fy = 1200.0
cx = 960.0
cy = 540.0

# 暴光参数 (在同一层级)
exposure_time = 8000.0
gain = 8.0
```
