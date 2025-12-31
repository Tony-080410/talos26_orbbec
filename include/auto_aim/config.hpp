#pragma once

#include <Eigen/Eigen>
#include <array>
#include <atomic>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <string>

#include "fcs/L2_perception/at_legacy_config.hpp"
#include "fcs/L3_estimation/armor_solver/config.hpp"
#include "spdlog_hook.hpp"
#include "toml_helper.hpp"

#include <toml++/toml.hpp>

namespace fcs {

/**
 * @brief 相机内参 (统一格式)
 */
struct CameraInfo {
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix;
    Eigen::Matrix<double, 1, 5> distort_coefficient;
    uint32_t width{1440};
    uint32_t height{1080};
};

/**
 * @brief 相机配置 (内参 + 外参)
 */
struct CameraConfig {
    // 相机内参
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix; // 3x3 row-major
    Eigen::Matrix<double, 1, 5> distort_coefficient;            // 1x5

    // 外参：相机到云台、云台到IMU
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> extrinsic_camera_gimbal; // 3x3 row-major
    Eigen::Matrix<double, 3, 1> extrinsic_camera_gimbal_tf;               // 3x1
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> extrinsic_gimbal_imu;    // 3x3 row-major

    // 统一格式的相机内参 (从上述矩阵填充)
    fcs::CameraInfo camera_info{};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        if (const auto _t = tbl["camera"].as_table()) {
            auto t = *_t;
            READ_MATRIX(c, t, camera_matrix);
            READ_MATRIX(d, t, distort_coefficient);
            READ_MATRIX(r, t, extrinsic_camera_gimbal);
            READ_MATRIX(tt, t, extrinsic_camera_gimbal_tf);
            READ_MATRIX(rr, t, extrinsic_gimbal_imu);

            // 填充 camera_info
            camera_info.camera_matrix       = camera_matrix;
            camera_info.distort_coefficient = distort_coefficient;

            return {};
        }
        // Fallback to old "solver" section for compatibility
        if (const auto _t = tbl["solver"].as_table()) {
            auto t = *_t;
            READ_MATRIX(c, t, camera_matrix);
            READ_MATRIX(d, t, distort_coefficient);
            READ_MATRIX(r, t, extrinsic_camera_gimbal);
            READ_MATRIX(tt, t, extrinsic_camera_gimbal_tf);
            READ_MATRIX(rr, t, extrinsic_gimbal_imu);

            camera_info.camera_matrix       = camera_matrix;
            camera_info.distort_coefficient = distort_coefficient;

            return {};
        }
        return std::unexpected("camera section not found");
    }
};

// 兼容别名
using AtVisionSolverConfig = CameraConfig;

// ============================================================================
// Foxglove Configuration
// ============================================================================

/**
 * @brief Foxglove 可视化服务器配置
 */
struct FoxgloveConfig {
    bool enabled{true};          // 默认开启
    std::string host{"0.0.0.0"}; // 默认监听所有接口
    uint16_t port{8765};         // 默认端口

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        if (const auto _t = tbl["foxglove"].as_table()) {
            auto t = *_t;
            READ_KEY_OPT(t, enabled);
            READ_KEY_OPT(t, host);
            READ_KEY_OPT(t, port);
        }
        // foxglove section is optional, use defaults if not found
        return {};
    }
};

// ============================================================================
// Unified Talos Configuration
// ============================================================================

/**
 * @brief 统一配置结构 (所有模块配置的聚合)
 */
struct TalosConfig {
    bool daedalus{false};
    CameraConfig camera{};
    AtLegacyConfig at_legacy{};
    L3::L3Config l3{};         // L3 estimation config (tracker + solver + trajectory)
    FoxgloveConfig foxglove{}; // Foxglove visualization config

    // 便捷访问器
    [[nodiscard]] const L3::TrackerConfig& tracker() const noexcept { return l3.tracker; }
    [[nodiscard]] const L3::SolverConfig& solver() const noexcept { return l3.solver; }
    [[nodiscard]] L3::TrackerConfig& tracker() noexcept { return l3.tracker; }
    [[nodiscard]] L3::SolverConfig& solver() noexcept { return l3.solver; }

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

/**
 * @brief 原子配置包装器 - 用于线程安全的配置读写
 *
 * 使用 atomic<shared_ptr<T>> 实现无锁读写：
 * - read(): 原子加载当前配置的快照
 * - update(): 原子替换整个配置
 *
 * 用法:
 *   AtomicConfig<TalosConfig> config(initial_config);
 *   auto snapshot = config.read();        // 读取快照
 *   config.update(new_config);            // 更新配置
 */
template <typename T>
class AtomicConfig {
public:
    AtomicConfig() noexcept
        : ptr_(std::make_shared<T>()) {}

    explicit AtomicConfig(T initial) noexcept
        : ptr_(std::make_shared<T>(std::move(initial))) {}

    /// 读取当前配置的快照 (线程安全)
    [[nodiscard]] std::shared_ptr<const T> read() const noexcept { return std::atomic_load(&ptr_); }

    /// 更新配置 (线程安全)
    void update(T new_config) noexcept {
        std::atomic_store(&ptr_, std::make_shared<T>(std::move(new_config)));
    }

    /// 更新配置 (使用 shared_ptr)
    void update(std::shared_ptr<T> new_ptr) noexcept {
        std::atomic_store(&ptr_, std::move(new_ptr));
    }

    /// 获取底层指针 (用于初始化时传递给 World)
    [[nodiscard]] std::shared_ptr<T> get_mutable() noexcept { return std::atomic_load(&ptr_); }

private:
    std::shared_ptr<T> ptr_;
};

// 类型别名
using AtomicTalosConfig = AtomicConfig<TalosConfig>;

// ============================================================================
// Implementation
// ============================================================================

inline std::expected<void, std::string> TalosConfig::read_from(const toml::table& tbl) noexcept {
    if (const auto _t = tbl["runtime"].as_table()) {
        auto t = *_t;
        READ_KEY(t, daedalus);
    } else {
        return std::unexpected("runtime section not found");
    }

    READ_SUB(camera, tbl);
    READ_SUB(at_legacy, tbl);
    READ_SUB(l3, tbl);       // L3 config (tracker + solver + trajectory)
    READ_SUB(foxglove, tbl); // Foxglove visualization config (optional)

    return {};
}

} // namespace fcs
