#pragma once

#include <Eigen/Core>
#include <atomic>
#include <cstdint>
#include <expected>
#include <memory>
#include <string>

#include "fcs/L2_perception/at_legacy_config.hpp"
#include "fcs/L3_estimation/armor_solver/config.hpp"
#include "toml_helper.hpp"

namespace fcs {

struct CameraConfig {
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix;
    Eigen::Matrix<double, 1, 5> distort_coefficient;
    uint32_t width{1440};
    uint32_t height{1080};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ(tbl, camera_matrix);
        READ(tbl, distort_coefficient);
        READ(tbl, width);
        READ(tbl, height);
        return {};
    }
};

struct FoxgloveConfig {
    bool enabled{true};
    std::string host{"0.0.0.0"};
    uint16_t port{8765};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ(tbl, enabled);
        READ(tbl, host);
        READ(tbl, port);
        return {};
    }
};

struct TalosConfig {
    bool daedalus{false};
    CameraConfig camera{};
    AtLegacyConfig at_legacy{};
    L3::L3Config l3{};
    FoxgloveConfig foxglove{};

    [[nodiscard]] const L3::TrackerConfig& tracker() const noexcept { return l3.tracker; }
    [[nodiscard]] const L3::SolverConfig& solver() const noexcept { return l3.solver; }
    [[nodiscard]] L3::TrackerConfig& tracker() noexcept { return l3.tracker; }
    [[nodiscard]] L3::SolverConfig& solver() noexcept { return l3.solver; }

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept {
        READ(tbl, daedalus);
        READ(tbl, camera);
        READ(tbl, at_legacy);
        READ(tbl, l3);
        READ(tbl, foxglove);
        return {};
    }
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

} // namespace fcs
