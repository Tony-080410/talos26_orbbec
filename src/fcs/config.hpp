#pragma once

#include <atomic>
#include <cstdint>
#include <expected>
#include <fmt/core.h>
#include <memory>
#include <optional>
#include <string>

#include "magic_enum.hpp"

#include "L2_perception/at_legacy_config.hpp"
#include "L2_perception/rune/rune_config.hpp"
#include "L3_estimation/energy_meter/energy_meter_systems.hpp"
#include "L3_estimation/tracker/config.hpp"
#include "L4_planning/config.hpp"
#include "L5_weapon/config.hpp"
#include "camera_config.hpp"

// Forward declaration to avoid including toml++/toml_helper in public headers.
namespace toml::inline v3 {
class table;
}

namespace fcs {

struct FoxgloveConfig {
    bool enabled{true};
    std::string host{"0.0.0.0"};
    uint16_t port{8765};

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

enum class RobotType { Infantry, Hero, Sentry };

struct HardwareConfig {
    CameraConfig camera{};
    bool untrust_roll{true};
    uint16_t imu_vendor_id{0x0483};
    uint16_t imu_product_id{0};

    [[nodiscard]] auto imu_product_id_opt() const noexcept {
        if (imu_product_id == 0) {
            return std::optional<uint16_t>{};
        }
        return std::optional<uint16_t>{imu_product_id};
    }

    std::expected<void, std::string> read_from(const toml::table& tbl) noexcept;
};

struct VisionConfig {
    core::trajectory::TrajectoryConfig trajectory{};
    L2::AtLegacyConfig at_legacy{};
    rune::RuneDetectorConfig rune_detector{};

    L3::L3Config l3{};
    energy_meter::EnergyMeterL3Config energy_meter{};

    L4::L4Config l4{};

    L5::L5Config l5{};

    [[nodiscard]] const L3::TrackerConfig& tracker() const noexcept { return l3.tracker; }
    [[nodiscard]] const L5::WeaponControllerConfig& weapon() const noexcept { return l5.weapon; }
    [[nodiscard]] L3::TrackerConfig& tracker() noexcept { return l3.tracker; }
    [[nodiscard]] L5::WeaponControllerConfig& weapon() noexcept { return l5.weapon; }

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
 *   AtomicConfig<VisionConfig> config(initial_config);
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
using AtomicVisionConfig = AtomicConfig<VisionConfig>;

} // namespace fcs

// ============================================================================
// fmt::formatter specializations
// ============================================================================

namespace fmt {

template <>
struct formatter<fcs::RobotType> : formatter<std::string_view> {
    auto format(fcs::RobotType type, format_context& ctx) const {
        return formatter<std::string_view>::format(magic_enum::enum_name(type), ctx);
    }
};

} // namespace fmt
