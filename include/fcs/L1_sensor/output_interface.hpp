#pragma once

#include "../../ipc/shm_client.hpp"
#include "../L5_weapon/fire_control.hpp"
#include "stm32.hpp"

#include <cmath>
#include <expected>
#include <memory>
#include <string>
#include <variant>

namespace fcs::L1 {

namespace detail {
constexpr double RAD_TO_DEG = 180.0 / M_PI;
}

// ============================================================================
// IpcOutput - 共享内存输出 (用于 Rust 模拟器)
// ============================================================================

class IpcOutput {
public:
    explicit IpcOutput(std::shared_ptr<ipc::ShmClient> client) noexcept;

    void send(const L5::WeaponCommand& cmd) const noexcept;

private:
    std::shared_ptr<ipc::ShmClient> client_; // 类型擦除，避免头文件依赖
};

// ============================================================================
// UsbOutput - USB 串口输出 (用于 STM32)
// ============================================================================

class UsbOutput {
public:
    explicit UsbOutput(
        std::shared_ptr<talos_gimbal::Stm32Impl<talos_gimbal::Stm32Parser>> device) noexcept;

    void send(const L5::WeaponCommand& cmd) const noexcept;

private:
    std::shared_ptr<talos_gimbal::Stm32Impl<talos_gimbal::Stm32Parser>> device_;
};

// ============================================================================
// OutputInterface - 输出接口 (静态多态)
// ============================================================================

class OutputInterface {
public:
    using OutputMode = std::variant<std::monostate, IpcOutput, UsbOutput>;

    OutputInterface() = default;

    explicit OutputInterface(OutputMode mode) noexcept
        : mode_(std::move(mode)) {}

    /**
     * @brief 发送武器控制命令
     */
    void send(const L5::WeaponCommand& cmd) noexcept {
        std::visit(
            [&cmd](auto& output) {
                if constexpr (!std::is_same_v<std::decay_t<decltype(output)>, std::monostate>) {
                    output.send(cmd);
                }
            },
            mode_);
    }

    /**
     * @brief 检查是否有效
     */
    [[nodiscard]] bool valid() const noexcept {
        return !std::holds_alternative<std::monostate>(mode_);
    }

    // ======== 工厂方法 ========

    /**
     * @brief 创建 IPC 输出模式
     */
    [[nodiscard]] static std::expected<OutputInterface, std::string>
        create_ipc(std::shared_ptr<ipc::ShmClient> client) noexcept;

    /**
     * @brief 创建 USB 输出模式
     */
    [[nodiscard]] static std::expected<OutputInterface, std::string> create_usb(
        uint16_t vendor_id = talos_gimbal::Stm32Impl<talos_gimbal::Stm32Parser>::VID,
        std::optional<uint16_t> product_id = std::nullopt) noexcept;

private:
    OutputMode mode_;
};

} // namespace fcs::L1
