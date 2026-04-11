#pragma once

#include "L5_weapon/fire_control.hpp"
#include "shm_client.hpp"
#include "talos_gimbal/mcu_device.hpp"

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
    std::shared_ptr<ipc::ShmClient> client_;
};

// ============================================================================
// McuOutput - MCU 输出 (USB/Serial 统一)
// ============================================================================

class McuOutput {
public:
    explicit McuOutput(std::shared_ptr<talos_gimbal::McuDeviceHandle> device) noexcept;

    void send(const L5::WeaponCommand& cmd) const noexcept;

private:
    std::shared_ptr<talos_gimbal::McuDeviceHandle> device_;
};

// ============================================================================
// OutputInterface - 输出接口 (静态多态)
// ============================================================================

class OutputInterface {
public:
    using OutputMode = std::variant<std::monostate, IpcOutput, McuOutput>;

    OutputInterface() = default;

    explicit OutputInterface(OutputMode mode) noexcept
        : mode_(std::move(mode)) {}

    void send(const L5::WeaponCommand& cmd) noexcept {
        std::visit(
            [&cmd](auto& output) {
                if constexpr (!std::is_same_v<std::decay_t<decltype(output)>, std::monostate>) {
                    output.send(cmd);
                }
            },
            mode_);
    }

    [[nodiscard]] bool valid() const noexcept {
        return !std::holds_alternative<std::monostate>(mode_);
    }

    [[nodiscard]] static std::expected<OutputInterface, std::string>
        create_ipc(std::shared_ptr<ipc::ShmClient> client) noexcept;

    [[nodiscard]] static std::expected<OutputInterface, std::string> create_usb(
        uint16_t vendor_id = talos_gimbal::Stm32Impl<talos_gimbal::Stm32Parser>::VID,
        std::optional<uint16_t> product_id = std::nullopt) noexcept;

    [[nodiscard]] static std::expected<OutputInterface, std::string> create_serial(
        const std::string& device_path = "/dev/ttyS4", int baud_rate = 115200) noexcept;

private:
    OutputMode mode_;
};

} // namespace fcs::L1
