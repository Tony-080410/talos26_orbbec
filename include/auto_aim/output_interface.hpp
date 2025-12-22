#pragma once

#include "aimer.hpp"
#include "ipc/shm_client.hpp"
#include "talos_gimbal/packet.hpp"
#include "talos_gimbal/stm32.hpp"
#include <cmath>
#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <variant>

namespace auto_aim {

namespace {
constexpr double RAD_TO_DEG = 180.0 / M_PI;
} // namespace

/**
 * @brief IPC 输出模式 (共享内存)
 */
struct IpcOutput {
    ipc::ShmClient client;

    void send(const GimbalCommand& cmd) noexcept {
        // 转换: rad → deg
        const auto yaw_deg    = static_cast<float>(cmd.yaw * RAD_TO_DEG);
        const auto pitch_deg  = static_cast<float>(cmd.pitch * RAD_TO_DEG);
        const auto distance_m = cmd.valid ? static_cast<float>(cmd.distance) : -1.0f;

        client.send_gimbal_cmd(yaw_deg, pitch_deg, distance_m, cmd.fire_advice);
    }
};

/**
 * @brief USB 输出模式 (串口)
 */
struct UsbOutput {
    talos_gimbal::Stm32Impl<talos_gimbal::Stm32Parser>* device{nullptr};

    void send(const GimbalCommand& cmd) noexcept {
        if (!device || !device->is_connected()) {
            return;
        }

        talos_gimbal::SendVisionData packet{
            .header =
                {
                         .sof = talos_gimbal::HeaderFrame::SoF(),
                         .len = sizeof(talos_gimbal::SendVisionData::data),
                         .id  = 0x02, // Vision data ID
                },
            .data =
                {
                         .fire_advice  = cmd.fire_advice,
                         .target_yaw   = static_cast<float>(cmd.yaw),
                         .target_pitch = static_cast<float>(cmd.pitch),
                         .ref_yaw_v    = 0.0f,
                         .ref_pitch_v  = 0.0f,
                         .ref_yaw_a    = 0.0f,
                         .ref_pitch_a  = 0.0f,
                         .distance     = static_cast<float>(cmd.distance),
                         },
            .eof = talos_gimbal::HeaderFrame::EoF(),
        };

        device->send_sync(reinterpret_cast<uint8_t*>(&packet), sizeof(packet));
    }
};

/**
 * @brief 输出接口 (使用 std::variant 实现静态多态)
 *
 * 支持两种输出模式:
 * 1. IPC: 通过共享内存发送到 Rust 模拟器
 * 2. USB: 通过串口发送到 STM32 (TODO)
 */
class OutputInterface {
public:
    using OutputMode = std::variant<IpcOutput, UsbOutput>;

    explicit OutputInterface(OutputMode mode) noexcept
        : mode_(std::move(mode)) {}

    /**
     * @brief 发送云台控制命令
     */
    void send(const GimbalCommand& cmd) noexcept {
        std::visit([&cmd](auto& output) { output.send(cmd); }, mode_);
    }

    /**
     * @brief 创建 IPC 输出模式
     */
    [[nodiscard]] static std::expected<OutputInterface, std::string> create_ipc() noexcept {
        auto client = ipc::ShmClient::connect();
        if (!client) {
            return std::unexpected(
                "failed to connect IPC client: " + std::string(to_string(client.error())));
        }

        return OutputInterface(IpcOutput{std::move(*client)});
    }

    /**
     * @brief 创建 USB 输出模式
     */
    [[nodiscard]] static std::expected<void, std::string> create_usb() {
        // TODO
        return std::unexpected("unavailable");
    }

private:
    OutputMode mode_;
};

} // namespace auto_aim
