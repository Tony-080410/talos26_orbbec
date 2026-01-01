#include "fcs/L1_sensor/output_interface.hpp"
#include "ipc/shm_client.hpp"
#include "talos_gimbal/packet.hpp"
#include "talos_gimbal/stm32.hpp"

#include <spdlog/spdlog.h>

namespace fcs::L1 {

namespace {
constexpr double RAD_TO_DEG = 180.0 / M_PI;
} // namespace

// ============================================================================
// IpcOutput 实现
// ============================================================================

IpcOutput::IpcOutput(std::shared_ptr<ipc::ShmClient> client) noexcept
    : client_(std::move(client)) {}

void IpcOutput::send(const L5::WeaponCommand& cmd) const noexcept {
    if (!client_) {
        return;
    }

    // 转换: rad → deg
    const auto yaw_deg    = static_cast<float>(cmd.yaw * RAD_TO_DEG);
    const auto pitch_deg  = -static_cast<float>(cmd.pitch * RAD_TO_DEG);
    const auto distance_m = cmd.valid ? static_cast<float>(cmd.distance) : -1.0f;

    client_->send_gimbal_cmd(yaw_deg, pitch_deg, distance_m, cmd.fire);
}

// ============================================================================
// UsbOutput 实现
// ============================================================================

UsbOutput::UsbOutput(
    std::shared_ptr<talos_gimbal::Stm32Impl<talos_gimbal::Stm32Parser>> device) noexcept
    : device_(std::move(device)) {}

void UsbOutput::send(const L5::WeaponCommand& cmd) const noexcept {
    if (!device_) {
        return;
    }

    if (!device_->is_connected()) {
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
                     .fire_advice  = cmd.fire,
                     .target_yaw   = static_cast<float>(cmd.yaw * RAD_TO_DEG),
                     .target_pitch = -static_cast<float>(cmd.pitch * RAD_TO_DEG),
                     .ref_yaw_v    = 0.0,
                     .ref_pitch_v  = 0.0,
                     .ref_yaw_a    = 0.0f,
                     .ref_pitch_a  = 0.0f,
                     .distance     = cmd.valid ? static_cast<float>(cmd.distance) : -1.0f,
                     },
        .eof = talos_gimbal::HeaderFrame::EoF(),
    };

    device_->send_sync(reinterpret_cast<uint8_t*>(&packet), sizeof(packet));
}

// ============================================================================
// OutputInterface 工厂方法
// ============================================================================

std::expected<OutputInterface, std::string>
    OutputInterface::create_ipc(std::shared_ptr<ipc::ShmClient> client) noexcept {
    spdlog::info("[OutputInterface] IPC mode connected");
    return OutputInterface(IpcOutput(std::move(client)));
}

std::expected<OutputInterface, std::string> OutputInterface::create_usb(
    const uint16_t vendor_id, const std::optional<uint16_t> product_id) noexcept {
    // 创建 STM32 设备 (libusb 枚举，不使用 device_path)
    auto device = std::make_unique<talos_gimbal::Stm32Impl<talos_gimbal::Stm32Parser>>();

    // 连接设备
    if (const auto result = device->connect(vendor_id, product_id); !result) {
        return std::unexpected("failed to connect USB device");
    }

    spdlog::info("[OutputInterface] USB mode connected");
    return OutputInterface(UsbOutput(std::move(device)));
}

} // namespace fcs::L1
