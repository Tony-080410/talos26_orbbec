#include "L1_sensor/output_interface.hpp"
#include "chiral/gimbal.hpp"
#include "shm_client.hpp"
#include "talos_gimbal/mcu_device.hpp"
#include "talos_gimbal/packet.hpp"

namespace fcs::L1 {

// ============================================================================
// IpcOutput 实现
// ============================================================================

IpcOutput::IpcOutput(std::shared_ptr<ipc::ShmClient> client) noexcept
    : client_(std::move(client)) {}

void IpcOutput::send(const L5::WeaponCommand& cmd) const noexcept {
    if (!client_) [[unlikely]] {
        return;
    }

    // 转换: rad → deg
    const auto yaw_deg    = static_cast<float>(cmd.yaw * detail::RAD_TO_DEG);
    const auto pitch_deg  = -static_cast<float>(cmd.pitch * detail::RAD_TO_DEG);
    const auto distance_m = static_cast<float>(cmd.distance);

    client_->send_gimbal_cmd(yaw_deg, pitch_deg, distance_m, cmd.fire);
}

// ============================================================================
// McuOutput 实现
// ============================================================================

McuOutput::McuOutput(std::shared_ptr<talos_gimbal::McuDeviceHandle> device) noexcept
    : device_(std::move(device)) {}

void McuOutput::send(const L5::WeaponCommand& cmd) const noexcept {
    if (!device_) [[unlikely]] {
        return;
    }

    if (!device_->is_connected()) [[unlikely]] {
        return;
    }

    // Serial只用于全向感知
    if (device_->backend() == talos_gimbal::McuBackend::Serial) {
        talos_gimbal::SendSimpleVisionData packet{
            .header =
                {
                         .sof = talos_gimbal::HeaderFrame::SoF(),
                         .len = sizeof(talos_gimbal::SendSimpleVisionData::data),
                         .id  = 0x04,
                         },
            .data =
                {
                         .target_yaw = static_cast<float>(cmd.yaw * detail::RAD_TO_DEG),
                         },
            .eof = talos_gimbal::HeaderFrame::EoF(),
        };
        device_->send_sync(reinterpret_cast<uint8_t*>(&packet), sizeof(packet));
    } else {
        talos_gimbal::SendVisionData packet{
            .header =
                {
                         .sof = talos_gimbal::HeaderFrame::SoF(),
                         .len = sizeof(talos_gimbal::SendVisionData::data),
                         .id  = 0x02,
                         },
            .data =
                {
                         .fire_advice  = cmd.fire,
                         .target_yaw   = static_cast<float>(cmd.yaw * detail::RAD_TO_DEG),
                         .target_pitch = -static_cast<float>(cmd.pitch * detail::RAD_TO_DEG),
                         .ref_yaw_v    = static_cast<float>(cmd.yaw_vel * detail::RAD_TO_DEG),
                         .ref_pitch_v  = -static_cast<float>(cmd.pitch_vel * detail::RAD_TO_DEG),
                         .ref_yaw_a    = static_cast<float>(cmd.yaw_accel * detail::RAD_TO_DEG),
                         .ref_pitch_a  = -static_cast<float>(cmd.pitch_accel * detail::RAD_TO_DEG),
                         .distance     = static_cast<float>(cmd.distance),
                         },
            .eof = talos_gimbal::HeaderFrame::EoF(),
        };
        device_->send_sync(reinterpret_cast<uint8_t*>(&packet), sizeof(packet));
    }
}
ChiralOutput::ChiralOutput(std::shared_ptr<talos::chiral::gimbal::TalosEndpoint> device) noexcept
    : device_(std::move(device)) {}

void ChiralOutput::send(const L5::WeaponCommand& cmd) const noexcept {
    if (!device_) [[unlikely]] {
        return;
    }

    device_->write(
        talos::chiral::gimbal::McuRequest{
            .timestamp_ns_system_clock = static_cast<int64_t>(cmd.timestamp_ns),
            .fire_advice               = cmd.fire,
            .target_yaw                = static_cast<float>(cmd.yaw * detail::RAD_TO_DEG),
            .target_pitch              = -static_cast<float>(cmd.pitch * detail::RAD_TO_DEG),
            .ref_yaw_v                 = static_cast<float>(cmd.yaw_vel * detail::RAD_TO_DEG),
            .ref_pitch_v               = -static_cast<float>(cmd.pitch_vel * detail::RAD_TO_DEG),
            .ref_yaw_a                 = static_cast<float>(cmd.yaw_accel * detail::RAD_TO_DEG),
            .ref_pitch_a               = -static_cast<float>(cmd.pitch_accel * detail::RAD_TO_DEG),
            .distance                  = static_cast<float>(cmd.distance),
        });
}

// ============================================================================
// OutputInterface 工厂方法
// ============================================================================

std::expected<OutputInterface, std::string>
    OutputInterface::create_ipc(std::shared_ptr<ipc::ShmClient> client) noexcept {
    return OutputInterface(IpcOutput(std::move(client)));
}

[[nodiscard]] static std::expected<OutputInterface, std::string>
    create_chiral(std::shared_ptr<talos::chiral::gimbal::TalosEndpoint> client) noexcept {
    return OutputInterface(ChiralOutput(std::move(client)));
}

std::expected<OutputInterface, std::string> OutputInterface::create_usb(
    const uint16_t vendor_id, const std::optional<uint16_t> product_id) noexcept {
    auto device = talos_gimbal::McuDeviceHandle::create_usb(vendor_id, product_id);
    if (!device) {
        return std::unexpected("failed to connect USB device");
    }
    return OutputInterface(McuOutput(std::move(device)));
}

std::expected<OutputInterface, std::string>
    OutputInterface::create_serial(const std::string& device_path, int baud_rate) noexcept {
    auto device = talos_gimbal::McuDeviceHandle::create_serial(device_path, baud_rate);
    if (!device) {
        return std::unexpected("failed to connect serial device");
    }
    return OutputInterface(McuOutput(std::move(device)));
}

} // namespace fcs::L1
