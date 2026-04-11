#pragma once

#include "serial.hpp"
#include "stm32.hpp"

namespace talos_gimbal {

enum class McuBackend : uint8_t {
    Usb,
    Serial,
};

using UsbDevice    = Stm32Impl<Stm32Parser>;
using SerialDevice = SerialImpl<Stm32Parser>;

class McuDeviceHandle {
public:
    // ──── Construction ────

    static std::shared_ptr<McuDeviceHandle>
        create_usb(uint16_t vendor_id, std::optional<uint16_t> product_id) {
        auto dev    = std::make_shared<UsbDevice>();
        auto result = dev->connect(vendor_id, product_id);
        if (!result) {
            SPDLOG_ERROR("USB connect failed: {}", result.error());
            return nullptr;
        }
        auto handle     = std::make_shared<McuDeviceHandle>();
        handle->device_ = std::move(dev);
        return handle;
    }

    static std::shared_ptr<McuDeviceHandle>
        create_serial(const std::string& device_path, int baud_rate) {
        auto dev    = std::make_shared<SerialDevice>();
        auto result = dev->connect(device_path, baud_rate);
        if (!result) {
            SPDLOG_ERROR("Serial connect failed: {}", result.error());
            return nullptr;
        }
        auto handle     = std::make_shared<McuDeviceHandle>();
        handle->device_ = std::move(dev);
        return handle;
    }

    // ──── Unified interface ────

    void handle_events() noexcept {
        std::visit([](auto& d) { d->handle_events(); }, device_);
    }

    bool send_sync(
        const uint8_t* data, const size_t size, const unsigned timeout_ms = 500) const noexcept {
        return std::visit([&](auto& d) { return d->send_sync(data, size, timeout_ms); }, device_);
    }

    bool is_connected() const noexcept {
        return std::visit([](auto& d) { return d->is_connected(); }, device_);
    }

    McuBackend backend() const noexcept {
        return std::holds_alternative<std::shared_ptr<UsbDevice>>(device_) ? McuBackend::Usb
                                                                           : McuBackend::Serial;
    }

private:
    std::variant<std::shared_ptr<UsbDevice>, std::shared_ptr<SerialDevice>> device_;
};

} // namespace talos_gimbal

// ============================================================================
// fmt::formatter specializations
// ============================================================================

namespace fmt {

template <>
struct formatter<talos_gimbal::McuBackend> : formatter<std::string_view> {
    auto format(talos_gimbal::McuBackend b, format_context& ctx) const {
        return formatter<std::string_view>::format(magic_enum::enum_name(b), ctx);
    }
};

} // namespace fmt
