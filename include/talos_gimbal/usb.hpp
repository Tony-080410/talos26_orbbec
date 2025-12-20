#pragma once

#include <expected>
#include <libusb-1.0/libusb.h>
#include <variant>


#include "packet.hpp"

namespace talos_gimbal {
    enum class usb_open_error : std::uint8_t { DetachDriver };
    using UsbInitError = std::variant<libusb_error, usb_open_error>;

    template<typename T>
    concept Device = requires(T &device, uint16_t vendor_id, std::optional<uint16_t> product_id, uint8_t *data,
                              size_t size, unsigned timeout) {
        // impl Drop
        requires std::destructible<T>;

        // impl !Copy
        requires !std::copyable<T>;

        { device.connect(vendor_id, product_id) } noexcept -> std::same_as<std::expected<void, UsbInitError>>;
        { device.send_sync(data, size, timeout) } noexcept -> std::same_as<bool>;
        { device.handle_events() } noexcept -> std::same_as<void>;
    };

    template<typename T>
    concept Parser = requires(const std::byte *data, size_t size) {
        { T::parse(data, size) } -> std::same_as<void>;
    };
} // namespace talos_gimbal
