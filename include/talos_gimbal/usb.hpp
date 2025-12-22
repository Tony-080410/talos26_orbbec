#pragma once

#include <expected>
#include <optional>
#include <libusb-1.0/libusb.h>
#include <span>
#include <variant>

#include "packet.hpp"

namespace talos_gimbal {
    enum class usb_open_error : std::uint8_t { DetachDriver };
    using UsbInitError = std::variant<libusb_error, usb_open_error>;

    constexpr std::string_view to_string(const libusb_error e) { return libusb_error_name(e); }
    constexpr std::string_view to_string(const usb_open_error e) {
        switch (e) {
            case usb_open_error::DetachDriver:
                return "DetachDriver";
            default:
                return "Unknown";
        }
    }
    inline std::string to_string(const UsbInitError &err) {
        return std::visit([](auto &&e) -> std::string { return std::string(to_string(e)); }, err);
    }

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
    concept Parser = requires(const std::span<std::byte> data) {
        { T::parse(data) } -> std::same_as<void>;
    };
} // namespace talos_gimbal
