#pragma once

#include <expected>
#include <libusb/libusb.h>
#include <optional>
#include <span>
#include <variant>

#include "packet.hpp"

#include <fmt/core.h>
#include <fmt/std.h>

namespace talos_gimbal {
enum class usb_open_error : std::uint8_t { DetachDriver };
using UsbInitError = std::variant<libusb_error, usb_open_error>;

template <typename T>
concept Device = requires(
    T& device, uint16_t vendor_id, std::optional<uint16_t> product_id, uint8_t* data, size_t size,
    unsigned timeout) {
    // impl Drop
    requires std::destructible<T>;

    // impl !Copy
    requires !std::copyable<T>;

    {
        device.connect(vendor_id, product_id)
    } noexcept -> std::same_as<std::expected<void, UsbInitError>>;
    { device.send_sync(data, size, timeout) } noexcept -> std::same_as<bool>;
    { device.handle_events() } noexcept -> std::same_as<void>;
};

template <typename T>
concept Parser = requires(const std::span<std::byte> data) {
    { T::parse(data) } -> std::same_as<void>;
};

} // namespace talos_gimbal

// ============================================================================
// fmt::formatter specializations
// ============================================================================

namespace fmt {

template <>
struct formatter<talos_gimbal::usb_open_error> : formatter<std::string_view> {
    auto format(const talos_gimbal::usb_open_error e, format_context& ctx) const {
        return formatter<std::string_view>::format(magic_enum::enum_name(e), ctx);
    }
};

template <>
struct formatter<talos_gimbal::UsbInitError> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    auto format(const talos_gimbal::UsbInitError& err, format_context& ctx) const {
        return std::visit(
            [&ctx](const auto& e) {
                if constexpr (std::is_same_v<std::decay_t<decltype(e)>, libusb_error>) {
                    return fmt::format_to(ctx.out(), "{}", libusb_error_name(e));
                } else {
                    return fmt::format_to(ctx.out(), "{}", magic_enum::enum_name(e));
                }
            },
            err);
    }
};

} // namespace fmt
