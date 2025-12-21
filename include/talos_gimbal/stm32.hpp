#pragma once

#include "talos_gimbal/usb.hpp"

// STL
#include <array>
#include <chrono>
#include <cstddef>
#include <span>
#include <thread>
// libusb
#include <libusb-1.0/libusb.h>

// spdlog
#include <spdlog/spdlog.h>

/* ------------------- error ------------------- */
namespace talos_gimbal {
    template<class F>
    class scoped_exit {
    public:
        explicit scoped_exit(F f) : f_(f), enabled_(true) {}
        ~scoped_exit() {
            if (enabled_)
                f_();
        }
        void disable() { enabled_ = false; }

        scoped_exit(const scoped_exit &) = delete;
        scoped_exit &operator=(const scoped_exit &) = delete;
        scoped_exit(scoped_exit &&) = delete;
        scoped_exit &operator=(scoped_exit &&) = delete;

    private:
        F f_;
        bool enabled_;
    };

    class Stm32Parser {
    public:
        Stm32Parser(const Stm32Parser &) = delete;
        Stm32Parser &operator=(const Stm32Parser &) = delete;
        static void parse(const std::span<const std::byte> data) noexcept {
            if (const auto size = data.size_bytes(); size < sizeof(HeaderFrame)) [[unlikely]] {
                spdlog::warn("Frame size too small, expected {}, got {}", sizeof(HeaderFrame), size);
                return;
            }
            const auto header_frame = reinterpret_cast<const HeaderFrame *>(data.data());
            if (header_frame->sof != HeaderFrame::SoF()) [[unlikely]] {
                spdlog::warn("Frame header invalid, expected {:X}, got {:X}", HeaderFrame::SoF(), header_frame->sof);
                return;
            }
            if (header_frame->id == 0x01) {
                //const auto received = reinterpret_cast<const ReceiveImuData *>(data.data());
                //print_imu(*received);
            }
        }
    };
    static_assert(Parser<Stm32Parser>);

    template<Parser Parser>
    struct Stm32Impl {
        static constexpr uint16_t VID = 0x0483;
        static constexpr uint8_t EP_OUT = 0x01;
        static constexpr uint8_t EP_IN = 0x81;
        static constexpr uint8_t INTERFACE_NUM = 1;

        explicit Stm32Impl() = default;
        ~Stm32Impl() noexcept { shutdown(); }

        Stm32Impl(const Stm32Impl &) = delete;
        Stm32Impl &operator=(const Stm32Impl &) = delete;
        Stm32Impl(Stm32Impl &&) = default;
        Stm32Impl &operator=(Stm32Impl &&) = default;

        // ─────────────────────────────────────────────
        // 公共接口
        // ─────────────────────────────────────────────

        std::expected<void, UsbInitError> connect(const uint16_t vendor_id = VID,
                                                  std::optional<uint16_t> product_id = std::nullopt) noexcept {
            if (!ctx_ && libusb_init(&ctx_) != LIBUSB_SUCCESS) {
                return std::unexpected(LIBUSB_ERROR_OTHER);
            }

            if (!product_id) {
                auto result = find_device(ctx_, vendor_id);
                if (!result)
                    return std::unexpected(result.error());
                product_id = *result;
            }
            pid_ = *product_id;

            if (auto err = open_device(vendor_id, *product_id)) {
                return std::unexpected(*err);
            }

            if (auto err = setup_transfer()) {
                return std::unexpected(*err);
            }

            if (auto err = register_hotplug(vendor_id, *product_id)) {
                return std::unexpected(*err);
            }

            state_ = State::Connected;
            first_rx_ = true;
            return {};
        }

        void handle_events() noexcept {
            if (state_ == State::Disconnected)
                return;

            timeval tv{0, 0};
            libusb_handle_events_timeout_completed(ctx_, &tv, nullptr);

            if (state_ == State::NeedsReconnect) {
                reconnect();
            }
        }

        bool send_sync(const uint8_t *data, const size_t size, const unsigned timeout_ms = 500) const noexcept {
            if (!handle_) {
                return false;
            }
            int actual = 0;
            const auto rc = libusb_bulk_transfer(handle_, EP_OUT, const_cast<uint8_t *>(data), static_cast<int>(size),
                                                 &actual, timeout_ms);
            return rc == 0 && actual == static_cast<int>(size);
        }

        bool is_connected() const noexcept { return state_ == State::Connected; }

    private:
        // ─────────────────────────────────────────────
        // 状态机
        // ─────────────────────────────────────────────
        enum class State { Disconnected, Connected, NeedsReconnect };

        // ─────────────────────────────────────────────
        // 内部实现
        // ─────────────────────────────────────────────

        std::optional<libusb_error> open_device(const uint16_t vid, const uint16_t pid) noexcept {
            handle_ = libusb_open_device_with_vid_pid(ctx_, vid, pid);
            if (!handle_)
                return LIBUSB_ERROR_NO_DEVICE;

            if (libusb_kernel_driver_active(handle_, INTERFACE_NUM)) {
                libusb_detach_kernel_driver(handle_, INTERFACE_NUM);
            }

            if (libusb_claim_interface(handle_, INTERFACE_NUM) != 0) {
                libusb_close(handle_);
                handle_ = nullptr;
                return LIBUSB_ERROR_BUSY;
            }
            return std::nullopt;
        }

        std::optional<libusb_error> setup_transfer() noexcept {
            rx_transfer_ = libusb_alloc_transfer(0);
            if (!rx_transfer_) {
                return LIBUSB_ERROR_NO_MEM;
            }

            libusb_fill_bulk_transfer(rx_transfer_, handle_, EP_IN, rx_buf_.data(), rx_buf_.size(),
                                      &Stm32Impl::on_rx_complete, this, 0);

            if (libusb_submit_transfer(rx_transfer_) != LIBUSB_SUCCESS) {
                libusb_free_transfer(rx_transfer_);
                rx_transfer_ = nullptr;
                return LIBUSB_ERROR_IO;
            }
            return std::nullopt;
        }

        std::optional<libusb_error> register_hotplug(const uint16_t vid, const uint16_t pid) noexcept {
            if (!libusb_has_capability(LIBUSB_CAP_HAS_HOTPLUG)) {
                return LIBUSB_ERROR_NOT_SUPPORTED;
            }

            auto rc = libusb_hotplug_register_callback(
                    ctx_,
                    static_cast<libusb_hotplug_event>(LIBUSB_HOTPLUG_EVENT_DEVICE_LEFT |
                                                      LIBUSB_HOTPLUG_EVENT_DEVICE_ARRIVED),
                    LIBUSB_HOTPLUG_NO_FLAGS, vid, pid, LIBUSB_HOTPLUG_MATCH_ANY, &Stm32Impl::on_hotplug, this,
                    &hp_handle_);
            return rc == LIBUSB_SUCCESS ? std::nullopt : std::optional(static_cast<libusb_error>(rc));
        }

        // ─────────────────────────────────────────────
        // 回调 (静态)
        // ─────────────────────────────────────────────

        static void LIBUSB_CALL on_rx_complete(libusb_transfer *tr) noexcept {
            auto self = static_cast<Stm32Impl *>(tr->user_data);

            if (tr->status == LIBUSB_TRANSFER_CANCELLED) [[unlikely]] {
                self->notify_transfer_done();
                return;
            }

            if (self->state_ != State::Connected) [[unlikely]] {
                self->notify_transfer_done();
                return;
            }

            if (tr->status != LIBUSB_TRANSFER_COMPLETED) [[unlikely]] {
                self->state_ = State::NeedsReconnect;
                self->notify_transfer_done();
                return;
            }

            // 处理数据
            if (tr->actual_length > 0 && !self->first_rx_) [[likely]] {
                Parser::parse(std::span(reinterpret_cast<std::byte *>(tr->buffer), tr->actual_length));
            }
            self->first_rx_ = false;

            // 继续接收
            if (libusb_submit_transfer(tr) != LIBUSB_SUCCESS) {
                self->state_ = State::NeedsReconnect;
                self->notify_transfer_done();
            }
        }

        static int LIBUSB_CALL on_hotplug(libusb_context *, libusb_device *, const libusb_hotplug_event event,
                                          void *user) noexcept {
            auto self = static_cast<Stm32Impl *>(user);
            if (event == LIBUSB_HOTPLUG_EVENT_DEVICE_LEFT) {
                self->state_ = State::NeedsReconnect;
            } else if (event == LIBUSB_HOTPLUG_EVENT_DEVICE_ARRIVED) {
                if (self->state_ == State::NeedsReconnect) {
                    // 延迟重连，让设备枚举完成
                    std::this_thread::sleep_for(std::chrono::milliseconds(300));
                }
            }
            return 0;
        }

        // ─────────────────────────────────────────────
        // 资源管理
        // ─────────────────────────────────────────────

        void notify_transfer_done() noexcept {
            std::lock_guard lock(mtx_);
            transfer_done_ = true;
            cv_.notify_one();
        }

        void wait_transfer_done() noexcept {
            std::unique_lock lock(mtx_);
            while (!transfer_done_) {
                // 必须在等待时处理事件，否则回调不会被调用
                lock.unlock();
                timeval tv{0, 10000};
                libusb_handle_events_timeout(ctx_, &tv);
                lock.lock();
            }
        }

        void close_device() noexcept {
            if (rx_transfer_) {
                transfer_done_ = false;
                libusb_cancel_transfer(rx_transfer_);
                wait_transfer_done();
                libusb_free_transfer(rx_transfer_);
                rx_transfer_ = nullptr;
            }
            if (handle_) {
                libusb_release_interface(handle_, INTERFACE_NUM);
                libusb_close(handle_);
                handle_ = nullptr;
            }
        }

        void reconnect() noexcept {
            close_device();
            if (!open_device(VID, pid_) && !setup_transfer()) {
                state_ = State::Connected;
                first_rx_ = true;
            }
        }

        void shutdown() noexcept {
            state_ = State::Disconnected;
            if (hp_handle_) {
                libusb_hotplug_deregister_callback(ctx_, hp_handle_);
                hp_handle_ = 0;
            }
            close_device();
            if (ctx_) {
                libusb_exit(ctx_);
                ctx_ = nullptr;
            }
        }

        [[nodiscard]] static std::expected<uint16_t, UsbInitError> find_device(libusb_context *ctx,
                                                                               const uint16_t vid) noexcept {
            libusb_device **list;
            auto cnt = libusb_get_device_list(ctx, &list);
            if (cnt < 0)
                return std::unexpected(static_cast<libusb_error>(cnt));

            scoped_exit cleanup{[&] { libusb_free_device_list(list, 1); }};

            uint16_t found_pid = 0;
            int count = 0;

            for (ssize_t i = 0; i < cnt; ++i) {
                libusb_device_descriptor desc{};
                if (libusb_get_device_descriptor(list[i], &desc) != 0)
                    continue;
                if (desc.idVendor != vid)
                    continue;
                found_pid = desc.idProduct;
                if (++count > 1)
                    return std::unexpected(LIBUSB_ERROR_OVERFLOW);
            }

            return count == 1 ? std::expected<uint16_t, UsbInitError>(found_pid)
                              : std::unexpected(LIBUSB_ERROR_NOT_FOUND);
        }

        // ─────────────────────────────────────────────
        // 成员变量
        // ─────────────────────────────────────────────
        libusb_context *ctx_ = nullptr;
        libusb_device_handle *handle_ = nullptr;
        libusb_transfer *rx_transfer_ = nullptr;
        libusb_hotplug_callback_handle hp_handle_ = 0;

        std::array<uint8_t, 64> rx_buf_{};
        uint16_t pid_ = 0;

        State state_ = State::Disconnected;
        bool first_rx_ = true;

        // 用于等待 transfer 取消完成
        std::mutex mtx_;
        std::condition_variable cv_;
        bool transfer_done_ = false;
    };
    static_assert(Device<Stm32Impl<Stm32Parser>>);
} // namespace talos_gimbal
