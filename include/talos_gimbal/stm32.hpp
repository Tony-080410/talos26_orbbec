#pragma once

#include "talos_gimbal/usb.hpp"

// STL
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <thread>
#include <vector>

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
        static void parse(const std::byte *data, size_t size) noexcept {
            if (size < sizeof(HeaderFrame)) [[unlikely]] {
                spdlog::warn("Frame size too small, expected {}, got {}", sizeof(HeaderFrame), size);
                return;
            }
            HeaderFrame header_frame{};
            std::memcpy(&header_frame, data, sizeof(HeaderFrame));
            if (header_frame.sof != HeaderFrame::SoF()) [[unlikely]] {
                spdlog::warn("Frame header invalid, expected {:X}, got {:X}", HeaderFrame::SoF(), header_frame.sof);
                return;
            }
            if (header_frame.id == 0x01) {
                const auto received = reinterpret_cast<const ReceiveImuData*>(data);
                print_imu(*received);
            }
        }
    };
    static_assert(Parser<Stm32Parser>);

    template<Parser Parser>
    struct Stm32Impl {
        explicit Stm32Impl() = default;
        static constexpr uint16_t VID = 0x0483; // STM32
        static constexpr uint8_t EP_OUT = 0x01;
        static constexpr uint8_t EP_IN = 0x81;
        static constexpr uint8_t INTERFACE_NUM = 1;
        Stm32Impl(const Stm32Impl &) = delete;
        Stm32Impl &operator=(const Stm32Impl &) = delete;

        Stm32Impl(Stm32Impl &&) = default;
        Stm32Impl &operator=(Stm32Impl &&) = default;

        std::expected<void, UsbInitError> connect(uint16_t vendor_id, std::optional<uint16_t> product_id) noexcept {
            // 只在首次连接时初始化 libusb context
            if (this->ctx_ == nullptr) {
                if (auto const init_result = libusb_init(&this->ctx_); init_result != LIBUSB_SUCCESS) {
                    return std::unexpected(static_cast<libusb_error>(init_result));
                }
            }
            if (!product_id) {
                auto result = find_device(this->ctx_, vendor_id);
                if (!result.has_value()) {
                    return std::unexpected(result.error());
                }
                product_id = result.value();
            }

            // 打开设备
            this->handle_ = libusb_open_device_with_vid_pid(this->ctx_, vendor_id, *product_id);
            if (this->handle_ == nullptr) {
                return std::unexpected(LIBUSB_ERROR_NO_DEVICE);
            }

            if (libusb_kernel_driver_active(this->handle_, INTERFACE_NUM)) {
                libusb_detach_kernel_driver(this->handle_, INTERFACE_NUM);
            }

            if (libusb_claim_interface(this->handle_, INTERFACE_NUM)) {
                return std::unexpected(LIBUSB_ERROR_BUSY);
            }

            // 分配 bulk IN transfer
            this->rx_transfer_ = libusb_alloc_transfer(0);
            if (this->rx_transfer_ == nullptr) {
                return std::unexpected(LIBUSB_ERROR_NO_MEM);
            }

            // 读取数据 callback
            libusb_fill_bulk_transfer(
                    this->rx_transfer_, this->handle_, EP_IN, this->rx_buf_.data(), this->rx_buf_.size(),
                    [](libusb_transfer *tr) {
                        auto self = static_cast<UsbConnectionState *>(tr->user_data);

                        // 传输被取消，通知cleanup可以继续
                        if (tr->status == LIBUSB_TRANSFER_CANCELLED) {
                            self->transfer_pending_ = false;
                            return;
                        }

                        if (!self->handling_events_) {
                            self->transfer_pending_ = false;
                            return;
                        }

                        // 设备中途中断
                        if (tr->status != LIBUSB_TRANSFER_COMPLETED) {
                            self->disconnected_ = true;
                            self->transfer_pending_ = false;
                            return;
                        }
                        if (const auto len = tr->actual_length; len > 0) {
                            if (self->first_rx_) {
                                self->first_rx_ = false;
                            } else {
                                Parser::parse(reinterpret_cast<std::byte *>(tr->buffer), static_cast<size_t>(len));
                            }
                        }

                        // 继续提交
                        if (const auto rc = libusb_submit_transfer(tr); rc != LIBUSB_SUCCESS) {
                            self->disconnected_ = true;
                            self->transfer_pending_ = false;
                        }
                    },
                    &this->state, 0);

            // Hot‑plug 回调
            if (!libusb_has_capability(LIBUSB_CAP_HAS_HOTPLUG)) {
                return std::unexpected(LIBUSB_ERROR_NOT_SUPPORTED);
            }

            libusb_hotplug_callback_handle hp_handle{};
            int rc = libusb_hotplug_register_callback(
                    this->ctx_,
                    static_cast<libusb_hotplug_event>(LIBUSB_HOTPLUG_EVENT_DEVICE_LEFT |
                                                      LIBUSB_HOTPLUG_EVENT_DEVICE_ARRIVED),
                    LIBUSB_HOTPLUG_NO_FLAGS, vendor_id, *product_id, LIBUSB_HOTPLUG_MATCH_ANY,
                    [](libusb_context * /*ctx*/, libusb_device * /*dev*/, const libusb_hotplug_event event,
                       void *user) -> int {
                        auto impl = static_cast<Stm32Impl *>(user);
                        if (event == LIBUSB_HOTPLUG_EVENT_DEVICE_LEFT) {
                            impl->state.disconnected_ = true;
                        } else if (event == LIBUSB_HOTPLUG_EVENT_DEVICE_ARRIVED) {
                            if (!impl->handle_) {
                                impl->try_reopen();
                                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                            }
                        }
                        return 0;
                    },
                    this, &hp_handle);

            if (libusb_submit_transfer(this->rx_transfer_) != LIBUSB_SUCCESS) {
                return std::unexpected(LIBUSB_ERROR_IO);
            }

            if (rc != LIBUSB_SUCCESS) {
                return std::unexpected(static_cast<libusb_error>(rc));
            }

            this->state.handling_events_ = true;
            this->state.first_rx_ = true; // 重置首包标志
            return {};
        }

        ~Stm32Impl() noexcept {
            state.handling_events_ = false;
            cleanup();
            if (ctx_)
                libusb_exit(ctx_);
        }

        // 事件循环里调用 — 单线程
        void handle_events() noexcept {
            timeval tv{0, 0};
            if (const auto result = libusb_handle_events_timeout_completed(ctx_, &tv, nullptr);
                result != LIBUSB_SUCCESS || state.disconnected_) {
                state.disconnected_ = true;
                cleanup();
                try_reopen();
            }
        }

        bool send_sync(uint8_t *data, const std::size_t size, const unsigned tout_ms = 500) const noexcept {
            if (!handle_) {
                return false;
            }
            auto actual = 0;
            const auto rc = libusb_bulk_transfer(handle_, EP_OUT, data, static_cast<int>(size), &actual, tout_ms);
            return rc == 0 && actual == static_cast<int>(size);
        }

    private:
        struct UsbConnectionState {
            libusb_context *ctx_ = nullptr;
            std::atomic_bool handling_events_{false};
            std::atomic_bool disconnected_{false};
            std::atomic_bool transfer_pending_{false};
            bool first_rx_{true};
        };
        // 查找 PID (若仅给 VID)
        [[nodiscard]] static std::expected<uint16_t, UsbInitError> find_device(libusb_context *ctx,
                                                                               const uint16_t vid) noexcept {
            libusb_device **list;
            const auto cnt = libusb_get_device_list(ctx, &list);
            if (cnt < 0) {
                // we assume this is a 'safe' operation :p
                return std::unexpected(static_cast<libusb_error>(cnt));
            }
            scoped_exit free{[&] { libusb_free_device_list(list, 1); }};

            std::vector<uint16_t> pids;
            pids.reserve(cnt);
            for (ssize_t i = 0; i < cnt; ++i) {
                libusb_device_descriptor d{};
                if (const auto descriptor = libusb_get_device_descriptor(list[i], &d); descriptor != 0) {
                    return std::unexpected(static_cast<libusb_error>(descriptor));
                }
                if (d.idVendor != vid) {
                    continue;
                }
                pids.push_back(d.idProduct);
            }
            if (pids.empty())
                return std::unexpected(LIBUSB_ERROR_NOT_FOUND);
            if (pids.size() > 1)
                return std::unexpected(LIBUSB_ERROR_OVERFLOW);
            return pids[0];
        }

        void cleanup() noexcept; // 关闭/释放资源
        bool try_reopen() noexcept;

        libusb_context *ctx_ = nullptr;
        libusb_device_handle *handle_ = nullptr;
        libusb_transfer *rx_transfer_ = nullptr;
        std::array<uint8_t, 64> rx_buf_{};
        UsbConnectionState state = {};
    };
    static_assert(Device<Stm32Impl<Stm32Parser>>);
    // 分配 bulk transfer

    // 清理所有资源
    template<Parser Parser>
    void Stm32Impl<Parser>::cleanup() noexcept {
        if (rx_transfer_) {
            // 取消传输并等待回调完成
            state.transfer_pending_ = true;
            libusb_cancel_transfer(rx_transfer_);
            // 等待回调被调用（状态变为 CANCELLED）
            while (state.transfer_pending_) {
                timeval tv{0, 10000}; // 10ms timeout
                libusb_handle_events_timeout(ctx_, &tv);
            }
            libusb_free_transfer(rx_transfer_);
            rx_transfer_ = nullptr;
        }
        if (handle_) {
            libusb_release_interface(handle_, INTERFACE_NUM);
            libusb_close(handle_);
            handle_ = nullptr;
        }
    }

    // 重连
    template<Parser Parser>
    bool Stm32Impl<Parser>::try_reopen() noexcept {
        while (state.handling_events_) {
            if (!connect(VID, std::nullopt)) {
                state.disconnected_ = false;
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        return false;
    }
} // namespace talos_gimbal
