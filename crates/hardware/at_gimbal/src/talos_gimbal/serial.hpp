#pragma once

#include "usb.hpp"

// STL
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <expected>
#include <string>
#include <vector>

// POSIX
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

// poll
#include <poll.h>

// spdlog
#include <spdlog/spdlog.h>

namespace talos_gimbal {

enum class SerialError : uint8_t {
    OpenFailed,
    GetAttrFailed,
    SetAttrFailed,
    NotConnected,
    WriteFailed,
};

template <Parser Parser>
class SerialImpl {
public:
    SerialImpl() = default;

    ~SerialImpl() noexcept { disconnect(); }

    SerialImpl(const SerialImpl&)            = delete;
    SerialImpl& operator=(const SerialImpl&) = delete;
    SerialImpl(SerialImpl&& other) noexcept
        : fd_(other.fd_)
        , connected_(other.connected_)
        , rx_buf_(std::move(other.rx_buf_)) {
        other.fd_        = -1;
        other.connected_ = false;
    }
    SerialImpl& operator=(SerialImpl&& other) noexcept {
        if (this != &other) {
            disconnect();
            fd_              = other.fd_;
            connected_       = other.connected_;
            rx_buf_          = std::move(other.rx_buf_);
            other.fd_        = -1;
            other.connected_ = false;
        }
        return *this;
    }

    [[nodiscard]] std::expected<void, SerialError>
        connect(const std::string& device_path = "/dev/ttyS4", int baud_rate = 115200) noexcept {
        fd_ = ::open(device_path.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
        if (fd_ < 0) {
            SPDLOG_ERROR("serial open '{}' failed: {}", device_path, std::strerror(errno));
            return std::unexpected(SerialError::OpenFailed);
        }

        termios tty{};
        if (::tcgetattr(fd_, &tty) != 0) {
            SPDLOG_ERROR("tcgetattr failed: {}", std::strerror(errno));
            ::close(fd_);
            fd_ = -1;
            return std::unexpected(SerialError::GetAttrFailed);
        }

        // Raw mode
        ::cfmakeraw(&tty);

        // 8N1
        tty.c_cflag &= ~CSTOPB;
        tty.c_cflag &= ~CRTSCTS;
        tty.c_cflag |= CLOCAL | CREAD;

        // No software flow control
        tty.c_iflag &= ~(IXON | IXOFF | IXANY);

        // Blocking reads: return when any data available
        tty.c_cc[VMIN]  = 0;
        tty.c_cc[VTIME] = 1;

        speed_t speed = baud_to_speed(baud_rate);
        ::cfsetispeed(&tty, speed);
        ::cfsetospeed(&tty, speed);

        if (::tcsetattr(fd_, TCSANOW, &tty) != 0) {
            SPDLOG_ERROR("tcsetattr failed: {}", std::strerror(errno));
            ::close(fd_);
            fd_ = -1;
            return std::unexpected(SerialError::SetAttrFailed);
        }

        ::tcflush(fd_, TCIOFLUSH);
        connected_ = true;
        SPDLOG_INFO("serial connected to {} @ {} baud", device_path, baud_rate);
        return {};
    }

    void disconnect() noexcept {
        if (fd_ >= 0) {
            connected_ = false;
            ::close(fd_);
            fd_ = -1;
        }
    }

    void handle_events() noexcept {
        if (!connected_ || fd_ < 0)
            return;

        // Poll for available data with zero timeout (non-blocking)
        struct pollfd pfd{.fd = fd_, .events = POLLIN, .revents = 0};
        while (::poll(&pfd, 1, 0) > 0 && (pfd.revents & POLLIN)) {
            uint8_t tmp[512];
            auto n = ::read(fd_, tmp, sizeof(tmp));
            if (n > 0) {
                rx_buf_.insert(rx_buf_.end(), tmp, tmp + n);
            } else if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
                connected_ = false;
                SPDLOG_WARN("serial read error: {}", std::strerror(errno));
                return;
            }
        }

        // Extract and parse frames
        while (try_parse_frame()) {}
    }

    bool send_sync(
        const uint8_t* data, const size_t size, const unsigned timeout_ms = 500) const noexcept {
        if (!connected_ || fd_ < 0)
            return false;

        size_t written      = 0;
        const auto deadline = [] {
            struct timespec ts;
            ::clock_gettime(CLOCK_MONOTONIC, &ts);
            return ts;
        }();
        const auto deadline_ns = deadline.tv_sec * 1'000'000'000LL + deadline.tv_nsec
                               + static_cast<long long>(timeout_ms) * 1'000'000LL;

        while (written < size) {
            struct pollfd pfd{.fd = fd_, .events = POLLOUT, .revents = 0};
            if (::poll(&pfd, 1, 10) <= 0) {
                // Check timeout
                struct timespec now;
                ::clock_gettime(CLOCK_MONOTONIC, &now);
                if (now.tv_sec * 1'000'000'000LL + now.tv_nsec >= deadline_ns)
                    return false;
                continue;
            }

            auto n = ::write(fd_, data + written, size - written);
            if (n < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK)
                    continue;
                SPDLOG_WARN("serial write error: {}", std::strerror(errno));
                return false;
            }
            written += static_cast<size_t>(n);
        }

        ::tcdrain(fd_);
        return true;
    }

    bool is_connected() const noexcept { return connected_; }

private:
    static speed_t baud_to_speed(int baud) noexcept {
        switch (baud) {
        case 9600: return B9600;
        case 19200: return B19200;
        case 38400: return B38400;
        case 57600: return B57600;
        case 115200: return B115200;
        case 230400: return B230400;
        case 460800: return 460800;
        case 921600: return 921600;
        default: return B115200;
        }
    }

    bool try_parse_frame() noexcept {
        // Find SOF
        auto sof_it = std::find(rx_buf_.begin(), rx_buf_.end(), HeaderFrame::SoF());
        if (sof_it == rx_buf_.end()) {
            rx_buf_.clear();
            return false;
        }

        // Discard garbage before SOF
        if (sof_it != rx_buf_.begin()) {
            rx_buf_.erase(rx_buf_.begin(), sof_it);
        }

        // Need at least header to determine frame length
        if (rx_buf_.size() < sizeof(HeaderFrame))
            return false;

        const auto* header = reinterpret_cast<const HeaderFrame*>(rx_buf_.data());
        if (header->sof != HeaderFrame::SoF()) {
            rx_buf_.erase(rx_buf_.begin());
            return true;
        }

        // Frame layout: HeaderFrame(3) + time_stamp(4) + payload(header->len bytes) + EOF(1)
        const size_t frame_size = sizeof(HeaderFrame) + sizeof(uint32_t) + header->len + 1;

        if (rx_buf_.size() < frame_size)
            return false;

        // Validate EOF
        if (rx_buf_[frame_size - 1] != HeaderFrame::EoF()) {
            rx_buf_.erase(rx_buf_.begin());
            return true;
        }

        Parser::parse(
            std::span<const std::byte>(
                reinterpret_cast<const std::byte*>(rx_buf_.data()), frame_size));

        rx_buf_.erase(rx_buf_.begin(), rx_buf_.begin() + static_cast<ptrdiff_t>(frame_size));
        return true;
    }

    int fd_         = -1;
    bool connected_ = false;
    std::vector<uint8_t> rx_buf_;
};

} // namespace talos_gimbal

// ============================================================================
// fmt::formatter specializations
// ============================================================================

namespace fmt {

template <>
struct formatter<talos_gimbal::SerialError> : formatter<std::string_view> {
    auto format(talos_gimbal::SerialError e, format_context& ctx) const {
        return formatter<std::string_view>::format(magic_enum::enum_name(e), ctx);
    }
};

} // namespace fmt
