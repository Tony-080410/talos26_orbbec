#pragma once

#include "L1_sensor/parcel.hpp"
#include "camera_config.hpp"
#include "core/time.hpp"
#include "hik_camera.hpp"
#include "shm_client.hpp"
#include <chrono>
#include <expected>
#include <fmt/core.h>
#include <fmt/std.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <utility>
#include <variant>

namespace fcs::L1 {

struct Timeout {};
struct Disconnected {
    std::string ctx;
};
struct ProducerNotAvailable {};

using InputError = std::variant<
    Timeout, Disconnected, ProducerNotAvailable, hikcamera::CameraError, ipc::ShmError>;

class IpcInput {
public:
    explicit IpcInput(std::shared_ptr<ipc::ShmClient> client, CameraConfig info) noexcept
        : client_(std::move(client))
        , info_(std::move(info)) {}

    ~IpcInput() { SPDLOG_INFO("IpcInput destructed"); }

    IpcInput& operator=(const IpcInput&) = delete;

    IpcInput(IpcInput&&) noexcept            = default;
    IpcInput& operator=(IpcInput&&) noexcept = default;

    [[nodiscard]] std::optional<Frame> try_recv() const noexcept {
        auto result = client_->recv_image();
        if (!result) {
            return std::nullopt;
        }
        // v.image is &'recv ImageData,
        // but we need to pass a &'mission ImageData
        // apparently 'mission outlives 'recv s.t. we must copy the data.
        // Use clone() to allocate once with correct size
        cv::Mat copied = result->image.clone();
        cv::cvtColor(copied, copied, cv::COLOR_RGB2BGR);
        return Frame{
            .seq          = result->seq,
            .timestamp_ns = result->timestamp_ns,
            .image        = std::move(copied),
        };
    }

    [[nodiscard]] std::expected<Frame, InputError>
        recv(const std::chrono::milliseconds timeout) const noexcept {
        using clock         = std::chrono::steady_clock;
        const auto deadline = clock::now() + timeout;

        while (clock::now() < deadline) {
            if (auto frame = try_recv()) {
                return *frame;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return std::unexpected(Timeout{});
    }

    [[nodiscard]] const CameraConfig& camera_info() const noexcept { return info_; }

private:
    std::shared_ptr<ipc::ShmClient> client_;
    CameraConfig info_;
};

class HikInput {
public:
    explicit HikInput(
        std::unique_ptr<hikcamera::ImageCapturer> camera, CameraConfig info,
        hikcamera::ImageCapturer::CameraProfile profile,
        std::chrono::duration<float, std::micro> exposure_time) noexcept
        : camera_(std::move(camera))
        , info_(std::move(info))
        , profile_(profile)
        , exposure_time_ns_(
              static_cast<uint64_t>(
                  std::chrono::duration_cast<std::chrono::nanoseconds>(exposure_time).count())) {}

    HikInput& operator=(const HikInput&) = delete;

    HikInput(HikInput&&) noexcept            = default;
    HikInput& operator=(HikInput&&) noexcept = default;

    [[nodiscard]] std::optional<Frame> try_recv() noexcept {
        using namespace std::chrono_literals;
        // NOTE: Using host timestamp (now_ns) instead of camera timestamp.
        // Camera-host clock sync is complex and error only visible in high-dynamic scenarios.
        auto now_   = now_ns();
        auto result = camera_->read(1ms);
        if (!result) {
            return std::nullopt;
        }
        return Frame{
            .seq          = seq_++,
            .timestamp_ns = now_,
            .image        = std::move(*result),
        };
    }

    [[nodiscard]] std::expected<Frame, InputError>
        recv(const std::chrono::milliseconds timeout) noexcept {
        // NOTE: Using host timestamp (now_ns) instead of camera timestamp.
        // Camera-host clock sync is complex and error only visible in high-dynamic scenarios.
        auto result = camera_->read(
            std::chrono::duration_cast<std::chrono::duration<unsigned int, std::milli>>(timeout));
        if (!result) {
            if (!camera_->valid()) {
                if (retry_-- > 0) {
                    auto cam = hikcamera::ImageCapturer::create(profile_);
                    if (!cam) {
                        return std::unexpected(cam.error());
                    }
                    camera_ = std::move(cam.value());
                    retry_  = 0;
                }
                return std::unexpected(Disconnected{result.error().what()});
            }
            return std::unexpected(result.error());
        }
        return Frame{
            .seq          = seq_++,
            .timestamp_ns = now_ns(),
            .image        = std::move(*result),
        };
    }

    [[nodiscard]] const CameraConfig& camera_info() const noexcept { return info_; }

private:
    /// Get timestamp compensated for exposure time (midpoint of exposure)
    /// This aligns with at_vision's timestamp: now() - exposure_time / 2
    [[nodiscard]] uint64_t now_ns() const noexcept {
        const auto now = clock::now_ns();
        // Subtract half of exposure time to get exposure midpoint
        return now - (exposure_time_ns_ / 2);
    }

    std::unique_ptr<hikcamera::ImageCapturer> camera_;
    CameraConfig info_;
    hikcamera::ImageCapturer::CameraProfile profile_;
    uint64_t exposure_time_ns_; // Exposure time in nanoseconds
    uint64_t seq_{0};
    uint64_t retry_{25565};
};

class CameraInterface {
public:
    using InputMode = std::variant<std::monostate, IpcInput, HikInput>;

    explicit CameraInterface() = default;

    explicit CameraInterface(InputMode mode) noexcept
        : mode_(std::move(mode)) {}

    CameraInterface(const CameraInterface&)            = delete;
    CameraInterface& operator=(const CameraInterface&) = delete;

    CameraInterface(CameraInterface&&) noexcept            = default;
    CameraInterface& operator=(CameraInterface&&) noexcept = default;

    [[nodiscard]] std::optional<Frame> try_recv() noexcept {
        return std::visit(
            [](auto& input) -> std::optional<Frame> {
                if constexpr (std::is_same_v<std::decay_t<decltype(input)>, std::monostate>) {
                    return std::nullopt;
                } else {
                    return input.try_recv();
                }
            },
            mode_);
    }

    [[nodiscard]] std::expected<Frame, InputError>
        recv(std::chrono::milliseconds timeout) noexcept {
        return std::visit(
            [timeout](auto& input) -> std::expected<Frame, InputError> {
                if constexpr (std::is_same_v<std::decay_t<decltype(input)>, std::monostate>) {
                    return std::unexpected(Disconnected{});
                } else {
                    return input.recv(timeout);
                }
            },
            mode_);
    }

    [[nodiscard]] const CameraConfig& camera_info() const noexcept {
        return std::visit(
            [](const auto& input) -> const CameraConfig& {
                if constexpr (std::is_same_v<std::decay_t<decltype(input)>, std::monostate>) {
                    static CameraConfig empty{};
                    return empty;
                } else {
                    return input.camera_info();
                }
            },
            mode_);
    }

    [[nodiscard]] static std::expected<CameraInterface, InputError>
        create_ipc(std::shared_ptr<ipc::ShmClient> client) noexcept;

    [[nodiscard]] static std::expected<CameraInterface, InputError>
        create_hik(const CameraConfig& config) noexcept;

private:
    InputMode mode_;
};

} // namespace fcs::L1

// ============================================================================
// fmt::formatter specializations
// ============================================================================

namespace fmt {

template <>
struct formatter<fcs::L1::Timeout> : formatter<std::string_view> {
    auto format(const fcs::L1::Timeout, format_context& ctx) const {
        return formatter<std::string_view>::format("timeout", ctx);
    }
};

template <>
struct formatter<fcs::L1::Disconnected> : formatter<std::string_view> {
    auto format(const fcs::L1::Disconnected d, format_context& ctx) const {
        return formatter<std::string_view>::format(fmt::format("disconnected: {}", d.ctx), ctx);
    }
};

template <>
struct formatter<fcs::L1::ProducerNotAvailable> : formatter<std::string_view> {
    auto format(const fcs::L1::ProducerNotAvailable, format_context& ctx) const {
        return formatter<std::string_view>::format("producer_not_available", ctx);
    }
};

template <>
struct formatter<fcs::L1::InputError> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    auto format(const fcs::L1::InputError& err, format_context& ctx) const {
        return std::visit(
            [&ctx](const auto& e) { return fmt::format_to(ctx.out(), "{}", e); }, err);
    }
};

} // namespace fmt
