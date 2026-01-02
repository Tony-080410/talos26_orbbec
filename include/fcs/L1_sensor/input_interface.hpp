#pragma once

#include "auto_aim/config.hpp"
#include "camera/hik_camera.hpp"
#include "core/time.hpp"
#include "fcs/L1_sensor/parcel.hpp"
#include "ipc/shm_client.hpp"
#include <chrono>
#include <expected>
#include <opencv2/core/mat.hpp>
#include <optional>
#include <utility>
#include <variant>

namespace fcs::L1 {

struct Timeout {};
struct Disconnected {};
struct ProducerNotAvailable {};

using InputError = std::variant<
    Timeout, Disconnected, ProducerNotAvailable, hikcamera::CameraError, ipc::ShmError>;

class IpcInput {
public:
    explicit IpcInput(std::shared_ptr<ipc::ShmClient> client, CameraConfig info) noexcept
        : client_(std::move(client))
        , info_(std::move(info)) {}

    [[nodiscard]] std::optional<Frame> try_recv() const noexcept {
        auto result = client_->recv_image();
        if (!result) {
            return std::nullopt;
        }
        // v.image is &'recv ImageData,
        // but we need to pass a &'mission ImageData
        // apparently 'mission outlives 'recv s.t. we must copy the data.
        auto copied = cv::Mat();
        result->image.copyTo(copied);
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
        std::chrono::duration<float, std::micro> exposure_time) noexcept
        : camera_(std::move(camera))
        , info_(std::move(info))
        , exposure_time_ns_(static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(exposure_time).count())) {}

    [[nodiscard]] std::optional<Frame> try_recv() noexcept {
        using namespace std::chrono_literals;
        auto result = camera_->read(1ms);
        if (!result) {
            return std::nullopt;
        }
        return Frame{
            .seq          = seq_++,
            .timestamp_ns = now_ns(),
            .image        = std::move(*result),
        };
    }

    [[nodiscard]] std::expected<Frame, InputError>
        recv(const std::chrono::milliseconds timeout) noexcept {
        auto result = camera_->read(
            std::chrono::duration_cast<std::chrono::duration<unsigned int, std::milli>>(timeout));
        if (!result) {
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
    uint64_t exposure_time_ns_; // Exposure time in nanoseconds
    uint64_t seq_{0};
};

class InputInterface {
public:
    using InputMode = std::variant<std::monostate, IpcInput, HikInput>;

    InputInterface() = default;
    explicit InputInterface(InputMode mode) noexcept
        : mode_(std::move(mode)) {}

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

    [[nodiscard]] static std::expected<InputInterface, InputError>
        create_ipc(std::shared_ptr<ipc::ShmClient> client) noexcept;

    [[nodiscard]] static std::expected<InputInterface, InputError> create_hik(
        const hikcamera::ImageCapturer::CameraProfile& profile, const CameraConfig& info) noexcept;

private:
    InputMode mode_;
};

} // namespace fcs::L1
