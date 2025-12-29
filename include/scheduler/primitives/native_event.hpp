#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

#ifdef __linux__
# include <climits>
# include <ctime>
# include <errno.h>
# include <linux/futex.h>
# include <sys/syscall.h>
# include <unistd.h>
#else
# include <condition_variable>
# include <mutex>
#endif

namespace primitive {

enum class WaitStatus : std::uint8_t { Signaled = 0, Timeout = 1, Closed = 2 };

/**
 * @brief Auto-reset event
 *
 * - `signal()` wakes 1 waiter; if no waiter, the next `wait()` consumes the signal immediately.
 * - `wait()` consumes the signal (auto reset).
 * - `close()` unblocks all waiters; subsequent waits return `WaitStatus::Closed`.
 */
class AutoResetEvent {
public:
    AutoResetEvent() noexcept = default;
    ~AutoResetEvent() { close(); }

    AutoResetEvent(const AutoResetEvent&)            = delete;
    AutoResetEvent& operator=(const AutoResetEvent&) = delete;
    AutoResetEvent(AutoResetEvent&&)                 = delete;
    AutoResetEvent& operator=(AutoResetEvent&&)      = delete;

    void signal() noexcept {
#ifdef __linux__
        std::uint32_t expected = UNSIGNALED;
        if (state_.compare_exchange_strong(
                expected, SIGNALED, std::memory_order_release, std::memory_order_relaxed)) {
            syscall(SYS_futex, &state_, FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0);
        }
#else
        {
            std::lock_guard lock(mutex_);
            if (state_ == State::Closed) {
                return;
            }
            state_ = State::Signaled;
        }
        cv_.notify_one();
#endif
    }

    void close() noexcept {
#ifdef __linux__
        const auto old = state_.exchange(CLOSED, std::memory_order_release);
        if (old != CLOSED) {
            syscall(SYS_futex, &state_, FUTEX_WAKE_PRIVATE, INT_MAX, nullptr, nullptr, 0);
        }
#else
        {
            std::lock_guard lock(mutex_);
            state_ = State::Closed;
        }
        cv_.notify_all();
#endif
    }

    [[nodiscard]] WaitStatus wait() noexcept {
#ifdef __linux__
        while (true) {
            std::uint32_t expected = SIGNALED;
            if (state_.compare_exchange_strong(
                    expected, UNSIGNALED, std::memory_order_acq_rel, std::memory_order_acquire)) {
                return WaitStatus::Signaled;
            }
            if (expected == CLOSED) {
                return WaitStatus::Closed;
            }

            syscall(SYS_futex, &state_, FUTEX_WAIT_PRIVATE, UNSIGNALED, nullptr, nullptr, 0);
        }
#else
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [this] { return state_ != State::Unsignaled; });
        if (state_ == State::Closed) {
            return WaitStatus::Closed;
        }
        state_ = State::Unsignaled;
        return WaitStatus::Signaled;
#endif
    }

    [[nodiscard]] WaitStatus wait_for(const std::chrono::nanoseconds timeout) noexcept {
        if (timeout.count() <= 0) {
            return WaitStatus::Timeout;
        }

#ifdef __linux__
        const auto deadline = std::chrono::steady_clock::now() + timeout;

        while (true) {
            std::uint32_t expected = SIGNALED;
            if (state_.compare_exchange_strong(
                    expected, UNSIGNALED, std::memory_order_acq_rel, std::memory_order_acquire)) {
                return WaitStatus::Signaled;
            }
            if (expected == CLOSED) {
                return WaitStatus::Closed;
            }

            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) {
                return WaitStatus::Timeout;
            }
            const auto remaining = deadline - now;

            struct timespec ts;
            const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(remaining).count();
            ts.tv_sec     = static_cast<time_t>(ns / 1'000'000'000LL);
            ts.tv_nsec    = static_cast<long>(ns % 1'000'000'000LL);

            const int ret =
                syscall(SYS_futex, &state_, FUTEX_WAIT_PRIVATE, UNSIGNALED, &ts, nullptr, 0);
            if (ret == -1 && errno == ETIMEDOUT) {
                return WaitStatus::Timeout;
            }
        }
#else
        std::unique_lock lock(mutex_);
        const bool ok = cv_.wait_for(lock, timeout, [this] { return state_ != State::Unsignaled; });
        if (!ok) {
            return WaitStatus::Timeout;
        }
        if (state_ == State::Closed) {
            return WaitStatus::Closed;
        }
        state_ = State::Unsignaled;
        return WaitStatus::Signaled;
#endif
    }

    [[nodiscard]] WaitStatus wait_for_ms(const std::uint64_t timeout_ms) noexcept {
        return wait_for(std::chrono::milliseconds(timeout_ms));
    }

    [[nodiscard]] bool is_closed() const noexcept {
#ifdef __linux__
        return state_.load(std::memory_order_acquire) == CLOSED;
#else
        std::lock_guard lock(mutex_);
        return state_ == State::Closed;
#endif
    }

private:
#ifdef __linux__
    static constexpr std::uint32_t UNSIGNALED = 0;
    static constexpr std::uint32_t SIGNALED   = 1;
    static constexpr std::uint32_t CLOSED     = 2;
    alignas(64) std::atomic<std::uint32_t> state_{UNSIGNALED};
#else
    enum class State : std::uint8_t { Unsignaled = 0, Signaled = 1, Closed = 2 };
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    State state_{State::Unsignaled};
#endif
};

/**
 * @brief Manual-reset event
 *
 * - `set()` wakes all waiters and stays signaled until `reset()`.
 * - `wait()` does NOT consume the signal (manual reset).
 * - `close()` unblocks all waiters; subsequent waits return `WaitStatus::Closed`.
 */
class ManualResetEvent {
public:
    ManualResetEvent() noexcept = default;
    ~ManualResetEvent() { close(); }

    ManualResetEvent(const ManualResetEvent&)            = delete;
    ManualResetEvent& operator=(const ManualResetEvent&) = delete;
    ManualResetEvent(ManualResetEvent&&)                 = delete;
    ManualResetEvent& operator=(ManualResetEvent&&)      = delete;

    void set() noexcept {
#ifdef __linux__
        std::uint32_t expected = UNSIGNALED;
        if (state_.compare_exchange_strong(
                expected, SIGNALED, std::memory_order_release, std::memory_order_relaxed)) {
            syscall(SYS_futex, &state_, FUTEX_WAKE_PRIVATE, INT_MAX, nullptr, nullptr, 0);
        }
#else
        {
            std::lock_guard lock(mutex_);
            if (state_ == State::Closed) {
                return;
            }
            state_ = State::Signaled;
        }
        cv_.notify_all();
#endif
    }

    void reset() noexcept {
#ifdef __linux__
        if (state_.load(std::memory_order_acquire) != CLOSED) {
            state_.store(UNSIGNALED, std::memory_order_release);
        }
#else
        std::lock_guard lock(mutex_);
        if (state_ != State::Closed) {
            state_ = State::Unsignaled;
        }
#endif
    }

    void close() noexcept {
#ifdef __linux__
        const auto old = state_.exchange(CLOSED, std::memory_order_release);
        if (old != CLOSED) {
            syscall(SYS_futex, &state_, FUTEX_WAKE_PRIVATE, INT_MAX, nullptr, nullptr, 0);
        }
#else
        {
            std::lock_guard lock(mutex_);
            state_ = State::Closed;
        }
        cv_.notify_all();
#endif
    }

    [[nodiscard]] WaitStatus wait() noexcept {
#ifdef __linux__
        while (true) {
            const auto s = state_.load(std::memory_order_acquire);
            if (s == SIGNALED) {
                return WaitStatus::Signaled;
            }
            if (s == CLOSED) {
                return WaitStatus::Closed;
            }
            syscall(SYS_futex, &state_, FUTEX_WAIT_PRIVATE, UNSIGNALED, nullptr, nullptr, 0);
        }
#else
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [this] { return state_ != State::Unsignaled; });
        if (state_ == State::Closed) {
            return WaitStatus::Closed;
        }
        return WaitStatus::Signaled;
#endif
    }

    [[nodiscard]] WaitStatus wait_for(const std::chrono::nanoseconds timeout) noexcept {
        if (timeout.count() <= 0) {
            return WaitStatus::Timeout;
        }

#ifdef __linux__
        const auto deadline = std::chrono::steady_clock::now() + timeout;

        while (true) {
            const auto s = state_.load(std::memory_order_acquire);
            if (s == SIGNALED) {
                return WaitStatus::Signaled;
            }
            if (s == CLOSED) {
                return WaitStatus::Closed;
            }

            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) {
                return WaitStatus::Timeout;
            }
            const auto remaining = deadline - now;

            struct timespec ts;
            const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(remaining).count();
            ts.tv_sec     = static_cast<time_t>(ns / 1'000'000'000LL);
            ts.tv_nsec    = static_cast<long>(ns % 1'000'000'000LL);

            const int ret =
                syscall(SYS_futex, &state_, FUTEX_WAIT_PRIVATE, UNSIGNALED, &ts, nullptr, 0);
            if (ret == -1 && errno == ETIMEDOUT) {
                return WaitStatus::Timeout;
            }
        }
#else
        std::unique_lock lock(mutex_);
        const bool ok = cv_.wait_for(lock, timeout, [this] { return state_ != State::Unsignaled; });
        if (!ok) {
            return WaitStatus::Timeout;
        }
        if (state_ == State::Closed) {
            return WaitStatus::Closed;
        }
        return WaitStatus::Signaled;
#endif
    }

    [[nodiscard]] WaitStatus wait_for_ms(const std::uint64_t timeout_ms) noexcept {
        return wait_for(std::chrono::milliseconds(timeout_ms));
    }

private:
#ifdef __linux__
    static constexpr std::uint32_t UNSIGNALED = 0;
    static constexpr std::uint32_t SIGNALED   = 1;
    static constexpr std::uint32_t CLOSED     = 2;
    alignas(64) std::atomic<std::uint32_t> state_{UNSIGNALED};
#else
    enum class State : std::uint8_t { Unsignaled = 0, Signaled = 1, Closed = 2 };
    std::mutex mutex_;
    std::condition_variable cv_;
    State state_{State::Unsignaled};
#endif
};

/**
 * @brief Backward-compatible event wrapper (historical name).
 *
 * Semantics:
 * - `wait()` is auto-reset (consumes a signal).
 * - `notify_one()` and `notify_all()` are both treated as "signal once" (matches old behavior).
 * - `wait_for(ms)` returns false only on timeout (Closed is treated as "woken").
 *
 * For explicit semantics, prefer `AutoResetEvent` / `ManualResetEvent`.
 */
class NativeEvent {
public:
    NativeEvent() noexcept = default;
    ~NativeEvent()         = default;

    NativeEvent(const NativeEvent&)            = delete;
    NativeEvent& operator=(const NativeEvent&) = delete;
    NativeEvent(NativeEvent&&)                 = delete;
    NativeEvent& operator=(NativeEvent&&)      = delete;

    void notify_one() noexcept { ev_.signal(); }
    void notify_all() noexcept { ev_.signal(); }

    void close() noexcept { ev_.close(); }

    void wait() noexcept { (void)ev_.wait(); }

    [[nodiscard]] bool wait_for(const std::uint64_t timeout_ms) noexcept {
        return ev_.wait_for_ms(timeout_ms) != WaitStatus::Timeout;
    }

    [[nodiscard]] WaitStatus wait_status() noexcept { return ev_.wait(); }

    [[nodiscard]] WaitStatus wait_for_status(const std::uint64_t timeout_ms) noexcept {
        return ev_.wait_for_ms(timeout_ms);
    }

    [[nodiscard]] bool is_closed() const noexcept { return ev_.is_closed(); }

private:
    AutoResetEvent ev_;
};

} // namespace primitive
