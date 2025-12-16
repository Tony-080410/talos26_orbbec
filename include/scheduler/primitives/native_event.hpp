#pragma once

#include <cstdint>
#include <atomic>
#include <chrono>

#ifdef __APPLE__

#elif __linux__
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <errno.h>
#else
#error "Unsupported platform - requires Linux or macOS"
#endif

namespace scheduler {

/**
 * @brief 平台原生事件原语 - 提供高性能的线程同步
 *
 * Linux: 使用futex (fast userspace mutex)
 * macOS: 使用mutex (dispatch_semaphore更慢)
 */
class NativeEvent {
public:
#ifdef __APPLE__
    NativeEvent() noexcept = default;

    // 禁止拷贝
    NativeEvent(const NativeEvent&) = delete;
    NativeEvent& operator=(const NativeEvent&) = delete;

    /**
     * @brief 等待事件（无超时）
     *
     * 使用double-checked locking模式避免虚假唤醒
     */
    inline void wait() noexcept {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return signaled_; });
        signaled_ = false;
    }

    /**
     * @brief 有限时间等待
     *
     * @return true 如果收到信号，false 如果超时
     */
    inline bool wait_for(uint64_t timeout_ms) noexcept {
        std::unique_lock<std::mutex> lock(mutex_);
        const bool result = cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] { return signaled_; });
        if (result) {
            signaled_ = false;
        }
        return result;
    }

    /**
     * @brief 唤醒单个等待者
     */
    inline void notify_one() noexcept {
        this->notify_all();
    }

    /**
     * @brief 唤醒所有等待者
     *
     * macOS dispatch_semaphore没有wake_all等价物，通过多次signal模拟，假设最多10个等待者
     * 那么比mutex更慢，所以使用mutex
     */
    inline void notify_all() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            signaled_ = true;
        }
        cv_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool signaled_{false};

#elif __linux__
    NativeEvent() noexcept = default;
    ~NativeEvent() noexcept = default;

    // 禁止拷贝
    NativeEvent(const NativeEvent&) = delete;
    NativeEvent& operator=(const NativeEvent&) = delete;

    // 允许移动
    NativeEvent(NativeEvent&& other) noexcept = default;
    NativeEvent& operator=(NativeEvent&& other) noexcept = default;

    /**
     * @brief 使用Linux futex等待事件
     *
     * futex性能特点:
     * - Hot path (已signaled): 单次CAS + acquire = 5-10ns
     * - Cold path (wait): 系统调用 ~300-500ns 但内核高效处理
     * - 无虚假唤醒 (kernel已检查state)
     */
    inline void wait() noexcept {
        while (true) {
            uint32_t expected = UNSIGNALED;
            if (state_.compare_exchange_strong(expected, UNSIGNALED,
                                               std::memory_order_acquire)) {
                break;  // CAS成功，说明signal已到达
            }

            // futex_wait: 内核会检查当前值是否仍为expected，
            // 如果不是则立即返回，避免race condition
            syscall(SYS_futex, &state_, FUTEX_WAIT_PRIVATE, UNSIGNALED,
                   nullptr, nullptr, 0);
        }
    }

    /**
     * @brief 有限时间futex等待
     *
     * @return true 如果收到信号，false 如果超时
     */
    inline bool wait_for(uint64_t timeout_ms) noexcept {
        struct timespec ts;
        ts.tv_sec = timeout_ms / 1000;
        ts.tv_nsec = (timeout_ms % 1000) * 1000000;

        while (true) {
            uint32_t expected = UNSIGNALED;
            if (state_.compare_exchange_strong(expected, UNSIGNALED,
                                               std::memory_order_acquire)) {
                return true;
            }

            int ret = syscall(SYS_futex, &state_, FUTEX_WAIT_PRIVATE, UNSIGNALED,
                             &ts, nullptr, 0);
            if (ret == -1 && errno == ETIMEDOUT) {
                return false;
            }
        }
    }

    /**
     * @brief 唤醒单个futex等待者
     *
     * FUTEX_WAKE_PRIVATE: 唤醒n个等待者，只需检查single futex word
     */
    inline void notify_one() noexcept {
        state_.store(SIGNALED, std::memory_order_release);
        syscall(SYS_futex, &state_, FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0);
    }

    /**
     * @brief 唤醒所有futex等待者
     */
    inline void notify_all() noexcept {
        state_.store(SIGNALED, std::memory_order_release);
        syscall(SYS_futex, &state_, FUTEX_WAKE_PRIVATE, INT_MAX, nullptr, nullptr, 0);
    }

private:
    static constexpr uint32_t UNSIGNALED = 0;
    static constexpr uint32_t SIGNALED = 1;

    alignas(64) std::atomic<uint32_t> state_{UNSIGNALED};
#endif

};

}  // namespace scheduler
