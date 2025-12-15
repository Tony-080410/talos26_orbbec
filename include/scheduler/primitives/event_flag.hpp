#pragma once

#include <atomic>
#include <mutex>
#include <condition_variable>
#include <chrono>

namespace scheduler {

/**
 * @brief 轻量级事件通知
 *
 * 用于线程间通知。Linux 可优化为 eventfd/futex。
 * 这里用 portable 实现 (condition_variable)。
 */
class EventFlag {
public:
    EventFlag() = default;

    // 禁止拷贝
    EventFlag(const EventFlag&) = delete;
    EventFlag& operator=(const EventFlag&) = delete;

    /**
     * @brief 发送通知，唤醒一个等待者
     */
    void notify() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            signaled_ = true;
        }
        cv_.notify_one();
    }

    /**
     * @brief 发送通知，唤醒所有等待者
     */
    void notify_all() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            signaled_ = true;
        }
        cv_.notify_all();
    }

    /**
     * @brief 阻塞等待通知
     */
    void wait() noexcept {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return signaled_; });
        signaled_ = false;
    }

    /**
     * @brief 非阻塞检查是否有通知
     * @return true 如果有通知
     */
    [[nodiscard]] bool try_wait() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (signaled_) {
            signaled_ = false;
            return true;
        }
        return false;
    }

    /**
     * @brief 带超时等待
     * @return true 如果收到通知, false 如果超时
     */
    template<typename Rep, typename Period>
    [[nodiscard]] bool wait_for(const std::chrono::duration<Rep, Period>& timeout) noexcept {
        std::unique_lock<std::mutex> lock(mutex_);
        const bool result = cv_.wait_for(lock, timeout, [this] { return signaled_; });
        if (result) {
            signaled_ = false;
        }
        return result;
    }

    /**
     * @brief 重置状态 (不等待)
     */
    void reset() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        signaled_ = false;
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool signaled_{false};
};

} // namespace scheduler
