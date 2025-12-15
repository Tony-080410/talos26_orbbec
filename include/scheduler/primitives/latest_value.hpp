#pragma once

#include <atomic>
#include <array>
#include <cstdint>

namespace scheduler {

/**
 * @brief Lock-free SPSC latest-value container for polling consumers
 *
 * Writer 写入最新值，Reader 轮询获取最新值。
 * 使用双缓冲 + 原子交换实现无锁。
 *
 * 特点:
 * - 写者永不阻塞
 * - 读者获取最新完整数据
 * - 带版本号避免重复处理
 */
template<typename T>
class LatestValue {
public:
    LatestValue() = default;

    template<typename... Args>
    explicit LatestValue(std::in_place_t, Args&&... args) {
        for (auto& slot : buffers_) {
            slot = T(std::forward<Args>(args)...);
        }
    }

    /**
     * @brief 写入新值 (单生产者)
     */
    void write(const T& val) noexcept {
        const auto idx = write_idx_;
        buffers_[idx] = val;
        // 发布: 设置 ready index + 递增版本
        uint64_t new_state = (version_.load(std::memory_order_relaxed) + 1) << 1 | idx;
        state_.store(new_state, std::memory_order_release);
        version_.fetch_add(1, std::memory_order_relaxed);
        write_idx_ = 1 - idx;  // 切换写缓冲
    }

    void write(T&& val) noexcept {
        const auto idx = write_idx_;
        buffers_[idx] = std::move(val);
        uint64_t new_state = (version_.load(std::memory_order_relaxed) + 1) << 1 | idx;
        state_.store(new_state, std::memory_order_release);
        version_.fetch_add(1, std::memory_order_relaxed);
        write_idx_ = 1 - idx;
    }

    /**
     * @brief 读取最新值 (可多消费者)
     */
    [[nodiscard]] T read() const noexcept {
        const auto state = state_.load(std::memory_order_acquire);
        const auto idx = state & 1;
        return buffers_[idx];
    }

    /**
     * @brief 带版本号读取，返回是否有新数据
     * @param out 输出值
     * @param last_version 上次读取的版本号，会被更新
     * @return true 如果有新数据
     */
    [[nodiscard]] bool try_read(T& out, uint64_t& last_version) const noexcept {
        const auto state = state_.load(std::memory_order_acquire);
        const auto current_version = state >> 1;
        if (current_version == last_version) {
            return false;
        }
        const auto idx = state & 1;
        out = buffers_[idx];
        last_version = current_version;
        return true;
    }

    /**
     * @brief 获取当前版本号
     */
    [[nodiscard]] uint64_t version() const noexcept {
        return state_.load(std::memory_order_acquire) >> 1;
    }

private:
    // 双缓冲，缓存行对齐
    alignas(64) std::array<T, 2> buffers_{};

    // 共享状态: [63:1] version, [0] ready_index
    alignas(64) std::atomic<uint64_t> state_{0};

    // 写者私有版本号
    alignas(64) std::atomic<uint64_t> version_{0};

    // 写者私有: 当前写入的 buffer index
    uint8_t write_idx_{0};
};

} // namespace scheduler
