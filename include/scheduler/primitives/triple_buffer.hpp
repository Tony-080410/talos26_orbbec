#pragma once

#include <array>
#include <atomic>
#include <new>
#include <optional>

namespace primitive {
/**
 * @brief Lock-free SPSC triple-buffer with zero-copy borrow semantics
 */
template <typename T>
class TripleBuffer {
public:
    TripleBuffer() = default;

    template <typename... Args>
    explicit TripleBuffer(std::in_place_t, Args&&... args) {
        for (auto& slot : slots_) {
            slot = T(std::forward<Args>(args)...);
        }
    }

    [[nodiscard]] T& borrow_mut() noexcept { return slots_[write_idx_]; }

    void publish() noexcept {
        // 设置 new_data flag，交换 ready 和 write
        const auto old = state_.exchange(write_idx_ | FLAG_NEW, std::memory_order_acq_rel);
        write_idx_     = old & INDEX_MASK;
    }

    [[nodiscard]] std::optional<const T*> borrow() noexcept {
        auto expected = state_.load(std::memory_order_acquire);

        // 没有新数据
        if (!(expected & FLAG_NEW)) {
            return std::nullopt;
        }

        // CAS: 尝试拿走 ready，把 read 还回去，清除 flag
        auto ready_idx = expected & INDEX_MASK;

        if (auto desired = read_idx_; !state_.compare_exchange_strong(
                expected, desired, std::memory_order_acq_rel, std::memory_order_acquire)) {
            // Producer 刚好又写了，expected 已更新，再试一次
            if (!(expected & FLAG_NEW)) {
                return std::nullopt;
            }
            ready_idx = expected & INDEX_MASK;
            desired   = read_idx_;
            if (!state_.compare_exchange_strong(
                    expected, desired, std::memory_order_acq_rel, std::memory_order_relaxed)) {
                // producer 在这么短时间内连续 publish 两次
                // 放弃这次
                return std::nullopt;
            }
        }

        read_idx_ = ready_idx;
        return &slots_[read_idx_];
    }

    [[nodiscard]] const T& current() const noexcept { return slots_[read_idx_]; }

    [[nodiscard]] bool has_new_frame() const noexcept {
        return state_.load(std::memory_order_acquire) & FLAG_NEW;
    }

private:
    static constexpr uint8_t FLAG_NEW   = 0x80;
    static constexpr uint8_t INDEX_MASK = 0x03;

    alignas(std::hardware_destructive_interference_size) std::array<T, 3> slots_{};

    // Writer 私有
    alignas(std::hardware_destructive_interference_size) uint8_t write_idx_{0};

    // 共享状态: [7]: new_data_flag, [1:0]: ready_index
    alignas(std::hardware_destructive_interference_size) std::atomic<uint8_t> state_{
        1}; // ready=1, no new data

    // Reader 私有
    alignas(std::hardware_destructive_interference_size) uint8_t read_idx_{2};
};
} // namespace primitive
