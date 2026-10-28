#pragma once

#include "shm_layout.hpp"
#include <atomic>
#include <optional>

namespace ipc {

/**
 * @brief TripleBuffer 原子操作模板
 *
 * 这是一个无锁 SPSC (Single Producer Single Consumer) 通信原语。
 * 基于 talos-cpp 现有的 triple_buffer.hpp 设计，扩展用于跨进程通信。
 *
 * 状态编码 (uint8_t):
 *   Bit 7: FLAG_NEW - 有新数据
 *   Bits 0-1: ready_index - 就绪槽位索引 (0-2)
 */
template <typename BufferType, typename SlotType>
class TripleBufferOps {
public:
    explicit TripleBufferOps(BufferType* buf)
        : buf_(buf) {}

    // ============ 生产者 API ============

    /**
     * @brief 获取可写槽位的可变引用
     */
    [[nodiscard]] SlotType& borrow_mut() noexcept { return buf_->slots[buf_->write_idx]; }

    /**
     * @brief 发布数据，使消费者可见
     *
     * 原子地交换 write_idx 和 ready_idx，并设置 FLAG_NEW
     */
    void publish() noexcept {
        // 设置 new_data flag，交换 ready 和 write
        const auto old =
            buf_->state.exchange(buf_->write_idx | FLAG_NEW, std::memory_order_acq_rel);
        buf_->write_idx = old & INDEX_MASK;
    }

    // ============ 消费者 API ============

    /**
     * @brief 尝试获取最新数据
     * @return 如果有新数据，返回指向数据的指针；否则返回 nullopt
     *
     * 原实现 CAS 只重试两次，producer 高频 publish 时直接放弃本帧，
     * 实测导致消费者只拿到 1/3 的帧（30fps 输入 → 10fps 消费）。
     * 改为循环重试：CAS 失败只说明 state 被 producer 更新了，
     * 重新读 state 再判断是否有新数据即可，不应丢帧。
     */
    [[nodiscard]] std::optional<const SlotType*> borrow() noexcept {
        for (;;) {
            auto expected = buf_->state.load(std::memory_order_acquire);

            // 没有新数据
            if (!(expected & FLAG_NEW)) {
                return std::nullopt;
            }

            const auto ready_idx = expected & INDEX_MASK;
            if (buf_->state.compare_exchange_strong(
                    expected, buf_->read_idx, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                buf_->read_idx = ready_idx;
                return &buf_->slots[ready_idx];
            }
            // CAS 失败：state 已被 producer 改写，expected 现在是新值，
            // 回到循环顶部重新判断 FLAG_NEW。
        }
    }

    /**
     * @brief 获取当前读取槽位 (不检查是否有新数据)
     */
    [[nodiscard]] const SlotType& current() const noexcept { return buf_->slots[buf_->read_idx]; }

    /**
     * @brief 检查是否有新数据可读
     */
    [[nodiscard]] bool has_new_data() const noexcept {
        return buf_->state.load(std::memory_order_acquire) & FLAG_NEW;
    }

private:
    BufferType* buf_;
};

// 便捷类型别名
using ImageOps  = TripleBufferOps<ImageTripleBuffer, ImageMeta>;
using PoseOps   = TripleBufferOps<PoseTripleBuffer, PoseMeta>;
using GimbalOps = TripleBufferOps<GimbalTripleBuffer, GimbalCmd>;

} // namespace ipc
