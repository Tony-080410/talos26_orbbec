#pragma once

#include "frame.hpp"
#include "native_event.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace primitive {

/**
 * @brief Image buffer pool with safe auto-return semantics.
 *
 * Key properties:
 * - `ImagePtr` is safe to copy/share; the buffer returns to the pool only when the last reference
 * is released.
 * - Optional blocking behavior when exhausted.
 * - Pool storage stays valid as long as any outstanding `ImagePtr` exists.
 */
class BufferPool {
public:
    enum class ExhaustPolicy : std::uint8_t {
        Allocate, // allocate an extra buffer when exhausted (does not return to pool)
        Block,    // block until a pooled buffer is returned
        Fail,     // return nullptr when exhausted
    };

    BufferPool(
        const std::size_t capacity, const int rows, const int cols, const int type,
        const ExhaustPolicy policy = ExhaustPolicy::Allocate)
        : state_(std::make_shared<State>(capacity, rows, cols, type, policy)) {}

    BufferPool(const BufferPool&)                = default;
    BufferPool& operator=(const BufferPool&)     = default;
    BufferPool(BufferPool&&) noexcept            = default;
    BufferPool& operator=(BufferPool&&) noexcept = default;

    /**
     * @brief Try acquire a pooled buffer (non-blocking).
     * @return ImagePtr (may be null if exhausted)
     */
    [[nodiscard]] ImagePtr try_acquire() noexcept { return acquire_impl(/*blocking=*/false); }

    /**
     * @brief Acquire a buffer according to ExhaustPolicy.
     * @return ImagePtr (may be null if ExhaustPolicy::Fail and exhausted)
     */
    [[nodiscard]] ImagePtr acquire() noexcept { return acquire_impl(/*blocking=*/true); }

    [[nodiscard]] std::size_t capacity() const noexcept { return state_->buffers.size(); }

    [[nodiscard]] std::size_t available_approx() const noexcept {
        return static_cast<std::size_t>(state_->available.load(std::memory_order_relaxed));
    }

    [[nodiscard]] std::uint64_t overflow_allocations() const noexcept {
        return state_->overflow_allocations.load(std::memory_order_relaxed);
    }

    [[nodiscard]] std::uint64_t failed_acquisitions() const noexcept {
        return state_->failed_acquisitions.load(std::memory_order_relaxed);
    }

private:
    static constexpr std::uint32_t kNullIndex = std::numeric_limits<std::uint32_t>::max();

    struct State {
        State(
            const std::size_t capacity_, const int rows_, const int cols_, const int type_,
            const ExhaustPolicy policy_)
            : rows(rows_)
            , cols(cols_)
            , type(type_)
            , policy(policy_) {
            buffers.reserve(capacity_);
            next.resize(capacity_, kNullIndex);

            for (std::size_t i = 0; i < capacity_; ++i) {
                buffers.push_back(std::make_unique<ImageBuffer>());
                buffers.back()->mat() = cv::Mat(rows, cols, type);
                next[i] = (i + 1 < capacity_) ? static_cast<std::uint32_t>(i + 1) : kNullIndex;
            }

            const std::uint32_t head_idx = capacity_ > 0 ? 0U : kNullIndex;
            head.store(pack_head(head_idx, 0), std::memory_order_relaxed);
            available.store(static_cast<std::uint32_t>(capacity_), std::memory_order_relaxed);
            if (capacity_ > 0) {
                available_event.set();
            }
        }

        const int rows;
        const int cols;
        const int type;
        const ExhaustPolicy policy;

        std::vector<std::unique_ptr<ImageBuffer>> buffers;
        std::vector<std::uint32_t> next;

        // Tagged freelist head to avoid ABA: low32 = index, high32 = tag.
        std::atomic<std::uint64_t> head{pack_head(kNullIndex, 0)};

        std::atomic<std::uint32_t> available{0};
        ManualResetEvent available_event;

        std::atomic<std::uint64_t> overflow_allocations{0};
        std::atomic<std::uint64_t> failed_acquisitions{0};

        static constexpr std::uint64_t
            pack_head(const std::uint32_t idx, const std::uint32_t tag) noexcept {
            return (static_cast<std::uint64_t>(tag) << 32) | static_cast<std::uint64_t>(idx);
        }

        static constexpr std::uint32_t head_index(const std::uint64_t packed) noexcept {
            return static_cast<std::uint32_t>(packed & 0xFFFF'FFFFULL);
        }

        static constexpr std::uint32_t head_tag(const std::uint64_t packed) noexcept {
            return static_cast<std::uint32_t>(packed >> 32);
        }

        [[nodiscard]] std::uint32_t pop_index() noexcept {
            std::uint64_t observed = head.load(std::memory_order_acquire);
            while (true) {
                const std::uint32_t idx = head_index(observed);
                if (idx == kNullIndex) {
                    return kNullIndex;
                }

                const std::uint32_t next_idx = next[idx];
                const std::uint64_t desired  = pack_head(next_idx, head_tag(observed) + 1);
                if (head.compare_exchange_weak(
                        observed, desired, std::memory_order_acq_rel, std::memory_order_acquire)) {
                    const auto prev = available.fetch_sub(1, std::memory_order_relaxed);
                    if (prev == 1) {
                        available_event.reset();
                    }
                    return idx;
                }
            }
        }

        void push_index(const std::uint32_t idx) noexcept {
            std::uint64_t observed = head.load(std::memory_order_acquire);
            while (true) {
                next[idx]                   = head_index(observed);
                const std::uint64_t desired = pack_head(idx, head_tag(observed) + 1);
                if (head.compare_exchange_weak(
                        observed, desired, std::memory_order_release, std::memory_order_acquire)) {
                    break;
                }
            }
            const auto prev = available.fetch_add(1, std::memory_order_relaxed);
            if (prev == 0) {
                available_event.set();
            }
        }
    };

    struct ReturnToPoolDeleter {
        std::shared_ptr<State> state;
        std::uint32_t index{kNullIndex};

        void operator()(ImageBuffer* /*ptr*/) noexcept {
            if (index != kNullIndex) {
                state->push_index(index);
            }
        }
    };

    [[nodiscard]] ImagePtr acquire_impl(const bool blocking) noexcept {
        while (true) {
            const std::uint32_t idx = state_->pop_index();
            if (idx != kNullIndex) {
                return ImagePtr(state_->buffers[idx].get(), ReturnToPoolDeleter{state_, idx});
            }

            switch (state_->policy) {
            case ExhaustPolicy::Allocate: {
                state_->overflow_allocations.fetch_add(1, std::memory_order_relaxed);
                return ImageBuffer::create(state_->rows, state_->cols, state_->type);
            }
            case ExhaustPolicy::Fail: {
                state_->failed_acquisitions.fetch_add(1, std::memory_order_relaxed);
                return {};
            }
            case ExhaustPolicy::Block: {
                if (!blocking) {
                    state_->failed_acquisitions.fetch_add(1, std::memory_order_relaxed);
                    return {};
                }
                (void)state_->available_event.wait();
                continue;
            }
            }
        }
    }

    std::shared_ptr<State> state_;
};

} // namespace primitive
