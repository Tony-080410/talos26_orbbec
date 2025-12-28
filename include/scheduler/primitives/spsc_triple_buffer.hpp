#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <new>
#include <optional>
#include <utility>

namespace primitive {

/**
 * @brief Lock-free SPSC triple buffer
 *
 * - Write<T>: move-only, single producer
 * - Read<T>: move-only, single consumer
 * - Data is moved in and moved out (zero-copy for movable types)
 */
template <typename T>
class SpscTripleBuffer {
private:
    static constexpr uint8_t FLAG_NEW   = 0x80;
    static constexpr uint8_t INDEX_MASK = 0x03;

    struct State {
        alignas(std::hardware_destructive_interference_size) std::array<T, 3> slots{};
        alignas(std::hardware_destructive_interference_size) uint8_t write_idx{0};
        alignas(std::hardware_destructive_interference_size) std::atomic<uint8_t> shared{1};
        alignas(std::hardware_destructive_interference_size) uint8_t read_idx{2};
    };

public:
    template <typename U>
    class Write;
    template <typename U>
    class Read;

    template <typename U>
    class Write {
    public:
        Write(const Write&)                = delete;
        Write& operator=(const Write&)     = delete;
        Write(Write&&) noexcept            = default;
        Write& operator=(Write&&) noexcept = default;

        /**
         * @brief Write data (move in)
         */
        void write(U data) noexcept {
            state_->slots[state_->write_idx] = std::move(data);
            publish();
        }

        /**
         * @brief Borrow mutable reference to write slot
         */
        [[nodiscard]] U& borrow_mut() noexcept { return state_->slots[state_->write_idx]; }

        /**
         * @brief Publish current write slot
         */
        void publish() noexcept {
            const auto old =
                state_->shared.exchange(state_->write_idx | FLAG_NEW, std::memory_order_acq_rel);
            state_->write_idx = old & INDEX_MASK;
        }

        [[nodiscard]] bool valid() const noexcept { return state_ != nullptr; }
        explicit operator bool() const noexcept { return valid(); }

    private:
        friend class SpscTripleBuffer;
        explicit Write(std::shared_ptr<State> s)
            : state_(std::move(s)) {}
        std::shared_ptr<State> state_;
    };

    template <typename U>
    class Read {
    public:
        Read(const Read&)                = delete;
        Read& operator=(const Read&)     = delete;
        Read(Read&&) noexcept            = default;
        Read& operator=(Read&&) noexcept = default;

        /**
         * @brief Try read (move out)
         * @return Data if available, nullopt if no new data
         */
        [[nodiscard]] std::optional<U> read() noexcept {
            auto expected = state_->shared.load(std::memory_order_acquire);
            if (!(expected & FLAG_NEW)) {
                return std::nullopt;
            }

            auto ready_idx = expected & INDEX_MASK;
            if (!state_->shared.compare_exchange_strong(
                    expected, state_->read_idx, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                if (!(expected & FLAG_NEW)) {
                    return std::nullopt;
                }
                ready_idx = expected & INDEX_MASK;
                if (!state_->shared.compare_exchange_strong(
                        expected, state_->read_idx, std::memory_order_acq_rel,
                        std::memory_order_relaxed)) {
                    return std::nullopt;
                }
            }

            state_->read_idx = ready_idx;
            return std::move(state_->slots[state_->read_idx]);
        }

        /**
         * @brief Check if new data available
         */
        [[nodiscard]] bool has_new() const noexcept {
            return state_->shared.load(std::memory_order_acquire) & FLAG_NEW;
        }

        /**
         * @brief Get current data (may be stale, no ownership transfer)
         */
        [[nodiscard]] const U& current() const noexcept { return state_->slots[state_->read_idx]; }

        [[nodiscard]] bool valid() const noexcept { return state_ != nullptr; }
        explicit operator bool() const noexcept { return valid(); }

    private:
        friend class SpscTripleBuffer;
        explicit Read(std::shared_ptr<State> s)
            : state_(std::move(s)) {}
        std::shared_ptr<State> state_;
    };

    SpscTripleBuffer()                                   = delete;
    SpscTripleBuffer(const SpscTripleBuffer&)            = delete;
    SpscTripleBuffer& operator=(const SpscTripleBuffer&) = delete;

    /**
     * @brief Create buffer, returns {Write<T>, Read<T>}
     */
    [[nodiscard]] static std::pair<Write<T>, Read<T>> create() {
        auto state = std::make_shared<State>();
        return {Write<T>(state), Read<T>(state)};
    }
};

} // namespace primitive
