#pragma once

/**
 * @file
 * @brief Lock-free single-producer multiple-consumer triple buffer
 *
 * This module provides a lock-free SPMC (Single Producer Multiple Consumers)
 * triple buffer implementation optimized for low-latency data transfer
 * between threads. It also keeps a bounded backlog of recent writes so active
 * readers can catch up on short bursts instead of immediately jumping to the
 * latest generation.
 *
 * ## Design
 *
 * - **RWSpinLock**: Readers don't block each other, only writer blocks
 * - **Readers**: Acquire shared lock (atomic increment)
 * - **Writer**: Acquires exclusive lock (waits for readers to drain)
 *
 * ## Performance characteristics
 *
 * - **Write**: ~50-100ns (wait for readers, then atomic swap)
 * - **Read**: ~20-40ns (atomic increment/decrement + copy)
 * - **Readers NEVER block each other** (unlike naive spinlock)
 *
 * ## Thread safety
 *
 * All operations are thread-safe. Multiple readers can read concurrently
 * without blocking each other.
 */

#include "spin.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <optional>
#include <utility>

namespace talos::primitive {

/**
 * @brief Lock-free single-producer multiple-consumer triple buffer
 *
 * A triple buffer provides thread-safe data transfer from one writer
 * to multiple readers. Each reader operates independently and sees
 * a consistent view of the data. Active readers can drain a small backlog
 * of retained generations; readers that fall farther behind skip to the
 * oldest retained generation.
 *
 * ## Memory model
 *
 * The buffer stores `shared_ptr<const T>`, allowing zero-copy transfers
 * when the producer and consumers share ownership of the underlying data.
 *
 * ## Usage
 *
 * ```cpp
 * auto [writer, reader] = SpmcTripleBuffer<Frame>::create();
 *
 * // Producer thread:
 * writer.write(Frame{...});
 *
 * // Consumer threads (each has their own reader):
 * auto reader2 = reader.clone();
 * if (auto frame = reader2.read()) {
 *     // process frame
 * }
 * ```
 *
 * ## Thread safety
 *
 * - Single writer, multiple readers
 * - Writers block until all readers release their locks
 * - Readers never block each other
 *
 * ## Performance
 *
 * - Write: O(1) atomic operations
 * - Read: O(backlog_capacity) bounded scan
 * - Memory: backlog_capacity * sizeof(shared_ptr) + atomic overhead
 */
template <typename T>
requires(std::movable<T> && std::copyable<T>) class SpmcTripleBuffer {
public:
    static constexpr std::size_t backlog_capacity = 64;

private:
    /**
     * @brief Reader-writer spinlock optimized for many readers
     *
     * ## Encoding
     *
     * - Bit 31: writer waiting/active flag
     * - Bits 0-30: current reader count
     *
     * ## Algorithm
     *
     * - Readers increment the counter (shared lock)
     * - Writers set the writer bit and wait for counter to reach 0
     */
    class RWSpinLock {
        static constexpr std::uint32_t WRITER_BIT  = 0x80000000u;
        static constexpr std::uint32_t READER_MASK = 0x7FFFFFFFu;
        std::atomic<std::uint32_t> state_{0};

    public:
        /**
         * @brief Acquire shared lock for reading
         *
         * Spins until the writer bit is clear, then atomically
         * increments the reader count.
         *
         * ## Thread safety
         *
         * Safe to call from multiple reader threads concurrently.
         */
        void read_lock() noexcept {
            while (true) {
                auto s = state_.load(std::memory_order_relaxed);
                // If writer is waiting/active, spin
                if (s & WRITER_BIT) {
                    SPIN_HINT();
                    continue;
                }
                // Try to increment reader count
                if (state_.compare_exchange_weak(
                        s, s + 1, std::memory_order_acquire, std::memory_order_relaxed)) {
                    return;
                }
            }
        }

        void read_unlock() noexcept { state_.fetch_sub(1, std::memory_order_release); }

        void write_lock() noexcept {
            // Set writer bit
            while (true) {
                auto s = state_.load(std::memory_order_relaxed);
                if (s & WRITER_BIT) {
                    // Another writer waiting
                    SPIN_HINT();
                    continue;
                }
                if (state_.compare_exchange_weak(
                        s, s | WRITER_BIT, std::memory_order_acquire, std::memory_order_relaxed)) {
                    break;
                }
            }
            // Wait for readers to drain
            while ((state_.load(std::memory_order_acquire) & READER_MASK) != 0) {
                SPIN_HINT();
            }
        }

        void write_unlock() noexcept { state_.fetch_and(~WRITER_BIT, std::memory_order_release); }
    };

    struct State {
        struct Slot {
            std::shared_ptr<const T> value{nullptr};
            std::uint64_t generation{0};
        };

        alignas(std::hardware_destructive_interference_size) RWSpinLock lock{};
        std::shared_ptr<const T> current{nullptr};
        std::array<Slot, backlog_capacity> backlog{};
        std::size_t write_index{0};
        alignas(std::hardware_destructive_interference_size) std::atomic<std::uint64_t> generation{
            0};

        [[nodiscard]] static std::uint64_t
            oldest_retained_generation(const std::uint64_t latest_generation) noexcept {
            if (latest_generation == 0) {
                return 0;
            }

            const auto retained = std::min<std::uint64_t>(latest_generation, backlog_capacity);
            return latest_generation - retained + 1;
        }
    };

public:
    template <typename U>
    class Write;
    template <typename U>
    class Read;

    /**
     * @brief Write handle for the producer
     *
     * Owned by the single producer thread. Provides thread-safe
     * write access to the buffer.
     */
    template <typename U>
    class Write {
    public:
        Write(const Write&)                = delete;
        Write& operator=(const Write&)     = delete;
        Write(Write&&) noexcept            = default;
        Write& operator=(Write&&) noexcept = default;

        /**
         * @brief Write data to the buffer
         *
         * Atomically publishes new data for all readers.
         *
         * ## Performance
         *
         * May spin briefly waiting for readers to release their locks.
         * The write operation itself is O(1).
         *
         * ## Thread safety
         *
         * Must only be called from the single producer thread.
         *
         * ## Panics
         *
         * Never panics. The operation is guaranteed to complete.
         */
        void write(U data) noexcept {
            auto ptr = std::make_shared<const U>(std::move(data));

            state_->lock.write_lock();
            const auto next_generation = state_->generation.load(std::memory_order_relaxed) + 1;
            auto& slot                 = state_->backlog[state_->write_index];
            slot.value                 = ptr;
            slot.generation            = next_generation;
            state_->current            = std::move(ptr);
            state_->write_index        = (state_->write_index + 1) % backlog_capacity;
            state_->generation.store(next_generation, std::memory_order_release);
            state_->lock.write_unlock();
        }

        [[nodiscard]] std::uint64_t generation() const noexcept {
            return state_->generation.load(std::memory_order_acquire);
        }

        [[nodiscard]] bool valid() const noexcept { return state_ != nullptr; }
        explicit operator bool() const noexcept { return valid(); }

    private:
        friend class SpmcTripleBuffer;
        explicit Write(std::shared_ptr<State> s)
            : state_(std::move(s)) {}

        std::shared_ptr<State> state_;
    };

    /**
     * @brief Read handle for consumers
     *
     * Copyable and movable. Each reader maintains its own generation
     * counter to track which data it has already seen.
     *
     * ## Thread safety
     *
     * Safe to use from any thread. Multiple readers can read
     * concurrently without blocking each other.
     */
    template <typename U>
    class Read {
    public:
        Read(const Read& other)
            : state_(other.state_)
            , last_gen_(other.last_gen_) {}

        Read& operator=(const Read& other) {
            if (this != &other) {
                state_    = other.state_;
                last_gen_ = other.last_gen_;
            }
            return *this;
        }

        Read(Read&&) noexcept            = default;
        Read& operator=(Read&&) noexcept = default;

        /**
         * @brief Clone this reader
         *
         * Creates a new reader that starts with the same generation
         * counter as this reader.
         *
         * ## Performance
         *
         * O(1) - just copies the shared state pointer.
         */
        [[nodiscard]] Read clone() const { return *this; }

        /**
         * @brief Read new data if available
         *
         * Returns new data if it hasn't been seen by this reader,
         * otherwise returns `std::nullopt`.
         *
         * ## Algorithm
         *
         * 1. Fast path: check generation counter without locking
         * 2. Slow path: acquire read lock and find the next retained generation
         * 3. Copy data out if new
         *
         * ## Thread safety
         *
         * Safe to call from any thread.
         *
         * ## Returns
         *
         * - `std::optional<U>` containing new data if available
         * - `std::nullopt` if no new data since last read
         */
        [[nodiscard]] std::optional<U> read() noexcept {
            // Fast path: check generation without lock
            if (const std::uint64_t gen = state_->generation.load(std::memory_order_acquire);
                gen <= last_gen_) {
                return std::nullopt;
            }

            // Acquire read lock (doesn't block other readers)
            state_->lock.read_lock();

            const std::uint64_t locked_gen = state_->generation.load(std::memory_order_acquire);
            std::uint64_t target_gen       = locked_gen;

            if (last_gen_ != 0 && locked_gen > last_gen_) {
                const auto oldest_gen = State::oldest_retained_generation(locked_gen);
                target_gen            = std::max(last_gen_ + 1, oldest_gen);
            }

            std::shared_ptr<const U> ptr{nullptr};
            for (const auto& slot : state_->backlog) {
                if (slot.generation == target_gen) {
                    ptr = slot.value;
                    break;
                }
            }
            if (!ptr) [[unlikely]] {
                ptr        = state_->current;
                target_gen = locked_gen;
            }

            state_->lock.read_unlock();

            if (!ptr) {
                return std::nullopt;
            }

            // Check generation again (might have been stale in fast path)
            if (target_gen <= last_gen_) {
                return std::nullopt;
            }

            last_gen_ = target_gen;
            return *ptr; // copy out
        }

        [[nodiscard]] std::optional<U> read_current() noexcept {
            state_->lock.read_lock();

            std::shared_ptr<const U> ptr = state_->current;
            std::uint64_t locked_gen     = state_->generation.load(std::memory_order_acquire);

            state_->lock.read_unlock();

            if (!ptr) {
                return std::nullopt;
            }

            last_gen_ = locked_gen;
            return *ptr;
        }

        [[nodiscard]] bool has_new() const noexcept {
            return state_->generation.load(std::memory_order_acquire) > last_gen_;
        }

        [[nodiscard]] std::uint64_t last_generation() const noexcept { return last_gen_; }

        [[nodiscard]] bool valid() const noexcept { return state_ != nullptr; }
        explicit operator bool() const noexcept { return valid(); }

    private:
        friend class SpmcTripleBuffer;
        explicit Read(std::shared_ptr<State> s)
            : state_(std::move(s)) {}

        std::shared_ptr<State> state_;
        std::uint64_t last_gen_{0};
    };

    SpmcTripleBuffer()                                   = delete;
    SpmcTripleBuffer(const SpmcTripleBuffer&)            = delete;
    SpmcTripleBuffer& operator=(const SpmcTripleBuffer&) = delete;

    [[nodiscard]] static std::pair<Write<T>, Read<T>> create() {
        auto state = std::make_shared<State>();
        return {Write<T>(state), Read<T>(state)};
    }
};

} // namespace talos::primitive
