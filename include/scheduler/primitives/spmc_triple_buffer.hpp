#pragma once

#include <atomic>
#include <memory>
#include <new>
#include <optional>
#include <utility>

namespace primitive {

/**
 * @brief Lock-free SPMC triple buffer
 *
 * - Write<T>: move-only, single producer, move in
 * - Read<T>: copyable, multiple consumers, copy out
 * - Overwrite automatically drops old data
 *
 * Uses spinlock to protect shared_ptr (Apple Clang doesn't support atomic<shared_ptr>)
 */
template <typename T>
class SpmcTripleBuffer {
private:
    class SpinLock {
    public:
        void lock() noexcept {
            while (flag_.test_and_set(std::memory_order_acquire)) {
                // spin
            }
        }
        void unlock() noexcept { flag_.clear(std::memory_order_release); }

    private:
        std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
    };

    struct State {
        alignas(std::hardware_destructive_interference_size) SpinLock lock;
        std::shared_ptr<const T> current{nullptr};
        alignas(std::hardware_destructive_interference_size) std::atomic<uint64_t> generation{0};
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
         * @brief Write data (move in), old data is dropped
         */
        void write(U data) noexcept {
            auto ptr = std::make_shared<const U>(std::move(data));
            {
                state_->lock.lock();
                state_->current = std::move(ptr);
                state_->lock.unlock();
            }
            state_->generation.fetch_add(1, std::memory_order_release);
        }

        /**
         * @brief Current generation
         */
        [[nodiscard]] uint64_t generation() const noexcept {
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
         * @brief Clone for another consumer
         */
        [[nodiscard]] Read clone() const { return *this; }

        /**
         * @brief Try read (copy out) if new data available
         * @return Copy of data if new, nullopt otherwise
         */
        [[nodiscard]] std::optional<U> read() noexcept {
            const uint64_t gen = state_->generation.load(std::memory_order_acquire);
            if (gen <= last_gen_) {
                return std::nullopt;
            }

            std::shared_ptr<const U> ptr;
            {
                state_->lock.lock();
                ptr = state_->current;
                state_->lock.unlock();
            }

            if (!ptr) {
                return std::nullopt;
            }

            last_gen_ = gen;
            return *ptr; // copy out
        }

        /**
         * @brief Force read current data (may be stale)
         */
        [[nodiscard]] std::optional<U> read_current() noexcept {
            std::shared_ptr<const U> ptr;
            {
                state_->lock.lock();
                ptr = state_->current;
                state_->lock.unlock();
            }

            if (!ptr) {
                return std::nullopt;
            }
            last_gen_ = state_->generation.load(std::memory_order_acquire);
            return *ptr;
        }

        /**
         * @brief Check if new data available
         */
        [[nodiscard]] bool has_new() const noexcept {
            return state_->generation.load(std::memory_order_acquire) > last_gen_;
        }

        /**
         * @brief Last read generation
         */
        [[nodiscard]] uint64_t last_generation() const noexcept { return last_gen_; }

        [[nodiscard]] bool valid() const noexcept { return state_ != nullptr; }
        explicit operator bool() const noexcept { return valid(); }

    private:
        friend class SpmcTripleBuffer;
        explicit Read(std::shared_ptr<State> s)
            : state_(std::move(s)) {}
        std::shared_ptr<State> state_;
        uint64_t last_gen_{0};
    };

    SpmcTripleBuffer()                                   = delete;
    SpmcTripleBuffer(const SpmcTripleBuffer&)            = delete;
    SpmcTripleBuffer& operator=(const SpmcTripleBuffer&) = delete;

    /**
     * @brief Create buffer, returns {Write<T>, Read<T>}
     */
    [[nodiscard]] static std::pair<Write<T>, Read<T>> create() {
        auto state = std::make_shared<State>();
        return {Write<T>(state), Read<T>(state)};
    }
};

} // namespace primitive
