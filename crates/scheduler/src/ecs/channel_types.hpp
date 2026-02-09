#pragma once

#include "../primitives/channel.hpp"
#include <atomic>
#include <optional>

namespace talos {

// ============================================================================
// Default Topic (used when no topic specified)
// ============================================================================

struct DefaultTopic {};

// ============================================================================
// Channel Kind: Single enum for type identification
// ============================================================================

enum class channel_kind {
    spsc_reader, // SPSC reader
    spsc_writer, // SPSC writer
    spmc_reader, // SPMC reader
    spmc_writer, // SPMC writer
    res,         // Read-only resource
    res_mut,     // Mutable resource
    atomic,      // Atomic variable
    local,       // System-local variable
};

// ============================================================================
// Channel Wrappers: Hold pointers to channel components
// ============================================================================

/// SPSC Reader wrapper
template <typename T, typename Topic = DefaultTopic>
struct spsc {
    using value_type = T;
    using topic_type = Topic;
    using ReaderType = primitive::SpscChannel<T>::Reader;

    static constexpr channel_kind kind = channel_kind::spsc_reader;

    ReaderType* ptr_ = nullptr;

    [[nodiscard]] std::optional<T> read() noexcept { return ptr_ ? ptr_->read() : std::nullopt; }
    [[nodiscard]] std::optional<T> read_current() noexcept {
        return ptr_ ? ptr_->read_current() : std::nullopt;
    }
    [[nodiscard]] bool has_new() const noexcept { return ptr_ && ptr_->has_new(); }
    [[nodiscard]] const T& current() const noexcept { return ptr_->current(); }
    [[nodiscard]] bool valid() const noexcept { return ptr_ != nullptr; }
};

/// SPSC Writer wrapper
template <typename T, typename Topic = DefaultTopic>
struct spsc_mut {
    using value_type = T;
    using topic_type = Topic;
    using WriterType = primitive::SpscChannel<T>::Writer;

    static constexpr channel_kind kind = channel_kind::spsc_writer;

    WriterType* ptr_    = nullptr;
    bool* written_flag_ = nullptr; // Set by scheduler before system runs

    void write(T value) noexcept {
        if (ptr_) [[likely]] {
            ptr_->write(std::move(value));
            if (written_flag_)
                *written_flag_ = true;
        }
    }
    [[nodiscard]] bool valid() const noexcept { return ptr_ != nullptr; }
};

/// SPMC Reader wrapper
template <typename T, typename Topic = DefaultTopic>
struct spmc {
    using value_type = T;
    using topic_type = Topic;
    using ReaderType = primitive::SpmcChannel<T>::Reader;

    static constexpr channel_kind kind = channel_kind::spmc_reader;

    ReaderType* ptr_ = nullptr;

    [[nodiscard]] std::optional<T> read() noexcept {
        if (ptr_) [[likely]] {
            return ptr_->read();
        } else {
            return std::nullopt;
        }
    }
    [[nodiscard]] std::optional<T> read_current() noexcept {
        if (ptr_) [[likely]] {
            return ptr_->read_current();
        } else {
            return std::nullopt;
        }
    }
    [[nodiscard]] bool has_new() const noexcept { return ptr_ && ptr_->has_new(); }
    [[nodiscard]] bool valid() const noexcept { return ptr_ != nullptr; }
};

/// SPMC Writer wrapper (uses channel's write directly)
template <typename T, typename Topic = DefaultTopic>
struct spmc_mut {
    using value_type  = T;
    using topic_type  = Topic;
    using ChannelType = primitive::SpmcChannel<T>;

    static constexpr channel_kind kind = channel_kind::spmc_writer;

    ChannelType* ptr_   = nullptr;
    bool* written_flag_ = nullptr; // Set by scheduler before system runs

    void write(T value) noexcept {
        if (ptr_) [[likely]] {
            ptr_->write(std::move(value));
            if (written_flag_)
                *written_flag_ = true;
        }
    }
    [[nodiscard]] bool valid() const noexcept { return ptr_ != nullptr; }
};

/// Atomic reference wrapper
template <typename T, typename Topic = DefaultTopic>
struct atomic_ref {
    using value_type = T;
    using topic_type = Topic;

    static constexpr channel_kind kind = channel_kind::atomic;

    std::atomic<T>* ptr_ = nullptr;

    [[nodiscard]] T load() const noexcept { return ptr_->load(std::memory_order_acquire); }
    void store(T value) noexcept { ptr_->store(value, std::memory_order_release); }
    [[nodiscard]] bool valid() const noexcept { return ptr_ != nullptr; }

    // Transparent access: forward . and -> to std::atomic<T>
    [[nodiscard]] std::atomic<T>* operator->() noexcept { return ptr_; }
    [[nodiscard]] const std::atomic<T>* operator->() const noexcept { return ptr_; }
    [[nodiscard]] std::atomic<T>& operator*() noexcept { return *ptr_; }
    [[nodiscard]] const std::atomic<T>& operator*() const noexcept { return *ptr_; }
};

// ============================================================================
// Local Variables (per-system storage)
// ============================================================================

/// System-local variable (mutable, per-system instance)
/// Unlike channels and resources, local<T> is NOT shared across systems
/// and does NOT participate in dependency analysis.
template <typename T>
struct local {
    using value_type = T;
    using topic_type = void;

    static constexpr channel_kind kind = channel_kind::local;

    T* ptr_ = nullptr;

    [[nodiscard]] T& operator*() noexcept { return *ptr_; }
    [[nodiscard]] T* operator->() noexcept { return ptr_; }
    [[nodiscard]] const T& operator*() const noexcept { return *ptr_; }
    [[nodiscard]] const T* operator->() const noexcept { return ptr_; }
    [[nodiscard]] bool valid() const noexcept { return ptr_ != nullptr; }
};

// ============================================================================
// Resource Types
// ============================================================================

/// Read-only resource
template <typename T>
struct res {
    using value_type = T;
    using topic_type = void;

    static constexpr channel_kind kind = channel_kind::res;

    const T* ptr_                         = nullptr;
    const std::atomic<uint64_t>* version_ = nullptr;
    mutable uint64_t last_seen_version_   = 0;

    [[nodiscard]] const T& operator*() const noexcept {
        if (version_) [[likely]]
            last_seen_version_ = version_->load(std::memory_order_acquire);
        return *ptr_;
    }
    [[nodiscard]] const T* operator->() const noexcept {
        if (version_) [[likely]]
            last_seen_version_ = version_->load(std::memory_order_acquire);
        return ptr_;
    }
    [[nodiscard]] bool valid() const noexcept { return ptr_ != nullptr; }

    [[nodiscard]] bool is_changed() const noexcept {
        if (!version_)
            return false;
        uint64_t current = version_->load(std::memory_order_acquire);
        if (current != last_seen_version_) {
            last_seen_version_ = current;
            return true;
        }
        return false;
    }
};

/// Mutable resource
template <typename T>
struct res_mut {
    using value_type = T;
    using topic_type = void;

    static constexpr channel_kind kind = channel_kind::res_mut;

    T* ptr_                             = nullptr;
    std::atomic<uint64_t>* version_     = nullptr;
    mutable uint64_t last_seen_version_ = 0;

    [[nodiscard]] T& operator*() noexcept {
        if (version_) [[likely]]
            version_->fetch_add(1, std::memory_order_release);
        return *ptr_;
    }
    [[nodiscard]] T* operator->() noexcept {
        if (version_) [[likely]]
            version_->fetch_add(1, std::memory_order_release);
        return ptr_;
    }
    [[nodiscard]] const T& operator*() const noexcept { return *ptr_; }
    [[nodiscard]] const T* operator->() const noexcept { return ptr_; }
    [[nodiscard]] bool valid() const noexcept { return ptr_ != nullptr; }

    [[nodiscard]] bool is_changed() const noexcept {
        if (!version_)
            return false;
        uint64_t current = version_->load(std::memory_order_acquire);
        if (current != last_seen_version_) {
            last_seen_version_ = current;
            return true;
        }
        return false;
    }

    // Transparent access: forward . and -> to T
    template <typename U>
    [[nodiscard]] auto operator->*(U T::* member) noexcept -> decltype(&(ptr_->*member)) {
        if (version_) [[likely]]
            version_->fetch_add(1, std::memory_order_release);
        return &(ptr_->*member);
    }

    template <typename U>
    [[nodiscard]] auto operator->*(U T::* member) const noexcept -> decltype(&(ptr_->*member)) {
        return &(ptr_->*member);
    }
};

} // namespace talos
