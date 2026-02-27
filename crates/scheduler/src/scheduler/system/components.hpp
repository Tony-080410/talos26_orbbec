#pragma once

#include "primitive/channel.hpp"
#include <atomic>
#include <optional>

namespace talos::scheduler::system {
template <typename T>
concept topic_type = std::is_same_v<T, void> || requires { sizeof(T) == 0; };

template <typename T>
concept component_kind = requires {
    typename T::value_type;
    typename T::topic_type;
    requires !std::is_void_v<typename T::value_type>;
    requires topic_type<typename T::topic_type>;
};

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

    static constexpr auto kind = channel_kind::spsc_reader;

    ReaderType* ptr_ = nullptr;

    [[nodiscard]] std::optional<T> read() noexcept { return ptr_ ? ptr_->read() : std::nullopt; }
    [[nodiscard]] std::optional<T> read_current() noexcept {
        return ptr_ ? ptr_->read_current() : std::nullopt;
    }
    [[nodiscard]] bool valid() const noexcept { return ptr_ != nullptr; }
};

/// SPSC Writer wrapper
template <typename T, typename Topic = DefaultTopic>
struct spsc_mut {
    using value_type = T;
    using topic_type = Topic;
    using WriterType = primitive::SpscChannel<T>::Writer;

    static constexpr auto kind = channel_kind::spsc_writer;

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

    static constexpr auto kind = channel_kind::spmc_reader;

    ReaderType* ptr_ = nullptr;

    [[nodiscard]] std::optional<T> read() noexcept {
        if (ptr_) [[likely]] {
            return ptr_->read();
        }
        return std::nullopt;
    }
    [[nodiscard]] std::optional<T> read_current() noexcept {
        if (ptr_) [[likely]] {
            return ptr_->read_current();
        }
        return std::nullopt;
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

    static constexpr auto kind = channel_kind::spmc_writer;

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

    static constexpr auto kind = channel_kind::local;

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

/// Read-only resource wrapper
template <typename T>
struct res {
    using value_type = T;
    using topic_type = void;

    static constexpr auto kind = channel_kind::res;

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
        if (const auto current = version_->load(std::memory_order_acquire);
            current != last_seen_version_) {
            last_seen_version_ = current;
            return true;
        }
        return false;
    }
};

/// Mutable resource wrapper
/// Bumps version on every non-const access so res holders see is_changed().
template <typename T>
struct res_mut {
    using value_type = T;
    using topic_type = void;

    static constexpr auto kind = channel_kind::res_mut;

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
        if (const auto current = version_->load(std::memory_order_acquire);
            current != last_seen_version_) {
            last_seen_version_ = current;
            return true;
        }
        return false;
    }
};

static_assert(component_kind<spsc<bool>>);
static_assert(component_kind<spsc_mut<bool>>);
static_assert(component_kind<spmc<bool>>);
static_assert(component_kind<spmc_mut<bool>>);
static_assert(component_kind<res<bool>>);
static_assert(component_kind<res_mut<bool>>);
static_assert(component_kind<local<bool>>);

} // namespace talos::scheduler::system