#pragma once

#include "../primitives/channel.hpp"

#include <atomic>
#include <optional>

namespace talos {

// ============================================================================
// Default Tag (used when no tag specified)
// ============================================================================

struct DefaultTag {};

// ============================================================================
// Channel Wrappers: Hold pointers to channel components
// ============================================================================

/// SPSC Reader wrapper
template <typename T, typename Tag = DefaultTag>
struct spsc {
    using value_type                 = T;
    using tag_type                   = Tag;
    using ReaderType                 = primitive::SpscChannel<T>::Reader;
    static constexpr bool is_spsc    = true;
    static constexpr bool is_spmc    = false;
    static constexpr bool is_reader  = true;
    static constexpr bool is_res     = false;
    static constexpr bool is_res_mut = false;
    static constexpr bool is_atomic  = false;

    ReaderType* ptr_ = nullptr;

    [[nodiscard]] std::optional<T> read() { return ptr_ ? ptr_->read() : std::nullopt; }
    [[nodiscard]] std::optional<T> read_current() {
        return ptr_ ? ptr_->read_current() : std::nullopt;
    }
    [[nodiscard]] bool has_new() const { return ptr_ && ptr_->has_new(); }
    [[nodiscard]] const T& current() const { return ptr_->current(); }
    [[nodiscard]] bool valid() const { return ptr_ != nullptr; }
};

/// SPSC Writer wrapper
template <typename T, typename Tag = DefaultTag>
struct spsc_mut {
    using value_type                 = T;
    using tag_type                   = Tag;
    using WriterType                 = primitive::SpscChannel<T>::Writer;
    static constexpr bool is_spsc    = true;
    static constexpr bool is_spmc    = false;
    static constexpr bool is_reader  = false;
    static constexpr bool is_res     = false;
    static constexpr bool is_res_mut = false;
    static constexpr bool is_atomic  = false;

    WriterType* ptr_    = nullptr;
    bool* written_flag_ = nullptr; // Set by scheduler before system runs

    void write(T value) {
        if (ptr_) {
            ptr_->write(std::move(value));
            if (written_flag_)
                *written_flag_ = true;
        }
    }
    [[nodiscard]] bool valid() const { return ptr_ != nullptr; }
};

/// SPMC Reader wrapper
template <typename T, typename Tag = DefaultTag>
struct spmc {
    using value_type                 = T;
    using tag_type                   = Tag;
    using ReaderType                 = primitive::SpmcChannel<T>::Reader;
    static constexpr bool is_spsc    = false;
    static constexpr bool is_spmc    = true;
    static constexpr bool is_reader  = true;
    static constexpr bool is_res     = false;
    static constexpr bool is_res_mut = false;
    static constexpr bool is_atomic  = false;

    ReaderType* ptr_ = nullptr;

    [[nodiscard]] std::optional<T> read() { return ptr_ ? ptr_->read() : std::nullopt; }
    [[nodiscard]] std::optional<T> read_current() {
        return ptr_ ? ptr_->read_current() : std::nullopt;
    }
    [[nodiscard]] bool has_new() const { return ptr_ && ptr_->has_new(); }
    [[nodiscard]] bool valid() const { return ptr_ != nullptr; }
};

/// SPMC Writer wrapper (uses channel's write directly)
template <typename T, typename Tag = DefaultTag>
struct spmc_mut {
    using value_type                 = T;
    using tag_type                   = Tag;
    using ChannelType                = primitive::SpmcChannel<T>;
    static constexpr bool is_spsc    = false;
    static constexpr bool is_spmc    = true;
    static constexpr bool is_reader  = false;
    static constexpr bool is_res     = false;
    static constexpr bool is_res_mut = false;
    static constexpr bool is_atomic  = false;

    ChannelType* ptr_   = nullptr;
    bool* written_flag_ = nullptr; // Set by scheduler before system runs

    void write(T value) {
        if (ptr_) {
            ptr_->write(std::move(value));
            if (written_flag_)
                *written_flag_ = true;
        }
    }
    [[nodiscard]] bool valid() const { return ptr_ != nullptr; }
};

/// Atomic reference wrapper
template <typename T, typename Tag = DefaultTag>
struct atomic_ref {
    using value_type                 = T;
    using tag_type                   = Tag;
    static constexpr bool is_spsc    = false;
    static constexpr bool is_spmc    = false;
    static constexpr bool is_reader  = true;
    static constexpr bool is_res     = false;
    static constexpr bool is_res_mut = false;
    static constexpr bool is_atomic  = true;

    std::atomic<T>* ptr_ = nullptr;

    [[nodiscard]] T load() const { return ptr_->load(std::memory_order_acquire); }
    void store(T value) { ptr_->store(value, std::memory_order_release); }
    [[nodiscard]] bool valid() const { return ptr_ != nullptr; }
};

// ============================================================================
// Resource Types
// ============================================================================

/// Read-only resource
template <typename T>
struct res {
    using value_type                 = T;
    using tag_type                   = void;
    static constexpr bool is_spsc    = false;
    static constexpr bool is_spmc    = false;
    static constexpr bool is_reader  = false;
    static constexpr bool is_res     = true;
    static constexpr bool is_res_mut = false;
    static constexpr bool is_atomic  = false;

    const T* ptr_ = nullptr;

    [[nodiscard]] const T& operator*() const noexcept { return *ptr_; }
    [[nodiscard]] const T* operator->() const noexcept { return ptr_; }
    [[nodiscard]] bool valid() const { return ptr_ != nullptr; }
};

/// Mutable resource
template <typename T>
struct res_mut {
    using value_type                 = T;
    using tag_type                   = void;
    static constexpr bool is_spsc    = false;
    static constexpr bool is_spmc    = false;
    static constexpr bool is_reader  = false;
    static constexpr bool is_res     = false;
    static constexpr bool is_res_mut = true;
    static constexpr bool is_atomic  = false;

    T* ptr_ = nullptr;

    [[nodiscard]] T& operator*() noexcept { return *ptr_; }
    [[nodiscard]] T* operator->() noexcept { return ptr_; }
    [[nodiscard]] const T& operator*() const noexcept { return *ptr_; }
    [[nodiscard]] const T* operator->() const noexcept { return ptr_; }
    [[nodiscard]] bool valid() const { return ptr_ != nullptr; }
};

} // namespace talos
