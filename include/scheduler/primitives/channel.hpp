#pragma once

#include "flow_graph.hpp"
#include "spmc_triple_buffer.hpp"
#include "spsc_triple_buffer.hpp"

#include <concepts>
#include <optional>
#include <type_traits>
#include <utility>

namespace primitive {

// ============================================================================
// Concepts: detect buffer type capabilities
// ============================================================================

namespace detail {

template <typename Buffer, typename T>
concept HasClone = requires(typename Buffer::template Read<T>& r) {
    { r.clone() } -> std::same_as<typename Buffer::template Read<T>>;
};

template <typename Buffer, typename T>
concept SpmcLike = HasClone<Buffer, T>;

template <typename Buffer, typename T>
concept SpscLike = !HasClone<Buffer, T>;

} // namespace detail

// ============================================================================
// TrackedWriter: wraps any buffer's Write<T>
// ============================================================================

template <typename T, typename Buffer>
class TrackedWriter : detail::LocStorage {
    using Inner = Buffer::template Write<T>;
    Inner inner_;

public:
    using value_type  = T;
    using buffer_type = Buffer;

    TrackedWriter(Inner w, const std::string_view file, const int line) noexcept
        : LocStorage(file, line)
        , inner_(std::move(w)) {}

    TrackedWriter(const TrackedWriter&)            = delete;
    TrackedWriter& operator=(const TrackedWriter&) = delete;
    TrackedWriter(TrackedWriter&&) noexcept        = default;
    TrackedWriter& operator=(TrackedWriter&&)      = default;

    void write(T data) noexcept { inner_.write(std::move(data)); }

    // SPSC only: borrow_mut + publish
    auto& borrow_mut() noexcept requires requires(Inner& w) { w.borrow_mut(); } {
        return inner_.borrow_mut();
    }

    void publish() noexcept requires requires(Inner& w) { w.publish(); } { inner_.publish(); }

    // SPMC only: generation
    [[nodiscard]] auto generation() const noexcept
        requires requires(const Inner& w) { w.generation(); } {
        return inner_.generation();
    }

    [[nodiscard]] bool valid() const noexcept { return inner_.valid(); }
    explicit operator bool() const noexcept { return valid(); }

    [[nodiscard]] NodeId location() const noexcept { return this->loc(); }

    // Access inner (for advanced use)
    [[nodiscard]] Inner& inner() noexcept { return inner_; }
    [[nodiscard]] const Inner& inner() const noexcept { return inner_; }
};

// ============================================================================
// TrackedReader: wraps any buffer's Read<T>
// ============================================================================

template <typename T, typename Buffer>
class TrackedReader : detail::LocStorage {
    using Inner = Buffer::template Read<T>;
    Inner inner_;

public:
    using value_type  = T;
    using buffer_type = Buffer;

    TrackedReader(Inner r, const std::string_view file, const int line) noexcept
        : LocStorage(file, line)
        , inner_(std::move(r)) {}

    // SPMC: copyable
    TrackedReader(const TrackedReader& other) requires detail::SpmcLike<Buffer, T>
        : LocStorage(other.loc().file, other.loc().line)
        , inner_(other.inner_) {}

    TrackedReader& operator=(const TrackedReader& other) requires detail::SpmcLike<Buffer, T> {
        if (this != &other) {
            static_cast<LocStorage&>(*this) = LocStorage(other.loc().file, other.loc().line);
            inner_                          = other.inner_;
        }
        return *this;
    }

    // SPSC: move-only
    TrackedReader(const TrackedReader&) requires detail::SpscLike<Buffer, T>            = delete;
    TrackedReader& operator=(const TrackedReader&) requires detail::SpscLike<Buffer, T> = delete;

    TrackedReader(TrackedReader&&) noexcept            = default;
    TrackedReader& operator=(TrackedReader&&) noexcept = default;

    [[nodiscard]] std::optional<T> read() noexcept { return inner_.read(); }

    [[nodiscard]] bool has_new() const noexcept { return inner_.has_new(); }

    // SPMC only: read_current, last_generation
    [[nodiscard]] auto read_current() noexcept requires requires(Inner& r) { r.read_current(); } {
        return inner_.read_current();
    }

    [[nodiscard]] auto last_generation() const noexcept
        requires requires(const Inner& r) { r.last_generation(); } {
        return inner_.last_generation();
    }

    // SPSC only: current (const ref, no ownership transfer)
    [[nodiscard]] const T& current() const noexcept
        requires requires(const Inner& r) { r.current(); } {
        return inner_.current();
    }

    [[nodiscard]] bool valid() const noexcept { return inner_.valid(); }
    explicit operator bool() const noexcept { return valid(); }

    [[nodiscard]] NodeId location() const noexcept { return this->loc(); }

    // Access inner
    [[nodiscard]] Inner& inner() noexcept { return inner_; }
    [[nodiscard]] const Inner& inner() const noexcept { return inner_; }
};

// ============================================================================
// TrackedChannel: unified wrapper for any triple buffer
// ============================================================================

template <typename T, typename Buffer>
class TrackedChannel : detail::LocStorage {
    using WriteInner = Buffer::template Write<T>;
    using ReadInner  = Buffer::template Read<T>;

    WriteInner writer_;
    ReadInner reader_;

public:
    using value_type  = T;
    using buffer_type = Buffer;
    using Writer      = TrackedWriter<T, Buffer>;
    using Reader      = TrackedReader<T, Buffer>;

    static constexpr bool is_spmc = detail::SpmcLike<Buffer, T>;
    static constexpr bool is_spsc = detail::SpscLike<Buffer, T>;

    TrackedChannel(WriteInner w, ReadInner r, const std::string_view file, const int line) noexcept
        : LocStorage(file, line)
        , writer_(std::move(w))
        , reader_(std::move(r)) {}

    TrackedChannel(const TrackedChannel&)            = delete;
    TrackedChannel& operator=(const TrackedChannel&) = delete;
    TrackedChannel(TrackedChannel&&) noexcept        = default;
    TrackedChannel& operator=(TrackedChannel&&)      = default;

    // ========================================================================
    // Writer interface (for convenience, can also use split())
    // ========================================================================

    void write(T data) noexcept { writer_.write(std::move(data)); }

    auto& borrow_mut() noexcept requires is_spsc { return writer_.borrow_mut(); }

    void publish() noexcept requires is_spsc { writer_.publish(); }

    [[nodiscard]] auto generation() const noexcept requires is_spmc { return writer_.generation(); }

    // ========================================================================
    // Clone reader (SPMC only): records edge
    // ========================================================================

    [[nodiscard]] Reader
        clone_reader(std::string_view file = __builtin_FILE(), int line = __builtin_LINE()) const
        requires is_spmc {
        register_flow_edge(this->loc(), NodeId{file, line}, typeid(T).name());
        return Reader(reader_.clone(), file, line);
    }

    // ========================================================================
    // Take reader (both): consumes the internal reader, records edge
    // ========================================================================

    [[nodiscard]] Reader
        take_reader(std::string_view file = __builtin_FILE(), int line = __builtin_LINE()) {
        register_flow_edge(this->loc(), NodeId{file, line}, typeid(T).name());
        return Reader(std::move(reader_), file, line);
    }

    // ========================================================================
    // Split: returns (Writer, Reader), records edge
    // ========================================================================

    struct SplitResult {
        Writer writer;
        Reader reader;
    };

    [[nodiscard]] SplitResult
        split(std::string_view file = __builtin_FILE(), int line = __builtin_LINE()) {
        register_flow_edge(this->loc(), NodeId{file, line}, typeid(T).name());
        return SplitResult{
            Writer(std::move(writer_), this->loc().file, this->loc().line),
            Reader(std::move(reader_), file, line)};
    }

    // ========================================================================
    // Location info
    // ========================================================================

    [[nodiscard]] NodeId producer_location() const noexcept { return this->loc(); }
};

// ============================================================================
// Factory function: tracked_create<Buffer, T>()
// Usage: tracked_create<SpmcTripleBuffer, Frame>()
// ============================================================================

template <template <typename> class Buffer, typename T>
[[nodiscard]] auto
    tracked_create(std::string_view file = __builtin_FILE(), int line = __builtin_LINE())
        -> TrackedChannel<T, Buffer<T>> {
    auto [w, r] = Buffer<T>::create();
    return TrackedChannel<T, Buffer<T>>(std::move(w), std::move(r), file, line);
}

// ============================================================================
// Type aliases for convenience
// ============================================================================

template <typename T>
using SpmcChannel = TrackedChannel<T, SpmcTripleBuffer<T>>;

template <typename T>
using SpscChannel = TrackedChannel<T, SpscTripleBuffer<T>>;

// Convenience factory functions
template <typename T>
[[nodiscard]] auto make_spmc_channel(
    const std::string_view file = __builtin_FILE(), const int line = __builtin_LINE())
    -> SpmcChannel<T> {
    return tracked_create<SpmcTripleBuffer, T>(file, line);
}

template <typename T>
[[nodiscard]] auto make_spsc_channel(
    const std::string_view file = __builtin_FILE(), const int line = __builtin_LINE())
    -> SpscChannel<T> {
    return tracked_create<SpscTripleBuffer, T>(file, line);
}

// ============================================================================
// Legacy alias: Channel<T> = SpmcChannel<T> for backward compatibility
// ============================================================================

template <typename T>
using Channel = SpmcChannel<T>;

} // namespace primitive
