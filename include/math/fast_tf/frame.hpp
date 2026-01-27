#pragma once

#include "buffer.hpp"
#include "matrix.hpp"
#include "types.hpp"
#include <string_view>
#include <utility>

namespace fast_tf {

constexpr size_t _base = __COUNTER__;

#define DECL_ROOT(name)                                                       \
    struct name##_fuxk_frame {                                                \
        static constexpr std::string_view frame_id = #name;                   \
        static constexpr size_t index              = __COUNTER__ - _base - 1; \
        using ancestor                             = void;                    \
    };

#define DECL(name, parent)                                                    \
    struct name##_fuxk_frame {                                                \
        static constexpr std::string_view frame_id = #name;                   \
        static constexpr size_t index              = __COUNTER__ - _base - 1; \
        using ancestor                             = parent##_fuxk_frame;     \
    };

DECL_ROOT(world);
DECL(odom, world);
DECL(gimbal_link, odom);
DECL(camera_link, gimbal_link);
struct camera_optical_fuxk_frame {
    static constexpr std::string_view frame_id = "camera_optical_frame";
    static constexpr size_t index              = __COUNTER__ - _base - 1;
    using ancestor                             = camera_link_fuxk_frame;
};
;
DECL(muzzle_link, gimbal_link);

// 防止外部使用
#undef DECL_ROOT
#undef DECL

using odom           = odom_fuxk_frame;
using gimbal         = gimbal_link_fuxk_frame;
using camera         = camera_link_fuxk_frame;
using camera_optical = camera_optical_fuxk_frame;
using muzzle         = muzzle_link_fuxk_frame;

constexpr size_t frame_count = __COUNTER__ - _base - 1;

template <typename T>
concept frame = requires {
    { T::frame_id } -> std::convertible_to<std::string_view>;
    { T::index } -> std::convertible_to<size_t>;
    typename T::ancestor;
};

template <typename T>
concept root_frame = frame<T> && std::is_void_v<typename T::ancestor>;

template <typename T>
concept non_root_frame = frame<T> && !std::is_void_v<typename T::ancestor>;

template <frame Child, frame Ancestor>
constexpr bool is_descendant_of() {
    if constexpr (std::is_same_v<Child, Ancestor>) {
        return true;
    } else if constexpr (std::is_void_v<typename Child::ancestor>) {
        return false;
    } else {
        return is_descendant_of<typename Child::ancestor, Ancestor>();
    }
}

template <frame Coordinate, typename T>
struct [[nodiscard]] Fuck {
    explicit Fuck(T t)
        : val(std::move(t)) {}
    explicit Fuck() = default;

    [[nodiscard]] const T& operator*() const noexcept { return val; }
    [[nodiscard]] T& operator*() noexcept { return val; }

    [[nodiscard]] const T* operator->() const noexcept { return &val; }
    [[nodiscard]] T* operator->() noexcept { return &val; }

    [[nodiscard]] auto operator*(const T& other) const noexcept { return this * *other; }

    template <frame OtherCoord>
    auto operator*(const Fuck<OtherCoord, T>& other) const {
        auto prod = val * (*other);
        return Fuck<Coordinate, decltype(prod)>(prod);
    }

    template <frame OtherCoord>
    auto operator+(const Fuck<OtherCoord, T>& other) const {
        auto prod = val + (*other);
        return Fuck<Coordinate, decltype(prod)>(prod);
    }

    template <frame OtherCoord>
    auto operator-(const Fuck<OtherCoord, T>& other) const {
        auto prod = val - (*other);
        return Fuck<Coordinate, decltype(prod)>(prod);
    }

    template <typename... Args>
    constexpr auto operator()(Args&&... args) const
        noexcept(noexcept(val(std::forward<Args>(args)...)))
            -> decltype(val(std::forward<Args>(args)...)) {
        return val(std::forward<Args>(args)...);
    }

private:
    T val;
};

template <typename Coordinate, typename T>
inline auto fucked(T&& t) {
    return Fuck<Coordinate, T>(std::forward<T>(t));
}

template <frame Coordinate>
using FuckTransform = Fuck<Coordinate, TransformMatrix4d>;

using FuckWorld = FuckTransform<world_fuxk_frame>;

using FuckOdom   = FuckTransform<odom_fuxk_frame>;
using FuckGimbal = FuckTransform<gimbal_link_fuxk_frame>;

using FuckMuzzle = FuckTransform<muzzle_link_fuxk_frame>;

using FuckCamera  = FuckTransform<camera_link_fuxk_frame>;
using FuckOptical = FuckTransform<camera_optical_fuxk_frame>;

template <frame Coordinate>
using FuckDouble = Fuck<Coordinate, double>;

template <frame Coordinate>
using FuckFloat = Fuck<Coordinate, float>;

template <frame Coordinate>
using FuckIntrinsicEulerRotd = Fuck<Coordinate, math_fuxk::Ros2EulerRotd>;

template <frame Coordinate>
using FuckIntrinsicEulerRotf = Fuck<Coordinate, math_fuxk::Ros2EulerRotf>;

// Default time window for coordinate system buffers (1 second)
inline constexpr uint64_t default_keep_duration_ns = 1'000'000'000;

// ============================================================================
// Coordinate System with Transactional Batch Updates
// ============================================================================

/// Coordinate system with transactional batch update support.
///
/// This class wraps an array of TimeRangeBuffers (one per frame) and provides
/// atomic batch updates to prevent "TF tearing" - a condition where readers
/// observe partially updated coordinate frames with inconsistent timestamps.
///
/// The global transaction lock (read-write mutex) ensures that:
/// - Writers (batch updates) acquire exclusive lock - all updates appear atomic
/// - Readers (multi-frame lookups) acquire shared lock - see consistent snapshot
/// - Single-frame reads can use individual buffer locks for low contention
///
/// Usage:
///   CoordinateSystem<1024> system;  // Capacity 1024, default 1s time window
///   system.update_transaction_batch(updates);  // Atomic batch update
///   auto result = lookup<odom>(system, ...);   // Acquires read lock internally
template <std::size_t Capacity, uint64_t KeepDurationNs = default_keep_duration_ns>
class CoordinateSystem {
public:
    using buffer_type = TimeRangeBuffer<Capacity, KeepDurationNs>;
    using mutex_type  = tbb::spin_rw_mutex;

    CoordinateSystem()                                   = default;
    CoordinateSystem(const CoordinateSystem&)            = delete;
    CoordinateSystem& operator=(const CoordinateSystem&) = delete;
    CoordinateSystem(CoordinateSystem&&)                 = delete;
    CoordinateSystem& operator=(CoordinateSystem&&)      = delete;

    // ========================================================================
    // Read Transaction (RAII read lock for multi-frame lookups)
    // ========================================================================

    /// RAII guard that acquires the transaction mutex in shared (read) mode.
    /// Use this to prevent writes during multi-frame read operations.
    /// Automatically releases the lock when destroyed.
    class ReadTransaction {
    public:
        explicit ReadTransaction(mutex_type& mtx)
            : lock_(mtx, false) {} // false = shared/read mode

        ReadTransaction(const ReadTransaction&)            = delete;
        ReadTransaction& operator=(const ReadTransaction&) = delete;
        ReadTransaction(ReadTransaction&&)                 = delete;
        ReadTransaction& operator=(ReadTransaction&&)      = delete;

        ~ReadTransaction() = default;

    private:
        mutex_type::scoped_lock lock_;
        friend class CoordinateSystem;
    };

    /// Create a read transaction for multi-frame lookups.
    /// The transaction prevents batch updates until it's destroyed,
    /// ensuring a consistent view across all frames.
    [[nodiscard]] ReadTransaction read_transaction() const {
        return ReadTransaction(const_cast<mutex_type&>(transaction_mutex_));
    }

    // ========================================================================
    // Single Frame Update (backward compatible)
    // ========================================================================

    /// Update a single frame (acquires transaction lock in write mode).
    /// This maintains the same semantics as the old direct buffer access
    /// but now participates in the transactional update protocol.
    template <non_root_frame Layer>
    void update_frame(FuckTransform<Layer> transform, timestamp_ns_t ns) {
        mutex_type::scoped_lock lock(transaction_mutex_); // Default is write mode
        buffers_[Layer::index].push(ns, *transform);
    }

    // ========================================================================
    // Transactional Batch Update (core API for preventing TF tearing)
    // ========================================================================

    /// Atomically update multiple frames in a single transaction.
    ///
    /// This is the primary API for preventing TF tearing. All updates are
    /// applied under the global transaction lock (exclusive mode), ensuring
    /// readers observe a consistent snapshot either before or after the batch.
    ///
    /// @param updates Vector of (frame_index, (transform, timestamp)) pairs
    ///
    /// Example usage:
    /// ```cpp
    /// std::vector<std::pair<size_t, std::pair<TransformMatrix4d, uint64_t>>> updates;
    /// updates.push_back({gimbal_link_fuxk_frame::index, {tf1, ts1}});
    /// updates.push_back({odom_fuxk_frame::index, {tf2, ts2}});
    /// system.update_transaction_batch(updates);  // Atomic operation
    /// ```
    void update_transaction_batch(
        const std::vector<std::pair<size_t, std::pair<TransformMatrix4d, timestamp_ns_t>>>&
            updates) {
        mutex_type::scoped_lock lock(transaction_mutex_); // Default is write mode
        for (const auto& [frame_idx, tf_and_ts] : updates) {
            const auto& [tf, ts] = tf_and_ts;
            buffers_[frame_idx].push(ts, tf);
        }
    }

    // ========================================================================
    // Buffer Access (for internal/advanced use)
    // ========================================================================

    /// Get buffer for a specific frame (const access).
    /// For multi-frame lookups, use read_transaction() to prevent tearing.
    /// For single-frame queries, direct access is acceptable.
    template <frame Layer>
    [[nodiscard]] const auto& get_buffer() const {
        return buffers_[Layer::index];
    }

    /// Get buffer for a specific frame (mutable access, use with caution).
    /// Direct buffer modification bypasses transaction lock - prefer update_frame().
    template <frame Layer>
    [[nodiscard]] auto& get_buffer() {
        return buffers_[Layer::index];
    }

    /// Array-style access for backward compatibility (same caveats as get_buffer).
    [[nodiscard]] auto& operator[](std::size_t idx) { return buffers_[idx]; }
    [[nodiscard]] const auto& operator[](std::size_t idx) const { return buffers_[idx]; }

private:
    std::array<buffer_type, frame_count> buffers_;
    mutable mutex_type transaction_mutex_; // Global transaction lock (read-write mutex)
};

template <non_root_frame Layer, std::size_t Capacity, uint64_t KeepDurationNs>
constexpr void update(
    CoordinateSystem<Capacity, KeepDurationNs>& system, FuckTransform<Layer> transform,
    timestamp_ns_t ns) {
    system.template update_frame<Layer>(transform, ns);
}

// ============================================================================
// Internal Helper: Timestamp Consistency Tracking
// ============================================================================

namespace detail {

/// Internal result type that carries both transform and max delta
template <frame To>
struct LookupWithConsistencyResult {
    FuckTransform<To> transform;
    uint64_t max_delta_ns;
};

/// Internal lookup with timestamp delta tracking and early exit on mismatch
template <frame To, frame From, std::size_t N>
constexpr auto lookup_with_consistency(
    const CoordinateSystem<N>& system, FuckTransform<From> tf_source, timestamp_ns_t ns,
    uint64_t max_delta_ns, uint64_t current_max_delta)
    -> std::expected<LookupWithConsistencyResult<To>, TransformResult> {
    static_assert(is_descendant_of<From, To>(), "From must be a descendant of To");

    if constexpr (std::is_same_v<From, To>) {
        return LookupWithConsistencyResult<To>{tf_source, current_max_delta};
    } else if constexpr (root_frame<From>) {
        return std::unexpected(TransformError::FrameNotFound);
    } else {
        const auto& buffer = system[From::index];

        // Get timestamp info for consistency check
        auto info_result = buffer.get_closest_timestamp_info(ns);
        if (!info_result) {
            return std::unexpected(info_result.error());
        }

        const uint64_t frame_delta = info_result->delta_ns;
        const uint64_t new_max_delta =
            (frame_delta > current_max_delta) ? frame_delta : current_max_delta;

        // Early exit: if this frame alone exceeds tolerance, return mismatch immediately
        if (new_max_delta > max_delta_ns) {
            TimestampMismatchInfo info{
                .frame_id     = From::frame_id,
                .max_delta_ns = new_max_delta,
                .query_ns     = ns,
                .actual_ns    = info_result->closest_ts};
            return std::unexpected(info);
        }

        // Get actual transform entry for interpolation
        auto entry_result = buffer.at_interpolated(ns);
        if (!entry_result) {
            TransformError err = entry_result.error();
            return std::unexpected(err);
        }

        FuckTransform<typename From::ancestor> parent_tf{
            tf_source->reparent_to(entry_result->transform)};
        auto parent_result = lookup_with_consistency<To, typename From::ancestor, N>(
            system, parent_tf, ns, max_delta_ns, new_max_delta);

        if (!parent_result) {
            return parent_result;
        }

        return LookupWithConsistencyResult<To>{
            parent_result->transform, parent_result->max_delta_ns};
    }
}

/// Internal lookup_clamped with timestamp delta tracking and early exit
template <frame To, frame From, std::size_t N>
constexpr auto lookup_clamped_with_consistency(
    const CoordinateSystem<N>& system, FuckTransform<From> tf_source, timestamp_ns_t ns,
    uint64_t max_delta_ns, uint64_t current_max_delta)
    -> std::expected<LookupWithConsistencyResult<To>, TransformResult> {
    static_assert(is_descendant_of<From, To>(), "From must be a descendant of To");
    if constexpr (std::is_same_v<From, To>) {
        return LookupWithConsistencyResult<To>{tf_source, current_max_delta};
    } else if constexpr (root_frame<From>) {
        return std::unexpected(TransformError::FrameNotFound);
    } else {
        const auto& buffer = system[From::index];

        // Get timestamp info for consistency check
        auto info_result = buffer.get_closest_timestamp_info(ns);
        if (!info_result) {
            return std::unexpected(info_result.error());
        }

        const uint64_t frame_delta = info_result->delta_ns;
        const uint64_t new_max_delta =
            (frame_delta > current_max_delta) ? frame_delta : current_max_delta;

        // Early exit: if this frame alone exceeds tolerance, return mismatch immediately
        if (new_max_delta > max_delta_ns) {
            TimestampMismatchInfo info{
                .frame_id     = From::frame_id,
                .max_delta_ns = new_max_delta,
                .query_ns     = ns,
                .actual_ns    = info_result->closest_ts};
            return std::unexpected(info);
        }

        // Get actual transform entry for clamped interpolation
        auto entry_result = buffer.at_interpolated_clamped(ns);
        if (!entry_result) {
            TransformError err = entry_result.error();
            return std::unexpected(err);
        }

        FuckTransform<typename From::ancestor> parent_tf{
            tf_source->reparent_to(entry_result->transform)};
        auto parent_result = lookup_clamped_with_consistency<To, typename From::ancestor, N>(
            system, parent_tf, ns, max_delta_ns, new_max_delta);

        if (!parent_result) {
            return parent_result;
        }

        return LookupWithConsistencyResult<To>{
            parent_result->transform, parent_result->max_delta_ns};
    }
}

} // namespace detail

// ============================================================================
// Public Lookup API
// ============================================================================

/// Lookup transform with optional timestamp consistency checking.
///
/// When max_delta_ns is provided (default: 100'000 ns = 0.1ms), checks that all frames
/// in the transform chain have data within max_delta_ns of the query timestamp.
/// This helps detect "TF tearing" where concurrent updates cause inconsistent timestamps.
///
/// This function acquires a read transaction lock to prevent batch updates during
/// multi-frame traversal, ensuring consistent snapshot semantics.
///
/// @param system The coordinate system containing all frame buffers
/// @param tf_source The source transform in the "From" frame
/// @param ns Query timestamp in nanoseconds
/// @param max_delta_ns Maximum allowed timestamp delta (default: 1ms)
/// @return Expected transform or error (may be TransformError or TimestampMismatchInfo)
template <frame To, frame From, std::size_t N>
auto lookup(
    const CoordinateSystem<N>& system, FuckTransform<From> tf_source, timestamp_ns_t ns,
    uint64_t max_delta_ns = 1'000'000) -> std::expected<FuckTransform<To>, TransformResult> {
    // Acquire read transaction to prevent batch updates during multi-frame traversal
    auto _read_tx = system.read_transaction();
    (void)_read_tx; // Suppress unused variable warning - RAII does the work

    auto result =
        detail::lookup_with_consistency<To, From, N>(system, tf_source, ns, max_delta_ns, 0);

    if (!result) {
        return std::unexpected(result.error());
    }

    return result->transform;
}

/// Lookup transform using clamped interpolation with optional timestamp consistency checking.
///
/// This never returns `FutureExtrapolationRequired`/`PastExtrapolationRequired` as long as the
/// buffer has data; it clamps the query time to the buffer's time range (equivalent to TF2
/// `TimePointZero` semantics for "latest available" when `ns` is newer than the newest sample).
///
/// This function acquires a read transaction lock to prevent batch updates during
/// multi-frame traversal, ensuring consistent snapshot semantics.
///
/// @param system The coordinate system containing all frame buffers
/// @param tf_source The source transform in the "From" frame
/// @param ns Query timestamp in nanoseconds
/// @param max_delta_ns Maximum allowed timestamp delta (default: 1ms)
/// @return Expected transform or error (may be TransformError or TimestampMismatchInfo)
template <frame To, frame From, std::size_t N>
auto lookup_clamped(
    const CoordinateSystem<N>& system, FuckTransform<From> tf_source, timestamp_ns_t ns,
    uint64_t max_delta_ns = 50'000'000) -> std::expected<FuckTransform<To>, TransformResult> {
    // Acquire read transaction to prevent batch updates during multi-frame traversal
    auto _read_tx = system.read_transaction();
    (void)_read_tx; // Suppress unused variable warning - RAII does the work

    auto result = detail::lookup_clamped_with_consistency<To, From, N>(
        system, tf_source, ns, max_delta_ns, 0);

    if (!result) {
        return std::unexpected(result.error());
    }

    return result->transform;
}

} // namespace fast_tf
