#pragma once

#include "error.hpp"
#include "matrix.hpp"
#include "types.hpp"
#include "validation.hpp"
#include <array>
#include <cassert>
#include <cstdint>
#include <expected>
#include <optional>
#include <tbb/spin_rw_mutex.h>

namespace fast_tf {

// ============================================================================
// Transform Entry
// ============================================================================

struct TransformEntry {
    timestamp_ns_t timestamp_ns;
    TransformMatrix4d transform;

    TransformEntry()
        : timestamp_ns(0) {}

    TransformEntry(timestamp_ns_t ts, const TransformMatrix4d& tf)
        : timestamp_ns(ts)
        , transform(tf) {}
};

// ============================================================================
// Circular Transform Buffer (Optimized)
// ============================================================================

/// Fixed-size circular buffer for transform entries.
/// Assumes timestamps are strictly increasing (typical for sensor data).
/// Zero heap allocation, O(log n) lookup.
template <std::size_t Capacity>
class CircularTransformBuffer {
public:
    static_assert(Capacity > 0, "Capacity must be greater than 0");
    static constexpr std::size_t capacity_value = Capacity;

    using value_type = TransformEntry;
    using mutex_type = tbb::spin_rw_mutex;

    CircularTransformBuffer() = default;

    CircularTransformBuffer(const CircularTransformBuffer&)            = delete;
    CircularTransformBuffer& operator=(const CircularTransformBuffer&) = delete;
    CircularTransformBuffer(CircularTransformBuffer&&)                 = default;
    CircularTransformBuffer& operator=(CircularTransformBuffer&&)      = default;

    // ========================================================================
    // Write
    // ========================================================================

    void push(timestamp_ns_t timestamp_ns, const TransformMatrix4d& transform) {
        mutex_type::scoped_lock lock(mutex_);

        // 数值验证（包括静态变换）
        auto validation = validate_transform(transform);
        if (!validation) {
            // 验证失败，静默丢弃
            // TODO: 可选的日志记录
            return;
        }

        // Debug: ensure timestamps are increasing
        if (newest_ts_ != 0 && timestamp_ns == newest_ts_) {
            return;
        }
        if (timestamp_ns != 0 && timestamp_ns < newest_ts_) {
            return;
        }

        buffer_[head_] = TransformEntry(timestamp_ns, transform);

        if (size_ == 0) {
            oldest_ts_ = timestamp_ns;
        }
        newest_ts_ = timestamp_ns;

        head_ = (head_ + 1) % Capacity;

        if (size_ < Capacity) {
            ++size_;
        } else {
            // Buffer full, oldest entry overwritten, update oldest_ts
            oldest_ts_ = buffer_[head_].timestamp_ns;
        }
    }

    // ========================================================================
    // Read
    // ========================================================================

    [[nodiscard]] std::expected<TransformEntry, TransformError> latest() const {
        mutex_type::scoped_lock lock(mutex_, false);

        if (size_ == 0) {
            return std::unexpected(TransformError::NoDataAvailable);
        }

        return buffer_[(head_ + Capacity - 1) % Capacity];
    }

    [[nodiscard]] std::expected<TransformEntry, TransformError>
        at_interpolated_clamped(timestamp_ns_t ts) const {
        mutex_type::scoped_lock lock(mutex_, false);

        if (size_ == 0) {
            return std::unexpected(TransformError::NoDataAvailable);
        }

        if (size_ == 1) {
            const auto& entry = buffer_[(head_ + Capacity - 1) % Capacity];
            if (entry.timestamp_ns == 0) {
                // static transform
                return TransformEntry{ts, entry.transform};
            }
            // Single data point: use it as-is (no extrapolation possible)
            return TransformEntry{ts, entry.transform};
        }

        // Binary search for the first entry with timestamp > ts
        // Result: lo points to entry just after ts (or size_ if ts >= newest)
        std::size_t lo = 0;
        std::size_t hi = size_;

        while (lo < hi) {
            std::size_t mid       = lo + (hi - lo) / 2;
            timestamp_ns_t mid_ts = at_logical(mid).timestamp_ns;

            if (mid_ts <= ts) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        // Determine interpolation/extrapolation pair
        const auto* prev = &at_logical(0);
        const auto* next = &at_logical(1);

        if (lo == 0) {
            // Extrapolate backward: use first two points
            prev = &at_logical(0);
            next = &at_logical(1);
        } else if (lo >= size_) {
            // Extrapolate forward: use last two points
            prev = &at_logical(size_ - 2);
            next = &at_logical(size_ - 1);
        } else {
            // Interpolate: between lo-1 and lo
            prev = &at_logical(lo - 1);
            next = &at_logical(lo);
        }

        // Exact match check
        if (prev->timestamp_ns == ts) {
            return *prev;
        }

        // Interpolate/extrapolate
        const uint64_t dt = next->timestamp_ns - prev->timestamp_ns;
        if (dt == 0) {
            return *prev; // 防御性编程：除零保护
        }
        const fp_t t = static_cast<fp_t>(ts - prev->timestamp_ns) / static_cast<fp_t>(dt);

        return TransformEntry(ts, TransformMatrix4d::lerp(prev->transform, next->transform, t));
    }

    [[nodiscard]] std::expected<TransformEntry, TransformError>
        at_interpolated(timestamp_ns_t ts) const {
        mutex_type::scoped_lock lock(mutex_, false);

        if (size_ == 0) {
            return std::unexpected(TransformError::NoDataAvailable);
        }

        if (size_ == 1) {
            const auto& entry = buffer_[(head_ + Capacity - 1) % Capacity];
            if (entry.timestamp_ns == ts) {
                return entry;
            }
            // static transform
            if (entry.timestamp_ns == 0) {
                return TransformEntry{ts, entry.transform};
            }
            return std::unexpected(TransformError::FutureExtrapolationRequired);
        }

        // Check bounds (no extrapolation)
        if (ts < oldest_ts_) {
            return std::unexpected(TransformError::PastExtrapolationRequired);
        }
        if (ts > newest_ts_) {
            return std::unexpected(TransformError::FutureExtrapolationRequired);
        }

        // Binary search
        std::size_t lo = 0;
        std::size_t hi = size_;

        while (lo < hi) {
            std::size_t mid       = lo + (hi - lo) / 2;
            timestamp_ns_t mid_ts = at_logical(mid).timestamp_ns;

            if (mid_ts <= ts) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        if (lo == 0) {
            // ts < all entries (shouldn't happen due to bounds check)
            return std::unexpected(TransformError::FutureExtrapolationRequired);
        }

        const auto& prev = at_logical(lo - 1);

        // Exact match
        if (prev.timestamp_ns == ts) {
            return prev;
        }

        if (lo >= size_) {
            // ts > all entries
            return std::unexpected(TransformError::FutureExtrapolationRequired);
        }

        const auto& next = at_logical(lo);

        // Interpolate
        const uint64_t dt = next.timestamp_ns - prev.timestamp_ns;
        if (dt == 0) {
            return prev; // 防御性编程：除零保护
        }
        const fp_t t = static_cast<fp_t>(ts - prev.timestamp_ns) / static_cast<fp_t>(dt);

        return TransformEntry(ts, TransformMatrix4d::lerp(prev.transform, next.transform, t));
    }

    // ========================================================================
    // Query
    // ========================================================================

    [[nodiscard]] std::optional<std::pair<timestamp_ns_t, timestamp_ns_t>> time_range() const {
        mutex_type::scoped_lock lock(mutex_, false);

        if (size_ == 0) {
            return std::nullopt;
        }

        return std::make_pair(oldest_ts_, newest_ts_);
    }

    [[nodiscard]] std::size_t size() const {
        mutex_type::scoped_lock lock(mutex_, false);
        return size_;
    }

    [[nodiscard]] constexpr std::size_t capacity() const { return Capacity; }

    [[nodiscard]] bool is_full() const {
        mutex_type::scoped_lock lock(mutex_, false);
        return size_ == Capacity;
    }

    [[nodiscard]] bool is_empty() const {
        mutex_type::scoped_lock lock(mutex_, false);
        return size_ == 0;
    }

    [[nodiscard]] bool contains_time(timestamp_ns_t ts) const {
        mutex_type::scoped_lock lock(mutex_, false);
        if (size_ == 0)
            return false;
        return ts >= oldest_ts_ && ts <= newest_ts_;
    }

    void clear() {
        mutex_type::scoped_lock lock(mutex_);
        head_      = 0;
        size_      = 0;
        oldest_ts_ = 0;
        newest_ts_ = 0;
    }

private:
    /// Get entry by logical index (0 = oldest, size_-1 = newest)
    [[nodiscard]] const TransformEntry& at_logical(std::size_t idx) const {
        // tail is at (head_ + Capacity - size_) % Capacity
        std::size_t physical = (head_ + Capacity - size_ + idx) % Capacity;
        return buffer_[physical];
    }

    std::array<TransformEntry, Capacity> buffer_{};
    std::size_t head_         = 0;
    std::size_t size_         = 0;
    timestamp_ns_t oldest_ts_ = 0;
    timestamp_ns_t newest_ts_ = 0;
    mutable mutex_type mutex_;
};

} // namespace fast_tf
