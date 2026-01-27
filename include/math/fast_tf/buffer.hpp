#pragma once

#include "error.hpp"
#include "matrix.hpp"
#include <algorithm>
#include <cstdint>
#include <expected>
#include <optional>
#include <tbb/spin_rw_mutex.h>
#include <utility>
#include <vector>

namespace fast_tf {

// Note: frame_count is defined in frame.hpp, CoordinateSystem is also defined there

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
// Time Range Buffer (Time-window + Capacity)
// ============================================================================

/// Time-windowed buffer with capacity limit.
/// Automatically removes data outside the time window on every push/operation.
/// Timestamp==0 represents static data (never expires, stored separately).
template <std::size_t Capacity, uint64_t KeepDurationNs>
class TimeRangeBuffer {
public:
    static_assert(Capacity > 0, "Capacity must be greater than 0");
    static_assert(KeepDurationNs > 0, "KeepDurationNs must be greater than 0");
    static constexpr std::size_t capacity_value = Capacity;

    using mutex_type = tbb::spin_rw_mutex;

    TimeRangeBuffer() = default;

    TimeRangeBuffer(const TimeRangeBuffer&)            = delete;
    TimeRangeBuffer& operator=(const TimeRangeBuffer&) = delete;
    TimeRangeBuffer(TimeRangeBuffer&&)                 = default;
    TimeRangeBuffer& operator=(TimeRangeBuffer&&)      = default;

    // ========================================================================
    // Write
    // ========================================================================

    void push(timestamp_ns_t timestamp_ns, const TransformMatrix4d& transform) {
        mutex_type::scoped_lock lock(mutex_);

        // Handle static transform (timestamp == 0)
        if (timestamp_ns == 0) {
            static_transform_ = TransformEntry(0, transform);
            has_static_       = true;
            return;
        }

        // Reject out-of-order timestamps
        if (!dynamic_entries_.empty() && timestamp_ns < dynamic_entries_.back().timestamp_ns) {
            return;
        }

        // Reject duplicate timestamps
        if (!dynamic_entries_.empty() && dynamic_entries_.back().timestamp_ns == timestamp_ns) {
            return;
        }

        // Add new entry
        dynamic_entries_.push_back(TransformEntry(timestamp_ns, transform));

        // Enforce capacity limit (remove oldest if needed)
        if (dynamic_entries_.size() > Capacity) {
            dynamic_entries_.erase(dynamic_entries_.begin());
        }

        // Remove entries outside time window
        prune_expired(timestamp_ns);

        // Update time range bounds
        update_time_bounds();
    }

    // ========================================================================
    // Read
    // ========================================================================

    [[nodiscard]] std::expected<TransformEntry, TransformError> latest() const {
        mutex_type::scoped_lock lock(mutex_, false);

        // Prefer static transform if available
        if (has_static_) {
            return static_transform_;
        }

        if (dynamic_entries_.empty()) {
            return std::unexpected(TransformError::NoDataAvailable);
        }

        return dynamic_entries_.back();
    }

    [[nodiscard]] std::expected<TransformEntry, TransformError>
        at_interpolated_clamped(timestamp_ns_t ts) const {
        mutex_type::scoped_lock lock(mutex_, false);

        // Static transform handles all timestamps
        if (has_static_) {
            return TransformEntry(ts, static_transform_.transform);
        }

        if (dynamic_entries_.empty()) {
            return std::unexpected(TransformError::NoDataAvailable);
        }

        if (dynamic_entries_.size() == 1) {
            return dynamic_entries_.front();
        }

        // Clamp to bounds
        const timestamp_ns_t oldest_ts = dynamic_entries_.front().timestamp_ns;
        const timestamp_ns_t newest_ts = dynamic_entries_.back().timestamp_ns;

        if (ts <= oldest_ts) {
            return dynamic_entries_.front();
        }
        if (ts >= newest_ts) {
            return dynamic_entries_.back();
        }

        // Binary search for interpolation
        return interpolate(ts);
    }

    [[nodiscard]] std::expected<TransformEntry, TransformError>
        at_interpolated(timestamp_ns_t ts) const {
        mutex_type::scoped_lock lock(mutex_, false);

        // Static transform handles all timestamps
        if (has_static_) {
            return TransformEntry(ts, static_transform_.transform);
        }

        if (dynamic_entries_.empty()) {
            return std::unexpected(TransformError::NoDataAvailable);
        }

        if (dynamic_entries_.size() == 1) {
            const auto& entry = dynamic_entries_.front();
            if (entry.timestamp_ns == ts) {
                return entry;
            }
            return std::unexpected(TransformError::FutureExtrapolationRequired);
        }

        // Check bounds (no extrapolation)
        const timestamp_ns_t oldest_ts = dynamic_entries_.front().timestamp_ns;
        const timestamp_ns_t newest_ts = dynamic_entries_.back().timestamp_ns;

        if (ts < oldest_ts) {
            return std::unexpected(TransformError::PastExtrapolationRequired);
        }
        if (ts > newest_ts) {
            return std::unexpected(TransformError::FutureExtrapolationRequired);
        }

        // Binary search for interpolation
        return interpolate(ts);
    }

    // ========================================================================
    // Query
    // ========================================================================

    struct ClosestTimestampInfo {
        timestamp_ns_t closest_ts;
        uint64_t delta_ns;
    };

    [[nodiscard]] std::expected<ClosestTimestampInfo, TransformError>
        get_closest_timestamp_info(timestamp_ns_t ts) const {
        mutex_type::scoped_lock lock(mutex_, false);

        // Static transform is always valid with delta=0
        if (has_static_) {
            return ClosestTimestampInfo{0, 0};
        }

        if (dynamic_entries_.empty()) {
            return std::unexpected(TransformError::NoDataAvailable);
        }

        const timestamp_ns_t oldest_ts = dynamic_entries_.front().timestamp_ns;
        const timestamp_ns_t newest_ts = dynamic_entries_.back().timestamp_ns;

        timestamp_ns_t closest;
        uint64_t delta_ns;

        if (ts <= oldest_ts) {
            closest  = oldest_ts;
            delta_ns = oldest_ts - ts;
        } else if (ts >= newest_ts) {
            closest  = newest_ts;
            delta_ns = ts - newest_ts;
        } else {
            // Find closest entry within range
            auto it = std::lower_bound(
                dynamic_entries_.begin(), dynamic_entries_.end(), ts,
                [](const TransformEntry& entry, timestamp_ns_t t) {
                    return entry.timestamp_ns < t;
                });

            if (it == dynamic_entries_.begin()) {
                closest  = it->timestamp_ns;
                delta_ns = it->timestamp_ns - ts;
            } else if (it == dynamic_entries_.end()) {
                --it;
                closest  = it->timestamp_ns;
                delta_ns = ts - it->timestamp_ns;
            } else {
                // Check both neighbors
                const uint64_t delta_prev = ts - (it - 1)->timestamp_ns;
                const uint64_t delta_next = it->timestamp_ns - ts;

                if (delta_prev <= delta_next) {
                    --it;
                    closest  = it->timestamp_ns;
                    delta_ns = delta_prev;
                } else {
                    closest  = it->timestamp_ns;
                    delta_ns = delta_next;
                }
            }
        }

        return ClosestTimestampInfo{closest, delta_ns};
    }

    [[nodiscard]] std::optional<std::pair<timestamp_ns_t, timestamp_ns_t>> time_range() const {
        mutex_type::scoped_lock lock(mutex_, false);

        if (has_static_) {
            return std::make_pair(0, UINT64_MAX);
        }

        if (dynamic_entries_.empty()) {
            return std::nullopt;
        }

        return std::make_pair(
            dynamic_entries_.front().timestamp_ns, dynamic_entries_.back().timestamp_ns);
    }

    [[nodiscard]] std::size_t size() const {
        mutex_type::scoped_lock lock(mutex_, false);
        return dynamic_entries_.size() + (has_static_ ? 1 : 0);
    }

    [[nodiscard]] constexpr std::size_t capacity() const { return Capacity; }

    [[nodiscard]] bool is_full() const {
        mutex_type::scoped_lock lock(mutex_, false);
        return dynamic_entries_.size() >= Capacity;
    }

    [[nodiscard]] bool is_empty() const {
        mutex_type::scoped_lock lock(mutex_, false);
        return dynamic_entries_.empty() && !has_static_;
    }

    [[nodiscard]] bool contains_time(timestamp_ns_t ts) const {
        mutex_type::scoped_lock lock(mutex_, false);

        // Static transform contains all timestamps
        if (has_static_) {
            return true;
        }

        if (dynamic_entries_.empty()) {
            return false;
        }

        return ts >= dynamic_entries_.front().timestamp_ns
            && ts <= dynamic_entries_.back().timestamp_ns;
    }

    void clear() {
        mutex_type::scoped_lock lock(mutex_);
        dynamic_entries_.clear();
        has_static_ = false;
    }

    /// Get the keep duration (time window) in nanoseconds
    [[nodiscard]] constexpr uint64_t keep_duration_ns() const { return KeepDurationNs; }

private:
    /// Remove entries older than (newest_ts - KeepDurationNs)
    void prune_expired(timestamp_ns_t current_ts) {
        const timestamp_ns_t expiry_threshold =
            (current_ts >= KeepDurationNs) ? current_ts - KeepDurationNs : 0;

        // Find first entry that is not expired
        auto it = std::find_if(
            dynamic_entries_.begin(), dynamic_entries_.end(),
            [expiry_threshold](const TransformEntry& entry) {
                return entry.timestamp_ns >= expiry_threshold;
            });

        // Remove expired entries
        if (it != dynamic_entries_.begin()) {
            dynamic_entries_.erase(dynamic_entries_.begin(), it);
        }
    }

    /// Update cached time bounds after modifications
    void update_time_bounds() {
        // Time bounds are computed on-demand from dynamic_entries_
    }

    /// Binary search interpolation helper
    [[nodiscard]] std::expected<TransformEntry, TransformError>
        interpolate(timestamp_ns_t ts) const {
        // Find first entry with timestamp > ts
        auto it = std::upper_bound(
            dynamic_entries_.begin(), dynamic_entries_.end(), ts,
            [](timestamp_ns_t t, const TransformEntry& entry) { return t < entry.timestamp_ns; });

        if (it == dynamic_entries_.begin() || it == dynamic_entries_.end()) {
            return std::unexpected(TransformError::FutureExtrapolationRequired);
        }

        const auto& prev = *(it - 1);
        const auto& next = *it;

        // Exact match check
        if (prev.timestamp_ns == ts) {
            return prev;
        }

        // Interpolate
        const uint64_t dt        = next.timestamp_ns - prev.timestamp_ns;
        const uint64_t time_diff = ts - prev.timestamp_ns;

        const fp_t t = static_cast<fp_t>(time_diff) / static_cast<fp_t>(dt);

        if (std::isnan(t) || std::isinf(t) || t < 0.0 || t > 1.0) {
            return std::unexpected(TransformError::FutureExtrapolationRequired);
        }

        return TransformEntry(ts, TransformMatrix4d::lerp_se3(prev.transform, next.transform, t));
    }

    // Dynamic (time-varying) entries
    std::vector<TransformEntry> dynamic_entries_;

    // Static transform (timestamp == 0, never expires)
    TransformEntry static_transform_{};
    bool has_static_ = false;

    mutable mutex_type mutex_;
};

} // namespace fast_tf
