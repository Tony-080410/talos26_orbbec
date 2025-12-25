#pragma once

#include "error.hpp"
#include "matrix.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <expected>
#include <optional>
#include <tbb/spin_rw_mutex.h>
#include <vector>

namespace fast_tf {

// ============================================================================
// Transform Entry
// ============================================================================

/// A timestamped transform
struct TransformEntry {
    timestamp_ns_t timestamp_ns;
    TransformMatrix4d transform;

    TransformEntry()
        : timestamp_ns(0) {}

    TransformEntry(timestamp_ns_t ts, const TransformMatrix4d& tf)
        : timestamp_ns(ts)
        , transform(tf) {}

    bool operator<(const TransformEntry& other) const { return timestamp_ns < other.timestamp_ns; }

    bool operator>(const TransformEntry& other) const { return timestamp_ns > other.timestamp_ns; }

    bool operator==(const TransformEntry& other) const {
        return timestamp_ns == other.timestamp_ns;
    }
};

// ============================================================================
// Circular Transform Buffer
// ============================================================================

/// Fixed-size circular buffer for transform entries
template <std::size_t Capacity>
class CircularTransformBuffer {
public:
    static_assert(Capacity > 0, "Capacity must be greater than 0");
    static constexpr std::size_t capacity_value = Capacity;

    using value_type = TransformEntry;
    using mutex_type = tbb::spin_rw_mutex;

    CircularTransformBuffer()
        : head_(0)
        , size_(0) {}

    // Non-copyable
    CircularTransformBuffer(const CircularTransformBuffer&)            = delete;
    CircularTransformBuffer& operator=(const CircularTransformBuffer&) = delete;

    // Movable
    CircularTransformBuffer(CircularTransformBuffer&&)            = default;
    CircularTransformBuffer& operator=(CircularTransformBuffer&&) = default;

    // ========================================================================
    // Write Operations
    // ========================================================================

    void push(timestamp_ns_t timestamp_ns, const TransformMatrix4d& transform) {
        typename mutex_type::scoped_lock lock(mutex_);

        buffer_[head_] = TransformEntry(timestamp_ns, transform);
        head_          = (head_ + 1) % Capacity;

        if (size_ < Capacity) {
            size_++;
        }
    }

    // ========================================================================
    // Read Operations (std::expected)
    // ========================================================================

    [[nodiscard]] std::expected<TransformEntry, TransformError> latest() const {
        typename mutex_type::scoped_lock lock(mutex_, false);

        if (size_ == 0) {
            return std::unexpected(TransformError::NoDataAvailable);
        }

        std::size_t latest_idx = (head_ + Capacity - 1) % Capacity;
        return buffer_[latest_idx];
    }

    [[nodiscard]] std::expected<TransformEntry, TransformError>
        at(timestamp_ns_t timestamp_ns) const {
        typename mutex_type::scoped_lock lock(mutex_, false);

        if (size_ == 0) {
            return std::unexpected(TransformError::NoDataAvailable);
        }

        for (std::size_t i = 0; i < size_; ++i) {
            std::size_t idx = (head_ + Capacity - size_ + i) % Capacity;
            if (buffer_[idx].timestamp_ns == timestamp_ns) {
                return buffer_[idx];
            }
        }

        return std::unexpected(TransformError::ExtrapolationRequired);
    }

    [[nodiscard]] std::expected<TransformEntry, TransformError>
        at_interpolated(timestamp_ns_t timestamp_ns) const {
        typename mutex_type::scoped_lock lock(mutex_, false);

        if (size_ == 0) {
            return std::unexpected(TransformError::NoDataAvailable);
        }

        if (size_ == 1) {
            std::size_t idx = (head_ + Capacity - 1) % Capacity;
            if (buffer_[idx].timestamp_ns == timestamp_ns) {
                return buffer_[idx];
            }
            return std::unexpected(TransformError::ExtrapolationRequired);
        }

        // Collect and sort entries
        std::vector<TransformEntry> entries;
        entries.reserve(size_);

        for (std::size_t i = 0; i < size_; ++i) {
            std::size_t idx = (head_ + Capacity - size_ + i) % Capacity;
            entries.push_back(buffer_[idx]);
        }

        std::sort(entries.begin(), entries.end());

        // Check bounds
        if (timestamp_ns < entries.front().timestamp_ns
            || timestamp_ns > entries.back().timestamp_ns) {
            return std::unexpected(TransformError::ExtrapolationRequired);
        }

        // Find target timestamp
        auto it =
            std::lower_bound(entries.begin(), entries.end(), TransformEntry(timestamp_ns, {}));

        if (it != entries.end() && it->timestamp_ns == timestamp_ns) {
            return *it;
        }

        // Interpolate
        if (it != entries.end() && it != entries.begin()) {
            auto prev = std::prev(it);

            uint64_t dt = it->timestamp_ns - prev->timestamp_ns;
            fp_t t = static_cast<fp_t>(timestamp_ns - prev->timestamp_ns) / static_cast<fp_t>(dt);

            TransformMatrix4d interpolated =
                TransformMatrix4d::lerp(prev->transform, it->transform, t);

            return TransformEntry(timestamp_ns, interpolated);
        }

        return std::unexpected(TransformError::ExtrapolationRequired);
    }

    /**
     * Get interpolated transform, clamping to nearest if extrapolation needed.
     * Unlike at_interpolated(), this never fails due to out-of-range timestamps.
     * @param timestamp_ns Target timestamp
     * @return Transform entry (interpolated or clamped to boundary)
     */
    [[nodiscard]] std::expected<TransformEntry, TransformError>
        at_interpolated_clamped(timestamp_ns_t timestamp_ns) const {
        typename mutex_type::scoped_lock lock(mutex_, false);

        if (size_ == 0) {
            return std::unexpected(TransformError::NoDataAvailable);
        }

        if (size_ == 1) {
            std::size_t idx = (head_ + Capacity - 1) % Capacity;
            // 只有一个数据点时，直接返回该点（允许外推）
            return buffer_[idx];
        }

        // Collect and sort entries
        std::vector<TransformEntry> entries;
        entries.reserve(size_);

        for (std::size_t i = 0; i < size_; ++i) {
            std::size_t idx = (head_ + Capacity - size_ + i) % Capacity;
            entries.push_back(buffer_[idx]);
        }

        std::sort(entries.begin(), entries.end());

        // Clamp to bounds instead of failing
        if (timestamp_ns < entries.front().timestamp_ns) {
            return entries.front();
        }
        if (timestamp_ns > entries.back().timestamp_ns) {
            return entries.back();
        }

        // Find target timestamp
        auto it =
            std::lower_bound(entries.begin(), entries.end(), TransformEntry(timestamp_ns, {}));

        if (it != entries.end() && it->timestamp_ns == timestamp_ns) {
            return *it;
        }

        // Interpolate
        if (it != entries.end() && it != entries.begin()) {
            auto prev = std::prev(it);

            uint64_t dt = it->timestamp_ns - prev->timestamp_ns;
            fp_t t = static_cast<fp_t>(timestamp_ns - prev->timestamp_ns) / static_cast<fp_t>(dt);

            TransformMatrix4d interpolated =
                TransformMatrix4d::lerp(prev->transform, it->transform, t);

            return TransformEntry(timestamp_ns, interpolated);
        }

        // Fallback (should not reach here)
        return entries.back();
    }

    // ========================================================================
    // Query Operations
    // ========================================================================

    std::optional<std::pair<timestamp_ns_t, timestamp_ns_t>> time_range() const {
        typename mutex_type::scoped_lock lock(mutex_, false);

        if (size_ == 0) {
            return std::nullopt;
        }

        timestamp_ns_t min_time = UINT64_MAX;
        timestamp_ns_t max_time = 0;

        for (std::size_t i = 0; i < size_; ++i) {
            std::size_t idx = (head_ + Capacity - size_ + i) % Capacity;
            min_time        = std::min(min_time, buffer_[idx].timestamp_ns);
            max_time        = std::max(max_time, buffer_[idx].timestamp_ns);
        }

        return std::make_pair(min_time, max_time);
    }

    [[nodiscard]] std::size_t size() const {
        typename mutex_type::scoped_lock lock(mutex_, false);
        return size_;
    }

    [[nodiscard]] constexpr std::size_t capacity() const { return Capacity; }

    [[nodiscard]] bool is_full() const {
        typename mutex_type::scoped_lock lock(mutex_, false);
        return size_ == Capacity;
    }

    [[nodiscard]] bool is_empty() const {
        typename mutex_type::scoped_lock lock(mutex_, false);
        return size_ == 0;
    }

    [[nodiscard]] bool contains_time(timestamp_ns_t timestamp_ns) const {
        auto range = time_range();
        if (range) {
            return timestamp_ns >= range->first && timestamp_ns <= range->second;
        }
        return false;
    }

    void clear() {
        typename mutex_type::scoped_lock lock(mutex_);
        head_ = 0;
        size_ = 0;
    }

    std::vector<TransformEntry> get_all_entries() const {
        typename mutex_type::scoped_lock lock(mutex_, false);

        std::vector<TransformEntry> entries;
        entries.reserve(size_);

        for (std::size_t i = 0; i < size_; ++i) {
            std::size_t idx = (head_ + Capacity - size_ + i) % Capacity;
            entries.push_back(buffer_[idx]);
        }

        std::sort(entries.begin(), entries.end());
        return entries;
    }

private:
    std::array<TransformEntry, Capacity> buffer_;
    std::size_t head_ = 0;
    std::size_t size_ = 0;
    mutable mutex_type mutex_;
};

// ============================================================================
// Buffer Capacity Helper
// ============================================================================

template <std::size_t N>
struct BufferCapacity {
    static constexpr std::size_t value = N;
    using buffer_type                  = CircularTransformBuffer<N>;
};

} // namespace fast_tf
