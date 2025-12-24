#pragma once

#include "frame_types.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cstdint>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <tbb/spin_rw_mutex.h>

namespace fast_tf {

// ============================================================================
// 变换条目（时间戳+变换矩阵）
// ============================================================================

struct TransformEntry {
    uint64_t timestamp_ns; // 纳秒级时间戳
    TransformMatrix4d transform;

    TransformEntry()
        : timestamp_ns(0) {}

    TransformEntry(uint64_t ts, const TransformMatrix4d& tf)
        : timestamp_ns(ts)
        , transform(tf) {}

    // 比较操作
    bool operator<(const TransformEntry& other) const { return timestamp_ns < other.timestamp_ns; }

    bool operator>(const TransformEntry& other) const { return timestamp_ns > other.timestamp_ns; }

    bool operator==(const TransformEntry& other) const {
        return timestamp_ns == other.timestamp_ns;
    }
};

// ============================================================================
// 固定大小的循环缓冲区
// ============================================================================

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

    // 禁用复制
    CircularTransformBuffer(const CircularTransformBuffer&)            = delete;
    CircularTransformBuffer& operator=(const CircularTransformBuffer&) = delete;

    // 默认移动构造函数
    CircularTransformBuffer(CircularTransformBuffer&&)            = default;
    CircularTransformBuffer& operator=(CircularTransformBuffer&&) = default;

    // 向缓冲区添加变换
    void push(uint64_t timestamp_ns, const TransformMatrix4d& transform) {
        typename mutex_type::scoped_lock lock(mutex_);

        buffer_[head_] = TransformEntry(timestamp_ns, transform);
        head_          = (head_ + 1) % Capacity;

        if (size_ < Capacity) {
            size_++;
        }
    }

    // 获取最新的变换
    std::optional<TransformEntry> latest() const {
        typename mutex_type::scoped_lock lock(mutex_, false);

        if (size_ == 0) {
            return std::nullopt;
        }

        std::size_t latest_idx = (head_ + Capacity - 1) % Capacity;
        return buffer_[latest_idx];
    }

    // 根据时间戳查询变换（精确查询）
    std::optional<TransformEntry> at(uint64_t timestamp_ns) const {
        typename mutex_type::scoped_lock lock(mutex_, false);

        for (std::size_t i = 0; i < size_; ++i) {
            std::size_t idx = (head_ + i) % Capacity;
            if (buffer_[idx].timestamp_ns == timestamp_ns) {
                return buffer_[idx];
            }
        }

        return std::nullopt;
    }

    // 根据时间戳查询变换（插值）
    std::optional<TransformEntry> at_interpolated(uint64_t timestamp_ns) const {
        typename mutex_type::scoped_lock lock(mutex_, false);

        if (size_ < 2) {
            if (size_ == 1) {
                std::size_t idx = (head_ + Capacity - 1) % Capacity;
                if (buffer_[idx].timestamp_ns == timestamp_ns) {
                    return buffer_[idx];
                }
            }
            return std::nullopt;
        }

        // 收集所有有效条目，按时间排序
        std::vector<TransformEntry> entries;
        entries.reserve(size_);

        for (std::size_t i = 0; i < size_; ++i) {
            std::size_t idx = (head_ + i) % Capacity;
            entries.push_back(buffer_[idx]);
        }

        std::sort(entries.begin(), entries.end());

        // 查找目标时间戳
        auto it =
            std::lower_bound(entries.begin(), entries.end(), TransformEntry(timestamp_ns, {}));

        if (it != entries.end() && it->timestamp_ns == timestamp_ns) {
            return *it;
        }

        // 如果不存在精确匹配，进行插值
        if (it != entries.end() && it != entries.begin()) {
            auto prev = std::prev(it);

            // 检查是否可以插值
            if (prev->timestamp_ns < timestamp_ns && timestamp_ns < it->timestamp_ns) {
                uint64_t dt = it->timestamp_ns - prev->timestamp_ns;
                double t    = static_cast<double>(timestamp_ns - prev->timestamp_ns)
                         / static_cast<double>(dt);

                TransformMatrix4d interpolated =
                    TransformMatrix4d::lerp(prev->transform, it->transform, t);

                return TransformEntry(timestamp_ns, interpolated);
            }
        }

        return std::nullopt;
    }

    // 获取时间范围（最早和最晚时间戳）
    std::optional<std::pair<uint64_t, uint64_t>> time_range() const {
        typename mutex_type::scoped_lock lock(mutex_, false);

        if (size_ == 0) {
            return std::nullopt;
        }

        uint64_t min_time = UINT64_MAX;
        uint64_t max_time = 0;

        for (std::size_t i = 0; i < size_; ++i) {
            std::size_t idx = (head_ + i) % Capacity;
            min_time        = std::min(min_time, buffer_[idx].timestamp_ns);
            max_time        = std::max(max_time, buffer_[idx].timestamp_ns);
        }

        return std::make_pair(min_time, max_time);
    }

    // 清空缓冲区
    void clear() {
        typename mutex_type::scoped_lock lock(mutex_);
        head_ = 0;
        size_ = 0;
    }

    // 获取缓冲区大小
    [[nodiscard]] std::size_t size() const {
        typename mutex_type::scoped_lock lock(mutex_, false);
        return size_;
    }

    // 获取缓冲区容量
    [[nodiscard]] constexpr std::size_t capacity() const { return Capacity; }

    // 检查缓冲区是否满
    [[nodiscard]] bool is_full() const {
        typename mutex_type::scoped_lock lock(mutex_, false);
        return size_ == Capacity;
    }

    // 检查缓冲区是否为空
    [[nodiscard]] bool is_empty() const {
        typename mutex_type::scoped_lock lock(mutex_, false);
        return size_ == 0;
    }

    // 检查某个时间戳是否在范围内
    [[nodiscard]] bool contains_time(uint64_t timestamp_ns) const {
        typename mutex_type::scoped_lock lock(mutex_, false);

        if (size_ == 0) {
            return false;
        }

        auto range = time_range();
        if (range) {
            return timestamp_ns >= range->first && timestamp_ns <= range->second;
        }

        return false;
    }

    // 获取所有条目的副本（用于调试和测试）
    std::vector<TransformEntry> get_all_entries() const {
        typename mutex_type::scoped_lock lock(mutex_, false);

        std::vector<TransformEntry> entries;
        entries.reserve(size_);

        for (std::size_t i = 0; i < size_; ++i) {
            std::size_t idx = (head_ + i) % Capacity;
            entries.push_back(buffer_[idx]);
        }

        std::sort(entries.begin(), entries.end());
        return entries;
    }

private:
    std::array<TransformEntry, Capacity> buffer_;
    std::size_t head_ = 0; // 下一个写入位置
    std::size_t size_ = 0; // 当前条目数
    mutable mutex_type mutex_;
};

// ============================================================================
// 便捷的树定义宏
// ============================================================================

template <std::size_t N>
struct BufferCapacity {
    static constexpr std::size_t value = N;
    using buffer_type                  = CircularTransformBuffer<N>;
};

// 别名（避免与模板类重复）
// TransformBuffer 已作为类定义存在，不需要别名

} // namespace fast_tf
