#pragma once

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <new>

namespace primitive {

/**
 * @brief 无锁延迟直方图
 *
 * 用于收集延迟分布，支持百分位数计算。
 * 使用对数分桶，覆盖 100ns ~ 100ms 范围。
 *
 * 桶分布 (对数):
 * - [0]: 0-100ns
 * - [1]: 100-200ns
 * - [2]: 200-500ns
 * - [3]: 500ns-1us
 * - [4]: 1-2us
 * - [5]: 2-5us
 * - [6]: 5-10us
 * - [7]: 10-20us
 * - [8]: 20-50us
 * - [9]: 50-100us
 * - [10]: 100-200us
 * - [11]: 200-500us
 * - [12]: 500us-1ms
 * - [13]: 1-2ms
 * - [14]: 2-5ms
 * - [15]: 5-10ms
 * - [16]: 10-20ms
 * - [17]: 20-50ms
 * - [18]: 50-100ms
 * - [19]: >100ms
 */
class LatencyHistogram {
public:
    static constexpr size_t kNumBuckets = 20;

    LatencyHistogram() = default;

    /**
     * @brief 记录一个延迟样本 (纳秒)
     */
    void record(const uint64_t latency_ns) noexcept {
        const size_t bucket = latency_to_bucket(latency_ns);
        buckets_[bucket].fetch_add(1, std::memory_order_relaxed);
        count_.fetch_add(1, std::memory_order_relaxed);
        sum_.fetch_add(latency_ns, std::memory_order_relaxed);

        // 更新 min/max (无锁)
        uint64_t old_min = min_.load(std::memory_order_relaxed);
        while (latency_ns < old_min
               && !min_.compare_exchange_weak(old_min, latency_ns, std::memory_order_relaxed)) {}

        uint64_t old_max = max_.load(std::memory_order_relaxed);
        while (latency_ns > old_max
               && !max_.compare_exchange_weak(old_max, latency_ns, std::memory_order_relaxed)) {}
    }

    /**
     * @brief 重置所有统计
     */
    void reset() noexcept {
        for (auto& b : buckets_) {
            b.store(0, std::memory_order_relaxed);
        }
        count_.store(0, std::memory_order_relaxed);
        sum_.store(0, std::memory_order_relaxed);
        min_.store(UINT64_MAX, std::memory_order_relaxed);
        max_.store(0, std::memory_order_relaxed);
    }

    /**
     * @brief 获取样本数
     */
    [[nodiscard]] uint64_t count() const noexcept { return count_.load(std::memory_order_relaxed); }

    /**
     * @brief 获取平均延迟 (纳秒)
     */
    [[nodiscard]] double mean() const noexcept {
        const auto c = count_.load(std::memory_order_relaxed);
        if (c == 0)
            return 0.0;
        return static_cast<double>(sum_.load(std::memory_order_relaxed)) / c;
    }

    /**
     * @brief 获取最小延迟 (纳秒)
     */
    [[nodiscard]] uint64_t min() const noexcept {
        const auto m = min_.load(std::memory_order_relaxed);
        return m == UINT64_MAX ? 0 : m;
    }

    /**
     * @brief 获取最大延迟 (纳秒)
     */
    [[nodiscard]] uint64_t max() const noexcept { return max_.load(std::memory_order_relaxed); }

    /**
     * @brief 获取百分位数 (纳秒)
     * @param p 百分位 (0.0-1.0)
     */
    [[nodiscard]] uint64_t percentile(const double p) const noexcept {
        const auto total = count_.load(std::memory_order_relaxed);
        if (total == 0)
            return 0;

        if (!(p > 0.0)) {
            return min();
        }
        if (!(p < 1.0)) {
            return max();
        }

        const auto target_rank =
            static_cast<uint64_t>(std::ceil(p * static_cast<double>(total))); // 1..total
        uint64_t cumulative = 0;

        for (size_t i = 0; i < kNumBuckets; ++i) {
            cumulative += buckets_[i].load(std::memory_order_relaxed);
            if (cumulative >= target_rank) {
                return bucket_to_latency(i);
            }
        }
        return bucket_to_latency(kNumBuckets - 1);
    }

    /**
     * @brief 快捷方法: p50
     */
    [[nodiscard]] uint64_t p50() const noexcept { return percentile(0.50); }

    /**
     * @brief 快捷方法: p90
     */
    [[nodiscard]] uint64_t p90() const noexcept { return percentile(0.90); }

    /**
     * @brief 快捷方法: p99
     */
    [[nodiscard]] uint64_t p99() const noexcept { return percentile(0.99); }

    /**
     * @brief 统计结果结构体
     */
    struct HistogramStats {
        uint64_t count{0};
        uint64_t min_ns{0};
        uint64_t max_ns{0};
        double mean_ns{0.0};
        double stddev_ns{0.0};
        uint64_t p50_ns{0};
        uint64_t p90_ns{0};
        uint64_t p95_ns{0};
        uint64_t p99_ns{0};
        uint64_t p999_ns{0};
    };

    /**
     * @brief 计算并返回完整统计信息
     *
     * 注意：这是一个近似计算（基于直方图桶），
     * 标准差使用加权方差公式近似。
     */
    [[nodiscard]] HistogramStats compute() const noexcept {
        HistogramStats stats;
        const auto c = count_.load(std::memory_order_relaxed);
        stats.count  = c;

        if (c == 0) {
            return stats;
        }

        stats.min_ns  = min();
        stats.max_ns  = max();
        stats.mean_ns = mean();

        const auto snapshot = this->snapshot();
        double variance_sum = 0.0;
        uint64_t cumulative = 0;

        // 计算加权百分位数和方差
        for (size_t i = 0; i < kNumBuckets; ++i) {
            const uint64_t bucket_count    = snapshot[i];
            const uint64_t bucket_latency  = bucket_to_latency(i);
            const uint64_t prev_cumulative = cumulative;
            cumulative += bucket_count;

            // 更新百分位数
            const double rank = static_cast<double>(cumulative) / static_cast<double>(c);
            if (rank >= 0.50 && stats.p50_ns == 0) {
                stats.p50_ns = bucket_latency;
            }
            if (rank >= 0.90 && stats.p90_ns == 0) {
                stats.p90_ns = bucket_latency;
            }
            if (rank >= 0.95 && stats.p95_ns == 0) {
                stats.p95_ns = bucket_latency;
            }
            if (rank >= 0.99 && stats.p99_ns == 0) {
                stats.p99_ns = bucket_latency;
            }
            if (rank >= 0.999 && stats.p999_ns == 0) {
                stats.p999_ns = bucket_latency;
            }

            // 加权方差: Σ count * (latency - mean)²
            const double diff = static_cast<double>(bucket_latency) - stats.mean_ns;
            variance_sum += static_cast<double>(bucket_count) * diff * diff;
        }

        // 标准差 = sqrt(方差 / count)
        stats.stddev_ns = std::sqrt(variance_sum / static_cast<double>(c));

        return stats;
    }

    /**
     * @brief 获取桶计数 (用于调试/可视化)
     */
    [[nodiscard]] std::array<uint64_t, kNumBuckets> snapshot() const noexcept {
        std::array<uint64_t, kNumBuckets> result{};
        for (size_t i = 0; i < kNumBuckets; ++i) {
            result[i] = buckets_[i].load(std::memory_order_relaxed);
        }
        return result;
    }

private:
    // 桶边界 (纳秒)
    static constexpr std::array<uint64_t, kNumBuckets> kBucketBounds = {
        100,       // 0: 0-100ns
        200,       // 1: 100-200ns
        500,       // 2: 200-500ns
        1000,      // 3: 500ns-1us
        2000,      // 4: 1-2us
        5000,      // 5: 2-5us
        10000,     // 6: 5-10us
        20000,     // 7: 10-20us
        50000,     // 8: 20-50us
        100000,    // 9: 50-100us
        200000,    // 10: 100-200us
        500000,    // 11: 200-500us
        1000000,   // 12: 500us-1ms
        2000000,   // 13: 1-2ms
        5000000,   // 14: 2-5ms
        10000000,  // 15: 5-10ms
        20000000,  // 16: 10-20ms
        50000000,  // 17: 20-50ms
        100000000, // 18: 50-100ms
        UINT64_MAX // 19: >100ms
    };

    static constexpr size_t latency_to_bucket(const uint64_t latency_ns) noexcept {
        // Binary search for O(log n) instead of O(n)
        size_t lo = 0, hi = kNumBuckets - 1;
        while (lo < hi) {
            const size_t mid = lo + (hi - lo) / 2;
            if (latency_ns < kBucketBounds[mid]) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    }

    static constexpr uint64_t bucket_to_latency(const size_t bucket) noexcept {
        if (bucket == 0) {
            return 50; // 中点
        }
        if (bucket >= kNumBuckets - 1) {
            // Last bucket is open-ended (>100ms); return its lower bound to avoid overflow.
            return kBucketBounds[kNumBuckets - 2];
        }
        // 返回桶的中点
        return (kBucketBounds[bucket - 1] + kBucketBounds[bucket]) / 2;
    }

    alignas(std::hardware_destructive_interference_size)
        std::array<std::atomic<uint64_t>, kNumBuckets> buckets_{};
    alignas(std::hardware_destructive_interference_size) std::atomic<uint64_t> count_{0};
    alignas(std::hardware_destructive_interference_size) std::atomic<uint64_t> sum_{0};
    alignas(std::hardware_destructive_interference_size) std::atomic<uint64_t> min_{UINT64_MAX};
    alignas(std::hardware_destructive_interference_size) std::atomic<uint64_t> max_{0};
};

} // namespace primitive
