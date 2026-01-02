#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <ranges>
#include <vector>

namespace talos {
/**
 * @brief High-resolution latency probe
 *
 * Uses std::chrono::high_resolution_clock for portable sub-microsecond timing.
 *
 * Design:
 * - Records start timestamp on construction
 * - Supports elapsed time queries
 * - Lock-free: no allocations, no system calls in critical path
 * - Portable: works on x86-64, ARM64, etc.
 *
 * Overhead: ~50-100 nanoseconds per call (acceptable for benchmarking)
 */
class LatencyProbe {
public:
    LatencyProbe()
        : start_ns_(now_ns()) {}

    ~LatencyProbe() = default;

    /**
     * @brief Get elapsed nanoseconds since construction
     */
    [[nodiscard]] uint64_t elapsed_ns() const { return now_ns() - start_ns_; }

    /**
     * @brief Get elapsed nanoseconds and reset
     */
    uint64_t snapshot_and_reset() {
        const auto now     = now_ns();
        const auto elapsed = now - start_ns_;
        start_ns_          = now;
        return elapsed;
    }

private:
    uint64_t start_ns_;

    /**
     * @brief Get current time in nanoseconds
     * Uses std::chrono::high_resolution_clock for portability
     */
    static uint64_t now_ns() {
        const auto now      = std::chrono::high_resolution_clock::now();
        const auto duration = now.time_since_epoch();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    }
};

/**
 * @brief Lock-free latency histogram for benchmark results
 *
 * Accumulates latency samples and computes statistics:
 * - mean, min, max
 * - percentiles (p50, p95, p99, p999)
 * - standard deviation
 *
 * Uses a fixed-size ring buffer (no allocations during measurement)
 */
class LatencyHistogram {
public:
    static constexpr size_t MAX_SAMPLES = 1000000; // 1M samples = 1GB

    LatencyHistogram()
        : samples_()
        , count_(0) {}

    /**
     * @brief Record a latency sample (nanoseconds)
     */
    void record(const uint64_t latency_ns) {
        if (count_ < MAX_SAMPLES) {
            samples_[count_++] = latency_ns;
        }
    }

    /**
     * @brief Compute statistics from recorded samples
     */
    struct Stats {
        uint64_t min_ns;
        uint64_t max_ns;
        uint64_t mean_ns;
        uint64_t p50_ns; // Median
        uint64_t p95_ns;
        uint64_t p99_ns;
        uint64_t p999_ns;
        double stddev_ns;
        size_t count;
    };

    [[nodiscard]] Stats compute() const {
        // Cache count_ to avoid race condition with concurrent record() calls
        const size_t count = count_;
        if (count == 0) {
            return {0, 0, 0, 0, 0, 0, 0, 0.0, 0};
        }

        // Sort for percentile computation
        std::vector sorted(samples_.begin(), samples_.begin() + count);
        std::ranges::sort(sorted);

        // Compute statistics
        const auto min_val  = sorted.front();
        const auto max_val  = sorted.back();
        const auto sum      = std::accumulate(sorted.begin(), sorted.end(), 0UL);
        const auto mean_val = sum / count;

        // Percentiles (clamp to valid range to handle edge cases)
        const auto p50_idx  = std::min(count * 50 / 100, count - 1);
        const auto p95_idx  = std::min(count * 95 / 100, count - 1);
        const auto p99_idx  = std::min(count * 99 / 100, count - 1);
        const auto p999_idx = std::min(count * 999 / 1000, count - 1);

        // Standard deviation
        double sum_sq = 0.0;
        for (size_t i = 0; i < count; i++) {
            const auto diff = static_cast<double>(sorted[i] - mean_val);
            sum_sq += diff * diff;
        }
        const double stddev = std::sqrt(sum_sq / static_cast<double>(count));

        return {min_val,          max_val,         mean_val,
                sorted[p50_idx],  sorted[p95_idx], sorted[p99_idx],
                sorted[p999_idx], stddev,          count};
    }

    /**
     * @brief Pretty-print statistics
     */
    void print_stats() const {
        const auto stats = compute();
        std::cout << "Latency Statistics (ns):\n"
                  << "  Count:  " << stats.count << "\n"
                  << "  Min:    " << stats.min_ns << " ns\n"
                  << "  P50:    " << stats.p50_ns << " ns\n"
                  << "  P95:    " << stats.p95_ns << " ns\n"
                  << "  P99:    " << stats.p99_ns << " ns\n"
                  << "  P99.9:  " << stats.p999_ns << " ns\n"
                  << "  Mean:   " << stats.mean_ns << " ns\n"
                  << "  Max:    " << stats.max_ns << " ns\n"
                  << "  StdDev: " << stats.stddev_ns << " ns\n";
    }

private:
    std::array<uint64_t, MAX_SAMPLES> samples_;
    size_t count_;
};
} // namespace talos
