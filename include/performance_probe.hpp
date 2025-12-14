#pragma once

#include <cstdint>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace spike {

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
  LatencyProbe() : start_ns_(now_ns()) {}

  ~LatencyProbe() {
    // Silent measurement (no I/O in critical path)
    // In real benchmark, this would record to lock-free histogram
  }

  /**
   * @brief Get elapsed nanoseconds since construction
   */
  inline uint64_t elapsed_ns() const {
    return now_ns() - start_ns_;
  }

  /**
   * @brief Get elapsed nanoseconds and reset
   */
  inline uint64_t snapshot_and_reset() {
    auto now = now_ns();
    auto elapsed = now - start_ns_;
    start_ns_ = now;
    return elapsed;
  }

 private:
  uint64_t start_ns_;

  /**
   * @brief Get current time in nanoseconds
   * Uses std::chrono::high_resolution_clock for portability
   */
  static inline uint64_t now_ns() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
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
  static constexpr size_t MAX_SAMPLES = 1000000;  // 1M samples = 1GB

  LatencyHistogram() : count_(0) {}

  /**
   * @brief Record a latency sample (nanoseconds)
   */
  inline void record(uint64_t latency_ns) {
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
    uint64_t p50_ns;    // Median
    uint64_t p95_ns;
    uint64_t p99_ns;
    uint64_t p999_ns;
    double stddev_ns;
    size_t count;
  };

  inline Stats compute() const {
    if (count_ == 0) {
      return {0, 0, 0, 0, 0, 0, 0, 0.0, 0};
    }

    // Sort for percentile computation
    std::vector<uint64_t> sorted(samples_.begin(), samples_.begin() + count_);
    std::sort(sorted.begin(), sorted.end());

    // Compute statistics
    auto min_val = sorted.front();
    auto max_val = sorted.back();
    auto sum = std::accumulate(sorted.begin(), sorted.end(), 0UL);
    auto mean_val = sum / count_;

    // Percentiles
    auto p50_idx = (count_ * 50) / 100;
    auto p95_idx = (count_ * 95) / 100;
    auto p99_idx = (count_ * 99) / 100;
    auto p999_idx = (count_ * 999) / 1000;

    // Standard deviation
    double sum_sq = 0.0;
    for (size_t i = 0; i < count_; i++) {
      double diff = sorted[i] - mean_val;
      sum_sq += diff * diff;
    }
    double stddev = std::sqrt(sum_sq / count_);

    return {
        min_val,
        max_val,
        mean_val,
        sorted[p50_idx],
        sorted[p95_idx],
        sorted[p99_idx],
        sorted[p999_idx],
        stddev,
        count_
    };
  }

  /**
   * @brief Pretty-print statistics
   */
  inline void print_stats() const {
    auto stats = compute();
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

} // namespace spike
