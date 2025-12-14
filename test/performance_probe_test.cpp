#include <gtest/gtest.h>
#include <thread>
#include <atomic>
#include <vector>
#include <chrono>
#include "performance_probe.hpp"

namespace spike::test {

// ============================================================================
// Test: Basic Timing Functionality
// ============================================================================

TEST(LatencyProbe, BasicTiming) {
  LatencyProbe probe;

  // Sleep for known duration
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  auto elapsed = probe.elapsed_ns();

  // Should be approximately 10ms (with some tolerance)
  EXPECT_GT(elapsed, 9'000'000);   // At least 9ms
  EXPECT_LT(elapsed, 15'000'000);  // At most 15ms (generous tolerance)
}

TEST(LatencyProbe, SubMicrosecondPrecision) {
  LatencyProbe probe;

  // Busy-wait for short duration to test precision
  auto start = std::chrono::high_resolution_clock::now();
  while (std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::high_resolution_clock::now() - start).count() < 1000) {
    // Busy spin for ~1 microsecond
  }

  auto elapsed = probe.elapsed_ns();

  // Should measure sub-microsecond resolution
  EXPECT_GT(elapsed, 500);     // At least 500ns
  EXPECT_LT(elapsed, 100'000); // Less than 100us
}

TEST(LatencyProbe, SnapshotAndReset) {
  LatencyProbe probe;

  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  auto first_elapsed = probe.snapshot_and_reset();

  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  auto second_elapsed = probe.elapsed_ns();

  // First measurement should be ~5ms
  EXPECT_GT(first_elapsed, 4'000'000);
  EXPECT_LT(first_elapsed, 8'000'000);

  // Second measurement should be ~5ms (not 10ms)
  EXPECT_GT(second_elapsed, 4'000'000);
  EXPECT_LT(second_elapsed, 8'000'000);
}

TEST(LatencyProbe, ZeroOverheadConstruction) {
  // Measure construction overhead
  auto start = std::chrono::high_resolution_clock::now();
  LatencyProbe probe;
  auto end = std::chrono::high_resolution_clock::now();

  auto overhead_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  // Construction should be extremely fast (< 1us)
  EXPECT_LT(overhead_ns, 1000);
}

// ============================================================================
// Test: Measurement Overhead
// ============================================================================

TEST(LatencyProbe, MeasurementOverhead) {
  constexpr int ITERATIONS = 10000;
  std::vector<uint64_t> overhead_samples;
  overhead_samples.reserve(ITERATIONS);

  for (int i = 0; i < ITERATIONS; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    LatencyProbe probe;
    [[maybe_unused]] auto elapsed = probe.elapsed_ns();
    auto end = std::chrono::high_resolution_clock::now();

    auto overhead_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    overhead_samples.push_back(overhead_ns);
  }

  // Compute mean overhead
  uint64_t sum = 0;
  for (auto sample : overhead_samples) {
    sum += sample;
  }
  uint64_t mean_overhead = sum / ITERATIONS;

  // Overhead should be < 500ns on average
  EXPECT_LT(mean_overhead, 500);

  std::cout << "Mean measurement overhead: " << mean_overhead << " ns\n";
}

// ============================================================================
// Test: Multithreading Safety
// ============================================================================

TEST(LatencyProbe, MultiThreadSafety) {
  constexpr int NUM_THREADS = 4;
  constexpr int ITERATIONS_PER_THREAD = 1000;

  std::vector<std::thread> threads;
  std::atomic<int> errors{0};

  for (int t = 0; t < NUM_THREADS; t++) {
    threads.emplace_back([&errors]() {
      for (int i = 0; i < ITERATIONS_PER_THREAD; i++) {
        LatencyProbe probe;

        // Simulate short workload
        std::this_thread::sleep_for(std::chrono::microseconds(10));

        auto elapsed = probe.elapsed_ns();

        // Should be at least 10us
        if (elapsed < 5'000) {
          errors++;
        }
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // No measurement errors
  EXPECT_EQ(errors.load(), 0);
}

TEST(LatencyProbe, ConcurrentProbes) {
  // Test that multiple probes can run concurrently without interference
  std::vector<std::thread> threads;
  std::atomic<bool> start{false};
  std::vector<uint64_t> measurements(4);

  for (int t = 0; t < 4; t++) {
    threads.emplace_back([&start, &measurements, t]() {
      // Wait for all threads to be ready
      while (!start.load()) {
        std::this_thread::yield();
      }

      LatencyProbe probe;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      measurements[t] = probe.elapsed_ns();
    });
  }

  // Start all threads simultaneously
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  start.store(true);

  for (auto& thread : threads) {
    thread.join();
  }

  // All measurements should be approximately 10ms
  for (size_t i = 0; i < measurements.size(); i++) {
    EXPECT_GT(measurements[i], 9'000'000) << "Thread " << i << " measurement too low";
    EXPECT_LT(measurements[i], 15'000'000) << "Thread " << i << " measurement too high";
  }
}

// ============================================================================
// Test: Histogram Statistics
// ============================================================================

TEST(LatencyHistogram, EmptyHistogram) {
  LatencyHistogram hist;
  auto stats = hist.compute();

  EXPECT_EQ(stats.count, 0);
  EXPECT_EQ(stats.min_ns, 0);
  EXPECT_EQ(stats.max_ns, 0);
  EXPECT_EQ(stats.mean_ns, 0);
}

TEST(LatencyHistogram, SingleSample) {
  LatencyHistogram hist;
  hist.record(1000);

  auto stats = hist.compute();

  EXPECT_EQ(stats.count, 1);
  EXPECT_EQ(stats.min_ns, 1000);
  EXPECT_EQ(stats.max_ns, 1000);
  EXPECT_EQ(stats.mean_ns, 1000);
  EXPECT_EQ(stats.p50_ns, 1000);
  EXPECT_EQ(stats.p95_ns, 1000);
  EXPECT_EQ(stats.p99_ns, 1000);
  EXPECT_DOUBLE_EQ(stats.stddev_ns, 0.0);
}

TEST(LatencyHistogram, UniformDistribution) {
  LatencyHistogram hist;

  // Record 100 samples: 1000, 1001, 1002, ..., 1099
  for (int i = 0; i < 100; i++) {
    hist.record(1000 + i);
  }

  auto stats = hist.compute();

  EXPECT_EQ(stats.count, 100);
  EXPECT_EQ(stats.min_ns, 1000);
  EXPECT_EQ(stats.max_ns, 1099);
  EXPECT_EQ(stats.mean_ns, 1049);  // (1000 + 1099) / 2

  // Percentiles
  EXPECT_NEAR(stats.p50_ns, 1050, 5);  // Median ~1050
  EXPECT_NEAR(stats.p95_ns, 1095, 5);  // P95 ~1095
  EXPECT_NEAR(stats.p99_ns, 1099, 2);  // P99 ~1099
}

TEST(LatencyHistogram, OutlierDetection) {
  LatencyHistogram hist;

  // Record 100 samples: 99 at 1000ns, 1 outlier at 10000ns
  for (int i = 0; i < 99; i++) {
    hist.record(1000);
  }
  hist.record(10000);

  auto stats = hist.compute();

  EXPECT_EQ(stats.count, 100);
  EXPECT_EQ(stats.min_ns, 1000);
  EXPECT_EQ(stats.max_ns, 10000);

  // P99 should capture the outlier
  EXPECT_EQ(stats.p99_ns, 10000);

  // P95 should still be 1000 (95% are 1000ns)
  EXPECT_EQ(stats.p95_ns, 1000);
}

TEST(LatencyHistogram, PercentilesAccuracy) {
  LatencyHistogram hist;

  // Record 1000 samples with known distribution
  for (int i = 0; i < 1000; i++) {
    hist.record(i);  // 0, 1, 2, ..., 999
  }

  auto stats = hist.compute();

  EXPECT_EQ(stats.count, 1000);
  EXPECT_EQ(stats.min_ns, 0);
  EXPECT_EQ(stats.max_ns, 999);
  EXPECT_EQ(stats.mean_ns, 499);  // (0 + 999) / 2

  // Verify percentiles
  EXPECT_NEAR(stats.p50_ns, 500, 10);   // P50 = 500
  EXPECT_NEAR(stats.p95_ns, 950, 10);   // P95 = 950
  EXPECT_NEAR(stats.p99_ns, 990, 10);   // P99 = 990
  EXPECT_NEAR(stats.p999_ns, 999, 10);  // P99.9 = 999
}

TEST(LatencyHistogram, LargeDataset) {
  LatencyHistogram hist;

  // Record 10,000 samples
  for (int i = 0; i < 10000; i++) {
    hist.record(1000 + (i % 100));  // Cycle 1000-1099
  }

  auto stats = hist.compute();

  EXPECT_EQ(stats.count, 10000);
  EXPECT_GE(stats.min_ns, 1000);
  EXPECT_LE(stats.max_ns, 1099);

  // Mean should be around 1050
  EXPECT_NEAR(stats.mean_ns, 1050, 10);
}

// ============================================================================
// Test: Real-world Latency Scenarios
// ============================================================================

TEST(LatencyProbe, TypicalWorkloadMeasurement) {
  LatencyHistogram hist;

  // Simulate 100 iterations of a typical workload
  for (int i = 0; i < 100; i++) {
    LatencyProbe probe;

    // Simulate variable workload (100-500 microseconds)
    int workload_us = 100 + (i % 5) * 100;
    std::this_thread::sleep_for(std::chrono::microseconds(workload_us));

    hist.record(probe.elapsed_ns());
  }

  auto stats = hist.compute();

  EXPECT_EQ(stats.count, 100);

  // Mean should be around 300us (middle of 100-500us range)
  EXPECT_GT(stats.mean_ns, 200'000);  // > 200us
  EXPECT_LT(stats.mean_ns, 600'000);  // < 600us

  std::cout << "Typical workload statistics:\n";
  std::cout << "  P50:  " << stats.p50_ns << " ns\n";
  std::cout << "  P95:  " << stats.p95_ns << " ns\n";
  std::cout << "  P99:  " << stats.p99_ns << " ns\n";
}

TEST(LatencyProbe, HighFrequencyMeasurements) {
  LatencyHistogram hist;
  constexpr int ITERATIONS = 1000;

  // High-frequency measurements (no sleep between)
  for (int i = 0; i < ITERATIONS; i++) {
    LatencyProbe probe;

    // Minimal workload - just measure probe overhead
    volatile int dummy = 0;
    for (int j = 0; j < 100; j++) {
      dummy += j;
    }
    (void)dummy;

    hist.record(probe.elapsed_ns());
  }

  auto stats = hist.compute();

  EXPECT_EQ(stats.count, ITERATIONS);

  // All measurements should be sub-microsecond
  EXPECT_LT(stats.p99_ns, 10'000);  // P99 < 10us

  std::cout << "High-frequency measurement overhead:\n";
  std::cout << "  Mean: " << stats.mean_ns << " ns\n";
  std::cout << "  P99:  " << stats.p99_ns << " ns\n";
}

} // namespace spike::test
