# Testing Guide

## Quick Start

### Build and Run All Tests

```bash
# Configure and build
cmake -B build -S .
cmake --build build -j8

# Run all tests
cd build
ctest --output-on-failure
```

**Expected Output**:
```
Test project /path/to/talos-cpp/build
    Start 1: TrackerSoATest
1/5 Test #1: TrackerSoATest ...................   Passed    0.00 sec
    Start 2: TripleBufferTest
2/5 Test #2: TripleBufferTest .................   Passed    0.52 sec
    Start 3: PerformanceProbeTest
3/5 Test #3: PerformanceProbeTest .............   Passed    0.11 sec
    Start 4: IntegrationTest
4/5 Test #4: IntegrationTest ..................   Passed    2.09 sec
    Start 5: LatencyBenchmark
5/5 Test #5: LatencyBenchmark .................   Passed    6.42 sec

100% tests passed, 0 tests failed out of 5
Total Test time (real) =   9.15 sec
```

---

## Test Suites

### 1. TrackerSoATest (22 tests)

**Purpose**: Memory layout, cache-line alignment, state machine

```bash
./bin/tracker_soa_test
```

**Key Tests**:
- Memory layout verification (64-byte alignment)
- ArmorSlots bitmask operations
- TrackerSoA state machine transitions
- TripleBuffer lock-free operations

### 2. TripleBufferTest (2 tests)

**Purpose**: Concurrent producer-consumer stress test

```bash
./bin/triple_buffer_test
```

**Key Tests**:
- ConcurrentProducerConsumer: 500ms stress test
- DataIntegrity: Sequential writes with race condition prevention

### 3. PerformanceProbeTest (15 tests)

**Purpose**: Latency measurement precision and thread safety

```bash
./bin/performance_probe_test
```

**Key Tests**:
- Sub-microsecond timing precision
- Measurement overhead (< 500ns)
- Multithreading safety (4 threads × 1000 iterations)
- Histogram statistics (percentiles, outliers)

### 4. IntegrationTest (7 tests)

**Purpose**: End-to-end pipeline validation

```bash
./bin/integration_test
```

**Key Tests**:
- DetectorToTrackerPipeline: Full workflow
- EndToEndLatency: P99 < 1µs validation
- StressTestAllComponents: 1-second high-frequency test

### 5. LatencyBenchmark (Google Benchmark)

**Purpose**: Microbenchmarks with statistical analysis

```bash
./bin/latency_benchmark
```

---

## Sanitizers

### AddressSanitizer (ASan)

Detect memory leaks, buffer overflows, use-after-free:

```bash
# Build with ASan
cmake -B build -S . -DENABLE_ASAN=ON
cmake --build build -j8

# Run tests
cd build
ctest --output-on-failure
```

**What it detects**:
- Memory leaks
- Heap buffer overflow
- Stack buffer overflow
- Use-after-free
- Use-after-return
- Double-free

### ThreadSanitizer (TSan)

Detect data races in concurrent code:

```bash
# Build with TSan
cmake -B build -S . -DENABLE_TSAN=ON
cmake --build build -j8

# Run concurrent tests only
cd build
ctest -R "TripleBuffer|Integration" --output-on-failure
```

**What it detects**:
- Data races
- Deadlocks
- Thread leaks

**Note**: TSan may report false positives with lock-free code. Review warnings carefully.

---

## Code Coverage

### Generate Coverage Report

```bash
# Install lcov (if not already installed)
brew install lcov  # macOS
# or
sudo apt-get install lcov  # Linux

# Build with coverage
cmake -B build -S . -DENABLE_COVERAGE=ON
cmake --build build -j8

# Generate report
cd build
make coverage

# View report
open coverage/index.html  # macOS
# or
xdg-open coverage/index.html  # Linux
```

**Coverage Target**: > 90% line coverage

**What it shows**:
- Line-by-line execution counts
- Branch coverage
- Function coverage
- Untested code paths

---

## Advanced Usage

### Run Specific Test

```bash
# Run single test suite
./bin/tracker_soa_test

# Run specific test
./bin/tracker_soa_test --gtest_filter=MemoryLayout.TrackerSoAExactSize

# Run all tests in a group
./bin/tracker_soa_test --gtest_filter=MemoryLayout.*
```

### Verbose Output

```bash
# Show all test names
./bin/tracker_soa_test --gtest_list_tests

# Verbose execution
./bin/tracker_soa_test --gtest_color=yes

# Repeat tests (check for flakiness)
./bin/triple_buffer_test --gtest_repeat=10
```

### Benchmark Options

```bash
# Run benchmarks with specific filter
./bin/latency_benchmark --benchmark_filter=TripleBuffer

# Set number of iterations
./bin/latency_benchmark --benchmark_repetitions=10

# Output to file
./bin/latency_benchmark --benchmark_out=results.json --benchmark_out_format=json
```

---

## Performance Validation

### Latency Targets

| Operation | Target | Test |
|-----------|--------|------|
| Triple-buffer write | < 100ns | IntegrationTest.EndToEndLatency |
| Triple-buffer read | < 100ns | IntegrationTest.EndToEndLatency |
| End-to-end (P99) | < 1µs | IntegrationTest.EndToEndLatency |
| State transition | < 50ns | TrackerSoATest.StateTransitions |

### Throughput Targets

| Metric | Target | Test |
|--------|--------|------|
| Producer rate | > 100K fps | IntegrationTest.StressTestAllComponents |
| Consumer rate | > 50K fps | IntegrationTest.StressTestAllComponents |
| Total throughput | > 1M fps | IntegrationTest.StressTestAllComponents |

---

## Troubleshooting

### Test Failures

#### "frames_consumed < 200" in IntegrationTest

**Cause**: Timing-sensitive test, may fail under high system load

**Fix**: Test uses relaxed thresholds (> 180), but if still failing:
```bash
# Ensure no background processes
# Re-run test
./bin/integration_test --gtest_filter=Integration.DetectorToTrackerPipeline
```

#### "Version decrement detected" in TripleBufferTest

**Cause**: Data race in lock-free code (serious issue)

**Fix**: This should NOT happen. Investigate:
```bash
# Run with TSan
cmake -B build -DENABLE_TSAN=ON && cmake --build build
./bin/triple_buffer_test
```

### Sanitizer Warnings

#### ASan: "Memory leak detected"

Check the leak location. If it's in:
- **Test code**: Fix the test
- **spdlog/third-party**: Likely false positive (static globals)

#### TSan: "Data race on atomic variable"

Lock-free code uses relaxed memory ordering. Check if:
- Race is on non-atomic variable → **BUG** (fix immediately)
- Race is on atomic with relaxed ordering → **Expected** (verify correctness)

### Performance Issues

#### Tests take > 20 seconds

**Possible causes**:
1. Debug build (should be Release for benchmarks)
2. High system load
3. Thermal throttling

**Fix**:
```bash
# Use Release build for benchmarks
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
cd build && ctest
```

---

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        compiler: [gcc-11, clang-14]
        build_type: [Debug, Release]
        sanitizer: [none, asan, tsan]

    steps:
      - uses: actions/checkout@v3

      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y cmake g++-11 clang-14 \
            libeigen3-dev libopencv-dev libspdlog-dev \
            libfmt-dev libtbb-dev libgtest-dev lcov

      - name: Configure
        run: |
          FLAGS=""
          if [ "${{ matrix.sanitizer }}" = "asan" ]; then
            FLAGS="-DENABLE_ASAN=ON"
          elif [ "${{ matrix.sanitizer }}" = "tsan" ]; then
            FLAGS="-DENABLE_TSAN=ON"
          fi
          cmake -B build -S . -DCMAKE_BUILD_TYPE=${{ matrix.build_type }} $FLAGS

      - name: Build
        run: cmake --build build -j$(nproc)

      - name: Test
        run: cd build && ctest --output-on-failure

      - name: Coverage
        if: matrix.build_type == 'Debug' && matrix.sanitizer == 'none'
        run: |
          cd build
          cmake . -DENABLE_COVERAGE=ON
          cmake --build . -j$(nproc)
          make coverage
          bash <(curl -s https://codecov.io/bash)
```

---

## Test Development

### Adding New Tests

1. Create test file in `test/` directory:
```cpp
#include <gtest/gtest.h>
#include "your_header.hpp"

TEST(YourComponent, BasicTest) {
  // Arrange
  YourComponent component;

  // Act
  auto result = component.do_something();

  // Assert
  EXPECT_EQ(result, expected_value);
}
```

2. Add to `CMakeLists.txt`:
```cmake
add_executable(your_test
    test/your_test.cpp
)

target_include_directories(your_test PRIVATE include)

target_link_libraries(your_test PRIVATE
    GTest::GTest
    GTest::Main
)

add_test(NAME YourTest COMMAND your_test)
```

3. Build and run:
```bash
cmake --build build
cd build && ctest -R YourTest
```

### Test Best Practices

1. **One assertion per test** (where possible)
2. **Use descriptive test names**: `TEST(Component, BehaviorUnderCondition)`
3. **Arrange-Act-Assert** pattern
4. **Clean up resources** with RAII or fixtures
5. **Avoid sleeps** (use synchronization primitives)
6. **Test edge cases**: empty, full, overflow, underflow

---

## References

- [Google Test Documentation](https://google.github.io/googletest/)
- [Google Benchmark Guide](https://github.com/google/benchmark)
- [AddressSanitizer Wiki](https://github.com/google/sanitizers/wiki/AddressSanitizer)
- [ThreadSanitizer Wiki](https://github.com/google/sanitizers/wiki/ThreadSanitizerCppManual)
- [lcov Documentation](http://ltp.sourceforge.net/coverage/lcov.php)

---

**Last Updated**: 2025-12-15
