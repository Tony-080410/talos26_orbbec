cmake_minimum_required(VERSION 3.14)
project(talos VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Performance optimization flags for microsecond-level latency
add_compile_options(-O3 -march=native)
add_compile_options(-Wall -Wextra -Wpedantic)

# Sanitizers (optional - enable via CMake flags)
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
option(ENABLE_COVERAGE "Enable code coverage" OFF)

if(ENABLE_ASAN)
  message(STATUS "AddressSanitizer enabled")
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address)
endif()

if(ENABLE_TSAN)
  message(STATUS "ThreadSanitizer enabled")
  add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
  add_link_options(-fsanitize=thread)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(TBB REQUIRED)
find_package(GTest REQUIRED)

# Benchmark is optional (not essential for Day 1)
find_package(benchmark QUIET)

include_directories(include)

# Main executable
add_executable(${PROJECT_NAME}
    src/main.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE include)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    spdlog::spdlog
    fmt::fmt
    Eigen3::Eigen
    TBB::tbb
)

# Unit tests
enable_testing()

add_executable(tracker_soa_test
    test/tracker_soa_test.cpp
)

target_include_directories(tracker_soa_test PRIVATE include)

target_link_libraries(tracker_soa_test PRIVATE
    GTest::GTest
    GTest::Main
)

add_test(NAME TrackerSoATest COMMAND tracker_soa_test)

add_executable(triple_buffer_test
    test/triple_buffer_test.cpp
)

target_include_directories(triple_buffer_test PRIVATE include)

target_link_libraries(triple_buffer_test PRIVATE
    GTest::GTest
    GTest::Main
)

add_test(NAME TripleBufferTest COMMAND triple_buffer_test)

# Performance probe tests
add_executable(performance_probe_test
    test/performance_probe_test.cpp
)

target_include_directories(performance_probe_test PRIVATE include)

target_link_libraries(performance_probe_test PRIVATE
    GTest::GTest
    GTest::Main
)

add_test(NAME PerformanceProbeTest COMMAND performance_probe_test)

# Integration tests
add_executable(integration_test
    test/integration_test.cpp
)

target_include_directories(integration_test PRIVATE include)

target_link_libraries(integration_test PRIVATE
    GTest::GTest
    GTest::Main
)

add_test(NAME IntegrationTest COMMAND integration_test)

# Latency benchmarks (only if benchmark library is found)
if(benchmark_FOUND)
  add_executable(latency_benchmark
      test/latency_benchmark.cpp
  )

  target_include_directories(latency_benchmark PRIVATE include)

  target_link_libraries(latency_benchmark PRIVATE
      benchmark::benchmark
      benchmark::benchmark_main
  )

  add_test(NAME LatencyBenchmark COMMAND latency_benchmark)
else()
  message(STATUS "benchmark library not found - benchmark tests will be skipped")
  message(STATUS "To enable: brew install google-benchmark")
endif()

if(ENABLE_COVERAGE)
    foreach(tgt ${PROJECT_NAME} tracker_soa_test triple_buffer_test performance_probe_test integration_test)
        target_compile_options(${tgt} PRIVATE -fprofile-arcs -ftest-coverage -O0 -g)
        target_link_options(${tgt} PRIVATE --coverage)
    endforeach()
endif()

# Coverage target (only if coverage is enabled)
if(ENABLE_COVERAGE)
  find_program(LCOV lcov)
  find_program(GENHTML genhtml)

  if(LCOV AND GENHTML)
    add_custom_target(coverage
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/coverage
      COMMAND ${LCOV} --directory . --zerocounters
      COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
      COMMAND ${LCOV} --directory . --capture --output-file coverage.info
      COMMAND ${LCOV} --remove coverage.info '/usr/*' '*/test/*' --output-file coverage.info
      COMMAND ${GENHTML} coverage.info --output-directory ${CMAKE_BINARY_DIR}/coverage
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      COMMENT "Generating code coverage report"
    )
    message(STATUS "Coverage target available: make coverage")
  else()
    message(STATUS "lcov/genhtml not found - coverage target unavailable")
    message(STATUS "To enable: brew install lcov")
  endif()
endif()

