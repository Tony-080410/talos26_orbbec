cmake_minimum_required(VERSION 3.14)

find_program(SCCACHE_PROGRAM sccache)

if(SCCACHE_PROGRAM)
    message(STATUS "sccache found: ${SCCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER   ${SCCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${SCCACHE_PROGRAM})
else()
    message(STATUS "sccache not found, building without compiler cache")
endif()

project(talos VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(TALOS_OFFLINE "Build without network" ON)

if (TALOS_OFFLINE)
    set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
endif()

# Performance optimization flags for microsecond-level latency
add_compile_options(-O3 -march=native)
add_compile_options(-Wall -Wextra -Wpedantic)

# Sanitizers (optional - enable via CMake flags)
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
option(ENABLE_COVERAGE "Enable code coverage" OFF)

if (ENABLE_ASAN)
    message(STATUS "AddressSanitizer enabled")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif ()

if (ENABLE_TSAN)
    message(STATUS "ThreadSanitizer enabled")
    add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
    add_link_options(-fsanitize=thread)
endif ()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(TBB REQUIRED)
find_package(GTest REQUIRED)

# Benchmark is optional (not essential for Day 1)
find_package(benchmark QUIET)

include(FetchContent)
FetchContent_Declare(
        tomlplusplus
        GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
        GIT_TAG v3.4.0
)
FetchContent_MakeAvailable(tomlplusplus)

include_directories(include)

# Scheduler library (header-only template, no .cpp compilation needed)
add_library(scheduler INTERFACE)

target_include_directories(scheduler INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(scheduler INTERFACE
        ${OpenCV_LIBS}
        spdlog::spdlog
        fmt::fmt
        TBB::tbb
)

# Main executable
add_executable(${PROJECT_NAME}
        src/main.cpp
        include/scheduler/primitives/buffer_pool.hpp
        include/scheduler/primitives/latency_stats.hpp
        include/toml_helper.hpp
        include/sp_vision/config.hpp
)

target_include_directories(${PROJECT_NAME} PRIVATE include)

target_link_libraries(${PROJECT_NAME} PRIVATE
        scheduler
        ${OpenCV_LIBS}
        spdlog::spdlog
        fmt::fmt
        Eigen3::Eigen
        TBB::tbb
        tomlplusplus::tomlplusplus
)

# sp_vision executable
add_executable(sp_vision
        src/sp_vision/sp_vision.cpp
        src/sp_vision/auto_aim/detector.cpp
)


target_include_directories(sp_vision PRIVATE include)

target_link_libraries(sp_vision PRIVATE
        scheduler
        ${OpenCV_LIBS}
        spdlog::spdlog
        fmt::fmt
        Eigen3::Eigen
        TBB::tbb
        tomlplusplus::tomlplusplus
)

# Detector benchmark (accuracy testing against dataset)
add_executable(detector_benchmark
        test/detector_benchmark.cpp
        src/sp_vision/auto_aim/detector.cpp
)

target_include_directories(detector_benchmark PRIVATE include)

target_link_libraries(detector_benchmark PRIVATE
        ${OpenCV_LIBS}
        spdlog::spdlog
        TBB::tbb
        fmt::fmt
        tomlplusplus::tomlplusplus
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

# UKF logic tests
add_executable(ukf_test
        test/ukf_test.cpp
)

target_include_directories(ukf_test PRIVATE include)

target_link_libraries(ukf_test PRIVATE
        GTest::GTest
        GTest::Main
        Eigen3::Eigen
)

add_test(NAME UKFTest COMMAND ukf_test)

# Scheduler tests
add_executable(scheduler_test
        test/scheduler_test.cpp
)

target_include_directories(scheduler_test PRIVATE include)

target_link_libraries(scheduler_test PRIVATE
        scheduler
        GTest::GTest
        GTest::Main
        ${OpenCV_LIBS}
        TBB::tbb
)

add_test(NAME SchedulerTest COMMAND scheduler_test)

# Integration scheduler test (comprehensive)
add_executable(integration_scheduler_test
        test/integration_scheduler_test.cpp
)

target_include_directories(integration_scheduler_test PRIVATE include)

target_link_libraries(integration_scheduler_test PRIVATE
        scheduler
        GTest::GTest
        GTest::Main
        ${OpenCV_LIBS}
        TBB::tbb
)

add_test(NAME IntegrationSchedulerTest COMMAND integration_scheduler_test)

# Latency benchmarks (only if benchmark library is found)
if (benchmark_FOUND)
    add_executable(latency_benchmark
            test/latency_benchmark.cpp
    )

    target_include_directories(latency_benchmark PRIVATE include)

    target_link_libraries(latency_benchmark PRIVATE
            benchmark::benchmark
            benchmark::benchmark_main
    )

    add_test(NAME LatencyBenchmark COMMAND latency_benchmark)

    # UKF benchmark (separate)
    add_executable(ukf_benchmark
            test/ukf_benchmark.cpp
    )

    target_include_directories(ukf_benchmark PRIVATE include)

    target_link_libraries(ukf_benchmark PRIVATE
            benchmark::benchmark
            Eigen3::Eigen
    )

    add_test(NAME UKFBenchmark COMMAND ukf_benchmark)

    # Scheduler benchmark
    add_executable(scheduler_benchmark
            test/scheduler_benchmark.cpp
    )

    target_include_directories(scheduler_benchmark PRIVATE include)

    target_link_libraries(scheduler_benchmark PRIVATE
            scheduler
            benchmark::benchmark
            ${OpenCV_LIBS}
            TBB::tbb
    )

    add_test(NAME SchedulerBenchmark COMMAND scheduler_benchmark)

    # Integration scheduler benchmark
    add_executable(integration_scheduler_benchmark
            test/integration_scheduler_benchmark.cpp
    )

    target_include_directories(integration_scheduler_benchmark PRIVATE include)

    target_link_libraries(integration_scheduler_benchmark PRIVATE
            scheduler
            benchmark::benchmark
            ${OpenCV_LIBS}
            TBB::tbb
    )

    add_test(NAME IntegrationSchedulerBenchmark COMMAND integration_scheduler_benchmark)

    # RealWorld scheduler benchmark (with camera)
    add_executable(realworld_scheduler_benchmark
            test/realworld_scheduler_benchmark.cpp
    )

    target_include_directories(realworld_scheduler_benchmark PRIVATE include)

    target_link_libraries(realworld_scheduler_benchmark PRIVATE
            scheduler
            ${OpenCV_LIBS}
            TBB::tbb
    )
else ()
    message(STATUS "benchmark library not found - benchmark tests will be skipped")
    message(STATUS "To enable: brew install google-benchmark")
endif ()

if (ENABLE_COVERAGE)
    foreach (tgt ${PROJECT_NAME} tracker_soa_test triple_buffer_test performance_probe_test integration_test ukf_test)
        target_compile_options(${tgt} PRIVATE -fprofile-arcs -ftest-coverage -O0 -g)
        target_link_options(${tgt} PRIVATE --coverage)
    endforeach ()
endif ()

# Coverage target (only if coverage is enabled)
if (ENABLE_COVERAGE)
    find_program(LCOV lcov)
    find_program(GENHTML genhtml)

    if (LCOV AND GENHTML)
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
    else ()
        message(STATUS "lcov/genhtml not found - coverage target unavailable")
        message(STATUS "To enable: brew install lcov")
    endif ()
endif ()

