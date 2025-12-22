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

# IPC shared memory tests (memory safety and tearing)
add_executable(ipc_test
        test/ipc_test.cpp
)

target_include_directories(ipc_test PRIVATE include/ipc)

target_link_libraries(ipc_test PRIVATE
        GTest::GTest
        GTest::Main
        ${OpenCV_LIBS}
)

add_test(NAME IPCTest COMMAND ipc_test)

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

# ============================================================================
# Fast TF Tests
# ============================================================================

# Frame types unit tests
add_executable(test_frame_types
        tests/test_frame_types.cpp
        src/fast_tf.cpp
)

target_include_directories(test_frame_types PRIVATE include)

target_link_libraries(test_frame_types PRIVATE
        fast_tf
        GTest::GTest
        GTest::Main
        Eigen3::Eigen
        TBB::tbb
)

add_test(NAME FrameTypesTest COMMAND test_frame_types)

# Frame hierarchy validation tests
add_executable(test_frame_hierarchy
        tests/test_frame_hierarchy.cpp
        src/fast_tf.cpp
)

target_include_directories(test_frame_hierarchy PRIVATE include)

target_link_libraries(test_frame_hierarchy PRIVATE
        fast_tf
        GTest::GTest
        GTest::Main
        Eigen3::Eigen
        TBB::tbb
)

add_test(NAME FrameHierarchyTest COMMAND test_frame_hierarchy)

# Static transform storage tests
add_executable(test_static_transforms
        tests/test_static_transforms.cpp
        src/fast_tf.cpp
)

target_include_directories(test_static_transforms PRIVATE include)

target_link_libraries(test_static_transforms PRIVATE
        fast_tf
        GTest::GTest
        GTest::Main
        Eigen3::Eigen
        TBB::tbb
)

add_test(NAME StaticTransformsTest COMMAND test_static_transforms)

# Circular transform buffer tests
add_executable(test_transform_buffer
        tests/test_transform_buffer.cpp
        src/fast_tf.cpp
)

target_include_directories(test_transform_buffer PRIVATE include)

target_link_libraries(test_transform_buffer PRIVATE
        fast_tf
        GTest::GTest
        GTest::Main
        Eigen3::Eigen
        TBB::tbb
)

add_test(NAME TransformBufferTest COMMAND test_transform_buffer)

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

    # Fast TF performance benchmarks
    add_executable(benchmark_transforms
            benchmarks/benchmark_transforms.cpp
            src/fast_tf.cpp
    )

    target_include_directories(benchmark_transforms PRIVATE include)

    target_link_libraries(benchmark_transforms PRIVATE
            fast_tf
            benchmark::benchmark
            benchmark::benchmark_main
            Eigen3::Eigen
            TBB::tbb
    )

    add_test(NAME TransformBenchmark COMMAND benchmark_transforms)

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
