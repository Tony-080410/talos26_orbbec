# Sanitizers (optional - enable via CMake flags)
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
option(ENABLE_COVERAGE "Enable code coverage" OFF)

# Sanitizer mutual exclusion check
if (ENABLE_ASAN AND ENABLE_TSAN)
    message(FATAL_ERROR "Cannot enable both AddressSanitizer and ThreadSanitizer simultaneously")
endif ()

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

# ============================================================================
# Test Helper Functions - Reduce boilerplate and ensure consistency
# ============================================================================

# Helper function to create tests with consistent setup
function(talos_add_test test_name source_file)
    add_executable(${test_name} ${source_file})
    target_link_libraries(${test_name} PRIVATE GTest::gtest_main Threads::Threads)
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# Helper for tests requiring OpenCV (core + imgproc)
function(talos_add_test_opencv test_name source_file)
    add_executable(${test_name} ${source_file})
    target_link_libraries(${test_name} PRIVATE GTest::gtest_main Threads::Threads ${OpenCV_LIBS})
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# Helper for fast_tf tests (need Eigen3 + TBB + OpenCV + lieplusplus)
function(talos_add_test_fast_tf test_name source_file)
    add_executable(${test_name} ${source_file})
    target_include_directories(${test_name} SYSTEM PRIVATE 3dparty/lieplusplus)
    target_link_libraries(${test_name} PRIVATE fast_tf GTest::gtest_main Eigen3::Eigen TBB::tbb ${OpenCV_LIBS})
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# Helper for scheduler tests
function(talos_add_test_scheduler test_name source_file)
    add_executable(${test_name} ${source_file})
    target_link_libraries(${test_name} PRIVATE scheduler GTest::gtest_main Threads::Threads)
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# Helper for FCS tests (needs fcs + fast_tf + tinympc + lieplusplus)
function(talos_add_test_fcs test_name source_file)
    add_executable(${test_name} ${source_file})
    target_include_directories(${test_name} SYSTEM PRIVATE 3dparty/lieplusplus)
    target_link_libraries(${test_name} PRIVATE
        fcs
        fast_tf
        tinympc
        lieplusplus
        GTest::gtest_main
        Eigen3::Eigen
        TBB::tbb
        ${OpenCV_LIBS}
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# ============================================================================
# Test Definitions
# ============================================================================

# Triple buffer tests
talos_add_test(triple_buffer_test test/triple_buffer_test.cpp)

# Scheduler primitives tests - needs OpenCV
talos_add_test_opencv(primitives_test test/primitives_test.cpp)

# Flow graph tests
talos_add_test(flow_graph_test test/flow_graph_test.cpp)

# Fast TF tests - needs Eigen3 + TBB + OpenCV + lieplusplus
talos_add_test_fast_tf(fast_tf_test test/fast_tf_test.cpp)

# Bevy-style ECS tests - needs TBB (special case, custom setup)
add_executable(bevy_ecs_test test/bevy_ecs_test.cpp)
target_link_libraries(bevy_ecs_test PRIVATE GTest::gtest_main Threads::Threads TBB::tbb)
add_test(NAME BevyEcsTest COMMAND bevy_ecs_test)

# Scheduler tests
talos_add_test_scheduler(scheduler_ecs_test test/scheduler_ecs_test.cpp)

# ROS2 compatibility layer tests - needs scheduler + special include paths
add_executable(rclcompat_test test/rclcompat_test.cpp)
target_include_directories(rclcompat_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/crates/scheduler/src)
target_link_libraries(rclcompat_test PRIVATE scheduler GTest::gtest_main Threads::Threads)
add_test(NAME RclCompatTest COMMAND rclcompat_test)

if (ENABLE_COVERAGE)
    message(STATUS "Coverage enabled for FCS tests")
endif ()
