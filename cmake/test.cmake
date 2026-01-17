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

# Triple buffer tests
add_executable(triple_buffer_test
    test/triple_buffer_test.cpp
)

target_include_directories(triple_buffer_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(triple_buffer_test PRIVATE
    GTest::gtest_main
    Threads::Threads
)

add_test(NAME TripleBufferTest COMMAND triple_buffer_test)

# Scheduler primitives tests
add_executable(primitives_test
    test/primitives_test.cpp
)

target_include_directories(primitives_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(primitives_test PRIVATE
    GTest::gtest_main
    Threads::Threads
    ${OpenCV_LIBS}
)

add_test(NAME PrimitivesTest COMMAND primitives_test)

# Flow graph tests
add_executable(flow_graph_test
    test/flow_graph_test.cpp
)

target_include_directories(flow_graph_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(flow_graph_test PRIVATE
    GTest::gtest_main
    Threads::Threads
)

add_test(NAME FlowGraphTest COMMAND flow_graph_test)

# Fast TF tests
add_executable(fast_tf_test
    test/fast_tf_test.cpp
)

target_include_directories(fast_tf_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(fast_tf_test SYSTEM PRIVATE
    3dparty/lieplusplus
)

target_link_libraries(fast_tf_test PRIVATE
    GTest::gtest_main
    Eigen3::Eigen
    TBB::tbb
    ${OpenCV_LIBS}
)

add_test(NAME FastTfTest COMMAND fast_tf_test)

# Bevy-style ECS tests
add_executable(bevy_ecs_test
    test/bevy_ecs_test.cpp
)

target_include_directories(bevy_ecs_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(bevy_ecs_test PRIVATE
    Threads::Threads
    TBB::tbb
)

add_test(NAME BevyEcsTest COMMAND bevy_ecs_test)

# Scheduler tests
add_executable(scheduler_ecs_test
    test/scheduler_ecs_test.cpp
)

target_include_directories(scheduler_ecs_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(scheduler_ecs_test PRIVATE
    scheduler
    Threads::Threads
)

add_test(NAME SchedulerEcsTest COMMAND scheduler_ecs_test)

# Jpda matcher tests
add_executable(jpda_matcher_test
    test/jpda_matcher_test.cpp
)

target_include_directories(jpda_matcher_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(jpda_matcher_test SYSTEM PRIVATE
    3dparty/lieplusplus
)

target_link_libraries(jpda_matcher_test PRIVATE
    GTest::gtest_main
    Eigen3::Eigen
    ${OpenCV_LIBS}
)

add_test(NAME JpdaMatcherTest COMMAND jpda_matcher_test)

if (ENABLE_COVERAGE)
    message(STATUS "Coverage enabled for FCS tests")
endif ()
