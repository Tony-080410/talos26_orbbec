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

# FCS Module Tests (commented out - source files not available)
# TODO: Re-enable when test files are restored

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

# Fast TF tests
add_executable(fast_tf_test
    test/fast_tf_test.cpp
)

target_include_directories(fast_tf_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(fast_tf_test PRIVATE
    GTest::gtest_main
    Eigen3::Eigen
    TBB::tbb
    ${OpenCV_LIBS}
)

add_test(NAME FastTfTest COMMAND fast_tf_test)

if (ENABLE_COVERAGE)
    message(STATUS "Coverage enabled for FCS tests")
endif ()
