cmake_minimum_required(VERSION 3.14)

include(cmake/compiler-spec.cmake)

project(talos VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(cmake/dependencies.cmake)
include(cmake/test.cmake)

# talos std library (header-only)
add_library(talos_std INTERFACE
        include/detector_state.hpp
        include/performance_probe.hpp
        include/spdlog_hook.hpp
        include/toml_helper.hpp
        include/tracker_soa.hpp
        include/triple_buffer.hpp
        include/ukf_logic.hpp
)

target_include_directories(talos_std INTERFACE include)

target_link_libraries(talos_std INTERFACE
        tomlplusplus::tomlplusplus
        Eigen3::Eigen
        spdlog::spdlog
        fmt::fmt
)


# scheduler library
add_library(scheduler INTERFACE)

target_include_directories(scheduler INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/scheduler
)

target_link_libraries(scheduler INTERFACE
        talos_std
        ${OpenCV_LIBS}
        TBB::tbb
)

# auto aim library
add_library(auto_aim SHARED
        src/auto_aim/aimer.cpp
        src/auto_aim/classifier.cpp
        src/auto_aim/detector.cpp
        src/auto_aim/extended_kalman_filter.cpp
        src/auto_aim/shooter.cpp
        src/auto_aim/solver.cpp
        src/auto_aim/target.cpp
        src/auto_aim/tracker.cpp
        src/auto_aim/trajectory.cpp
)

target_include_directories(auto_aim PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include/auto_aim
)

# Add foxglove headers for fast_tf support
target_include_directories(auto_aim SYSTEM PUBLIC
        ${foxglove_SOURCE_DIR}/include
        ${foxglove_SOURCE_DIR}/include/foxglove
)

# Add OpenVINO include directories if found
if (OpenVINO_FOUND)
    # Get include path from OpenVINO runtime target
    if (TARGET openvino::runtime)
        get_target_property(_openvino_include openvino::runtime INTERFACE_INCLUDE_DIRECTORIES)
        if (_openvino_include)
            target_include_directories(auto_aim PUBLIC ${_openvino_include})
        else ()
            # Fallback to known OpenVINO installation path
            target_include_directories(auto_aim PUBLIC
                    ${OpenVINO_DIR}/../../../include
            )
        endif ()
    endif ()
endif ()

target_link_libraries(auto_aim PUBLIC
        talos_std
        scheduler
        ${OpenCV_LIBS}
        $<$<TARGET_EXISTS:openvino::runtime>:openvino::runtime>
)

# ipc library
add_library(ipc INTERFACE)

target_include_directories(ipc INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/ipc
)

target_link_libraries(ipc INTERFACE
        ${OpenCV_LIBS}
)

# talos gimbal library
add_library(talos_gimbal SHARED
        src/talos_gimbal/usb.cpp
)

target_include_directories(talos_gimbal PUBLIC
        ${LIBUSB_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include/talos_gimbal
)

target_link_libraries(talos_gimbal PRIVATE
        talos_std
        ${LIBUSB_LINK_LIBRARIES}
)

# hikcamera library (HIK SDK only available on Linux x86_64)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    add_library(hikcamera SHARED
            src/camera/hik_camera.cpp
    )

    target_include_directories(hikcamera PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${HIK_SDK_ROOT}/include
    )

    target_link_directories(hikcamera PRIVATE
            ${HIK_SDK_ROOT}/lib
    )

    target_link_libraries(hikcamera PUBLIC
            talos_std
            spdlog::spdlog
            ${OpenCV_LIBS}
    )

    target_link_libraries(hikcamera PRIVATE
            MvCameraControl
            FormatConversion
    )

    # Suppress warnings from external SDK headers
    target_compile_options(hikcamera PRIVATE -Wno-documentation -Wno-documentation-unknown-command -Wno-switch-enum)
else ()
    message(STATUS "Building hikcamera with stub (HIK SDK only available on Linux x86_64)")
    add_library(hikcamera SHARED
            src/camera/hik_camera_stub.cpp
    )

    target_include_directories(hikcamera PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_link_libraries(hikcamera PUBLIC
            talos_std
            spdlog::spdlog
            ${OpenCV_LIBS}
    )
endif ()

# Main executable
add_executable(${PROJECT_NAME}
        src/main.cpp
        src/strategy.cpp
        src/strategy_wrapper.cpp
        ${FOXGLOVE_SOURCES}
        include/toml_helper.hpp
)

target_include_directories(${PROJECT_NAME} PRIVATE include)
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ${foxglove_SOURCE_DIR}/include)
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ${foxglove_SOURCE_DIR}/include/foxglove)

target_link_libraries(${PROJECT_NAME} PRIVATE
        talos_gimbal
        scheduler
        auto_aim
        hikcamera
        ${OpenCV_LIBS}
        spdlog::spdlog
        fmt::fmt
        Eigen3::Eigen
        TBB::tbb
        tomlplusplus::tomlplusplus
        ${LIBUSB_LINK_LIBRARIES}
        ${foxglove_SOURCE_DIR}/lib/libfoxglove.a
        "$<$<PLATFORM_ID:Darwin>:-framework Security>"
        "$<$<PLATFORM_ID:Darwin>:-framework CoreFoundation>"
        "$<$<PLATFORM_ID:Darwin>:-framework SystemConfiguration>"
)

# sp_vision executable
add_executable(sp_vision
        src/sp_vision/sp_vision.cpp
        ${FOXGLOVE_SOURCES}
)


target_include_directories(sp_vision PRIVATE include)
target_include_directories(sp_vision PRIVATE ${foxglove_SOURCE_DIR}/include)
target_include_directories(sp_vision PRIVATE ${foxglove_SOURCE_DIR}/include/foxglove)

target_link_libraries(sp_vision PRIVATE
        talos_gimbal
        ${LIBUSB_LINK_LIBRARIES}
        scheduler
        auto_aim
        ${OpenCV_LIBS}
        spdlog::spdlog
        fmt::fmt
        Eigen3::Eigen
        TBB::tbb
        tomlplusplus::tomlplusplus
        $<$<TARGET_EXISTS:openvino::runtime>:openvino::runtime>
        ${foxglove_SOURCE_DIR}/lib/libfoxglove.a
        nlohmann_json::nlohmann_json
        "$<$<PLATFORM_ID:Darwin>:-framework Security>"
        "$<$<PLATFORM_ID:Darwin>:-framework CoreFoundation>"
        "$<$<PLATFORM_ID:Darwin>:-framework SystemConfiguration>"
)

# Fast TF library (header-only + compiled support)
add_library(fast_tf INTERFACE)

target_include_directories(fast_tf INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(fast_tf INTERFACE
        Eigen3::Eigen
        TBB::tbb
)

# Fast TF compiled support
add_library(fast_tf_compiled OBJECT
        src/fast_tf.cpp
)

target_include_directories(fast_tf_compiled PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(fast_tf_compiled PUBLIC
        fast_tf
)

# Infantry Robot TF Integration Example
# Demonstrates type-safe frame hierarchy + Foxglove SDK integration
add_executable(infantry_tf_example
        examples/infantry_tf_example.cpp
        src/fast_tf.cpp
        ${FOXGLOVE_SOURCES}
)

target_include_directories(infantry_tf_example PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${foxglove_SOURCE_DIR}/include
)

target_compile_definitions(infantry_tf_example PRIVATE
        FOXGLOVE_SDK_AVAILABLE
)

target_link_libraries(infantry_tf_example PRIVATE
        fast_tf
        Eigen3::Eigen
        TBB::tbb
        spdlog::spdlog
        ${foxglove_SOURCE_DIR}/lib/libfoxglove.a
        "$<$<PLATFORM_ID:Darwin>:-framework Security>"
        "$<$<PLATFORM_ID:Darwin>:-framework CoreFoundation>"
        "$<$<PLATFORM_ID:Darwin>:-framework SystemConfiguration>"
)

# Detector benchmark (accuracy testing against dataset)
add_executable(detector_benchmark
        test/detector_benchmark.cpp
        ${FOXGLOVE_SOURCES}
)

target_include_directories(detector_benchmark PRIVATE include)
target_include_directories(detector_benchmark SYSTEM PRIVATE ${foxglove_SOURCE_DIR}/include)

target_link_libraries(detector_benchmark PRIVATE
        auto_aim
        ${foxglove_SOURCE_DIR}/lib/libfoxglove.a
        "$<$<PLATFORM_ID:Darwin>:-framework Security>"
        "$<$<PLATFORM_ID:Darwin>:-framework CoreFoundation>"
        "$<$<PLATFORM_ID:Darwin>:-framework SystemConfiguration>"
)

# Detector parameter tuning tool (multi-objective optimization)
add_executable(detector_tuner
        test/detector_tuner.cpp
        src/auto_aim/detector_tuner.cpp
)

target_include_directories(detector_tuner PRIVATE include)

target_link_libraries(detector_tuner PRIVATE
        auto_aim
        TBB::tbb
        ${OpenCV_LIBS}
)

# CMake package configuration for hikcamera
if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    include(CMakePackageConfigHelpers)

    install(TARGETS hikcamera
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            RUNTIME DESTINATION bin
    )

    install(DIRECTORY include/camera DESTINATION include)
endif ()