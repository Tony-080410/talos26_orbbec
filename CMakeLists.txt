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
# include(cmake/calibration.cmake)

get_filename_component(PROJECT_SOURCE_DIR_ABS "${CMAKE_SOURCE_DIR}" REALPATH)

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-fmacro-prefix-map=${PROJECT_SOURCE_DIR_ABS}/=)
endif ()

# Create tomlplusplus implementation library
add_library(toml_impl STATIC
        ${tomlplusplus_SOURCE_DIR}/src/toml.cpp
)

target_include_directories(toml_impl PUBLIC
        ${tomlplusplus_SOURCE_DIR}/include
)

target_compile_definitions(toml_impl PUBLIC
        TOML_HEADER_ONLY=0
        TOML_EXCEPTIONS=0
)

# talos std library (header-only)
add_library(talos_std INTERFACE
        include/performance_probe.hpp
        include/spdlog_hook.hpp
        include/toml_helper.hpp
)

target_include_directories(talos_std INTERFACE include)
target_include_directories(talos_std INTERFACE 3dparty/magic_enum)

target_link_libraries(talos_std INTERFACE
        toml_impl
        Eigen3::Eigen
        spdlog::spdlog
        fmt::fmt
)

# scheduler library
add_library(scheduler SHARED
        src/scheduler/scheduler.cpp
)

target_include_directories(scheduler PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include/scheduler
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(scheduler PUBLIC
        talos_std
        ${OpenCV_LIBS}
        TBB::tbb
)

# auto aim library (header-only during refactoring)
add_library(auto_aim INTERFACE)

target_include_directories(auto_aim INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/auto_aim
        ${foxglove_SOURCE_DIR}/include
        ${foxglove_SOURCE_DIR}/include/foxglove
)

target_link_libraries(auto_aim INTERFACE
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

# talos gimbal library (header-only during refactoring)
add_library(talos_gimbal INTERFACE)

target_include_directories(talos_gimbal INTERFACE
        ${LIBUSB_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include/talos_gimbal
)

target_link_libraries(talos_gimbal INTERFACE
        talos_std
        ${LIBUSB_LINK_LIBRARIES}
)

# FCS library (Fire Control System - refactored architecture)
add_library(fcs SHARED
        src/fcs/camera_config.cpp
        src/fcs/L1_sensor/input_interface.cpp
        src/fcs/L1_sensor/output_interface.cpp
        src/fcs/L2_perception/at_legacy_detector_backend.cpp
        src/fcs/L2_perception/at_legacy_nn_backend.cpp
        src/fcs/L2_perception/at_legacy_nn_tup_backend.cpp
        src/fcs/L2_perception/at_legacy_traditional_backend.cpp
        src/fcs/L2_perception/at_legacy_traditional_classifier.cpp
        src/fcs/L2_perception/at_legacy_light_corner_corrector.cpp
        src/fcs/L5_weapon/enhanced/weapon_controller.cpp
)

target_include_directories(fcs PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/fcs
)

# Add foxglove headers for fast_tf support
target_include_directories(fcs SYSTEM PUBLIC
        ${foxglove_SOURCE_DIR}/include
        ${foxglove_SOURCE_DIR}/include/foxglove
        3dparty/lieplusplus
)

message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

target_link_libraries(fcs PUBLIC
        talos_std
        talos_gimbal
        ipc
        hikcamera
        scheduler
        Ceres::ceres
        ${OpenCV_LIBS}
        Eigen3::Eigen
        spdlog::spdlog
        ${LIBUSB_LINK_LIBRARIES}
        $<$<TARGET_EXISTS:openvino::runtime>:openvino::runtime>
)

# Use precompiled headers for fcs library (speeds up compilation by 50-70%)
target_precompile_headers(fcs PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/fcs/pch.hpp
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
        ${FOXGLOVE_SOURCES}
)

target_include_directories(${PROJECT_NAME} PRIVATE include)
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ${foxglove_SOURCE_DIR}/include)
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ${foxglove_SOURCE_DIR}/include/foxglove)

target_link_libraries(${PROJECT_NAME} PRIVATE
        talos_gimbal
        scheduler
        auto_aim
        hikcamera
        fcs
        ipc
        fast_tf
        ${OpenCV_LIBS}
        spdlog::spdlog
        fmt::fmt
        Eigen3::Eigen
        TBB::tbb

        nlohmann_json::nlohmann_json
        ${LIBUSB_LINK_LIBRARIES}
        ${foxglove_SOURCE_DIR}/lib/libfoxglove.a
        "$<$<PLATFORM_ID:Darwin>:-framework Security>"
        "$<$<PLATFORM_ID:Darwin>:-framework CoreFoundation>"
        "$<$<PLATFORM_ID:Darwin>:-framework SystemConfiguration>"
        "$<$<PLATFORM_ID:Darwin>:-mmacosx-version-min=15.5>"
)

# sp_vision executable (removed during refactoring)

# Fast TF library (explicit template instantiations + robot factory)
add_library(fast_tf STATIC
        src/math/fast_tf/infantry.cpp
        src/math/fast_tf/hero.cpp
        src/math/fast_tf/sentry.cpp
        src/math/fast_tf/robot_factory.cpp
)

target_include_directories(fast_tf PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(fast_tf PUBLIC
        Eigen3::Eigen
        TBB::tbb
        ${OpenCV_LIBS}
)

# NOTE: detector_benchmark and detector_tuner removed - test files deleted during refactoring

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
