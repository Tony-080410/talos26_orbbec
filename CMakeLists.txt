cmake_minimum_required(VERSION 3.14)

include(cmake/compiler-spec.cmake)

project(talos VERSION 0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Add project cmake directory to module search path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# RPATH configuration for micromamba environment
# Ensure micromamba libraries are found before system libraries
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)

# Get micromamba lib path from compiler location
if(CMAKE_CXX_COMPILER MATCHES "micromamba")
    get_filename_component(MICROMAMBA_LIB_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)
    get_filename_component(MICROMAMBA_LIB_DIR ${MICROMAMBA_LIB_DIR} DIRECTORY)
    set(MICROMAMBA_LIB_DIR "${MICROMAMBA_LIB_DIR}/lib")

    # Set RPATH to prioritize micromamba libraries
    set(CMAKE_INSTALL_RPATH "${CMAKE_BINARY_DIR};${MICROMAMBA_LIB_DIR}")
    set(CMAKE_BUILD_RPATH "${CMAKE_BINARY_DIR};${MICROMAMBA_LIB_DIR}")
endif()

include(cmake/dependencies.cmake)
include(cmake/test.cmake)
include(cmake/calibration.cmake)

get_filename_component(PROJECT_SOURCE_DIR_ABS "${CMAKE_SOURCE_DIR}" REALPATH)

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-fmacro-prefix-map=${PROJECT_SOURCE_DIR_ABS}/=)
endif ()

# Create tomlplusplus implementation library
add_library(toml_impl STATIC
        ${tomlplusplus_SOURCE_DIR}/src/toml.cpp
)

silence_target_warnings(toml_impl)

target_include_directories(toml_impl SYSTEM PUBLIC
        ${tomlplusplus_SOURCE_DIR}/include
)

target_compile_definitions(toml_impl PUBLIC
        TOML_HEADER_ONLY=0
        TOML_EXCEPTIONS=0
)

# Create tinympc implementation library
add_library(tinympc STATIC
        # TinyMPC solver
        3dparty/tinympc/admm.cpp
        3dparty/tinympc/tiny_api.cpp
        3dparty/tinympc/rho_benchmark.cpp
)

silence_target_warnings(tinympc)

target_include_directories(tinympc SYSTEM PUBLIC
        3dparty/tinympc
)

target_link_libraries(tinympc PUBLIC Eigen3::Eigen)

add_library(magic_enum INTERFACE)

target_include_directories(magic_enum INTERFACE 3dparty/magic_enum)

target_compile_definitions(magic_enum INTERFACE
        MAGIC_ENUM_RANGE_MIN=0
        MAGIC_ENUM_RANGE_MAX=16
)

add_library(lieplusplus INTERFACE)

target_include_directories(lieplusplus INTERFACE 3dparty/lieplusplus)

# scheduler library
add_library(scheduler SHARED
        src/scheduler/scheduler.cpp
        src/scheduler/rclcompat/timer_factory.cpp
)

target_include_directories(scheduler PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include/scheduler
)

target_link_libraries(scheduler PUBLIC
        magic_enum
        spdlog::spdlog
        fmt::fmt
        TBB::tbb
)

# Foxglove C++ wrapper library (compiled sources + prebuilt core lib)
add_library(foxglove_cpp SHARED
        ${FOXGLOVE_SOURCES}
)

silence_target_warnings(foxglove_cpp)

target_include_directories(foxglove_cpp SYSTEM PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include  # Override foxglove headers with local fixes
        ${foxglove_SOURCE_DIR}/include
        ${foxglove_SOURCE_DIR}/include/foxglove
)

target_link_libraries(foxglove_cpp PUBLIC
        ${foxglove_SOURCE_DIR}/lib/libfoxglove.a
        "$<$<PLATFORM_ID:Darwin>:-framework Security>"
        "$<$<PLATFORM_ID:Darwin>:-framework CoreFoundation>"
        "$<$<PLATFORM_ID:Darwin>:-framework SystemConfiguration>"
        "$<$<PLATFORM_ID:Darwin>:-mmacosx-version-min=15.5>"
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
        ${LIBUSB_LINK_LIBRARIES}
)

# FCS library source files
set(FCS_SOURCES
        src/auto_aim/config.cpp
        src/fcs/camera_config.cpp
        src/fcs/L1_sensor/camera_interface.cpp
        src/fcs/L1_sensor/output_interface.cpp
        src/fcs/runtime/l1_l2_setup.cpp
        src/fcs/L2_perception/at_legacy_config.cpp
        src/fcs/L2_perception/at_legacy_detector_backend.cpp
        src/fcs/L2_perception/at_legacy_nn_backend.cpp
        src/fcs/L2_perception/at_legacy_nn_tup_backend.cpp
        src/fcs/L2_perception/at_legacy_systems.cpp
        src/fcs/L2_perception/rune/rune_detector.cpp
        src/fcs/L2_perception/rune/rune_config.cpp
        src/fcs/L2_perception/rune/rune_systems.cpp
        src/fcs/L2_perception/at_legacy_traditional_backend.cpp
        src/fcs/L2_perception/at_legacy_traditional_classifier.cpp
        src/fcs/L2_perception/at_legacy_light_corner_corrector.cpp
        src/fcs/L3_estimation/tracker_systems.cpp
        src/fcs/L3_estimation/energy_meter_solver/angle_fitter.cpp
        src/fcs/L3_estimation/energy_meter_solver/energy_meter_tracker.cpp
        src/fcs/L3_estimation/energy_meter/energy_meter_systems.cpp
        src/fcs/L4_planning/gimbal_planner/trajectory_compensator.cpp
        src/fcs/L4_planning/trajectory_config.cpp
        src/fcs/L4_planning/config.cpp
        src/fcs/L4_planning/mpc_planner/config.cpp
        src/fcs/L4_planning/planning_systems.cpp
        src/fcs/L4_planning/plan_mux/plan_mux_systems.cpp
        src/fcs/L4_planning/rune_planner/rune_planner_systems.cpp
        src/fcs/L5_weapon/config.cpp
        src/fcs/L5_weapon/enhanced/weapon_controller.cpp
        src/fcs/L5_weapon/enhanced/weapon_systems.cpp
)

# Conditionally add TensorRT backend source files if TensorRT is available
if(TensorRT_FOUND)
    list(APPEND FCS_SOURCES src/fcs/L2_perception/at_legacy_trt_backend.cpp)
    message(STATUS "TensorRT backend enabled")
else()
    message(STATUS "TensorRT backend disabled (TensorRT not found)")
endif()

# Note: CUDA preprocessing kernels are disabled due to nvcc compiler compatibility
# The CPU fallback path will be used instead
# TODO: Re-enable CUDA when using a compatible CUDA/host compiler version

# FCS library (Fire Control System - refactored architecture)
add_library(fcs SHARED ${FCS_SOURCES})

target_include_directories(fcs PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include/fcs
        ${CMAKE_CURRENT_SOURCE_DIR}/include/auto_aim
        $<$<TARGET_EXISTS:TensorRT::TensorRT>:${TensorRT_INCLUDE_DIRS}>
)

# Add CUDA runtime if available (for linking with pre-built TensorRT)
if(CUDA_FOUND)
    target_link_libraries(fcs PUBLIC ${CUDA_LIBRARIES})
    target_include_directories(fcs SYSTEM PUBLIC ${CUDA_INCLUDE_DIRS})
    # Disable CUDA runtime preprocessing due to nvcc compatibility issues
    target_compile_definitions(fcs PUBLIC TALOS_HAS_CUDA_RUNTIME=0)
    message(STATUS "Linking with CUDA runtime: ${CUDA_LIBRARIES}")
    message(STATUS "CUDA preprocessing kernels DISABLED (using CPU fallback)")
else()
    target_compile_definitions(fcs PUBLIC TALOS_HAS_CUDA_RUNTIME=0)
endif()

# Add TensorRT support if available
if(TensorRT_FOUND)
    target_compile_definitions(fcs PUBLIC TALOS_HAS_TENSORRT=1)
    message(STATUS "TensorRT support enabled")
else()
    target_compile_definitions(fcs PUBLIC TALOS_HAS_TENSORRT=0)
endif()

# Suppress Eigen/NEON false positive array-bounds warnings
target_compile_options(fcs PRIVATE -Wno-error=array-bounds)

target_link_libraries(fcs PUBLIC
        toml_impl
        fast_tf
        scheduler
        Eigen3::Eigen
        Ceres::ceres
        tinympc
        lieplusplus

        talos_gimbal
        hikcamera
        ipc

        ${OpenCV_LIBS}
        $<$<TARGET_EXISTS:openvino::runtime>:openvino::runtime>
        $<$<TARGET_EXISTS:TensorRT::TensorRT>:TensorRT::TensorRT>
        $<$<TARGET_EXISTS:TensorRT::nvonnxparser>:TensorRT::nvonnxparser>
)

# Use precompiled headers for fcs library (speeds up compilation by 50-70%)
target_precompile_headers(fcs PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/fcs/pch.hpp
)

add_library(fcs_visualization SHARED src/fcs_visualization/foxglove_systems.cpp)

target_include_directories(fcs_visualization PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/fcs
        ${CMAKE_CURRENT_SOURCE_DIR}/include/auto_aim
)

# Add foxglove headers for fast_tf support
target_include_directories(fcs_visualization SYSTEM PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include/fcs_visualization
)

target_link_libraries(fcs_visualization PUBLIC
        fcs
        scheduler
        foxglove_cpp
        ${OpenCV_LIBS}
        Eigen3::Eigen
        spdlog::spdlog
)

target_precompile_headers(fcs_visualization PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/fcs_visualization/pch.hpp
)

# hikcamera library - auto-detect architecture and link appropriate SDK
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Detect architecture and map to SDK directory
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        set(HIK_SDK_ARCH_DIR "amd64")
        set(HIK_SDK_ARCH_NAME "x86_64")
    elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        set(HIK_SDK_ARCH_DIR "arm64")
        set(HIK_SDK_ARCH_NAME "aarch64")
    else ()
        set(HIK_SDK_ARCH_DIR "")
        set(HIK_SDK_ARCH_NAME "")
    endif ()

    # Check if SDK libraries exist for this architecture
    set(HIK_SDK_LIB_DIR "${HIK_SDK_ROOT}/lib/${HIK_SDK_ARCH_DIR}")
    if (HIK_SDK_ARCH_DIR AND EXISTS "${HIK_SDK_LIB_DIR}")
        message(STATUS "Building hikcamera with HIK SDK for ${HIK_SDK_ARCH_NAME} (lib: ${HIK_SDK_LIB_DIR})")
        add_library(hikcamera SHARED
                src/camera/hik_camera.cpp
        )

        target_include_directories(hikcamera PUBLIC
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${HIK_SDK_ROOT}/include
        )

        target_link_directories(hikcamera PRIVATE
                ${HIK_SDK_LIB_DIR}
        )

        target_link_libraries(hikcamera PUBLIC
                magic_enum
                Eigen3::Eigen
                spdlog::spdlog
                ${OpenCV_LIBS}
        )

        target_link_libraries(hikcamera PRIVATE
                MvCameraControl
                FormatConversion
        )

        # Set rpath to find the SDK libraries at runtime
        set_target_properties(hikcamera PROPERTIES
            INSTALL_RPATH "${HIK_SDK_LIB_DIR}"
            BUILD_WITH_INSTALL_RPATH TRUE
        )

        # Suppress warnings from external SDK headers
        target_compile_options(hikcamera PRIVATE -Wno-documentation -Wno-documentation-unknown-command -Wno-switch-enum)
    else ()
        message(STATUS "Building hikcamera with stub (HIK SDK not available for ${CMAKE_SYSTEM_PROCESSOR} or lib directory not found)")
        add_library(hikcamera SHARED
                src/camera/hik_camera_stub.cpp
        )

        target_include_directories(hikcamera PUBLIC
                ${CMAKE_CURRENT_SOURCE_DIR}/include
        )

        target_link_libraries(hikcamera PUBLIC
                magic_enum
                Eigen3::Eigen
                spdlog::spdlog
                ${OpenCV_LIBS}
        )
    endif ()
else ()
    message(STATUS "Building hikcamera with stub (HIK SDK only available on Linux)")
    add_library(hikcamera SHARED
            src/camera/hik_camera_stub.cpp
    )

    target_include_directories(hikcamera PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_link_libraries(hikcamera PUBLIC
            magic_enum
            Eigen3::Eigen
            spdlog::spdlog
            ${OpenCV_LIBS}
    )
endif ()

# Main executable
add_executable(${PROJECT_NAME}
        src/main.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE include)

target_link_libraries(${PROJECT_NAME} PRIVATE
        fcs
        fcs_visualization
        fast_tf
        spdlog::spdlog
        fmt::fmt
        "$<$<PLATFORM_ID:Darwin>:-mmacosx-version-min=15.5>"
)

# Fast TF library (explicit template instantiations + robot factory)
add_library(fast_tf SHARED
        src/math/fast_tf/infantry.cpp
        src/math/fast_tf/hero.cpp
        src/math/fast_tf/validation.cpp
        src/math/fast_tf/sentry.cpp
        src/math/fast_tf/robot_factory.cpp
)

target_include_directories(fast_tf PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include/math/fast_tf
)

target_include_directories(fast_tf SYSTEM PUBLIC
        3dparty/lieplusplus
)

target_link_libraries(fast_tf PUBLIC
        magic_enum
        Eigen3::Eigen
        TBB::tbb
        fmt::fmt
        ${OpenCV_LIBS}
)

target_precompile_headers(fast_tf PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/math/fast_tf/pch.hpp
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
