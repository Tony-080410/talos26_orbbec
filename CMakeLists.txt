cmake_minimum_required(VERSION 3.14)

include(cmake/compiler-spec.cmake)
include(cmake/crate-utils.cmake)

project(talos VERSION 0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

get_filename_component(PROJECT_SOURCE_DIR_ABS "${CMAKE_SOURCE_DIR}" REALPATH)

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-fmacro-prefix-map=${PROJECT_SOURCE_DIR_ABS}/=)
endif ()

# ============================================================================
# AddressSanitizer Option (Memory Leak Detection)
# ============================================================================
option(ENABLE_ASAN "Enable AddressSanitizer for memory leak detection" OFF)

if(ENABLE_ASAN)
    message(STATUS "AddressSanitizer: ENABLED")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address)
else()
    message(STATUS "AddressSanitizer: DISABLED")
endif()

# Add project cmake directory to module search path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# RPATH configuration for micromamba environment
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)

if(CMAKE_CXX_COMPILER MATCHES "micromamba")
    get_filename_component(MICROMAMBA_LIB_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)
    get_filename_component(MICROMAMBA_LIB_DIR ${MICROMAMBA_LIB_DIR} DIRECTORY)
    set(MICROMAMBA_LIB_DIR "${MICROMAMBA_LIB_DIR}/lib")
    set(CMAKE_INSTALL_RPATH "${CMAKE_BINARY_DIR};${MICROMAMBA_LIB_DIR}")
    set(CMAKE_BUILD_RPATH "${CMAKE_BINARY_DIR};${MICROMAMBA_LIB_DIR}")
endif()

include(cmake/dependencies.cmake)
include(cmake/test.cmake)
include(cmake/calibration.cmake)

# ============================================================================
# Third-party libraries (header-only / vendored)
# ============================================================================

# tomlplusplus implementation
add_library(toml_impl STATIC
        ${tomlplusplus_SOURCE_DIR}/src/toml.cpp
)

crate_silence_warnings(toml_impl)

target_include_directories(toml_impl SYSTEM PUBLIC
        ${tomlplusplus_SOURCE_DIR}/include
)

target_compile_definitions(toml_impl PUBLIC
        TOML_HEADER_ONLY=0
        TOML_EXCEPTIONS=0
)

# TinyMPC solver
add_library(tinympc STATIC
        3dparty/tinympc/admm.cpp
        3dparty/tinympc/tiny_api.cpp
        3dparty/tinympc/rho_benchmark.cpp
)

crate_silence_warnings(tinympc)

target_include_directories(tinympc SYSTEM PUBLIC 3dparty/tinympc)
target_link_libraries(tinympc PUBLIC Eigen3::Eigen)

# Magic enum
add_library(magic_enum INTERFACE)
target_include_directories(magic_enum INTERFACE 3dparty/magic_enum)
target_compile_definitions(magic_enum INTERFACE
        MAGIC_ENUM_RANGE_MIN=0
        MAGIC_ENUM_RANGE_MAX=16
)

# Lie++ (SO3 manifold library)
add_library(lieplusplus INTERFACE)
target_include_directories(lieplusplus INTERFACE 3dparty/lieplusplus)

# Foxglove C++ wrapper library
add_library(foxglove_cpp SHARED ${FOXGLOVE_SOURCES})
crate_silence_warnings(foxglove_cpp)

target_include_directories(foxglove_cpp SYSTEM PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include_foxglove
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

# ============================================================================
# Crates (Rust-style workspace)
# ============================================================================

# Core utilities
add_subdirectory(crates/toml)
add_subdirectory(crates/log)
add_subdirectory(crates/math)

# Math and geometry
add_subdirectory(crates/fast_tf)

# Scheduler framework
add_subdirectory(crates/primitive)
add_subdirectory(crates/scheduler)

# Chiral bidirectional communication
add_subdirectory(crates/chiral)

# Hardware drivers
add_subdirectory(crates/hardware/at_gimbal)
add_subdirectory(crates/hardware/hik_camera_driver)
add_subdirectory(crates/hardware_daedalus)

# ============================================================================
# FCS library (Fire Control System)
# ============================================================================

set(FCS_SOURCES
        src/fcs/config.cpp
        src/fcs/camera_config.cpp
        src/fcs/chiral/chiral_collector_system.cpp
        src/fcs/core/trajectory/solver/trajectory_solver.cpp
        src/fcs/L1_sensor/camera_interface.cpp
        src/fcs/L1_sensor/output_interface.cpp
        src/fcs/runtime/l1_l2_setup.cpp
        src/fcs/L2_perception/armor/at_legacy_config.cpp
        src/fcs/L2_perception/armor/at_legacy_detector_backend.cpp
        src/fcs/L2_perception/armor/at_legacy_systems.cpp
        src/fcs/L2_perception/rune/rune_detector.cpp
        src/fcs/L2_perception/rune/rune_config.cpp
        src/fcs/L2_perception/rune/rune_systems.cpp
        src/fcs/L2_perception/armor/backends/at_legacy_traditional_backend.cpp
        src/fcs/L2_perception/armor/backends/at_legacy_traditional_classifier.cpp
        src/fcs/L2_perception/armor/at_legacy_pnp_sort.cpp
        src/fcs/L2_perception/armor/backends/at_legacy_corner_refiner.cpp
        src/fcs/L3_estimation/tracker_systems.cpp
        src/fcs/L3_estimation/extended_kalman_filter.cpp
        src/fcs/L3_estimation/energy_meter_solver/angle_fitter.cpp
        src/fcs/L3_estimation/energy_meter_solver/energy_meter_tracker.cpp
        src/fcs/L3_estimation/energy_meter/energy_meter_systems.cpp
        src/fcs/L4_planning/config.cpp
        src/fcs/L4_planning/aimer/aimer.cpp
        src/fcs/L4_planning/aimer/aimer_systems.cpp
        src/fcs/L4_planning/planning/fire_decision.cpp
        src/fcs/L4_planning/planning_systems.cpp
        src/fcs/L5_weapon/config.cpp
        src/fcs/L5_weapon/enhanced/trajectory_optimizer.cpp
        src/fcs/L5_weapon/enhanced/weapon_systems.cpp
)

if(OpenVINO_FOUND)
    list(APPEND FCS_SOURCES
        src/fcs/L2_perception/armor/backends/at_legacy_nn_backend.cpp
        src/fcs/L2_perception/armor/backends/at_legacy_nn_tup_backend.cpp)
    message(STATUS "OpenVINO backends enabled")
else()
    message(STATUS "OpenVINO backends disabled (OpenVINO not found)")
endif()

if(TensorRT_FOUND)
    list(APPEND FCS_SOURCES src/fcs/L2_perception/armor/backends/at_legacy_trt_backend.cpp)
    message(STATUS "TensorRT backend enabled")
else()
    message(STATUS "TensorRT backend disabled (TensorRT not found)")
endif()

if(AXERA_FOUND)
    list(APPEND FCS_SOURCES src/fcs/L2_perception/armor/backends/at_legacy_axera_backend.cpp)
    message(STATUS "Axera backend enabled")
else()
    message(STATUS "Axera backend disabled (AXERA not found)")
endif()

add_library(fcs SHARED ${FCS_SOURCES})

target_include_directories(fcs PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/fcs
        $<$<TARGET_EXISTS:TensorRT::TensorRT>:${TensorRT_INCLUDE_DIRS}>
)

if(CUDA_FOUND)
    # Modern CMake: use imported targets instead of variables
    target_link_libraries(fcs PUBLIC ${CUDA_LIBRARIES})
    target_include_directories(fcs SYSTEM PUBLIC ${CUDA_INCLUDE_DIRS})
    target_compile_definitions(fcs PUBLIC TALOS_HAS_CUDA_RUNTIME=0)
    message(STATUS "Linking with CUDA runtime: ${CUDA_LIBRARIES}")
else()
    target_compile_definitions(fcs PUBLIC TALOS_HAS_CUDA_RUNTIME=0)
endif()

if(TensorRT_FOUND)
    target_compile_definitions(fcs PUBLIC TALOS_HAS_TENSORRT=1)
    message(STATUS "TensorRT support enabled")
else()
    target_compile_definitions(fcs PUBLIC TALOS_HAS_TENSORRT=0)
endif()

if(OpenVINO_FOUND)
    target_compile_definitions(fcs PUBLIC TALOS_HAS_OPENVINO=1)
    message(STATUS "OpenVINO support enabled")
else()
    target_compile_definitions(fcs PUBLIC TALOS_HAS_OPENVINO=0)
endif()

if(AXERA_FOUND)
    target_compile_definitions(fcs PUBLIC TALOS_HAS_AXERA=1)
    target_include_directories(fcs SYSTEM PUBLIC ${AXERA_INCLUDE_DIRS})
    target_link_libraries(fcs PUBLIC ${AXERA_LIBRARIES})
    message(STATUS "Axera support enabled")
else()
    target_compile_definitions(fcs PUBLIC TALOS_HAS_AXERA=0)
    message(STATUS "Axera support disabled")
endif()

target_compile_options(fcs PRIVATE -Wno-error=array-bounds)

target_link_libraries(fcs PUBLIC
        toml_std
        fast_tf
        scheduler
        Eigen3::Eigen
        Ceres::ceres
        tinympc
        lieplusplus
        talos_gimbal
        hikcamera
        ipc
        chiral
        ${OpenCV_LIBS}
        $<$<TARGET_EXISTS:openvino::runtime>:openvino::runtime>
        $<$<TARGET_EXISTS:TensorRT::TensorRT>:TensorRT::TensorRT>
        $<$<TARGET_EXISTS:TensorRT::nvonnxparser>:TensorRT::nvonnxparser>
)

target_precompile_headers(fcs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/fcs/pch.hpp)

# FCS visualization
add_library(fcs_visualization SHARED src/fcs_visualization/foxglove_systems.cpp)

target_include_directories(fcs_visualization PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/fcs
)

target_include_directories(fcs_visualization SYSTEM PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/fcs_visualization
)

target_link_libraries(fcs_visualization PUBLIC
        fcs
        scheduler
        foxglove_cpp
        ${OpenCV_LIBS}
        Eigen3::Eigen
        spdlog::spdlog
)

target_precompile_headers(fcs_visualization PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/fcs_visualization/pch.hpp)

# FCS tests (auto-discovered from src/fcs/tests/)
if(BUILD_TESTING)
    file(GLOB FCS_TESTS "${CMAKE_CURRENT_SOURCE_DIR}/src/fcs/tests/*_test.cpp")
    foreach(test_file ${FCS_TESTS})
        get_filename_component(test_name ${test_file} NAME_WE)
        add_executable(${test_name} ${test_file})
        target_link_libraries(${test_name} PRIVATE
            GTest::gtest_main
            Threads::Threads
            fcs
            fast_tf
        )
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()

# ============================================================================
# Main executable
# ============================================================================

add_executable(${PROJECT_NAME} src/main.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE
        fcs
        fcs_visualization
        fast_tf
        talos_log
        toml_std
        spdlog::spdlog
        fmt::fmt
        "$<$<PLATFORM_ID:Darwin>:-mmacosx-version-min=15.5>"
)

add_executable(talos-playground src/playground.cpp)

target_include_directories(talos-playground PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/crates/hardware_daedalus/src
)

target_link_libraries(talos-playground PRIVATE
        fast_tf
        Eigen3::Eigen
        lieplusplus
        ipc
        talos_log
        spdlog::spdlog
        ${OpenCV_LIBS}
)
