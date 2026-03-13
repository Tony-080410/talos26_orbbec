set(CMAKE_WARN_DEPRECATED OFF)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 OLD)
endif()

find_package(OpenCV REQUIRED)

# Include FetchContent helper
include(${CMAKE_CURRENT_LIST_DIR}/fetch-content-helper.cmake)

fetch_dependency(NAME Eigen3
                 REPO gitlab.com/libeigen/eigen
                 VERSION 5.0.1)

fetch_dependency(NAME fmt
                 REPO fmtlib/fmt
                 VERSION 12.1.0)

set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "" FORCE)
set(SPDLOG_ENABLE_PCH ON CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_PIC ON CACHE BOOL "" FORCE)

fetch_dependency(NAME spdlog
                 REPO gabime/spdlog
                 VERSION v1.17.0)

fetch_dependency(NAME nlohmann_json
                 REPO nlohmann/json
                 VERSION v3.12.0)

fetch_dependency(NAME tomlplusplus
                 REPO marzer/tomlplusplus
                 VERSION v3.4.0)

find_package(OpenVINO QUIET COMPONENTS Runtime)
if (OpenVINO_FOUND)
    message(STATUS "OpenVINO found at: ${OpenVINO_DIR}")
else ()
    message(STATUS "OpenVINO not found - OpenVINO backends will be disabled")
endif ()
find_package(TBB REQUIRED)
find_package(GTest REQUIRED)
find_package(Ceres REQUIRED)

# TensorRT configuration with better error handling
find_package(CUDA QUIET)
if (CUDA_FOUND)
    message(STATUS "CUDA found at: ${CUDA_TOOLKIT_ROOT_DIR}")
else ()
    message(STATUS "CUDA not found - TensorRT backend will be disabled")
endif ()

find_package(TensorRT QUIET MODULE)
if (TensorRT_FOUND)
    message(STATUS "TensorRT found at: ${TensorRT_DIR}")
else ()
    message(STATUS "TensorRT not found - TensorRT backend will be disabled")
endif ()

find_package(benchmark QUIET)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBUSB REQUIRED libusb-1.0)

# HIK Camera SDK configuration
set(HIK_SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/3dparty/hik_sdk")

if (EXISTS "${HIK_SDK_ROOT}")
    message(STATUS "HIK SDK found at: ${HIK_SDK_ROOT}")
else ()
    message(STATUS "HIK SDK not found at: ${HIK_SDK_ROOT}")
endif ()

# Axera AX650 SDK configuration
# SDK should be deployed to /soc/ with env vars set (see docs/ax650.md)
# CMAKE_PREFIX_PATH, CMAKE_INCLUDE_PATH, CMAKE_LIBRARY_PATH will be searched automatically

find_path(AXERA_INCLUDE_DIRS
    NAMES ax_sys_api.h ax_engine_api.h
    PATH_SUFFIXES "soc/include" "include"
    DOC "Axera SDK include directory"
)

find_library(AXERA_SYS_LIBRARY
    NAMES ax_sys
    DOC "Axera system library"
)

find_library(AXERA_ENGINE_LIBRARY
    NAMES ax_engine
    DOC "Axera engine library"
)

find_library(AXERA_IVPS_LIBRARY
    NAMES ax_ivps
    DOC "Axera IVPS library"
)

if (AXERA_INCLUDE_DIRS AND AXERA_SYS_LIBRARY AND AXERA_ENGINE_LIBRARY AND AXERA_IVPS_LIBRARY)
    set(AXERA_LIBRARIES "${AXERA_SYS_LIBRARY}" "${AXERA_ENGINE_LIBRARY}" "${AXERA_IVPS_LIBRARY}")
    set(AXERA_FOUND TRUE)
    message(STATUS "Axera SDK found:")
    message(STATUS "  Include: ${AXERA_INCLUDE_DIRS}")
    message(STATUS "  Libraries: ${AXERA_LIBRARIES}")
else ()
    set(AXERA_FOUND FALSE)
    message(STATUS "Axera SDK not found - ensure /soc/ is deployed and env vars are set")
    message(STATUS "  See docs/ax650.md for deployment instructions")
endif ()

# Foxglove platform detection
if (APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(FOXGLOVE_PLATFORM "aarch64-apple-darwin")
elseif (UNIX AND NOT APPLE)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        set(FOXGLOVE_PLATFORM "aarch64-unknown-linux-gnu")
    else ()
        set(FOXGLOVE_PLATFORM "x86_64-unknown-linux-gnu")
    endif ()
else ()
    message(FATAL_ERROR "Unsupported platform. Only macOS and Linux are supported.")
endif ()

message(STATUS "Foxglove platform: ${FOXGLOVE_PLATFORM}")

set(FOXGLOVE_VERSION "v0.17.1")

# Fetch foxglove (custom URL + ZIP_NAME)
fetch_dependency(NAME foxglove
                 ZIP_URL "https://github.com/foxglove/foxglove-sdk/releases/download/sdk%2F${FOXGLOVE_VERSION}/foxglove-${FOXGLOVE_VERSION}-cpp-${FOXGLOVE_PLATFORM}.zip"
                 ZIP_NAME "foxglove-${FOXGLOVE_VERSION}-cpp-${FOXGLOVE_PLATFORM}.zip")

# Get foxglove source directory (FetchContent variable may not be set yet)
FetchContent_GetProperties(foxglove SOURCE_DIR foxglove_SOURCE_DIR)

# Foxglove SDK sources
#
# Avoid `file(GLOB ... CONFIGURE_DEPENDS ...)` here: on some setups it forces Ninja to run the
# glob-check + CMake regeneration on *every* incremental build, which hurts edit/compile iteration.
#
# This list is stable for the pinned SDK version above; update it only when bumping Foxglove.
set(FOXGLOVE_SOURCES
    "${foxglove_SOURCE_DIR}/src/channel.cpp"
    "${foxglove_SOURCE_DIR}/src/context.cpp"
    "${foxglove_SOURCE_DIR}/src/error.cpp"
    "${foxglove_SOURCE_DIR}/src/foxglove.cpp"
    "${foxglove_SOURCE_DIR}/src/mcap.cpp"
    "${foxglove_SOURCE_DIR}/src/schemas.cpp"
    "${foxglove_SOURCE_DIR}/src/server.cpp"
    "${foxglove_SOURCE_DIR}/src/server/connection_graph.cpp"
    "${foxglove_SOURCE_DIR}/src/server/fetch_asset.cpp"
    "${foxglove_SOURCE_DIR}/src/server/parameter.cpp"
    "${foxglove_SOURCE_DIR}/src/server/service.cpp"
)

function(silence_target_warnings tgt)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(${tgt} PRIVATE -Wno-everything)
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(${tgt} PRIVATE -w)
    elseif (MSVC)
        target_compile_options(${tgt} PRIVATE /W0)
    endif()
endfunction()
