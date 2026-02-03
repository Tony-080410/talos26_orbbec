option(TALOS_OFFLINE "Build without network" ON)

set(CMAKE_WARN_DEPRECATED OFF)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 OLD)
endif()

if (TALOS_OFFLINE)
    set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
endif ()

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)

# OpenVINO configuration with better error handling
find_package(OpenVINO QUIET COMPONENTS Runtime)
find_package(TBB REQUIRED)
find_package(GTest REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Ceres REQUIRED)
if (OpenVINO_FOUND)
    message(STATUS "OpenVINO found at: ${OpenVINO_DIR}")
else ()
    message(WARNING "OpenVINO not found - auto_aim features may be disabled")
endif ()

# TensorRT configuration with better error handling
find_package(CUDA QUIET)
if (CUDA_FOUND)
    message(STATUS "CUDA found at: ${CUDA_TOOLKIT_ROOT_DIR}")
else ()
    message(WARNING "CUDA not found - TensorRT backend will be disabled")
endif ()

find_package(TensorRT QUIET MODULE)
if (TensorRT_FOUND)
    message(STATUS "TensorRT found at: ${TensorRT_DIR}")
else ()
    message(WARNING "TensorRT not found - TensorRT backend will be disabled")
endif ()

find_package(benchmark QUIET)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBUSB REQUIRED libusb-1.0)

# HIK Camera SDK configuration
set(HIK_SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/3dparty/hik_sdk")

if (EXISTS "${HIK_SDK_ROOT}")
    message(STATUS "HIK SDK found at: ${HIK_SDK_ROOT}")
else ()
    message(WARNING "HIK SDK not found at: ${HIK_SDK_ROOT}")
endif ()

include(FetchContent)

# Check for local tomlplusplus cache
set(TOMLPLUSPLUS_LOCAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3dparty/tomlplusplus")
if (EXISTS "${TOMLPLUSPLUS_LOCAL_DIR}")
    message(STATUS "Using local tomlplusplus at: ${TOMLPLUSPLUS_LOCAL_DIR}")
    FetchContent_Declare(
            tomlplusplus
            SOURCE_DIR ${TOMLPLUSPLUS_LOCAL_DIR}
    )
else ()
    message(STATUS "Fetching tomlplusplus from remote repository")
    FetchContent_Declare(
            tomlplusplus
            GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
            GIT_TAG v3.4.0
    )
endif ()

FetchContent_MakeAvailable(tomlplusplus)

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
set(FOXGLOVE_SHA256_aarch64-apple-darwin "0799fcb999616fe25d16d7e4b2894adb6f7fa97ce9671bd908d8412505887bec")
set(FOXGLOVE_SHA256_aarch64-unknown-linux-gnu "54cbc57c60a550473e0a75f8f360efb6fa245ebb7f582db5efa64f1c297e554f")
set(FOXGLOVE_SHA256_x86_64-unknown-linux-gnu "54cbc57c60a550473e0a75f8f360efb6fa245ebb7f582db5efa64f1c297e554f")

# Check for local Foxglove cache
set(FOXGLOVE_LOCAL_ZIP "${CMAKE_CURRENT_SOURCE_DIR}/3dparty/foxglove-${FOXGLOVE_VERSION}-cpp-${FOXGLOVE_PLATFORM}.zip")
set(FOXGLOVE_SHA ${FOXGLOVE_SHA256_${FOXGLOVE_PLATFORM}})

if (EXISTS "${FOXGLOVE_LOCAL_ZIP}")
    message(STATUS "Using local Foxglove at: ${FOXGLOVE_LOCAL_ZIP}")
    FetchContent_Declare(
            foxglove
            URL ${FOXGLOVE_LOCAL_ZIP}
            # URL_HASH SHA256=${FOXGLOVE_SHA}
    )
else ()
    set(FOXGLOVE_URL "https://github.com/foxglove/foxglove-sdk/releases/download/sdk%2F${FOXGLOVE_VERSION}/foxglove-${FOXGLOVE_VERSION}-cpp-${FOXGLOVE_PLATFORM}.zip")
    message(STATUS "Downloading Foxglove from: ${FOXGLOVE_URL}")
    FetchContent_Declare(
            foxglove
            URL ${FOXGLOVE_URL}
            # URL_HASH SHA256=${FOXGLOVE_SHA}
            # DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
endif ()

FetchContent_MakeAvailable(foxglove)

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
