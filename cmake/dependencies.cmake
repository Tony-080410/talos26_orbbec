option(TALOS_OFFLINE "Build without network" ON)

if (TALOS_OFFLINE)
    set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
endif ()

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(TBB REQUIRED)
find_package(GTest REQUIRED)
find_package(OpenVINO QUIET)
find_package(benchmark QUIET)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBUSB REQUIRED libusb-1.0)

include(FetchContent)

FetchContent_Declare(
        tomlplusplus
        GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
        GIT_TAG v3.4.0
)

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

set(FOXGLOVE_VERSION "v0.16.1")
set(FOXGLOVE_SHA256_aarch64-apple-darwin "7bb818be22396dce6f2430df3451bb2deb4faf9bcff4cd007d737172e0eec764")
set(FOXGLOVE_SHA256_aarch64-unknown-linux-gnu "20dd412fa8be481e6ed84d88879f368e1a55a146e9a98f936a05931742e1dea5")
set(FOXGLOVE_SHA256_x86_64-unknown-linux-gnu "d8afbb8f99b75d7ad0f1b2885d35cacead28e6f2e36bd4d1a30e17413ebab896")

set(FOXGLOVE_URL "https://github.com/foxglove/foxglove-sdk/releases/download/sdk%2F${FOXGLOVE_VERSION}/foxglove-${FOXGLOVE_VERSION}-cpp-${FOXGLOVE_PLATFORM}.zip")
set(FOXGLOVE_SHA ${FOXGLOVE_SHA256_${FOXGLOVE_PLATFORM}})

message(STATUS "Downloading Foxglove from: ${FOXGLOVE_URL}")

FetchContent_Declare(
        foxglove
        URL ${FOXGLOVE_URL}
        URL_HASH SHA256=${FOXGLOVE_SHA}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(foxglove)

