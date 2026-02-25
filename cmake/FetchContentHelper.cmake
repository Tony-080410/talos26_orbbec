# FetchContent helper: local zip cache first, remote fallback
#
# Usage (standard GitHub archive):
#   fetch_dependency(NAME fmt
#                    REPO fmtlib/fmt
#                    VERSION 12.1.0)
#   -> Checks: 3dparty/fmt-12.1.0.zip
#   -> URL: github.com/fmtlib/fmt/archive/refs/tags/12.1.0.zip
#
# Usage (custom URL + local name):
#   fetch_dependency(NAME foxglove
#                    REPO foxglove/foxglove-sdk
#                    VERSION v0.17.1
#                    ZIP_URL https://github.com/.../release.zip
#                    ZIP_NAME foxglove-v0.17.1-cpp-x86_64.zip)
#   -> Checks: 3dparty/foxglove-v0.17.1-cpp-x86_64.zip
#   -> URL: custom ZIP_URL

include(FetchContent)

function(fetch_dependency)
    cmake_parse_arguments(ARG
        ""
        "NAME;REPO;VERSION;ZIP_URL;ZIP_NAME"
        ""
        ${ARGN})

    if(NOT ARG_NAME)
        message(FATAL_ERROR "fetch_dependency: NAME is required")
    endif()

    # Build default zip filename and URL
    if(NOT ARG_ZIP_NAME)
        set(ARG_ZIP_NAME "${ARG_NAME}-${ARG_VERSION}.zip")
    endif()

    if(NOT ARG_ZIP_URL)
        if(NOT ARG_REPO OR NOT ARG_VERSION)
            message(FATAL_ERROR "fetch_dependency: Must specify (REPO + VERSION) or ZIP_URL")
        endif()
        set(ARG_ZIP_URL "https://github.com/${ARG_REPO}/archive/refs/tags/v${ARG_VERSION}.zip")
    endif()

    set(LOCAL_ZIP "${CMAKE_CURRENT_SOURCE_DIR}/3dparty/${ARG_ZIP_NAME}")

    if(EXISTS "${LOCAL_ZIP}")
        message(STATUS "Using local ${ARG_NAME} at: ${LOCAL_ZIP}")
        FetchContent_Declare(${ARG_NAME} URL ${LOCAL_ZIP})
    else()
        message(STATUS "Downloading ${LOCAL_ZIP} ${ARG_NAME} from ${ARG_ZIP_URL}")
        FetchContent_Declare(${ARG_NAME} URL ${ARG_ZIP_URL})
    endif()

    FetchContent_MakeAvailable(${ARG_NAME})
endfunction()
