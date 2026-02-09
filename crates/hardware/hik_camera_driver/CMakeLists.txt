# Detect architecture and map to SDK directory
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
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
        set(HIK_USE_SDK TRUE)
    else ()
        message(STATUS "Building hikcamera with stub (HIK SDK not available for ${CMAKE_SYSTEM_PROCESSOR} or lib directory not found)")
        set(HIK_USE_SDK FALSE)
    endif ()
else ()
    message(STATUS "Building hikcamera with stub (HIK SDK only available on Linux)")
    set(HIK_USE_SDK FALSE)
endif ()

if (HIK_USE_SDK)
    crate(
        NAME hikcamera
        TYPE SHARED
        SOURCES
            src/hik_camera.cpp
        DEPS
            magic_enum
            talos_math
            talos_log
            ${OpenCV_LIBS}
    )

    target_include_directories(hikcamera SYSTEM PUBLIC ${HIK_SDK_ROOT}/include)

    target_link_directories(hikcamera PRIVATE ${HIK_SDK_LIB_DIR})

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
    crate_silence_warnings(hikcamera)
else ()
    crate(
        NAME hikcamera
        TYPE SHARED
        SOURCES
            src/hik_camera_stub.cpp
        DEPS
            magic_enum
            talos_math
            talos_log
            ${OpenCV_LIBS}
    )
endif ()
