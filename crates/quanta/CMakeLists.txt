# FFmpeg targets are configured centrally in cmake/dependencies.cmake.

crate(
    NAME quanta
    TYPE SHARED
    SOURCES
        src/transcoder.cpp
        src/stream_encoder.cpp
        src/stream_transport.cpp
        src/encoder_backend_ffmpeg.cpp
        src/decoder_backend_ffmpeg.cpp
    DEPS
        PkgConfig::AVCODEC
        PkgConfig::AVFORMAT
        PkgConfig::AVFILTER
        PkgConfig::SWSCALE
        PkgConfig::SWRESAMPLE
        PkgConfig::AVUTIL
)
if(AXERA_FOUND)
    target_compile_definitions(quanta PUBLIC TALOS_HAS_AXERA=1)
    target_include_directories(quanta SYSTEM PUBLIC ${AXERA_INCLUDE_DIRS})
    target_sources(quanta PRIVATE
        src/encoder_backend_ax.cpp
    )
    # spdlog::spdlog is needed by ax_log_redirect.cpp.
    # Must appear BEFORE ${AXERA_LIBRARIES} so our AX_SYS_LogPrint wins at link time.
    target_link_libraries(quanta PUBLIC spdlog::spdlog ${AXERA_LIBRARIES})
    message(STATUS "Axera venc enabled (spdlog log intercept active)")
else()
    target_compile_definitions(quanta PUBLIC TALOS_HAS_AXERA=0)
    message(STATUS "Axera venc disabled")
endif()

# MVP executable: file-to-file transcoder
add_executable(quanta-transcode src/main.cpp)
target_link_libraries(quanta-transcode PRIVATE quanta)

# Stream encoder debug tool: generates test pattern -> HEVC Annex-B file
add_executable(quanta-stream-test src/debug_main.cpp)
target_link_libraries(quanta-stream-test PRIVATE quanta)

# Loopback transport benchmark: sender -> 50Hz MTU simulation -> receiver
add_executable(quanta-stream-loopback src/loopback_main.cpp)
target_link_libraries(quanta-stream-loopback PRIVATE quanta)

# Visual demo: encoder input and decoder output side-by-side in a live window
add_executable(quanta-stream-demo src/window_demo.cpp)
target_link_libraries(quanta-stream-demo PRIVATE quanta opencv_core opencv_imgproc opencv_highgui opencv_videoio)
