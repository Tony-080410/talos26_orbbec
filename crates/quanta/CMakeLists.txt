# FFmpeg targets are configured centrally in cmake/dependencies.cmake.

crate(
    NAME quanta
    TYPE SHARED
    SOURCES
        src/transcoder.cpp
        src/stream_encoder.cpp
        src/stream_transport.cpp
    DEPS
        PkgConfig::AVCODEC
        PkgConfig::AVFORMAT
        PkgConfig::AVFILTER
        PkgConfig::SWSCALE
        PkgConfig::SWRESAMPLE
        PkgConfig::AVUTIL
)

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
