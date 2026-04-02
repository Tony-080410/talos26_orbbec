# FFmpeg discovery (required for quanta — video encode/decode)
pkg_check_modules(AVCODEC REQUIRED IMPORTED_TARGET libavcodec)
pkg_check_modules(AVFORMAT REQUIRED IMPORTED_TARGET libavformat)
pkg_check_modules(AVFILTER REQUIRED IMPORTED_TARGET libavfilter)
pkg_check_modules(SWSCALE REQUIRED IMPORTED_TARGET libswscale)
pkg_check_modules(AVUTIL REQUIRED IMPORTED_TARGET libavutil)

crate(
    NAME quanta
    TYPE SHARED
    SOURCES
        src/transcoder.cpp
        src/stream_encoder.cpp
    DEPS
        PkgConfig::AVCODEC
        PkgConfig::AVFORMAT
        PkgConfig::AVFILTER
        PkgConfig::SWSCALE
        PkgConfig::AVUTIL
)

# MVP executable: file-to-file transcoder
add_executable(quanta-transcode src/main.cpp)
target_link_libraries(quanta-transcode PRIVATE quanta)

# Stream encoder debug tool: generates test pattern -> H.264 file
add_executable(quanta-stream-test src/debug_main.cpp)
target_link_libraries(quanta-stream-test PRIVATE quanta)
