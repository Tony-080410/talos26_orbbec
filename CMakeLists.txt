cmake_minimum_required(VERSION 3.14)

include(cmake/compiler-spec.cmake)

project(talos VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(cmake/dependencies.cmake)
include(cmake/test.cmake)

# talos std library (header-only)
add_library(talos_std INTERFACE
        include/detector_state.hpp
        include/performance_probe.hpp
        include/spdlog_hook.hpp
        include/toml_helper.hpp
        include/tracker_soa.hpp
        include/triple_buffer.hpp
        include/ukf_logic.hpp
)

target_include_directories(talos_std INTERFACE include)

target_link_libraries(talos_std INTERFACE
        tomlplusplus::tomlplusplus
        Eigen3::Eigen
        spdlog::spdlog
        fmt::fmt
)


# scheduler library
add_library(scheduler INTERFACE)

target_include_directories(scheduler INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/scheduler
)

target_link_libraries(scheduler INTERFACE
        talos_std
        ${OpenCV_LIBS}
        TBB::tbb
)

# auto aim library
add_library(auto_aim SHARED
        src/auto_aim/classifier.cpp
        src/auto_aim/detector.cpp
        src/auto_aim/extended_kalman_filter.cpp
        src/auto_aim/solver.cpp
        src/auto_aim/target.cpp
        src/auto_aim/tracker.cpp
)

target_include_directories(auto_aim PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include/auto_aim
)

target_link_libraries(auto_aim PUBLIC
        talos_std
        scheduler
        ${OpenCV_LIBS}
        $<$<TARGET_EXISTS:openvino::runtime>:openvino::runtime>
)

# ipc library
add_library(ipc INTERFACE)

target_include_directories(ipc INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/ipc
)

target_link_libraries(ipc INTERFACE
        ${OpenCV_LIBS}
)

# talos gimbal library
add_library(talos_gimbal SHARED
        src/talos_gimbal/usb.cpp
)

target_include_directories(talos_gimbal PUBLIC
        ${LIBUSB_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include/talos_gimbal
)

target_link_libraries(talos_gimbal PRIVATE
        talos_std
        ${LIBUSB_LINK_LIBRARIES}
)

# Main executable
add_executable(${PROJECT_NAME}
        src/main.cpp
        include/toml_helper.hpp
)

target_include_directories(${PROJECT_NAME} PRIVATE include)

target_link_libraries(${PROJECT_NAME} PRIVATE
        talos_gimbal
        scheduler
        ${OpenCV_LIBS}
        spdlog::spdlog
        fmt::fmt
        Eigen3::Eigen
        TBB::tbb
        tomlplusplus::tomlplusplus
        ${LIBUSB_LINK_LIBRARIES}
)

# sp_vision executable
add_executable(sp_vision
        src/sp_vision/sp_vision.cpp
        ${FOXGLOVE_SOURCES}
)


target_include_directories(sp_vision PRIVATE include)
target_include_directories(sp_vision PRIVATE ${foxglove_SOURCE_DIR}/include)
target_include_directories(sp_vision PRIVATE ${foxglove_SOURCE_DIR}/include/foxglove)

target_link_libraries(sp_vision PRIVATE
        talos_gimbal
        ${LIBUSB_LINK_LIBRARIES}
        scheduler
        auto_aim
        ${OpenCV_LIBS}
        spdlog::spdlog
        fmt::fmt
        Eigen3::Eigen
        TBB::tbb
        tomlplusplus::tomlplusplus
        $<$<TARGET_EXISTS:openvino::runtime>:openvino::runtime>
        ${foxglove_SOURCE_DIR}/lib/libfoxglove.a
)

# Detector benchmark (accuracy testing against dataset)
add_executable(detector_benchmark
        test/detector_benchmark.cpp
)

target_include_directories(detector_benchmark PRIVATE include)

target_link_libraries(detector_benchmark PRIVATE
        auto_aim
)


