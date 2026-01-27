option(BUILD_UNI_KF_TOOLS "Build uni_kf standalone tools" ON)

if (BUILD_UNI_KF_TOOLS)
    add_executable(uni_kf_super_benchmark
        tools/uni_kf_super_benchmark.cpp
    )

    target_include_directories(uni_kf_super_benchmark PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_include_directories(uni_kf_super_benchmark SYSTEM PRIVATE
        3dparty/lieplusplus
    )

    target_link_libraries(uni_kf_super_benchmark PRIVATE
        Eigen3::Eigen
        Threads::Threads
    )
endif()
