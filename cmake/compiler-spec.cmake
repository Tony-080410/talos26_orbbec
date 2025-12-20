find_program(SCCACHE_PROGRAM sccache)

if (SCCACHE_PROGRAM)
    message(STATUS "sccache found: ${SCCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER ${SCCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${SCCACHE_PROGRAM})
else ()
    message(STATUS "sccache not found, building without compiler cache")
endif ()

add_compile_options(
        -O3
        -march=native
        -mtune=native

        # LTO
        -flto=thin

        # 数学优化
        # -ffast-math
        # -fno-math-errno
        -ffinite-math-only

        # 循环优化
        -funroll-loops
        -fvectorize
        -fslp-vectorize

        # 内联优化
        -finline-functions
        -finline-hint-functions

        # 其他优化
        -fomit-frame-pointer
        -fstrict-aliasing
        -fstrict-enums
)

add_link_options(-flto=thin)

add_compile_options(
        # 基础警告
        -Wpedantic -Werror
)
