option(ENABLE_SCCACHE "Enable compiler caching via sccache" ON)

if (ENABLE_SCCACHE)
    find_program(SCCACHE_PROGRAM sccache)

    if (SCCACHE_PROGRAM)
        message(STATUS "Sccache found at: ${SCCACHE_PROGRAM}")
        if (NOT CMAKE_C_COMPILER_LAUNCHER)
            set(CMAKE_C_COMPILER_LAUNCHER ${SCCACHE_PROGRAM})
        endif ()
        if (NOT CMAKE_CXX_COMPILER_LAUNCHER)
            set(CMAKE_CXX_COMPILER_LAUNCHER ${SCCACHE_PROGRAM})
        endif ()
    else ()
        message(STATUS "sccache not found, building without compiler cache")
    endif ()
else ()
    message(STATUS "sccache disabled")
endif ()

add_compile_options(-fPIC)

# Common optimization options (work for both GCC and Clang)
#
# Host-native tuning is only valid for native builds. During cross-compilation it
# can either be rejected outright or silently tune code for the build host
# instead of the target SoC.
if (CMAKE_CROSSCOMPILING)
    add_compile_options(-funroll-loops)
else ()
    add_compile_options(
        -march=native
        -mtune=native
        -funroll-loops
    )
endif ()

# Clang-specific optimization options
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(
        -fvectorize
        -fslp-vectorize
    )
endif()

# Common warning options (work for both GCC and Clang)
add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Werror=pedantic

    -Werror=enum-conversion
    # -Wswitch-default
    -Werror=implicit-fallthrough

    -Werror=misleading-indentation
    -Werror=non-virtual-dtor
    -Werror=overloaded-virtual
    -Werror=suggest-override

    -Werror=null-dereference
    -Werror=array-bounds
    -Werror=vla
    -Werror=alloca
    -Werror=pointer-arith
    -Werror=format=2
    -Werror=format-security
    -Werror=cast-align
    -Werror=cast-qual
    -Werror=redundant-decls
    -Werror=undef

    -Werror=return-type
    -Werror=uninitialized
)

# Clang-specific warning options
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(
        # Type safety
        -Werror=bool-conversion
        -Werror=string-plus-int

        -Werror=header-hygiene
        -Werror=move
        -Werror=assign-enum
        -Werror=bad-function-cast

        -Wrange-loop-analysis
        -Wthread-safety
        -Wthread-safety-analysis
    )
endif()
