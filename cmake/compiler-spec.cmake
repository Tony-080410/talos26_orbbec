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

add_compile_options(
    -march=native
    -mtune=native
    -funroll-loops
    -fvectorize
    -fslp-vectorize
)

add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Werror=pedantic

    # Type safety
    -Werror=bool-conversion
    -Wenum-conversion

    -Werror=string-plus-int
    -Werror=switch-enum
    -Wswitch-default
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
    -Werror=header-hygiene
    -Werror=move
    -Wrange-loop-analysis
    -Wthread-safety
    -Wthread-safety-analysis

    -Werror=return-type
    -Werror=uninitialized
    -Werror=assign-enum
    -Werror=bad-function-cast
)
