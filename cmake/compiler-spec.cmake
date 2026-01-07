option(ENABLE_SCCACHE "Enable compiler caching via sccache" ON)

if (ENABLE_SCCACHE)
    find_program(SCCACHE_PROGRAM sccache)

    if (SCCACHE_PROGRAM)
        message(STATUS "sccache found: ${SCCACHE_PROGRAM}")
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
# add_compile_options(
#         -fsanitize=address -fno-omit-frame-pointer
# )

# add_link_options(
#         -fsanitize=address
# )
# add_compile_options(
#         -O3
#         -march=native
#         -mtune=native

#         # LTO
#         $<$<CONFIG:Release>:-flto=thin>

#         # 数学优化
#         # -ffast-math
#         # -fno-math-errno
#         # -ffinite-math-only

#         # 循环优化
#         -funroll-loops
#         -fvectorize
#         -fslp-vectorize

#         # 内联优化
#         $<$<CONFIG:Release>:-finline-functions>
#         $<$<CONFIG:Release>:-finline-hint-functions>

#         # 其他优化
#         $<$<CONFIG:Release>:-fomit-frame-pointer>
#         -fstrict-aliasing
#         -fstrict-enums
# )

# add_link_options($<$<CONFIG:Release>:-flto=thin>)

# add_compile_options(
#         # 基础警告
#         -Wall -Wextra -Wpedantic -Werror

#         # 未定义行为
#         -Wuninitialized
#         -Wcast-align
#         -Wcast-qual
#         -Wformat=2
#         -Wformat-security
#         -Wnull-dereference
#         -Wstrict-overflow=5
#         -Wstring-plus-int

#         -Wimplicit-fallthrough           # switch 缺少 break
#         -Wmissing-include-dirs           # 缺少的 include 目录

#         # 内存安全
#         -Warray-bounds
#         -Wstack-protector
#         -Wvla
#         -Walloca
#         -Wpointer-arith
#         -Wwrite-strings

#         -Wshadow
#         -Wundef
#         -Wswitch-enum
#         -Wredundant-decls                # 冗余声明
#         -Winline                         # 内联失败
#         -Wdisabled-optimization          # 优化失败
#         -Wunused                         # 未使用的变量/函数
#         -Wunused-parameter               # 未使用的参数
#         -Wunused-result                  # 忽略返回值

#         # C++ 特有
#         -Wold-style-cast
#         -Woverloaded-virtual
#         -Wnon-virtual-dtor
#         -Weffc++
#         -Wsuggest-override
#         -Wzero-as-null-pointer-constant
#         -Wsuggest-override
#         -Wextra-semi
#         -Wdelete-non-virtual-dtor
#         -Wctor-dtor-privacy
#         -Wno-c++98-compat
#         -Wmissing-field-initializers

#         # ========== Clang 特有 ==========
#         -Wthread-safety                  # 线程安全分析
#         -Wthread-safety-analysis         # 详细线程安全分析
#         -Wthread-safety-precise          # 精确线程安全检查
#         -Wthread-safety-negative         # 负面线程安全检查
#         -Wconsumed                       # 资源消费检查
#         -Wdocumentation                  # 文档注释检查
#         -Wdocumentation-pedantic         # 严格文档注释
#         -Wheader-hygiene                 # 头文件卫生
#         -Wmove                           # 移动语义问题
#         -Wrange-loop-analysis            # 范围循环分析
#         -Wcomma                          # 逗号运算符误用
#         -Wassign-enum                    # 枚举赋值
#         -Watomic-implicit-seq-cst        # 原子操作隐式顺序
#         -Wbad-function-cast              # 错误的函数转换
#         -Wfloat-equal                    # 浮点数相等比较
#         -Wmissing-variable-declarations  # 缺少变量声明
#         -Wunreachable-code               # 不可达代码
#         -Wunreachable-code-aggressive    # 激进的不可达代码检查

#         -Wno-c++98-compat                # 不检查 C++98 兼容性
#         -Wno-c++98-compat-pedantic
#         -Wno-exit-time-destructors       # 允许全局析构
#         -Wno-global-constructors         # 允许全局构造
# )
