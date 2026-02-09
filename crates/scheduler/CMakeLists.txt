crate(
    NAME scheduler
    TYPE SHARED
    DEPS
        magic_enum
        talos_log
        fmt::fmt
        TBB::tbb
)

# 导出符号
target_compile_definitions(scheduler PRIVATE SCHEDULER_EXPORTS)
