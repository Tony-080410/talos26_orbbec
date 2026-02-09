crate(
    NAME fast_tf
    TYPE SHARED
    DEPS
        magic_enum
        talos_math
        TBB::tbb
        fmt::fmt
        ${OpenCV_LIBS}
        lieplusplus
)

crate_add_pch(fast_tf)
