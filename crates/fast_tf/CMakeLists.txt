crate(
    NAME fast_tf
    TYPE INTERFACE
    DEPS
        magic_enum
        talos_math
        TBB::tbb
        fmt::fmt
        ${OpenCV_LIBS}
        lieplusplus
)
