crate(
    NAME fast_tf
    TYPE STATIC
    DEPS
        talos_math
        fmt::fmt
        lieplusplus
        TBB::tbb
        ${OpenCV_LIBS}
)
