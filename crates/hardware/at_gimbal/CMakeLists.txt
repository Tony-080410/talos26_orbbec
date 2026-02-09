crate(
    NAME talos_gimbal
    TYPE INTERFACE
    DEPS
        ${LIBUSB_LINK_LIBRARIES}
)

target_include_directories(talos_gimbal INTERFACE ${LIBUSB_INCLUDE_DIRS})
