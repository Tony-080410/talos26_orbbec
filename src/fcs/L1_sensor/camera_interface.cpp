#include "L1_sensor/camera_interface.hpp"

namespace fcs::L1 {

std::expected<CameraInterface, InputError>
    CameraInterface::create_ipc(std::shared_ptr<ipc::ShmClient> client) noexcept {
    // 从 IPC 获取相机内参
    const auto& ipc_info = client->camera_info();
    CameraConfig info;
    info.camera_matrix << ipc_info.fx, 0.0, ipc_info.cx, 0.0, ipc_info.fy, ipc_info.cy, 0.0, 0.0,
        1.0;
    info.distort_coefficient << ipc_info.distortion[0], ipc_info.distortion[1],
        ipc_info.distortion[2], ipc_info.distortion[3], ipc_info.distortion[4];
    info.width  = ipc_info.width;
    info.height = ipc_info.height;

    return CameraInterface(IpcInput(client, info));
}

std::expected<CameraInterface, InputError>
    CameraInterface::create_hik(const CameraConfig& config) noexcept {
    hikcamera::ImageCapturer::CameraProfile profile;
    profile.trigger_mode = config.profile.trigger_mode;
    profile.invert_image = config.profile.invert_image;
    profile.exposure_time =
        std::chrono::duration<float, std::micro>(config.profile.exposure_time_us);
    profile.gain        = config.profile.gain;
    profile.rotate_type = config.profile.rotate_angle;

    auto result = hikcamera::ImageCapturer::create(profile);
    if (!result) {
        return std::unexpected(result.error());
    }
    return CameraInterface(HikInput(std::move(*result), config, profile, profile.exposure_time));
}

} // namespace fcs::L1
