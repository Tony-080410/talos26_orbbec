#include "fcs/L1_sensor/input_interface.hpp"

namespace fcs::L1 {

std::expected<InputInterface, InputError>
    InputInterface::create_ipc(std::shared_ptr<ipc::ShmClient> client) noexcept {
    // 从 IPC 获取相机内参
    const auto& ipc_info = client->camera_info();
    CameraConfig info;
    info.camera_matrix << ipc_info.fx, 0.0, ipc_info.cx, 0.0, ipc_info.fy, ipc_info.cy, 0.0, 0.0,
        1.0;
    info.distort_coefficient << ipc_info.distortion[0], ipc_info.distortion[1],
        ipc_info.distortion[2], ipc_info.distortion[3], ipc_info.distortion[4];
    info.width  = ipc_info.width;
    info.height = ipc_info.height;

    return InputInterface(IpcInput(client, info));
}

std::expected<InputInterface, InputError> InputInterface::create_hik(
    const hikcamera::ImageCapturer::CameraProfile& profile, const CameraConfig& info) noexcept {
    auto result = hikcamera::ImageCapturer::create(profile);
    if (!result) {
        return std::unexpected(result.error());
    }
    return InputInterface(HikInput(std::move(*result), info, profile.exposure_time));
}

std::expected<InputInterface, InputError>
    InputInterface::create_hik(const CameraConfig& config) noexcept {
    hikcamera::ImageCapturer::CameraProfile profile;
    profile.trigger_mode = config.profile.trigger_mode;
    profile.invert_image = config.profile.invert_image;
    profile.exposure_time =
        std::chrono::duration<float, std::micro>(config.profile.exposure_time_us);
    profile.gain = config.profile.gain;

    auto result = hikcamera::ImageCapturer::create(profile);
    if (!result) {
        return std::unexpected(result.error());
    }
    return InputInterface(HikInput(std::move(*result), config, profile.exposure_time));
}

} // namespace fcs::L1
