#include "hik_camera.hpp"
#include "CameraParams.h"
#include "MvErrorDefine.h"

#include <cstring>
#include <memory>
#include <mutex>

#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <ratio>
#include <tuple>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <MvCameraControl.h>
#include <PixelType.h>

namespace hikcamera {

std::string error_code_to_message(unsigned int code) {
    switch (code) {
    // Success
    case MV_OK: return "Success";

    // General errors
    case MV_E_HANDLE: return "Error or invalid handle";
    case MV_E_SUPPORT: return "Not supported function";
    case MV_E_BUFOVER: return "Buffer overflow";
    case MV_E_CALLORDER: return "Function calling order error";
    case MV_E_PARAMETER: return "Incorrect parameter";
    case MV_E_RESOURCE: return "Applying resource failed";
    case MV_E_NODATA: return "No data";
    case MV_E_PRECONDITION: return "Precondition error, or running environment changed";
    case MV_E_VERSION: return "Version mismatches";
    case MV_E_NOENOUGH_BUF: return "Insufficient memory";
    case MV_E_ABNORMAL_IMAGE:
        return "Abnormal image, maybe incomplete image because of lost packet";
    case MV_E_LOAD_LIBRARY: return "Load library failed";
    case MV_E_NOOUTBUF: return "No available buffer";
    case MV_E_ENCRYPT: return "Encryption error";
    case MV_E_OPENFILE: return "Open file error";
    case MV_E_BUF_IN_USE: return "Buffer already in use";
    case MV_E_BUF_INVALID: return "Buffer address invalid";
    case MV_E_NOALIGN_BUF: return "Buffer alignment error";
    case MV_E_NOENOUGH_BUF_NUM: return "Insufficient cache count";
    case MV_E_PORT_IN_USE: return "Port is in use";
    case MV_E_IMAGE_DECODEC: return "Decoding error (SDK verification image exception)";
    case MV_E_UINT32_LIMIT: return "Image size exceeds unsigned int return limit";
    case MV_E_IMAGE_HEIGHT: return "Image height anomaly (discard incomplete image)";
    case MV_E_NOENOUGH_DDR: return "Insufficient DDR cache";
    case MV_E_NOENOUGH_STREAM: return "Insufficient stream channel";
    case MV_E_NORESPONSE: return "No response from device";
    case MV_E_UNKNOW: return "Unknown error";

    // GenICam errors
    case MV_E_GC_GENERIC: return "General error (GenICam)";
    case MV_E_GC_ARGUMENT: return "Illegal parameters (GenICam)";
    case MV_E_GC_RANGE: return "The value is out of range (GenICam)";
    case MV_E_GC_PROPERTY: return "Property error (GenICam)";
    case MV_E_GC_RUNTIME: return "Running environment error (GenICam)";
    case MV_E_GC_LOGICAL: return "Logical error (GenICam)";
    case MV_E_GC_ACCESS: return "Node accessing condition error (GenICam)";
    case MV_E_GC_TIMEOUT: return "Timeout (GenICam)";
    case MV_E_GC_DYNAMICCAST: return "Transformation exception (GenICam)";
    case MV_E_GC_UNKNOW: return "GenICam unknown error";

    // GigE errors
    case MV_E_NOT_IMPLEMENTED: return "The command is not supported by device";
    case MV_E_INVALID_ADDRESS: return "The target address being accessed does not exist";
    case MV_E_WRITE_PROTECT: return "The target address is not writable";
    case MV_E_ACCESS_DENIED: return "No permission";
    case MV_E_BUSY: return "Device is busy, or network disconnected";
    case MV_E_PACKET: return "Network data packet error";
    case MV_E_NETER: return "Network error";
    case MV_E_SUPPORT_MODIFY_DEVICE_IP: return "Current mode not support modify IP";
    case MV_E_KEY_VERIFICATION: return "Switch key verification error";
    case MV_E_IP_CONFLICT: return "Device IP conflict";

    // USB errors
    case MV_E_USB_READ: return "Reading USB error";
    case MV_E_USB_WRITE: return "Writing USB error";
    case MV_E_USB_DEVICE: return "Device exception";
    case MV_E_USB_GENICAM: return "GenICam error (USB)";
    case MV_E_USB_BANDWIDTH: return "Insufficient bandwidth";
    case MV_E_USB_DRIVER: return "Driver mismatch or unmounted drive";
    case MV_E_USB_UNKNOW: return "USB unknown error";

    // Upgrade errors
    case MV_E_UPG_FILE_MISMATCH: return "Firmware mismatches";
    case MV_E_UPG_LANGUSGE_MISMATCH: return "Firmware language mismatches";
    case MV_E_UPG_CONFLICT: return "Upgrading conflicted";
    case MV_E_UPG_INNER_ERR: return "Camera internal error during upgrade";
    case MV_E_UPG_UNKNOW: return "Unknown error during upgrade";

    default: return fmt::format("Unknown error code: 0x{:08X}", code);
    }
}

// SDK 全局初始化/清理实现
// 使用 Meyer's Singleton 确保线程安全和正确的初始化顺序
namespace {
class SDKInitializer {
public:
    static SDKInitializer& instance() noexcept {
        static SDKInitializer init;
        return init;
    }

    std::expected<void, CameraError> initialize() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (initialized_) {
            return {}; // 已经初始化，直接返回成功
        }

        unsigned int ret = MV_CC_Initialize();
        if (ret != MV_OK) {
            return std::unexpected(
                CameraError{"Failed to initialize HIK SDK: " + error_code_to_message(ret), ret});
        }

        initialized_ = true;
        return {};
    }

    void finalize() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (initialized_) {
            MV_CC_Finalize();
            initialized_ = false;
        }
    }

private:
    SDKInitializer() noexcept = default;
    ~SDKInitializer() noexcept { finalize(); }

    std::mutex mutex_;
    bool initialized_ = false;
};
} // namespace

std::expected<void, CameraError> initialize_sdk() noexcept {
    return SDKInitializer::instance().initialize();
}

void finalize_sdk() noexcept { SDKInitializer::instance().finalize(); }

class ImageCapturer::Impl {
public:
    std::expected<void, CameraError> init(
        const CameraProfile& profile, const char* user_defined_name,
        const SyncMode& sync_mode) noexcept {
        SPDLOG_INFO("[HIK] Starting camera init...");
        auto device_info = search_camera(user_defined_name);
        if (!device_info) {
            SPDLOG_ERROR("[HIK] Failed to search camera: {}", device_info.error().message);
            return std::unexpected(device_info.error());
        }
        SPDLOG_INFO("[HIK] Camera found, initializing...");

        auto init_result = init_camera(**device_info, profile, sync_mode);
        if (!init_result) {
            SPDLOG_ERROR("[HIK] Camera init failed: {}", init_result.error().message);
            return init_result;
        }
        SPDLOG_INFO("[HIK] Camera init OK, clearing stale buffers...");

        // 清除缓冲区中的残留帧（处理程序异常退出后的脏数据）
        // 相比 read() 等待超时，ClearImageBuffer 更直接且不会阻塞
        unsigned int clear_ret = MV_CC_ClearImageBuffer(camera_handle_);
        if (clear_ret != MV_OK) {
            SPDLOG_WARN("[HIK] Failed to clear image buffer: {}", error_code_to_message(clear_ret));
            // 非致命错误，继续
        }

        return {};
    }

    ~Impl() noexcept { uninit_camera(); }

    std::tuple<int, int> get_image_size() const noexcept { return {image_width_, image_height_}; }

    std::expected<cv::Mat, CameraError>
        read(std::chrono::duration<unsigned int, std::milli> timeout) noexcept {
        MV_FRAME_OUT stImageInfo;

        // MV_CC_GetImageBuffer timeout 单位是毫秒（nMsec）
        // 注意：不要乘以 1000000，timeout.count() 已经是毫秒数
        unsigned int ret =
            MV_CC_GetImageBuffer(camera_handle_, &stImageInfo, timeout.count() * 100000);
        if (ret != MV_OK) {
            logger_->error(
                "Failed to get image buffer: {} [code=0x{:08X}]", error_code_to_message(ret), ret);
            return std::unexpected(
                CameraError{"Failed to get image buffer: " + error_code_to_message(ret), ret});
        }

        // RAII 包装器确保 MV_CC_FreeImageBuffer 被调用一次
        struct ImageBufferGuard {
            void* handle;
            MV_FRAME_OUT* frame;
            bool released = false;

            ~ImageBufferGuard() noexcept {
                if (!released && frame != nullptr && handle != nullptr) {
                    MV_CC_FreeImageBuffer(handle, frame);
                }
            }

            void release() noexcept { released = true; }
        };

        ImageBufferGuard buffer_guard{camera_handle_, &stImageInfo};

        // 获取 Bayer 像素格式对应的 OpenCV 解拜耳代码
        const auto bayer_code = get_bayer_conversion_code(stImageInfo.stFrameInfo.enPixelType);
        if (!bayer_code) {
            return std::unexpected(
                CameraError{
                    "Unsupported Bayer format: "
                        + std::to_string(stImageInfo.stFrameInfo.enPixelType),
                    0});
        }

        // 包装原始 Bayer 数据（不拷贝数据，只创建 cv::Mat header）
        const int width  = stImageInfo.stFrameInfo.nWidth;
        const int height = stImageInfo.stFrameInfo.nHeight;
        cv::Mat raw_bayer{height, width, CV_8UC1, stImageInfo.pBufAddr};

        // 使用 OpenCV 的 EA（Edge-Aware，边缘感知）算法解拜耳
        // 相比普通插值更注重边缘方向，能减少边缘处的彩边和模糊
        // 在画质与速度之间通常比 VNG 更适合实时视觉场景
        cv::Mat img;
        cv::cvtColor(raw_bayer, img, *bayer_code);

        // 缓存图像尺寸（供 get_width_height() 使用）
        image_width_  = width;
        image_height_ = height;

        // 旋转图像（如果需要）
        if (rotate_type_ != RotateType::None) {
            try {
                switch (rotate_type_) {
                case RotateType::None: break;
                case RotateType::Clockwise90: cv::rotate(img, img, cv::ROTATE_90_CLOCKWISE); break;
                case RotateType::Clockwise180: cv::rotate(img, img, cv::ROTATE_180); break;
                case RotateType::Clockwise270:
                    cv::rotate(img, img, cv::ROTATE_90_COUNTERCLOCKWISE);
                    break;
                }
            } catch (const cv::Exception& e) {
                logger_->error("OpenCV rotation failed: {}", e.what());
                return std::unexpected(CameraError{"OpenCV rotation failed"});
            }
        }

        // buffer_guard 析构时会自动调用 MV_CC_FreeImageBuffer
        return img;
    }

    std::expected<void, CameraError> software_trigger_on() noexcept {
        auto ret = MV_CC_SetCommandValue(camera_handle_, "TriggerSoftware");
        if (MV_OK != ret) {
            logger_->error("Failed To send software trigger nRet [{}]", ret);
            return std::unexpected(
                CameraError{"Failed to send software trigger", static_cast<unsigned int>(ret)});
        }
        return {};
    }

    std::expected<void, CameraError> set_frame_rate_inner_trigger_mode(float frame_rate) noexcept {
        // 在触发模式下，帧率由触发信号控制，AcquisitionFrameRate 设置无效
        if (trigger_mode_) {
            return std::unexpected(CameraError{"Cannot set frame rate in trigger mode", 0});
        }

        auto ret = MV_CC_SetBoolValue(camera_handle_, "AcquisitionFrameRateEnable", true);
        if (MV_OK != ret) {
            logger_->error("Failed To set frame rate control enable nRet [{}]", ret);
            return std::unexpected(
                CameraError{
                    "Failed to set frame rate control enable", static_cast<unsigned int>(ret)});
        }
        ret = MV_CC_SetFloatValue(camera_handle_, "AcquisitionFrameRate", frame_rate);
        if (MV_OK != ret) {
            logger_->error("Failed To set frame rate nRet [{}]", ret);
            return std::unexpected(
                CameraError{"Failed to set frame rate", static_cast<unsigned int>(ret)});
        }
        return {};
    }

    std::expected<void, CameraError> stop_grabbing() noexcept {
        if (camera_handle_ == nullptr) {
            return {};
        }
        unsigned int ret = MV_CC_StopGrabbing(camera_handle_);
        if (ret != MV_OK) {
            logger_->error("Failed to stop grabbing: {}", error_code_to_message(ret));
            return std::unexpected(
                CameraError{"Failed to stop grabbing: " + error_code_to_message(ret), ret});
        }
        return {};
    }

private:
    RotateType rotate_type_ = RotateType::None;

    // 缓存图像尺寸（在第一次 read() 后填充）
    int image_width_  = 0;
    int image_height_ = 0;

    // 记录当前是否处于触发模式（在 init_camera 时确定）
    bool trigger_mode_ = false;

#define SDK_RET_ASSERT(ret, message)                                                      \
    do {                                                                                  \
        if ((ret) != MV_OK) {                                                             \
            logger_->error(message " nRet [{}]", ret);                                    \
            return std::unexpected(CameraError{message, static_cast<unsigned int>(ret)}); \
        }                                                                                 \
    } while (false)

    bool is_same_device_name(MV_CC_DEVICE_INFO* pstMVDevInfo, const char* targetName) noexcept {
        if (nullptr == pstMVDevInfo) {
            logger_->error("The Pointer of pstMVDevInfo is NULL!");
            return false;
        }
        const unsigned char* deviceName;
        if (pstMVDevInfo->nTLayerType == MV_GIGE_DEVICE) {
            deviceName = pstMVDevInfo->SpecialInfo.stGigEInfo.chUserDefinedName;
        } else if (pstMVDevInfo->nTLayerType == MV_USB_DEVICE) {
            deviceName = pstMVDevInfo->SpecialInfo.stUsb3VInfo.chUserDefinedName;
        } else {
            return false;
        }

        return strcmp(reinterpret_cast<const char*>(deviceName), targetName) == 0;
    }

    bool print_device_info(MV_CC_DEVICE_INFO* pstMVDevInfo) noexcept {
        if (nullptr == pstMVDevInfo) {
            logger_->error("The Pointer of pstMVDevInfo is NULL!");
            return false;
        }
        if (pstMVDevInfo->nTLayerType == MV_GIGE_DEVICE) {
            unsigned nIp1 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24);
            unsigned nIp2 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16);
            unsigned nIp3 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8);
            unsigned nIp4 = (pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff);
            logger_->info("DeviceIp: {}.{}.{}.{}", nIp1, nIp2, nIp3, nIp4);
            logger_->info(
                "UserDefinedName: {}", reinterpret_cast<const char*>(
                                           pstMVDevInfo->SpecialInfo.stGigEInfo.chUserDefinedName));
        } else if (pstMVDevInfo->nTLayerType == MV_USB_DEVICE) {
            logger_->info(
                "UserDefinedName: {}",
                reinterpret_cast<const char*>(
                    pstMVDevInfo->SpecialInfo.stUsb3VInfo.chUserDefinedName));
            logger_->info(
                "Serial Number: {}", reinterpret_cast<const char*>(
                                         pstMVDevInfo->SpecialInfo.stUsb3VInfo.chSerialNumber));
            logger_->info("Device Number: {}", pstMVDevInfo->SpecialInfo.stUsb3VInfo.nDeviceNumber);
        } else {
            logger_->error("Neither a GigE camera nor a USB camera.");
            return false;
        }

        return true;
    }

    std::expected<MV_CC_DEVICE_INFO*, CameraError>
        search_camera(const char* user_defined_name) noexcept {
        MV_CC_DEVICE_INFO_LIST device_list;
        memset(&device_list, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
        unsigned int ret = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &device_list);
        if (ret != MV_OK) {
            return std::unexpected(
                CameraError{"Failed to enum Devices: " + error_code_to_message(ret), ret});
        }
        if (device_list.nDeviceNum == 0) {
            return std::unexpected(CameraError{"Find No Devices.", 0});
        }

        if (user_defined_name == nullptr) {
            if (device_list.nDeviceNum > 1) {
                return std::unexpected(
                    CameraError{
                        fmt::format(
                            "Must pass in the device name because {} devices were found.",
                            device_list.nDeviceNum),
                        0});
            }
            return device_list.pDeviceInfo[0];
        } else {
            for (unsigned int i = 0; i < device_list.nDeviceNum; i++) {
                if (is_same_device_name(device_list.pDeviceInfo[i], user_defined_name))
                    return device_list.pDeviceInfo[i];
            }
            return std::unexpected(
                CameraError{
                    fmt::format(
                        "{} devices was found, but no device matches the name passed in: {}",
                        device_list.nDeviceNum, user_defined_name),
                    0});
        }
    }

    std::expected<void, CameraError> init_camera(
        MV_CC_DEVICE_INFO& device_info, const CameraProfile& profile,
        const SyncMode& sync_mode) noexcept {
        auto pDeviceInfo = &device_info;
        rotate_type_     = profile.rotate_type;

        unsigned int ret;

        // 使用 MV_CC_CreateHandle 而非已废弃的 MV_CC_CreateHandleWithoutLog
        // 参考文档：Deprecated camera initialization related APIs: MV_CC_CreateHandleWithoutLog()
        ret = MV_CC_CreateHandle(&camera_handle_, pDeviceInfo);
        SDK_RET_ASSERT(ret, "Failed to create handle.");
        FinalAction destroy_handle{[this]() { MV_CC_DestroyHandle(camera_handle_); }};

        ret = MV_CC_OpenDevice(camera_handle_);
        SDK_RET_ASSERT(ret, "Failed to open device.");
        FinalAction close_device{[this]() { MV_CC_CloseDevice(camera_handle_); }};

        if (pDeviceInfo->nTLayerType == MV_GIGE_DEVICE) {
            int nPacketSize = MV_CC_GetOptimalPacketSize(camera_handle_);
            if (nPacketSize <= 0) {
                logger_->error("Get invalid packet Size: {}", nPacketSize);
                return std::unexpected(CameraError{"Invalid packet size", 0});
            }

            // 使用 MV_CC_SetIntValueEx 而非 MV_CC_SetIntValue（与官方示例保持一致）
            ret = MV_CC_SetIntValueEx(camera_handle_, "GevSCPSPacketSize", nPacketSize);
            SDK_RET_ASSERT(ret, "Failed to set packet Size.");
        }

        // Set pixel format to 8-bit BayerRG for ADC depth
        ret = MV_CC_SetEnumValue(camera_handle_, "PixelFormat", PixelType_Gvsp_BayerRG8);
        SDK_RET_ASSERT(ret, "Failed to set pixel format to BayerRG8.");

        // 根据 profile 和 sync_mode 统一设置 TriggerMode，避免重复设置
        bool trigger_on = profile.trigger_mode || (sync_mode == SyncMode::SOFTWARE);
        trigger_mode_   = trigger_on;
        ret             = MV_CC_SetEnumValue(
            camera_handle_, "TriggerMode", trigger_on ? MV_TRIGGER_MODE_ON : MV_TRIGGER_MODE_OFF);
        SDK_RET_ASSERT(ret, "Failed to set trigger Mode.");

        ret = MV_CC_SetBoolValue(camera_handle_, "ReverseX", profile.invert_image);
        SDK_RET_ASSERT(ret, "Failed to set reverse x.");
        ret = MV_CC_SetBoolValue(camera_handle_, "ReverseY", profile.invert_image);
        SDK_RET_ASSERT(ret, "Failed to set reverse y.");

        ret = MV_CC_SetEnumValue(camera_handle_, "ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF);
        SDK_RET_ASSERT(ret, "Failed to set auto exposure.");

        ret = MV_CC_SetFloatValue(camera_handle_, "ExposureTime", profile.exposure_time.count());
        SDK_RET_ASSERT(ret, "Failed to set exposure time.");

        ret = MV_CC_SetFloatValue(camera_handle_, "Gain", profile.gain);
        SDK_RET_ASSERT(ret, "Failed to set gain.");

        ret = MV_CC_SetBoolValue(camera_handle_, "AcquisitionFrameRateEnable", false);
        SDK_RET_ASSERT(ret, "Failed to set acquisition frame rate enable.");

        ret = MV_CC_SetBayerCvtQuality(camera_handle_, 1);
        SDK_RET_ASSERT(ret, "Failed to set bayer cvt quality.");

        // 文档建议：在 MV_CC_StartGrabbing() 之前调用 MV_CC_SetImageNodeNum()
        // 设置合适的缓冲节点数量，影响因子包括：相机帧率、图像分辨率、计算机性能
        // 注意：此处使用默认值，如需优化可根据实际情况调整
        ret = MV_CC_SetImageNodeNum(camera_handle_, 5);
        SDK_RET_ASSERT(ret, "Failed to set image node num.");

        // 设置取流策略为 LatestImagesOnly（实时场景优先取最新帧，丢弃积压）
        // 注意：必须在 StartGrabbing 之前调用；Linux 下仅 USB 设备支持
        ret = MV_CC_SetGrabStrategy(camera_handle_, MV_GrabStrategy_LatestImagesOnly);
        if (ret != MV_OK) {
            logger_->warn(
                "Failed to set grab strategy (may not be supported on this "
                "platform/device): {}",
                error_code_to_message(ret));
        }

        // 注册设备断连回调，及时感知相机离线（必须在 StartGrabbing 之前）
        ret = MV_CC_RegisterExceptionCallBack(camera_handle_, Impl::on_exception_callback, this);
        if (ret != MV_OK) {
            logger_->warn("Failed to register exception callback: {}", error_code_to_message(ret));
        }

        // 注册流异常回调，及时感知丢包/断流/图像异常（必须在 StartGrabbing 之前）
        ret = MV_CC_RegisterStreamExceptionCallBack(
            camera_handle_, Impl::on_stream_exception_callback, this);
        if (ret != MV_OK) {
            logger_->warn(
                "Failed to register stream exception callback: {}", error_code_to_message(ret));
        }

        ret = MV_CC_StartGrabbing(camera_handle_);
        SDK_RET_ASSERT(ret, "Failed to start grabbing.");

        // SOFTWARE 模式下设置 TriggerSource 为 SOFTWARE
        if (sync_mode == SyncMode::SOFTWARE) {
            ret = MV_CC_SetEnumValue(camera_handle_, "TriggerSource", MV_TRIGGER_SOURCE_SOFTWARE);
            SDK_RET_ASSERT(ret, "Failed to set trigger source. : soft trigger");
        }

        destroy_handle.disable();
        close_device.disable();
        return {};
    }

    // 将 HIK SDK 的 Bayer 像素格式映射到 OpenCV 的解拜耳代码
    // 参考：OpenCV cvtColor 文档中 Bayer 模式定义
    static std::expected<int, CameraError>
        get_bayer_conversion_code(MvGvspPixelType enType) noexcept {
        switch (enType) {
        // 8-bit Bayer 格式（推荐使用）
        case PixelType_Gvsp_BayerRG8: return cv::COLOR_BayerBG2BGR_EA;
        case PixelType_Gvsp_BayerBG8: return cv::COLOR_BayerRG2BGR_EA;
        case PixelType_Gvsp_BayerGR8: return cv::COLOR_BayerGB2BGR_EA;
        case PixelType_Gvsp_BayerGB8: return cv::COLOR_BayerGR2BGR_EA;

        // 10-bit Bayer 格式（需要预处理）
        case PixelType_Gvsp_BayerRG10:
        case PixelType_Gvsp_BayerRG10_Packed:
        case PixelType_Gvsp_BayerBG10:
        case PixelType_Gvsp_BayerBG10_Packed:
        case PixelType_Gvsp_BayerGR10:
        case PixelType_Gvsp_BayerGR10_Packed:
        case PixelType_Gvsp_BayerGB10:
        case PixelType_Gvsp_BayerGB10_Packed:
            return std::unexpected(
                CameraError{"10-bit Bayer not supported, please set to 8-bit mode", 0});

        // 12-bit Bayer 格式（需要预处理）
        case PixelType_Gvsp_BayerRG12:
        case PixelType_Gvsp_BayerRG12_Packed:
        case PixelType_Gvsp_BayerBG12:
        case PixelType_Gvsp_BayerBG12_Packed:
        case PixelType_Gvsp_BayerGR12:
        case PixelType_Gvsp_BayerGR12_Packed:
        case PixelType_Gvsp_BayerGB12:
        case PixelType_Gvsp_BayerGB12_Packed:
            return std::unexpected(
                CameraError{"12-bit Bayer not supported, please set to 8-bit mode", 0});

        // 已解拜耳的格式（非 Bayer）
        case PixelType_Gvsp_BGR8_Packed:
        case PixelType_Gvsp_YUV422_Packed:
        case PixelType_Gvsp_YUV422_YUYV_Packed:
            return std::unexpected(CameraError{"Already debayered format, not Bayer", 0});

        default:
            return std::unexpected(
                CameraError{"Unknown pixel format: " + std::to_string(enType), 0});
        }
    }

    void uninit_camera() noexcept {
        // 如果句柄未初始化（构造失败路径），无需清理
        if (camera_handle_ == nullptr) {
            return;
        }

        // 按照官方文档的顺序清理资源
        // StopGrabbing → CloseDevice → DestroyHandle

        unsigned int nRet = MV_CC_StopGrabbing(camera_handle_);
        if (MV_OK != nRet) {
            logger_->error("Failed to stop grabbing. nRet [{}]", error_code_to_message(nRet));
            // 即使失败也继续尝试清理，确保尽可能释放资源
        }

        nRet = MV_CC_CloseDevice(camera_handle_);
        if (MV_OK != nRet) {
            logger_->error("Failed to close device. nRet [{}]", error_code_to_message(nRet));
            // 即使失败也继续尝试销毁句柄
        }

        nRet = MV_CC_DestroyHandle(camera_handle_);
        if (MV_OK != nRet) {
            logger_->error("Failed to destroy handle. nRet [{}]", error_code_to_message(nRet));
            // 句柄销毁失败是最严重的情况，但我们已经尽力了
        }

        camera_handle_ = nullptr;
    }

    template <typename Func>
    struct FinalAction {
        explicit FinalAction(Func func) noexcept
            : clean_{func}
            , enabled_(true) {}

        FinalAction(const FinalAction&)            = delete;
        FinalAction& operator=(const FinalAction&) = delete;
        FinalAction(FinalAction&&)                 = delete;
        FinalAction& operator=(FinalAction&&)      = delete;

        ~FinalAction() noexcept {
            if (enabled_)
                clean_();
        }

        void disable() noexcept { enabled_ = false; }

    private:
        Func clean_;
        bool enabled_;
    };

    void* camera_handle_ = nullptr;

    // 设备断连回调（通过 MV_CC_RegisterExceptionCallBack 注册）
    static void __stdcall on_exception_callback(unsigned int nMsgType, void* pUser) noexcept {
        auto* self = static_cast<Impl*>(pUser);
        if (nMsgType == MV_EXCEPTION_DEV_DISCONNECT) {
            self->logger_->error("[HIK] Device disconnected!");
        } else {
            self->logger_->warn("[HIK] Device exception: {}", exception_type_to_string(nMsgType));
        }
    }

    // 流异常回调（通过 MV_CC_RegisterStreamExceptionCallBack 注册）
    static void __stdcall
        on_stream_exception_callback(MV_CC_STREAM_EXCEPTION_INFO* pstInfo, void* pUser) noexcept {
        auto* self = static_cast<Impl*>(pUser);
        self->logger_->error(
            "[HIK] Stream exception: {}, serial={}",
            stream_exception_type_to_string(pstInfo->enExceptionType),
            reinterpret_cast<const char*>(pstInfo->chSerialNumber));
    }

    static const char* exception_type_to_string(unsigned int nMsgType) noexcept {
        switch (nMsgType) {
        case MV_EXCEPTION_DEV_DISCONNECT: return "device disconnected";
        case MV_EXCEPTION_VERSION_CHECK: return "SDK/driver version mismatch";
        default: return "unknown";
        }
    }

    static const char* stream_exception_type_to_string(MV_CC_STREAM_EXCEPTION_TYPE type) noexcept {
        switch (type) {
        case MV_CC_STREAM_EXCEPTION_ABNORMAL_IMAGE: return "abnormal image (discarded)";
        case MV_CC_STREAM_EXCEPTION_LIST_OVERFLOW:
            return "buffer list full (frames not consumed in time)";
        case MV_CC_STREAM_EXCEPTION_LIST_EMPTY: return "buffer list empty";
        case MV_CC_STREAM_EXCEPTION_RECONNECTION: return "stream reconnection triggered";
        case MV_CC_STREAM_EXCEPTION_DISCONNECTED:
            return "stream reconnection failed, grabbing stopped";
        case MV_CC_STREAM_EXCEPTION_DEVICE: return "device exception, grabbing stopped";
        case MV_CC_STREAM_EXCEPTION_PARTIAL_IMAGE:
            return "partial image (insufficient line height, discarded)";
        case MV_CC_STREAM_EXCEPTION_IMAGE_BUFFER_OVERFLOW:
            return "image data exceeds buffer capacity (discarded)";
        default: return "unknown";
        }
    }

    std::shared_ptr<spdlog::logger> logger_ = []() {
        auto logger = spdlog::get("hikcamera");
        if (!logger) {
            logger = spdlog::stdout_color_mt("hikcamera");
        }
        return logger;
    }();
};

// Factory method implementation
std::expected<std::unique_ptr<ImageCapturer>, CameraError> ImageCapturer::create(
    const CameraProfile& profile, const char* user_defined_name,
    const SyncMode& sync_mode) noexcept {
    auto capturer    = std::make_unique<ImageCapturer>();
    auto init_result = capturer->init(profile, user_defined_name, sync_mode);
    if (!init_result) {
        return std::unexpected(init_result.error());
    }
    SPDLOG_INFO("Hikvision camera initialized successfully.");
    return capturer;
}

ImageCapturer::ImageCapturer() noexcept
    : impl_(std::make_unique<ImageCapturer::Impl>()) {}

ImageCapturer::~ImageCapturer() = default;

std::expected<void, CameraError> ImageCapturer::init(
    const CameraProfile& profile, const char* user_defined_name,
    const SyncMode& sync_mode) noexcept {
    return impl_->init(profile, user_defined_name, sync_mode);
}

std::expected<void, CameraError> ImageCapturer::software_trigger_on() noexcept {
    return impl_->software_trigger_on();
}

std::expected<void, CameraError>
    ImageCapturer::set_frame_rate_inner_trigger_mode(float frame_rate) noexcept {
    return impl_->set_frame_rate_inner_trigger_mode(frame_rate);
}

std::expected<void, CameraError> ImageCapturer::stop_grabbing() noexcept {
    return impl_->stop_grabbing();
}

std::expected<cv::Mat, CameraError>
    ImageCapturer::read(std::chrono::duration<unsigned int, std::milli> timeout) noexcept {
    return impl_->read(timeout);
}

std::expected<std::tuple<int, int>, CameraError> ImageCapturer::get_width_height() const noexcept {
    return impl_->get_image_size();
}

} // namespace hikcamera
