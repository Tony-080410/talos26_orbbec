#include "camera/hik_camera.hpp"
#include "CameraParams.h"
#include "MvErrorDefine.h"
#include "fcs/camera_config.hpp"

#include <cstdlib>
#include <cstring>

#include <chrono>
#include <memory>
#include <ratio>
#include <thread>
#include <tuple>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <MvCameraControl.h>
#include <opencv2/opencv.hpp>

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

class ImageCapturer::Impl {
public:
    std::expected<void, CameraError> init(
        const CameraProfile& profile, const char* user_defined_name, const SyncMode& sync_mode) {
        SPDLOG_INFO("[HIK] Starting camera init...");
        auto device_info = search_camera(user_defined_name);
        if (!device_info) {
            SPDLOG_ERROR("[HIK] Failed to search camera");
            return std::unexpected(CameraError{"Failed to search camera", 0});
        }
        SPDLOG_INFO("[HIK] Camera found, initializing...");

        auto init_result = init_camera(*device_info, profile, sync_mode);
        if (!init_result) {
            SPDLOG_ERROR("[HIK] Camera init failed: {}", init_result.error().message);
            return init_result;
        }
        SPDLOG_INFO("[HIK] Camera init OK, draining old frames...");

        // 先榨干缓冲区里的脏数据（处理程序异常退出后的残留帧）
        try {
            using namespace std::chrono_literals;
            auto read_result = read(1000ms);
            if (!read_result) {
                uninit_camera();
                return std::unexpected(read_result.error());
            }
        } catch (...) {
            uninit_camera();
            return std::unexpected(CameraError{"Failed to read initial frame", 0});
        }
        return {};
    }

    ~Impl() { uninit_camera(); }

    std::tuple<int, int> get_image_size() const {
        return {convert_parameter_.nWidth, convert_parameter_.nHeight};
    }

    std::expected<cv::Mat, CameraError>
        read(std::chrono::duration<unsigned int, std::milli> timeout) {
        MV_FRAME_OUT stImageInfo;

        unsigned int ret =
            MV_CC_GetImageBuffer(camera_handle_, &stImageInfo, timeout.count() * 1000000);
        if (ret != MV_OK) {
            logger_->error("Image getting timeout. nRet [{}]", ret);
            return std::unexpected(CameraError{"Image getting timeout", ret});
        }
        cv::Mat img{stImageInfo.stFrameInfo.nHeight, stImageInfo.stFrameInfo.nWidth, CV_8UC3};
        // Only consider the situation where the size and format of each frame of image passed in by
        // the camera remain unchanged.

        if (!is_rgb_pixel_type(stImageInfo.stFrameInfo.enPixelType))
            return std::unexpected(CameraError{"RGB camera needed", 0});

        convert_parameter_.nWidth      = stImageInfo.stFrameInfo.nWidth;    // image width
        convert_parameter_.nHeight     = stImageInfo.stFrameInfo.nHeight;   // image height
        convert_parameter_.nSrcDataLen = stImageInfo.stFrameInfo.nFrameLen; // input data size
        convert_parameter_.enSrcPixelType =
            stImageInfo.stFrameInfo.enPixelType;                            // input pixel format
        convert_parameter_.enDstPixelType = PixelType_Gvsp_BGR8_Packed;     // output pixel format
        convert_parameter_.pDstBuffer     = img.data;                       // output data buffer
        convert_parameter_.nDstBufferSize = stImageInfo.stFrameInfo.nWidth
                                          * stImageInfo.stFrameInfo.nHeight
                                          * 3;                              // output buffer size

        convert_parameter_.pSrcData = stImageInfo.pBufAddr;                 // input data buffer

        ret = MV_CC_ConvertPixelType(camera_handle_, &convert_parameter_);

        if (rotate_type_ != fcs::RotateType::None) {
            try {
                switch (rotate_type_) {
                case fcs::RotateType::Clockwise90:
                    cv::rotate(img, img, cv::ROTATE_90_CLOCKWISE);
                    break;
                case fcs::RotateType::Clockwise180: cv::rotate(img, img, cv::ROTATE_180); break;
                case fcs::RotateType::Clockwise270:
                    cv::rotate(img, img, cv::ROTATE_90_COUNTERCLOCKWISE);
                    break;
                default:
                    logger_->error("Unsupported rotate type for Hikvision camera.");
                    MV_CC_FreeImageBuffer(camera_handle_, &stImageInfo);
                    return std::unexpected(CameraError{"Unsupported rotate type", 0});
                }
            } catch (const cv::Exception& e) {
                logger_->error("OpenCV rotation failed: {}", e.what());
                MV_CC_FreeImageBuffer(camera_handle_, &stImageInfo);
                return std::unexpected(CameraError{"OpenCV rotation failed"});
            }
        }
        if (ret != MV_OK) {
            logger_->error("Failed to convert pixel type. nRet [{}]", ret);
            MV_CC_FreeImageBuffer(camera_handle_, &stImageInfo);
            return std::unexpected(CameraError{"Failed to convert pixel type", ret});
        }

        // Double buffer swapness
        MV_CC_FreeImageBuffer(camera_handle_, &stImageInfo);

        return img;
    }

    std::expected<void, CameraError> software_trigger_on() {
        auto ret = MV_CC_SetCommandValue(camera_handle_, "TriggerSoftware");
        if (MV_OK != ret) {
            logger_->error("Failed To send software trigger nRet [{}]", ret);
            return std::unexpected(
                CameraError{"Failed to send software trigger", static_cast<unsigned int>(ret)});
        }
        return {};
    }

    std::expected<void, CameraError> set_frame_rate_inner_trigger_mode(float frame_rate) {
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

private:
    fcs::RotateType rotate_type_ = fcs::RotateType::None;
#define SDK_RET_ASSERT(ret, message, error_code)                                          \
    do {                                                                                  \
        if ((ret) != MV_OK) {                                                             \
            logger_->error(message " nRet [{}]", ret);                                    \
            return std::unexpected(CameraError{message, static_cast<unsigned int>(ret)}); \
        }                                                                                 \
    } while (false)

    bool is_same_device_name(MV_CC_DEVICE_INFO* pstMVDevInfo, const char* targetName) {
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

    bool print_device_info(MV_CC_DEVICE_INFO* pstMVDevInfo) {
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

    MV_CC_DEVICE_INFO* search_camera(const char* user_defined_name) {
        MV_CC_DEVICE_INFO_LIST device_list;
        memset(&device_list, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
        unsigned int ret = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &device_list);
        if (ret != MV_OK) {
            logger_->error("Failed to enum Devices. nRet [{}]", ret);
            return nullptr;
        }
        if (device_list.nDeviceNum == 0) {
            logger_->error("Find No Devices.");
            return nullptr;
        }

        if (user_defined_name == nullptr) {
            if (device_list.nDeviceNum > 1) {
                logger_->error(
                    "Must pass in the device name because {} devices were found.",
                    device_list.nDeviceNum);
                return nullptr;
            }
            return device_list.pDeviceInfo[0];
        } else {
            for (unsigned int i = 0; i < device_list.nDeviceNum; i++) {
                if (is_same_device_name(device_list.pDeviceInfo[i], user_defined_name))
                    return device_list.pDeviceInfo[i];
            }
            logger_->error(
                "{} devices was found, but no device matches the name passed in: {}",
                device_list.nDeviceNum, user_defined_name);
            return nullptr;
        }
    }

    std::expected<void, CameraError> init_camera(
        MV_CC_DEVICE_INFO& device_info, const CameraProfile& profile, const SyncMode& sync_mode) {
        auto pDeviceInfo = &device_info;
        rotate_type_     = profile.rotate_type;

        unsigned int ret;

        ret = MV_CC_CreateHandleWithoutLog(&camera_handle_, pDeviceInfo);
        SDK_RET_ASSERT(ret, "Failed to create handle.", ret);
        FinalAction destroy_handle{[this]() { MV_CC_DestroyHandle(camera_handle_); }};

        ret = MV_CC_OpenDevice(camera_handle_);
        SDK_RET_ASSERT(ret, "Failed to open device.", ret);
        FinalAction close_device{[this]() { MV_CC_CloseDevice(camera_handle_); }};

        if (pDeviceInfo->nTLayerType == MV_GIGE_DEVICE) {
            int nPacketSize = MV_CC_GetOptimalPacketSize(camera_handle_);
            if (nPacketSize <= 0) {
                logger_->error("Get invalid packet Size: {}", nPacketSize);
                return std::unexpected(CameraError{"Invalid packet size", 0});
            }

            ret = MV_CC_SetIntValue(camera_handle_, "GevSCPSPacketSize", nPacketSize);
            SDK_RET_ASSERT(ret, "Failed to set packet Size.", ret);
        }

        // Set pixel format to 8-bit BayerRG for ADC depth
        ret = MV_CC_SetEnumValue(camera_handle_, "PixelFormat", PixelType_Gvsp_BayerRG8);
        SDK_RET_ASSERT(ret, "Failed to set pixel format to BayerRG8.", ret);

        ret = MV_CC_SetEnumValue(
            camera_handle_, "TriggerMode",
            profile.trigger_mode ? MV_TRIGGER_MODE_ON : MV_TRIGGER_MODE_OFF);
        SDK_RET_ASSERT(ret, "Failed to set trigger Mode.", ret);

        ret = MV_CC_SetBoolValue(camera_handle_, "ReverseX", profile.invert_image);
        SDK_RET_ASSERT(ret, "Failed to set reverse x.", ret);
        ret = MV_CC_SetBoolValue(camera_handle_, "ReverseY", profile.invert_image);
        SDK_RET_ASSERT(ret, "Failed to set reverse y.", ret);

        ret = MV_CC_SetEnumValue(camera_handle_, "ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF);
        SDK_RET_ASSERT(ret, "Failed to set auto exposure.", ret);

        ret = MV_CC_SetFloatValue(camera_handle_, "ExposureTime", profile.exposure_time.count());
        SDK_RET_ASSERT(ret, "Failed to set exposure time.", ret);

        ret = MV_CC_SetFloatValue(camera_handle_, "Gain", profile.gain);
        SDK_RET_ASSERT(ret, "Failed to set gain.", ret);

        ret = MV_CC_SetBoolValue(camera_handle_, "AcquisitionFrameRateEnable", false);
        SDK_RET_ASSERT(ret, "Failed to set acquisition frame rate enable.", ret);

        ret = MV_CC_SetBayerCvtQuality(camera_handle_, 1);
        SDK_RET_ASSERT(ret, "Failed to set bayer cvt quality.", ret);

        ret = MV_CC_StartGrabbing(camera_handle_);
        SDK_RET_ASSERT(ret, "Failed to start grabbing.", ret);

        if (sync_mode == SyncMode::SOFTWARE) {
            ret = MV_CC_SetEnumValue(camera_handle_, "TriggerMode", MV_TRIGGER_MODE_ON);
            SDK_RET_ASSERT(ret, "Failed to start TriggerMode. : soft trigger", ret);

            ret = MV_CC_SetEnumValue(camera_handle_, "TriggerSource", MV_TRIGGER_SOURCE_SOFTWARE);
            SDK_RET_ASSERT(ret, "Failed to set trigger source. : soft trigger", ret);
        }

        destroy_handle.disable();
        close_device.disable();
        return {};
    }

    static bool is_rgb_pixel_type(MvGvspPixelType enType) {
        switch (enType) {
        case PixelType_Gvsp_BGR8_Packed:
        case PixelType_Gvsp_YUV422_Packed:
        case PixelType_Gvsp_YUV422_YUYV_Packed:
        case PixelType_Gvsp_BayerGR8:
        case PixelType_Gvsp_BayerRG8:
        case PixelType_Gvsp_BayerGB8:
        case PixelType_Gvsp_BayerBG8:
        case PixelType_Gvsp_BayerGB10:
        case PixelType_Gvsp_BayerGB10_Packed:
        case PixelType_Gvsp_BayerBG10:
        case PixelType_Gvsp_BayerBG10_Packed:
        case PixelType_Gvsp_BayerRG10:
        case PixelType_Gvsp_BayerRG10_Packed:
        case PixelType_Gvsp_BayerGR10:
        case PixelType_Gvsp_BayerGR10_Packed:
        case PixelType_Gvsp_BayerGB12:
        case PixelType_Gvsp_BayerGB12_Packed:
        case PixelType_Gvsp_BayerBG12:
        case PixelType_Gvsp_BayerBG12_Packed:
        case PixelType_Gvsp_BayerRG12:
        case PixelType_Gvsp_BayerRG12_Packed:
        case PixelType_Gvsp_BayerGR12:
        case PixelType_Gvsp_BayerGR12_Packed: return true;
        default: return false;
        }
    }

    void uninit_camera() {
        unsigned int nRet;
        nRet = MV_CC_StopGrabbing(camera_handle_);
        if (MV_OK != nRet)
            logger_->error("Failed to stop grabbing. nRet [{}]", nRet);

        nRet = MV_CC_CloseDevice(camera_handle_);
        if (MV_OK != nRet)
            logger_->error("Failed to close device. nRet [{}]", nRet);

        nRet = MV_CC_DestroyHandle(camera_handle_);
        if (MV_OK != nRet)
            logger_->error("Failed to destroy handle. nRet [{}]", nRet);
    }

    template <typename Func>
    struct FinalAction {
        explicit FinalAction(Func func)
            : clean_{func}
            , enabled_(true) {}

        FinalAction(const FinalAction&)            = delete;
        FinalAction& operator=(const FinalAction&) = delete;
        FinalAction(FinalAction&&)                 = delete;
        FinalAction& operator=(FinalAction&&)      = delete;

        ~FinalAction() {
            if (enabled_)
                clean_();
        }

        void disable() { enabled_ = false; }

    private:
        Func clean_;
        bool enabled_;
    };

    void* camera_handle_ = nullptr;
    MV_CC_PIXEL_CONVERT_PARAM convert_parameter_;

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
    const CameraProfile& profile, const char* user_defined_name, const SyncMode& sync_mode) {
    auto capturer    = std::make_unique<ImageCapturer>();
    auto init_result = capturer->init(profile, user_defined_name, sync_mode);
    if (!init_result) {
        return std::unexpected(init_result.error());
    }
    SPDLOG_INFO("Hikvision camera initialized successfully.");
    return capturer;
}

ImageCapturer::ImageCapturer()
    : impl_(std::make_unique<ImageCapturer::Impl>()) {}

ImageCapturer::~ImageCapturer() = default;

std::expected<void, CameraError> ImageCapturer::init(
    const CameraProfile& profile, const char* user_defined_name, const SyncMode& sync_mode) {
    return impl_->init(profile, user_defined_name, sync_mode);
}

std::expected<void, CameraError> ImageCapturer::software_trigger_on() {
    return impl_->software_trigger_on();
}

std::expected<void, CameraError>
    ImageCapturer::set_frame_rate_inner_trigger_mode(float frame_rate) {
    return impl_->set_frame_rate_inner_trigger_mode(frame_rate);
}

std::expected<cv::Mat, CameraError>
    ImageCapturer::read(std::chrono::duration<unsigned int, std::milli> timeout) {
    return impl_->read(timeout);
}

std::expected<std::tuple<int, int>, CameraError> ImageCapturer::get_width_height() const {
    return impl_->get_image_size();
}

} // namespace hikcamera
