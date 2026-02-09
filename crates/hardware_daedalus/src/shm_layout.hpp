#pragma once

#include <atomic>
#include <cstdint>

namespace ipc {

// ============ 常量定义 ============

static constexpr size_t CACHE_LINE_SIZE = 64;
static constexpr uint32_t SHM_MAGIC     = 0x54414C05; // "TAL05"
static constexpr uint32_t SHM_VERSION   = 1;

// 图像配置
static constexpr uint32_t IMAGE_WIDTH    = 1440;
static constexpr uint32_t IMAGE_HEIGHT   = 1080;
static constexpr uint32_t IMAGE_CHANNELS = 3;                                     // RGB8
static constexpr size_t IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS; // 4,665,600 bytes
static constexpr size_t IMAGE_POOL_SIZE = IMAGE_SIZE * 3; // 3 buffers for triple buffering

// 共享内存文件名 (将存放在 /tmp 目录)
static constexpr auto SHM_NAME_META       = "talos_ipc_meta";
static constexpr auto SHM_NAME_IMAGE_POOL = "talos_ipc_image_pool";

// ============ TripleBuffer 原子状态 ============

static constexpr uint8_t FLAG_NEW   = 0x80; // Bit 7: 新数据标志
static constexpr uint8_t INDEX_MASK = 0x03; // Bits 0-1: ready index

// ============ 消息类型枚举 ============

enum class MessageType : uint8_t {
    Image      = 0,
    Pose       = 1,
    GimbalCmd  = 2,
    CameraInfo = 3,
};

// ============ 数据结构定义 (C ABI 兼容) ============

/**
 * @brief 图像元数据 - 32 bytes
 */
struct alignas(32) ImageMeta {
    uint64_t seq;          // 帧序号
    uint64_t timestamp_ns; // 纳秒时间戳
    uint32_t width;        // 图像宽度
    uint32_t height;       // 图像高度
    uint8_t buffer_id;     // 图像数据在 pool 中的 slot (0-2)
    uint8_t format;        // 0=RGB8, 1=BGR8, 2=GRAY8
    uint8_t _pad[6];       // 填充到 32 字节
};
static_assert(sizeof(ImageMeta) == 32, "ImageMeta must be 32 bytes");

/**
 * @brief 位姿数据 - 64 bytes
 *
 * 使用 float 以适配 64 字节对齐
 */
struct alignas(64) PoseMeta {
    uint64_t timestamp_ns;
    float position[3];   // x, y, z (meters)
    float quaternion[4]; // w, x, y, z (normalized)
    uint8_t frame_id;    // 0=map, 1=odom, 2=gimbal, 3=camera
    uint8_t _pad[27];    // 填充到 64 字节
};
static_assert(sizeof(PoseMeta) == 64, "PoseMeta must be 64 bytes");

/**
 * @brief 云台控制命令 - 32 bytes
 */
struct alignas(32) GimbalCmd {
    uint64_t timestamp_ns;
    float yaw_deg;       // 目标偏航角 (度)
    float pitch_deg;     // 目标俯仰角 (度)
    float distance_m;    // 目标距离 (米), -1 表示无效
    uint8_t fire_advice; // 1=建议开火, 0=不开火
    uint8_t _pad[11];    // 填充到 32 字节
};
static_assert(sizeof(GimbalCmd) == 32, "GimbalCmd must be 32 bytes");

/**
 * @brief 相机内参 - 128 bytes
 */
struct alignas(64) CameraInfo {
    uint64_t timestamp_ns;
    double fx, fy;        // 焦距
    double cx, cy;        // 主点
    double distortion[5]; // k1, k2, p1, p2, k3
    uint32_t width;
    uint32_t height;
    uint8_t _pad[24];
};
static_assert(sizeof(CameraInfo) == 128, "CameraInfo must be 128 bytes");

// ============ TripleBuffer 槽位 ============

/**
 * @brief 图像 Topic TripleBuffer 布局 - 192 bytes
 */
struct alignas(CACHE_LINE_SIZE) ImageTripleBuffer {
    // 原子状态 (64 bytes for cache line alignment)
    alignas(CACHE_LINE_SIZE) std::atomic<uint8_t> state{1}; // ready=1, no new data
    uint8_t write_idx{0};                                   // 生产者私有
    uint8_t read_idx{2};                                    // 消费者私有
    uint8_t _pad1[61];

    // 三个元数据槽位 (3 × 32 = 96 bytes)
    ImageMeta slots[3]{};
};
static_assert(sizeof(ImageTripleBuffer) == 192, "ImageTripleBuffer size mismatch");

/**
 * @brief 位姿 Topic TripleBuffer 布局 - 256 bytes
 */
struct alignas(CACHE_LINE_SIZE) PoseTripleBuffer {
    alignas(CACHE_LINE_SIZE) std::atomic<uint8_t> state{1};
    uint8_t write_idx{0};
    uint8_t read_idx{2};
    uint8_t _pad1[61];

    PoseMeta slots[3]{};
};
static_assert(sizeof(PoseTripleBuffer) == 256, "PoseTripleBuffer size mismatch");

/**
 * @brief 云台命令 Topic TripleBuffer 布局 - 192 bytes (160 content + alignment padding)
 */
struct alignas(CACHE_LINE_SIZE) GimbalTripleBuffer {
    alignas(CACHE_LINE_SIZE) std::atomic<uint8_t> state{1};
    uint8_t write_idx{0};
    uint8_t read_idx{2};
    uint8_t _pad1[61];

    GimbalCmd slots[3]{};
};
static_assert(sizeof(GimbalTripleBuffer) == 192, "GimbalTripleBuffer size mismatch");

// ============ 共享内存头部 ============

/**
 * @brief 共享内存区域头部 - 64 bytes
 */
struct alignas(CACHE_LINE_SIZE) ShmHeader {
    uint32_t magic;        // 魔数: 0x54414C05
    uint32_t version;      // 协议版本
    uint64_t created_ns;   // 创建时间戳
    uint64_t heartbeat_ns; // 心跳时间戳 (生产者定期更新)
    uint32_t image_width;
    uint32_t image_height;
    uint8_t _pad[32];
};
static_assert(sizeof(ShmHeader) == 64, "ShmHeader must be 64 bytes");

/**
 * @brief 元数据共享内存区域布局
 *
 * 内存布局:
 * ┌─────────────────────────────────────┐
 * │ ShmHeader (64 bytes)                │ offset 0
 * ├─────────────────────────────────────┤
 * │ ImageTripleBuffer (192 bytes)       │ offset 64
 * ├─────────────────────────────────────┤
 * │ PoseTripleBuffer[5] (1280 bytes)    │ offset 256
 * │   [0] gimbal_pose                   │
 * │   [1] odom_pose                     │
 * │   [2] muzzle_pose                   │
 * │   [3] camera_pose                   │
 * │   [4] reserved                      │
 * ├─────────────────────────────────────┤
 * │ GimbalTripleBuffer (192 bytes)      │ offset 1536
 * ├─────────────────────────────────────┤
 * │ CameraInfo (128 bytes)              │ offset 1728
 * ├─────────────────────────────────────┤
 * │ _pad (192 bytes)                    │ offset 1856
 * └─────────────────────────────────────┘
 * Total: 2048 bytes
 */
struct ShmMetaRegion {
    ShmHeader header;
    ImageTripleBuffer image;
    PoseTripleBuffer poses[5]; // gimbal, odom, muzzle, camera, reserved
    GimbalTripleBuffer gimbal_cmd;
    CameraInfo camera_info;
    uint8_t _pad[192];         // 填充到 2048 字节
};
static_assert(sizeof(ShmMetaRegion) == 2048, "ShmMetaRegion must be 2048 bytes");

// Pose 索引
enum PoseIndex : uint8_t {
    POSE_GIMBAL = 0,
    POSE_ODOM   = 1,
    POSE_MUZZLE = 2,
    POSE_CAMERA = 3,
};

} // namespace ipc
