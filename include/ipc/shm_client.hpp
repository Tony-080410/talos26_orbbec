#pragma once

#include <chrono>
#include <expected>
#include <opencv2/core.hpp>
#include <optional>

#include "shm_layout.hpp"
#include "shm_region.hpp"
#include "shm_triple_buffer.hpp"

namespace ipc {

    /**
     * @brief 共享内存 IPC 客户端
     *
     * 提供高级 API 用于:
     * - 订阅图像流 (Rust → C++)
     * - 订阅位姿数据 (Rust → C++)
     * - 发布云台控制命令 (C++ → Rust)
     *
     * 使用示例:
     * ```cpp
     * auto client = ShmClient::connect();
     * if (!client) {
     *     std::cerr << "连接失败: " << to_string(client.error()) << std::endl;
     *     return;
     * }
     *
     * // 订阅图像
     * if (auto frame = client->recv_image()) {
     *     cv::Mat img = frame->image;
     *     // 处理图像...
     * }
     *
     * // 发布云台命令
     * client->send_gimbal_cmd(15.0f, -8.0f, 3.5f, true);
     * ```
     */
    class ShmClient {
    public:
        /**
         * @brief 接收到的图像帧
         */
        struct ImageFrame {
            cv::Mat image; // 零拷贝 cv::Mat (直接指向共享内存)
            uint64_t seq; // 帧序号
            uint64_t timestamp_ns; // 时间戳
        };

        /**
         * @brief 接收到的位姿数据
         */
        struct Pose {
            double x, y, z; // 位置 (米)
            double qw, qx, qy, qz; // 四元数
            uint64_t timestamp_ns;
        };

        ~ShmClient() = default;

        // Move-only
        ShmClient(ShmClient &&) = default;
        ShmClient &operator=(ShmClient &&) = default;
        ShmClient(const ShmClient &) = delete;
        ShmClient &operator=(const ShmClient &) = delete;

        /**
         * @brief 连接到共享内存 (消费者模式)
         *
         * 消费者连接到已由 Rust 模拟器创建的共享内存区域
         */
        [[nodiscard]] static std::expected<ShmClient, ShmError> connect() {
            // 打开元数据区域
            auto meta_result = ShmRegion::open(SHM_NAME_META, sizeof(ShmMetaRegion));
            if (!meta_result) {
                return std::unexpected(meta_result.error());
            }

            // 打开图像池
            auto pool_result = ShmRegion::open(SHM_NAME_IMAGE_POOL, IMAGE_POOL_SIZE);
            if (!pool_result) {
                return std::unexpected(pool_result.error());
            }

            // 验证魔数和版本
            const auto *meta = meta_result->as<ShmMetaRegion>();
            if (meta->header.magic != SHM_MAGIC) {
                return std::unexpected(ShmError::InvalidSize); // 实际应该是 InvalidMagic
            }
            if (meta->header.version != SHM_VERSION) {
                return std::unexpected(ShmError::InvalidSize); // 实际应该是 VersionMismatch
            }

            return ShmClient(std::move(*meta_result), std::move(*pool_result));
        }

        /**
         * @brief 创建共享内存 (生产者模式，仅用于测试)
         */
        [[nodiscard]] static std::expected<ShmClient, ShmError> create() {
            // 创建元数据区域
            auto meta_result = ShmRegion::create(SHM_NAME_META, sizeof(ShmMetaRegion));
            if (!meta_result) {
                return std::unexpected(meta_result.error());
            }

            // 创建图像池
            auto pool_result = ShmRegion::create(SHM_NAME_IMAGE_POOL, IMAGE_POOL_SIZE);
            if (!pool_result) {
                return std::unexpected(pool_result.error());
            }

            // 初始化头部
            auto *meta = meta_result->as<ShmMetaRegion>();
            meta->header.magic = SHM_MAGIC;
            meta->header.version = SHM_VERSION;
            meta->header.created_ns =
                    static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
            meta->header.image_width = IMAGE_WIDTH;
            meta->header.image_height = IMAGE_HEIGHT;

            return ShmClient(std::move(*meta_result), std::move(*pool_result));
        }

        // ============ 图像订阅 ============

        /**
         * @brief 尝试接收最新图像帧
         * @return 如果有新图像，返回 ImageFrame；否则返回 nullopt
         *
         * 返回的 cv::Mat 直接指向共享内存，零拷贝。
         * 注意：cv::Mat 在下次 recv_image() 调用后可能失效！
         */
        [[nodiscard]] std::optional<ImageFrame> recv_image() const {
            ImageOps ops(&meta_->image);
            const auto slot = ops.borrow();
            if (!slot) {
                return std::nullopt;
            }

            const auto &img_meta = **slot;

            // 计算图像数据在 pool 中的偏移
            uint8_t *img_data = image_pool_ + img_meta.buffer_id * IMAGE_SIZE;

            // 确定 OpenCV 类型
            int cv_type = CV_8UC3; // 默认 RGB8
            if (img_meta.format == 1)
                cv_type = CV_8UC3; // BGR8
            else if (img_meta.format == 2)
                cv_type = CV_8UC1; // GRAY8

            // 创建零拷贝 cv::Mat
            const cv::Mat image(static_cast<int>(img_meta.height), static_cast<int>(img_meta.width), cv_type, img_data);

            return ImageFrame{
                    .image = image,
                    .seq = img_meta.seq,
                    .timestamp_ns = img_meta.timestamp_ns,
            };
        }

        /**
         * @brief 检查是否有新图像
         */
        [[nodiscard]] bool has_new_image() const {
            const ImageOps ops(&meta_->image);
            return ops.has_new_data();
        }

        // ============ 位姿订阅 ============

        /**
         * @brief 尝试接收指定类型的位姿
         * @param index 位姿类型 (POSE_GIMBAL, POSE_ODOM, POSE_MUZZLE, POSE_CAMERA)
         */
        [[nodiscard]] std::optional<Pose> recv_pose(const PoseIndex index) const {
            if (index > 4)
                return std::nullopt;

            PoseOps ops(&meta_->poses[index]);
            const auto slot = ops.borrow();
            if (!slot) {
                return std::nullopt;
            }

            const auto &pose = **slot;
            return Pose{
                    .x = pose.position[0],
                    .y = pose.position[1],
                    .z = pose.position[2],
                    .qw = pose.quaternion[0],
                    .qx = pose.quaternion[1],
                    .qy = pose.quaternion[2],
                    .qz = pose.quaternion[3],
                    .timestamp_ns = pose.timestamp_ns,
            };
        }

        // ============ 云台命令发布 ============

        /**
         * @brief 发布云台控制命令
         * @param yaw_deg 目标偏航角 (度)
         * @param pitch_deg 目标俯仰角 (度)
         * @param distance_m 目标距离 (米)，-1 表示无效
         * @param fire_advice 是否建议开火
         */
        void send_gimbal_cmd(const float yaw_deg, const float pitch_deg, const float distance_m,
                             const bool fire_advice) const {
            GimbalOps ops(&meta_->gimbal_cmd);
            auto &cmd = ops.borrow_mut();

            cmd.timestamp_ns = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
            cmd.yaw_deg = yaw_deg;
            cmd.pitch_deg = pitch_deg;
            cmd.distance_m = distance_m;
            cmd.fire_advice = fire_advice ? 1 : 0;

            ops.publish();
        }

        // ============ 相机信息 ============

        /**
         * @brief 获取相机内参 (直接访问，不使用 TripleBuffer)
         */
        [[nodiscard]] const CameraInfo &camera_info() const { return meta_->camera_info; }

        // ============ 诊断 ============

        /**
         * @brief 获取共享内存头部信息
         */
        [[nodiscard]] const ShmHeader &header() const { return meta_->header; }

        /**
         * @brief 更新心跳 (生产者调用)
         */
        void update_heartbeat() const {
            meta_->header.heartbeat_ns =
                    static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
        }

        /**
         * @brief 检查生产者是否存活 (消费者调用)
         * @param timeout_ns 超时时间 (纳秒)
         */
        [[nodiscard]] bool is_producer_alive(const uint64_t timeout_ns = 1'000'000'000) const {
            const auto now = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
            return now - meta_->header.heartbeat_ns < timeout_ns;
        }

        // ============ 图像发布 (生产者模式，用于测试) ============

        /**
         * @brief 发布图像 (生产者模式)
         * @param image 图像数据
         * @param seq 帧序号
         * @param timestamp_ns 时间戳
         */
        void publish_image(const cv::Mat &image, const uint64_t seq, const uint64_t timestamp_ns) const {
            ImageOps ops(&meta_->image);
            auto &meta = ops.borrow_mut();

            // 选择下一个 buffer
            uint8_t buffer_id = meta.buffer_id;
            buffer_id = (buffer_id + 1) % 3;

            // 复制图像数据到 pool
            uint8_t *dst = image_pool_ + buffer_id * IMAGE_SIZE;
            if (image.isContinuous()) {
                std::memcpy(dst, image.data, IMAGE_SIZE);
            } else {
                for (int row = 0; row < image.rows; ++row) {
                    std::memcpy(dst + row * image.cols * image.channels(), image.ptr(row),
                                image.cols * image.channels());
                }
            }

            // 更新元数据
            meta.seq = seq;
            meta.timestamp_ns = timestamp_ns;
            meta.width = static_cast<uint32_t>(image.cols);
            meta.height = static_cast<uint32_t>(image.rows);
            meta.buffer_id = buffer_id;
            meta.format = image.type() == CV_8UC1 ? 2 : 1; // BGR8 或 GRAY8

            ops.publish();
        }

    private:
        ShmClient(ShmRegion meta_region, ShmRegion pool_region) :
            meta_region_(std::move(meta_region)), pool_region_(std::move(pool_region)),
            meta_(meta_region_.as<ShmMetaRegion>()), image_pool_(static_cast<uint8_t *>(pool_region_.data())) {}

        ShmRegion meta_region_;
        ShmRegion pool_region_;
        ShmMetaRegion *meta_;
        uint8_t *image_pool_;
    };

} // namespace ipc
