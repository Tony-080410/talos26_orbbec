#pragma once

#include <Eigen/Core>
#include <variant>
#include <vector>

#include "detector_state.hpp"
#include "scheduler/types.hpp"
#include "tracker_soa.hpp"
#include "types.hpp"

namespace auto_aim {

// ============ 预处理类型 ============

/**
 * @brief 预处理类型枚举
 *
 * 检测器通过静态方法声明自己需要的预处理类型，
 * 预处理器根据此类型决定执行哪些预处理步骤。
 */
enum class PreprocessType : uint8_t {
    None,          // NN 检测器不需要预处理
    GrayOnly,      // 只需要灰度图
    GrayAndBinary, // 灰度 + 二值化（传统检测器）
};

// ============ 预处理后的帧 ============

/**
 * @brief 预处理后的帧数据
 *
 * 替代 Captured 作为 detect() 的输入，
 * 包含原始 RGB 图像以及可选的灰度/二值化图像。
 */
struct PreprocessedFrame {
    scheduler::ImagePtr rgb;    // 原始 RGB
    scheduler::ImagePtr gray;   // 灰度图（如果需要）
    scheduler::ImagePtr binary; // 二值化（如果需要）
    uint64_t timestamp_ns{0};
    uint64_t frame_seq{0};
};

// ============ 检测结果 Variant ============

/**
 * @brief NN 检测器额外信息
 *
 * 当 NN 检测器需要输出额外信息时使用。
 * 包含基础的 ArmorSlots 以保持与传统检测器的兼容性。
 */
struct NNDetectionExtra {
    talos::ArmorSlots slots;          // 复用现有结构
    std::array<float, 10> raw_scores; // NN 原始分数（可选）
};

/**
 * @brief 检测结果 Variant
 *
 * - std::monostate: 检测失败或无结果
 * - ArmorSlots: 传统检测器 / NN 通用输出
 * - NNDetectionExtra: NN 需要额外信息时使用
 */
using DetectionResult = std::variant<std::monostate, talos::ArmorSlots, NNDetectionExtra>;

// ============ 解算结果 Variant ============

/**
 * @brief BA 优化位姿结果（未来扩展）
 */
struct BAPoses3D {
    Poses3D poses;                   // 复用现有结构
    std::array<float, 10> residuals; // BA 残差
    bool converged{false};           // 是否收敛
};

/**
 * @brief 解算结果 Variant
 *
 * - std::monostate: 解算失败或无结果
 * - Poses3D: PnP 解算
 * - BAPoses3D: BA 优化（未来扩展）
 */
using SolveResult = std::variant<std::monostate, Poses3D, BAPoses3D>;

// ============ 追踪输出 ============

/**
 * @brief 追踪器输出
 *
 * 只包含共享输出，内部状态由各 Tracker 自己持有。
 * 使用 TrackerSoA 保证 64 字节缓存行对齐。
 */
using TrackerOutput = talos::TrackerSoA;

// ============ 调试数据 ============

/**
 * @brief 调试数据结构
 *
 * 各流水线阶段可以写入调试信息，
 * 由独立的 debug 线程通过 triple buffer 消费。
 */
struct DebugData {
    // 帧信息
    uint64_t frame_seq{0};
    uint64_t timestamp_ns{0};

    // 预处理调试
    scheduler::ImagePtr binary_img;

    // 检测调试
    scheduler::ImagePtr detection_viz;          // 检测可视化
    std::vector<talos::Lightbar> all_lightbars; // 所有灯条（含被过滤的）

    // 解算调试
    std::vector<cv::Point2f> reprojected_pts; // 重投影点

    // 追踪调试
    talos::TrackerState tracker_state{talos::IDLE};
    Eigen::Vector3d predicted_pos{0, 0, 0};

    // 输出调试
    float gimbal_yaw{0};
    float gimbal_pitch{0};

    void clear() noexcept {
        frame_seq    = 0;
        timestamp_ns = 0;
        binary_img.reset();
        detection_viz.reset();
        all_lightbars.clear();
        reprojected_pts.clear();
        tracker_state = talos::IDLE;
        predicted_pos.setZero();
        gimbal_yaw   = 0;
        gimbal_pitch = 0;
    }
};

// ============ Variant 辅助函数 ============

/**
 * @brief 从 DetectionResult 提取 ArmorSlots 指针
 * @return 指向 ArmorSlots 的指针，如果是 monostate 则返回 nullptr
 */
[[nodiscard]] inline const talos::ArmorSlots* get_armor_slots(const DetectionResult& det) noexcept {
    if (std::holds_alternative<talos::ArmorSlots>(det)) {
        return &std::get<talos::ArmorSlots>(det);
    }
    if (std::holds_alternative<NNDetectionExtra>(det)) {
        return &std::get<NNDetectionExtra>(det).slots;
    }
    return nullptr;
}

/**
 * @brief 从 SolveResult 提取 Poses3D 指针
 * @return 指向 Poses3D 的指针，如果是 monostate 则返回 nullptr
 */
[[nodiscard]] inline const Poses3D* get_poses3d(const SolveResult& solve) noexcept {
    if (std::holds_alternative<Poses3D>(solve)) {
        return &std::get<Poses3D>(solve);
    }
    if (std::holds_alternative<BAPoses3D>(solve)) {
        return &std::get<BAPoses3D>(solve).poses;
    }
    return nullptr;
}

/**
 * @brief 检查 DetectionResult 是否有效（非 monostate）
 */
[[nodiscard]] inline bool has_detection(const DetectionResult& det) noexcept {
    return !std::holds_alternative<std::monostate>(det);
}

/**
 * @brief 检查 SolveResult 是否有效（非 monostate）
 */
[[nodiscard]] inline bool has_solve(const SolveResult& solve) noexcept {
    return !std::holds_alternative<std::monostate>(solve);
}

} // namespace auto_aim
