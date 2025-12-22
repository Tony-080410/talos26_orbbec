/**
 * Fast TF 改进版示例 - 简化版
 * 展示新的 expected 和友好的树定义方式
 */

#include <iostream>
#include <iomanip>
#include "fast_tf/core.hpp"

using namespace fast_tf;

// ============================================================================
// 使用改进的帧定义宏定义坐标系
// ============================================================================

DEFINE_ROOT_FRAME(world, 100);         // 定义根帧
DEFINE_CHILD_FRAME(base_link, world, 50);   // 定义子帧
DEFINE_CHILD_FRAME(camera, base_link, 30);  // 定义孙帧

// ============================================================================
// 示例1：使用新的帧定义语法
// ============================================================================

void example_new_frame_syntax() {
    std::cout << "\n=== 示例1：改进的帧定义语法 ===\n";

    // 打印帧信息
    std::cout << "世界坐标系: " << world_Frame::name << "\n";
    std::cout << "  缓冲大小: " << world_Frame::buffer_size << "\n";

    std::cout << "\nBase Link 坐标系: " << base_link_Frame::name << "\n";
    std::cout << "  缓冲大小: " << base_link_Frame::buffer_size << "\n";

    std::cout << "\n摄像头坐标系: " << camera_Frame::name << "\n";
    std::cout << "  缓冲大小: " << camera_Frame::buffer_size << "\n";
}

// ============================================================================
// 示例2：使用 expected 进行错误处理
// ============================================================================

expected<TransformMatrix4d, std::string> create_valid_transform(
    double roll, double pitch, double yaw,
    double x, double y, double z) {
    // 创建成功的结果
    return expected<TransformMatrix4d, std::string>(
        TransformMatrix4d::from_rpy(roll, pitch, yaw, x, y, z));
}

void example_expected_handling() {
    std::cout << "\n=== 示例2：使用 expected 进行错误处理 ===\n";

    // 有效的变换
    auto valid_result = create_valid_transform(0.1, 0.2, 0.3, 1.0, 2.0, 3.0);

    if (valid_result.has_value()) {
        std::cout << "✓ 创建了有效的变换\n";
        auto trans = valid_result.value().translation();
        std::cout << "  平移: (" << trans.x() << ", " << trans.y() << ", "
                  << trans.z() << ")\n";
    }

    // 演示错误情况
    std::cout << "\n演示错误处理：\n";
    expected<TransformMatrix4d, std::string> invalid_result(
        expected<TransformMatrix4d, std::string>::error_tag,
        "Invalid input values");

    if (invalid_result.has_error()) {
        std::cout << "✓ 捕获错误: " << invalid_result.error() << "\n";
    }

    // 使用 map 进行链式转换
    std::cout << "\n使用 map 进行链式处理：\n";
    auto norm_result = create_valid_transform(0, 0, 0, 1, 2, 3)
                           .map([](const auto& tf) {
                               auto trans = tf.translation();
                               return trans.norm();
                           });

    if (norm_result.has_value()) {
        std::cout << "✓ 平移向量的范数: " << norm_result.value() << "\n";
    }
}

// ============================================================================
// 示例3：缓冲区查询使用 expected
// ============================================================================

expected<TransformMatrix4d, std::string> safe_query_transform(
    CircularTransformBuffer<100>& buffer, uint64_t timestamp_ns) {

    auto entry = buffer.at_interpolated(timestamp_ns);

    if (entry.has_value()) {
        return expected<TransformMatrix4d, std::string>(entry->transform);
    } else {
        return expected<TransformMatrix4d, std::string>(
            expected<TransformMatrix4d, std::string>::error_tag,
            "Transform not found at timestamp: " + std::to_string(timestamp_ns));
    }
}

void example_buffer_expected() {
    std::cout << "\n=== 示例3：缓冲区查询使用 expected ===\n";

    CircularTransformBuffer<10> buffer;

    // 填充缓冲区
    for (int i = 0; i < 10; ++i) {
        TransformMatrix4d tf =
            TransformMatrix4d::from_rpy(0, 0, 0, i * 0.1, 0, 0);
        buffer.push(i * 1000, tf);
    }

    std::cout << "缓冲区已填充 10 个条目\n\n";

    // 查询存在的变换
    auto result1 = safe_query_transform(buffer, 5000);
    if (result1.has_value()) {
        std::cout << "✓ 找到时间戳 5000 的变换\n";
    }

    // 查询不存在的变换
    auto result2 = safe_query_transform(buffer, 99999);
    if (result2.has_error()) {
        std::cout << "✓ 如预期：" << result2.error() << "\n";
    }

    // 使用 map_error 进行错误转换
    auto mapped = safe_query_transform(buffer, 99999).map_error([](const auto& err) {
        return "错误: " + err;
    });

    if (mapped.has_error()) {
        std::cout << "✓ 映射后的错误: " << mapped.error() << "\n";
    }
}

// ============================================================================
// 示例4：预定义帧名称
// ============================================================================

void example_predefined_frames() {
    std::cout << "\n=== 示例4：预定义的常用帧名称 ===\n";

    std::cout << "预定义的标准帧：\n";
    std::cout << "  世界坐标: " << predef::WORLD << "\n";
    std::cout << "  里程计: " << predef::ODOM << "\n";
    std::cout << "  基座链接: " << predef::BASE_LINK << "\n";
    std::cout << "  摄像头: " << predef::CAMERA << "\n";
    std::cout << "  激光雷达: " << predef::LIDAR << "\n";
    std::cout << "  IMU: " << predef::IMU << "\n";
}

// ============================================================================
// 示例5：编译期帧ID
// ============================================================================

void example_frame_ids() {
    std::cout << "\n=== 示例5：编译期帧ID ===\n";

    // 创建强类型的帧ID
    struct WorldIdInfo {
        static constexpr std::string_view name = predef::WORLD;
    };
    struct BaseLinkIdInfo {
        static constexpr std::string_view name = predef::BASE_LINK;
    };
    struct CameraIdInfo {
        static constexpr std::string_view name = predef::CAMERA;
    };

    using WorldId = FrameId<WorldIdInfo>;
    using BaseLinkId = FrameId<BaseLinkIdInfo>;
    using CameraId = FrameId<CameraIdInfo>;

    std::cout << "帧ID （编译期计算的哈希值）：\n";
    std::cout << "  " << predef::WORLD << ": " << WorldId::id() << "\n";
    std::cout << "  " << predef::BASE_LINK << ": " << BaseLinkId::id() << "\n";
    std::cout << "  " << predef::CAMERA << ": " << CameraId::id() << "\n";

    // 这些ID在编译期就已经确定
    static_assert(WorldId::id() != BaseLinkId::id(),
                  "Frame IDs must be unique");
    std::cout << "\n✓ 编译期验证：帧ID唯一\n";
}

// ============================================================================
// 示例6：expected vs optional
// ============================================================================

void example_expected_vs_optional() {
    std::cout << "\n=== 示例6：expected vs optional ===\n";

    // expected 可以携带错误信息
    expected<int, std::string> result1(42);
    if (result1.has_value()) {
        std::cout << "expected<int, string>: " << result1.value() << "\n";
    }

    expected<int, std::string> result2(
        expected<int, std::string>::error_tag, "Something went wrong");
    if (result2.has_error()) {
        std::cout << "expected error: " << result2.error() << "\n";
    }

    std::cout << "\nexpected 的优势：\n";
    std::cout << "  • 携带错误信息（而非 optional 的空值）\n";
    std::cout << "  • 支持 map 和 map_error 的链式操作\n";
    std::cout << "  • 类型安全的错误处理\n";
    std::cout << "  • 与 Rust 的 Result<T, E> 兼容的设计\n";
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║      Fast TF 改进版API示例                           ║\n";
    std::cout << "║  使用 expected<T, E> 和友好的树定义语法               ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";

    example_new_frame_syntax();
    example_expected_handling();
    example_buffer_expected();
    example_predefined_frames();
    example_frame_ids();
    example_expected_vs_optional();

    std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║           示例执行完成！                               ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";

    return 0;
}
