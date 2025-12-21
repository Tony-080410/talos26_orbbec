/**
 * Fast TF 系统示例程序
 * 展示如何使用编译期验证的变换树系统
 */

#include <iostream>
#include <iomanip>
#include "fast_tf/core.hpp"

using namespace fast_tf;

// ============================================================================
// 示例1: 基础变换设置和查询
// ============================================================================

void example_basic_transforms() {
    std::cout << "\n=== 示例1: 基础变换设置和查询 ===\n";

    // 创建一个简单的树结构
    using SimpleTree = TransformTreeStructure<
        TreeNode<5, NullParent, BufferCapacity<100>>,  // "world"（5个字符）
        TreeNode<9, ParentRef<5>, BufferCapacity<50>>  // "base_link"（9个字符）
    >;

    SimpleTree tree;
    std::cout << "创建了树结构，包含 " << SimpleTree::num_nodes << " 个节点\n";

    // 创建变换矩阵
    TransformMatrix4d tf_world_to_base =
        TransformMatrix4d::from_rpy(0.0, 0.0, 0.0, 1.0, 2.0, 3.0);

    std::cout << "创建了变换 (RPY=(0,0,0), 平移=(1,2,3))\n";

    // 显示变换
    auto trans = tf_world_to_base.translation();
    std::cout << "平移向量: (" << trans.x() << ", " << trans.y() << ", "
              << trans.z() << ")\n";
}

// ============================================================================
// 示例2: 循环缓冲区管理
// ============================================================================

void example_circular_buffer() {
    std::cout << "\n=== 示例2: 循环缓冲区管理 ===\n";

    CircularTransformBuffer<10> buffer;
    std::cout << "创建了容量为10的循环缓冲区\n";

    // 添加变换
    for (int i = 0; i < 15; ++i) {
        double x = static_cast<double>(i);
        TransformMatrix4d tf =
            TransformMatrix4d::from_rpy(0, 0, 0, x, 0, 0);

        buffer.push(i * 1000, tf);
    }

    std::cout << "推送了15个变换到大小为10的缓冲区\n";
    std::cout << "当前缓冲区大小: " << buffer.size() << "\n";
    std::cout << "缓冲区已满: " << (buffer.is_full() ? "是" : "否") << "\n";

    // 查询时间范围
    auto range = buffer.time_range();
    if (range) {
        std::cout << "时间范围: [" << range->first << ", " << range->second
                  << "] (纳秒)\n";
    }

    // 精确查询
    auto entry = buffer.at(14000);
    if (entry) {
        std::cout << "查询到时间戳=14000的变换\n";
    } else {
        std::cout << "时间戳=14000的变换已被覆盖（循环缓冲区行为）\n";
    }
}

// ============================================================================
// 示例3: 变换插值
// ============================================================================

void example_interpolation() {
    std::cout << "\n=== 示例3: 变换插值 ===\n";

    CircularTransformBuffer<100> buffer;

    // 添加两个关键帧
    TransformMatrix4d tf_start =
        TransformMatrix4d::from_rpy(0, 0, 0, 0, 0, 0);
    TransformMatrix4d tf_end =
        TransformMatrix4d::from_rpy(0, 0, 0, 10, 0, 0);

    buffer.push(0, tf_start);
    buffer.push(10000, tf_end);

    std::cout << "添加了起点和终点变换\n";

    // 在多个时间点查询（插值）
    std::cout << std::fixed << std::setprecision(2);

    for (int i = 0; i <= 10; ++i) {
        uint64_t ts = i * 1000;
        auto entry = buffer.at_interpolated(ts);

        if (entry) {
            auto trans = entry->transform.translation();
            std::cout << "t=" << std::setw(5) << i << ": x=" << std::setw(6)
                      << trans.x() << ", y=" << std::setw(6) << trans.y()
                      << ", z=" << std::setw(6) << trans.z() << "\n";
        }
    }
}

// ============================================================================
// 示例4: 编译期字符串和哈希
// ============================================================================

void example_compile_time_strings() {
    std::cout << "\n=== 示例4: 编译期字符串和哈希 ===\n";

    constexpr CompileTimeString<5> world("world");
    constexpr CompileTimeString<9> base_link("base_link");
    constexpr CompileTimeString<6> camera("camera");

    std::cout << "编译期字符串:\n";
    std::cout << "  - world: \"" << world.view() << "\" (哈希: "
              << world.hash() << ")\n";
    std::cout << "  - base_link: \"" << base_link.view() << "\" (哈希: "
              << base_link.hash() << ")\n";
    std::cout << "  - camera: \"" << camera.view() << "\" (哈希: "
              << camera.hash() << ")\n";

    // 编译期比较（仅对相同大小可用）
    constexpr CompileTimeString<5> world2("world");
    static_assert(world == world2, "编译期字符串相等比较");
    std::cout << "\n编译期比较: world == world ✓\n";

    // 运行期比较
    bool is_less = std::string(world.view()) < std::string(base_link.view());
    if (is_less) {
        std::cout << "运行期比较: world < base_link ✓\n";
    } else {
        std::cout << "运行期比较: world >= base_link\n";
    }
}

// ============================================================================
// 示例5: 矩阵运算
// ============================================================================

void example_matrix_operations() {
    std::cout << "\n=== 示例5: 矩阵运算 ===\n";

    // 创建两个变换
    TransformMatrix4d tf1 =
        TransformMatrix4d::from_rpy(0, 0, 0, 1, 0, 0);  // 向X移动1单位
    TransformMatrix4d tf2 =
        TransformMatrix4d::from_rpy(0, 0, 0, 2, 0, 0);  // 向X移动2单位

    std::cout << "变换1: 向X轴平移1单位\n";
    std::cout << "变换2: 向X轴平移2单位\n";

    // 矩阵乘法（变换合成）
    TransformMatrix4d tf_combined = tf1 * tf2;
    auto trans = tf_combined.translation();

    std::cout << "\n合成变换 (tf1 * tf2): 平移 = ("
              << trans.x() << ", " << trans.y() << ", " << trans.z()
              << ")\n";
    std::cout << "（期望：(3, 0, 0)）\n";

    // 逆变换
    TransformMatrix4d tf1_inv = tf1.inverse();
    auto trans_inv = tf1_inv.translation();

    std::cout << "\n变换1的逆: 平移 = ("
              << trans_inv.x() << ", " << trans_inv.y() << ", "
              << trans_inv.z() << ")\n";
    std::cout << "（期望：(-1, 0, 0)）\n";

    // 验证：tf1 * tf1^{-1} = Identity
    TransformMatrix4d result = tf1 * tf1_inv;
    auto result_trans = result.translation();

    std::cout << "\n验证：tf1 * tf1^{-1} = Identity\n";
    std::cout << "平移: (" << result_trans.x() << ", " << result_trans.y()
              << ", " << result_trans.z() << ")\n";
    std::cout << "（期望：(0, 0, 0)）\n";
}

// ============================================================================
// 示例6: 四元数变换
// ============================================================================

void example_quaternion_transforms() {
    std::cout << "\n=== 示例6: 四元数变换 ===\n";

    // 创建四元数变换（单位四元数 + 平移）
    TransformMatrix4d tf =
        TransformMatrix4d::from_quaternion(1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0);

    std::cout << "从四元数创建变换:\n";
    std::cout << "  四元数: (w=1, x=0, y=0, z=0) [单位四元数]\n";
    std::cout << "  平移: (1, 2, 3)\n";

    // 提取四元数
    auto quat = tf.quaternion();
    std::cout << "\n提取的四元数: (w=" << quat.w() << ", x=" << quat.x()
              << ", y=" << quat.y() << ", z=" << quat.z() << ")\n";

    // 获取平移
    auto trans = tf.translation();
    std::cout << "提取的平移: (" << trans.x() << ", " << trans.y() << ", "
              << trans.z() << ")\n";
}

// ============================================================================
// 示例7: 框架管理
// ============================================================================

void example_frame_registry() {
    std::cout << "\n=== 示例7: 框架管理 ===\n";

    FrameRegistry::clear();

    // 注册坐标系
    uint64_t world_id = hash_string_view("world");
    uint64_t base_id = hash_string_view("base_link");

    FrameRegistry::register_frame("world", world_id);
    FrameRegistry::register_frame("base_link", base_id);

    std::cout << "注册了坐标系: world 和 base_link\n";

    // 注册别名
    FrameRegistry::register_alias("robot_base", "base_link");
    FrameRegistry::register_alias("odom", "world");

    std::cout << "注册了别名:\n";
    std::cout << "  robot_base -> base_link\n";
    std::cout << "  odom -> world\n";

    // 解析别名
    std::string_view resolved1 =
        FrameRegistry::resolve_alias("robot_base");
    std::string_view resolved2 = FrameRegistry::resolve_alias("odom");

    std::cout << "\n别名解析:\n";
    std::cout << "  robot_base 解析为: " << resolved1 << "\n";
    std::cout << "  odom 解析为: " << resolved2 << "\n";
}

// ============================================================================
// 示例8: 性能基准
// ============================================================================

void example_performance_benchmark() {
    std::cout << "\n=== 示例8: 性能基准 ===\n";

    // 测试1: 缓冲区写入性能
    {
        std::cout << "\n测试1: 缓冲区写入性能\n";

        CircularTransformBuffer<1000> buffer;
        TransformMatrix4d tf =
            TransformMatrix4d::from_rpy(0, 0, 0, 1, 2, 3);

        auto start = std::chrono::steady_clock::now();

        for (int i = 0; i < 10000; ++i) {
            buffer.push(i * 1000, tf);
        }

        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                           end - start)
                           .count();

        std::cout << "推送10000个变换: " << duration << " 微秒\n";
        std::cout << "平均每个操作: "
                  << (static_cast<double>(duration) / 10000.0) << " 微秒\n";
    }

    // 测试2: 查询性能
    {
        std::cout << "\n测试2: 缓冲区查询性能\n";

        CircularTransformBuffer<100> buffer;
        TransformMatrix4d tf =
            TransformMatrix4d::from_rpy(0, 0, 0, 1, 2, 3);

        for (int i = 0; i < 100; ++i) {
            buffer.push(i * 1000, tf);
        }

        auto start = std::chrono::steady_clock::now();

        int result_count = 0;
        for (int i = 0; i < 10000; ++i) {
            auto entry = buffer.at_interpolated(50000 + (i % 50000));
            if (entry) {
                result_count++;
            }
        }

        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                           end - start)
                           .count();

        std::cout << "执行10000个查询: " << duration << " 微秒\n";
        std::cout << "平均每个查询: "
                  << (static_cast<double>(duration) / 10000.0) << " 微秒\n";
        std::cout << "成功查询: " << result_count << "\n";
    }

    // 测试3: 矩阵乘法性能
    {
        std::cout << "\n测试3: 矩阵乘法性能\n";

        TransformMatrix4d tf1 =
            TransformMatrix4d::from_rpy(0, 0, 0, 1, 2, 3);
        TransformMatrix4d tf2 =
            TransformMatrix4d::from_rpy(0.1, 0.2, 0.3, 2, 3, 4);

        auto start = std::chrono::steady_clock::now();

        for (int i = 0; i < 100000; ++i) {
            auto result = tf1 * tf2;
            (void)result;  // 防止被优化掉
        }

        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                           end - start)
                           .count();

        std::cout << "执行100000个矩阵乘法: " << duration << " 微秒\n";
        std::cout << "平均每个乘法: "
                  << (static_cast<double>(duration) / 100000.0) << " 微秒\n";
    }
}

// ============================================================================
// 主程序
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "    Fast TF - 编译期变换树系统示例    \n";
    std::cout << "========================================\n";

    example_basic_transforms();
    example_circular_buffer();
    example_interpolation();
    example_compile_time_strings();
    example_matrix_operations();
    example_quaternion_transforms();
    example_frame_registry();
    example_performance_benchmark();

    std::cout << "\n========================================\n";
    std::cout << "          示例程序执行完成！          \n";
    std::cout << "========================================\n";

    return 0;
}
