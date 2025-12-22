#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include "fast_tf/core.hpp"

using namespace fast_tf;

// ============================================================================
// 编译期字符串测试
// ============================================================================

TEST(CompileTimeStringTest, ConstructionAndComparison) {
    constexpr CompileTimeString<5> world("world");
    constexpr CompileTimeString<5> world2("world");
    constexpr CompileTimeString<6> camera("camera");

    // 编译期比较
    static_assert(world == world2);
    static_assert(world != camera);
    static_assert(world < camera);

    // 运行期检查
    EXPECT_EQ(world.view(), std::string_view("world"));
    EXPECT_EQ(world2.view(), std::string_view("world"));
    EXPECT_NE(world.view(), camera.view());
}

TEST(CompileTimeStringTest, HashFunction) {
    constexpr CompileTimeString<5> world("world");
    constexpr CompileTimeString<6> camera("camera");

    // 不同的字符串应该有不同的哈希值（大概率）
    EXPECT_NE(world.hash(), camera.hash());

    // 相同的字符串应该有相同的哈希值
    CompileTimeString<5> world2("world");
    EXPECT_EQ(world.hash(), world2.hash());
}

// ============================================================================
// 变换矩阵测试
// ============================================================================

TEST(TransformMatrixTest, IdentityMatrix) {
    TransformMatrix4d tf;
    auto mat = tf.matrix();

    // 验证单位矩阵
    EXPECT_EQ(mat(0, 0), 1.0);
    EXPECT_EQ(mat(1, 1), 1.0);
    EXPECT_EQ(mat(2, 2), 1.0);
    EXPECT_EQ(mat(3, 3), 1.0);
    EXPECT_EQ(mat(0, 3), 0.0);
    EXPECT_EQ(mat(1, 3), 0.0);
    EXPECT_EQ(mat(2, 3), 0.0);
}

TEST(TransformMatrixTest, FromRPY) {
    // RPY = (0, 0, 0)，平移 = (1, 2, 3)
    TransformMatrix4d tf = TransformMatrix4d::from_rpy(0, 0, 0, 1, 2, 3);

    auto trans = tf.translation();
    EXPECT_NEAR(trans.x(), 1.0, 1e-10);
    EXPECT_NEAR(trans.y(), 2.0, 1e-10);
    EXPECT_NEAR(trans.z(), 3.0, 1e-10);

    // 旋转应该是单位矩阵
    auto rot = tf.rotation();
    EXPECT_NEAR(rot(0, 0), 1.0, 1e-10);
    EXPECT_NEAR(rot(1, 1), 1.0, 1e-10);
    EXPECT_NEAR(rot(2, 2), 1.0, 1e-10);
}

TEST(TransformMatrixTest, FromQuaternion) {
    // 单位四元数 + 平移
    TransformMatrix4d tf = TransformMatrix4d::from_quaternion(
        1.0, 0.0, 0.0, 0.0,  // 单位四元数 (w, x, y, z)
        1.0, 2.0, 3.0         // 平移
    );

    auto trans = tf.translation();
    EXPECT_NEAR(trans.x(), 1.0, 1e-10);
    EXPECT_NEAR(trans.y(), 2.0, 1e-10);
    EXPECT_NEAR(trans.z(), 3.0, 1e-10);
}

TEST(TransformMatrixTest, MatrixMultiplication) {
    // 创建两个变换
    TransformMatrix4d tf1 = TransformMatrix4d::from_rpy(0, 0, 0, 1, 0, 0);
    TransformMatrix4d tf2 = TransformMatrix4d::from_rpy(0, 0, 0, 2, 0, 0);

    // 相乘
    TransformMatrix4d result = tf1 * tf2;
    auto trans = result.translation();

    // 应该得到 (3, 0, 0)
    EXPECT_NEAR(trans.x(), 3.0, 1e-10);
    EXPECT_NEAR(trans.y(), 0.0, 1e-10);
    EXPECT_NEAR(trans.z(), 0.0, 1e-10);
}

TEST(TransformMatrixTest, Inverse) {
    TransformMatrix4d tf = TransformMatrix4d::from_rpy(0, 0, 0, 1, 2, 3);
    TransformMatrix4d tf_inv = tf.inverse();

    // tf * tf_inv 应该等于单位矩阵
    TransformMatrix4d result = tf * tf_inv;
    auto mat = result.matrix();

    EXPECT_NEAR(mat(0, 0), 1.0, 1e-10);
    EXPECT_NEAR(mat(1, 1), 1.0, 1e-10);
    EXPECT_NEAR(mat(2, 2), 1.0, 1e-10);
    EXPECT_NEAR(mat(0, 3), 0.0, 1e-10);
    EXPECT_NEAR(mat(1, 3), 0.0, 1e-10);
    EXPECT_NEAR(mat(2, 3), 0.0, 1e-10);
}

TEST(TransformMatrixTest, LinearInterpolation) {
    TransformMatrix4d tf1 = TransformMatrix4d::from_rpy(0, 0, 0, 0, 0, 0);
    TransformMatrix4d tf2 = TransformMatrix4d::from_rpy(0, 0, 0, 2, 0, 0);

    // 在中点插值
    TransformMatrix4d tf_mid = TransformMatrix4d::lerp(tf1, tf2, 0.5);
    auto trans = tf_mid.translation();

    EXPECT_NEAR(trans.x(), 1.0, 1e-10);
    EXPECT_NEAR(trans.y(), 0.0, 1e-10);
    EXPECT_NEAR(trans.z(), 0.0, 1e-10);
}

// ============================================================================
// 循环缓冲区测试
// ============================================================================

TEST(CircularBufferTest, PushAndRetrieve) {
    CircularTransformBuffer<10> buffer;

    TransformMatrix4d tf = TransformMatrix4d::from_rpy(0, 0, 0, 1, 2, 3);
    buffer.push(1000, tf);

    EXPECT_EQ(buffer.size(), 1);
    EXPECT_FALSE(buffer.is_empty());

    auto entry = buffer.at(1000);
    ASSERT_TRUE(entry.has_value());
    EXPECT_EQ(entry->timestamp_ns, 1000);
}

TEST(CircularBufferTest, CircularBehavior) {
    CircularTransformBuffer<3> buffer;

    TransformMatrix4d tf1 = TransformMatrix4d::from_rpy(0, 0, 0, 1, 0, 0);
    TransformMatrix4d tf2 = TransformMatrix4d::from_rpy(0, 0, 0, 2, 0, 0);
    TransformMatrix4d tf3 = TransformMatrix4d::from_rpy(0, 0, 0, 3, 0, 0);
    TransformMatrix4d tf4 = TransformMatrix4d::from_rpy(0, 0, 0, 4, 0, 0);

    buffer.push(1000, tf1);
    buffer.push(2000, tf2);
    buffer.push(3000, tf3);

    EXPECT_EQ(buffer.size(), 3);
    EXPECT_TRUE(buffer.is_full());

    // 添加第四个时应该覆盖第一个
    buffer.push(4000, tf4);
    EXPECT_EQ(buffer.size(), 3);

    // 第一个应该被覆盖
    EXPECT_FALSE(buffer.at(1000).has_value());
    EXPECT_TRUE(buffer.at(2000).has_value());
    EXPECT_TRUE(buffer.at(3000).has_value());
    EXPECT_TRUE(buffer.at(4000).has_value());
}

TEST(CircularBufferTest, TimeRange) {
    CircularTransformBuffer<10> buffer;

    TransformMatrix4d tf = TransformMatrix4d::from_rpy(0, 0, 0, 1, 0, 0);

    buffer.push(1000, tf);
    buffer.push(2000, tf);
    buffer.push(3000, tf);

    auto range = buffer.time_range();
    ASSERT_TRUE(range.has_value());
    EXPECT_EQ(range->first, 1000);
    EXPECT_EQ(range->second, 3000);
}

TEST(CircularBufferTest, Interpolation) {
    CircularTransformBuffer<10> buffer;

    TransformMatrix4d tf1 = TransformMatrix4d::from_rpy(0, 0, 0, 0, 0, 0);
    TransformMatrix4d tf2 = TransformMatrix4d::from_rpy(0, 0, 0, 2, 0, 0);

    buffer.push(1000, tf1);
    buffer.push(3000, tf2);

    // 在中间时间戳查询（应该插值）
    auto entry = buffer.at_interpolated(2000);
    ASSERT_TRUE(entry.has_value());

    auto trans = entry->transform.translation();
    EXPECT_NEAR(trans.x(), 1.0, 1e-2);  // 应该接近 1.0
}

// ============================================================================
// 变换输入概念测试
// ============================================================================

TEST(TransformInputTest, RPYFormat) {
    RPYTransformData rpy(0.1, 0.2, 0.3, 1.0, 2.0, 3.0);

    EXPECT_TRUE(rpy.is_valid());
    EXPECT_NEAR(rpy.roll, 0.1, 1e-10);
    EXPECT_NEAR(rpy.pitch, 0.2, 1e-10);
    EXPECT_NEAR(rpy.yaw, 0.3, 1e-10);
}

TEST(TransformInputTest, QuaternionFormat) {
    QuaternionTransformData quat(1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0);

    quat.normalize();  // 归一化
    EXPECT_TRUE(quat.is_valid());
}

// ============================================================================
// 框架注册表测试
// ============================================================================

TEST(FrameRegistryTest, RegisterAndRetrieve) {
    FrameRegistry::clear();

    uint64_t world_id = hash_string_view("world");
    FrameRegistry::register_frame("world", world_id);

    std::string_view name = FrameRegistry::get_frame_name(world_id);
    EXPECT_EQ(name, "world");
}

TEST(FrameRegistryTest, AliasResolution) {
    FrameRegistry::clear();

    FrameRegistry::register_alias("robot_base", "base_link");

    std::string_view resolved = FrameRegistry::resolve_alias("robot_base");
    EXPECT_EQ(resolved, "base_link");

    // 非别名应该返回原名称
    std::string_view resolved2 = FrameRegistry::resolve_alias("base_link");
    EXPECT_EQ(resolved2, "base_link");
}

// ============================================================================
// 高级变换树API测试（简化版本）
// ============================================================================

TEST(TransformTreeTest, CreationAndClearing) {
    // 创建一个简单的树结构
    using SimpleTree = TransformTreeStructure<
        TreeNode<5, NullParent, BufferCapacity<100>>,  // "world"
        TreeNode<9, ParentRef<5>, BufferCapacity<50>>   // "base_link"
    >;

    SimpleTree tree;
    EXPECT_EQ(SimpleTree::num_nodes, 2);

    tree.clear();
}

// ============================================================================
// 性能测试
// ============================================================================

TEST(PerformanceTest, BufferPushThroughput) {
    CircularTransformBuffer<1000> buffer;
    TransformMatrix4d tf = TransformMatrix4d::from_rpy(0, 0, 0, 1, 2, 3);

    // 推送1000个条目
    for (int i = 0; i < 1000; ++i) {
        buffer.push(i * 1000, tf);
    }

    EXPECT_TRUE(buffer.is_full());
}

TEST(PerformanceTest, InterpolationSpeed) {
    CircularTransformBuffer<100> buffer;

    TransformMatrix4d tf1 = TransformMatrix4d::from_rpy(0, 0, 0, 0, 0, 0);
    TransformMatrix4d tf2 = TransformMatrix4d::from_rpy(0, 0, 0, 2, 0, 0);

    for (int i = 0; i < 50; ++i) {
        buffer.push(i * 2000, i % 2 == 0 ? tf1 : tf2);
    }

    // 查询1000次
    for (int i = 0; i < 1000; ++i) {
        auto entry = buffer.at_interpolated(50000 + i * 100);
        // 不检查结果，只测速度
        (void)entry;
    }
}

// ============================================================================
// 编译期验证测试
// ============================================================================

// 这些测试验证编译期约束
TEST(CompileTimeTest, ConceptConstraints) {
    // TransformInput 概念
    static_assert(TransformInput<RPYTransformData>);
    static_assert(TransformInput<QuaternionTransformData>);

    // FloatingPoint 概念
    static_assert(FloatingPoint<double>);
    static_assert(FloatingPoint<float>);

    // Timestamp 概念（uint64_t应该可以作为时间戳）
    static_assert(Timestamp<uint64_t>);
    static_assert(Timestamp<int64_t>);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
