# Fast TF - 高级编译期变换树系统

一个高性能的 ROS2 tf2 替代方案，具有完整的编译期验证和零运行时开销的树结构验证。

## 概述

Fast TF 是一个使用现代 C++23 实现的变换树系统，提供：

- **编译期验证**：所有树结构和坐标系名称在编译期验证，编译错误即可捕获问题
- **完全类型安全**：通过 C++23 概念确保正确的变换输入格式
- **零运行时开销**：树结构验证完全在编译期完成
- **高性能**：使用 SIMD 优化的 Eigen 矩阵运算、TBB 并发容器和自旋读写互斥体
- **灵活的变换表示**：支持 RPY（欧拉角）和四元数两种格式
- **固定大小缓冲区**：使用编译期确定的循环缓冲区避免动态分配
- **线程安全**：所有操作都使用 TBB 的自旋读写互斥体保护

## 核心特性

### 1. 编译期树结构定义

```cpp
using MyTree = TransformTreeStructure<
    TreeNode<5, NullParent, BufferCapacity<100>>,          // "world"
    TreeNode<9, ParentRef<5>, BufferCapacity<50>>,         // "base_link"
    TreeNode<6, ParentRef<9>, BufferCapacity<30>>          // "camera"
>;

MyTree tree;  // 编译期验证的树
```

### 2. 变换输入支持

#### RPY 格式（横滚-俯仰-偏航角）

```cpp
tree.set_transform("camera", "base_link", timestamp_ns,
    0.1,    // roll
    0.2,    // pitch
    0.3,    // yaw
    1.0, 2.0, 3.0  // tx, ty, tz
);
```

#### 四元数格式

```cpp
tree.set_transform_quat("camera", "base_link", timestamp_ns,
    1.0, 0.0, 0.0, 0.0,  // w, x, y, z
    1.0, 2.0, 3.0         // tx, ty, tz
);
```

### 3. 变换查询

#### 精确查询

```cpp
auto result = tree.lookup_transform("camera", "base_link", timestamp_ns);
if (result) {
    auto matrix = result->transform.matrix();
    auto trans = result->transform.translation();
}
```

#### 查询选项

```cpp
TransformQueryOptions options;
options.allow_interpolation = true;
options.allow_extrapolation = false;
options.max_age_ns = 1000000000;  // 1秒

auto result = tree.lookup_transform("camera", "base_link", timestamp_ns, options);
```

### 4. 时间插值

系统自动支持两个关键帧之间的线性插值：

```cpp
// 缓冲区自动处理插值
auto entry = buffer.at_interpolated(timestamp_ns);
```

## API 参考

### 变换矩阵 (TransformMatrix4d)

```cpp
// 创建
TransformMatrix4d tf = TransformMatrix4d::from_rpy(roll, pitch, yaw, x, y, z);
TransformMatrix4d tf = TransformMatrix4d::from_quaternion(w, x, y, z, tx, ty, tz);

// 操作
TransformMatrix4d result = tf1 * tf2;              // 矩阵乘法
TransformMatrix4d inv = tf.inverse();              // 逆变换
TransformMatrix4d lerp = TransformMatrix4d::lerp(tf1, tf2, t);  // 线性插值

// 提取信息
Eigen::Vector3d trans = tf.translation();          // 平移向量
Eigen::Matrix3d rot = tf.rotation();               // 旋转矩阵
Eigen::Quaterniond q = tf.quaternion();            // 四元数
const Eigen::Matrix4d& mat = tf.matrix();          // 4x4矩阵
```

### 循环缓冲区 (CircularTransformBuffer<N>)

```cpp
CircularTransformBuffer<100> buffer;

// 写入
buffer.push(timestamp_ns, transform_matrix);

// 查询
auto entry = buffer.at(timestamp_ns);              // 精确查询
auto entry = buffer.at_interpolated(timestamp_ns); // 插值查询
auto entry = buffer.latest();                      // 最新值

// 查询信息
std::size_t size = buffer.size();
std::size_t cap = buffer.capacity();
bool full = buffer.is_full();
auto range = buffer.time_range();                  // 时间范围

// 管理
buffer.clear();
```

### 变换树 (TransformTree<TreeStructure>)

```cpp
TransformTree<MyTree> tree;

// 设置变换
tree.set_transform(from, to, ts, roll, pitch, yaw, tx, ty, tz);
tree.set_transform_quat(from, to, ts, w, x, y, z, tx, ty, tz);

// 查询变换
auto result = tree.lookup_transform(target, source, timestamp_ns);
auto result = tree.get_latest_transform(target, source);

// 坐标系管理
tree.register_frame(name, parent_name);
tree.register_alias(alias, canonical_name);

// 诊断
std::vector<std::string> frames = tree.get_all_frames();
auto parent = tree.get_parent_frame(name);
auto children = tree.get_child_frames(name);
bool exists = tree.has_frame(name);

// 并行查询
auto results = tree.parallel_lookup_transforms(frame_pairs, timestamp_ns);

// 清理
tree.clear();
```

## C++23 概念

系统使用以下概念进行编译期约束：

```cpp
// 变换输入（RPY或四元数）
concept TransformInput = RPYTransform<T> || QuaternionTransform<T>;

// 坐标系标识符
concept FrameIdentifier;

// 时间查询
concept TemporalQuery;

// 变换缓冲区
concept TransformBuffer;

// 树节点
concept TreeNode;

// 有效的变换树
concept ValidTransformTree;
```

## 编译期字符串

系统支持编译期字符串，用于坐标系名称：

```cpp
constexpr CompileTimeString<5> world("world");
constexpr CompileTimeString<9> base_link("base_link");

// 编译期比较和哈希
static_assert(world.hash() != base_link.hash());
static_assert(world < base_link);

// 运行期使用
std::cout << world.view() << "\n";  // "world"
```

## 性能特性

### 编译期优化

1. **树结构验证**：零运行时开销
2. **路径查找**：编译期可以验证坐标系连接
3. **类型检查**：转换失配导致编译错误

### 运行期优化

1. **SIMD**：Eigen 自动使用 AVX2/AVX-512
2. **无锁读取**：TBB 自旋读写互斥体
3. **固定大小分配**：避免动态内存分配
4. **并行查询**：使用 TBB 并行化多个查询

## 使用示例

### 基础用法

```cpp
#include "fast_tf/core.hpp"

using namespace fast_tf;

// 定义树结构
using MyTree = TransformTreeStructure<
    TreeNode<5, NullParent, BufferCapacity<100>>,    // "world"
    TreeNode<9, ParentRef<5>, BufferCapacity<50>>,   // "base_link"
    TreeNode<6, ParentRef<9>, BufferCapacity<30>>    // "camera"
>;

// 创建树
MyTree tree;

// 设置变换
uint64_t now = 1000000000;  // 纳秒
tree.set_transform("camera", "base_link", now, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3);

// 查询变换
auto result = tree.lookup_transform("camera", "base_link", now);
if (result) {
    std::cout << "Found transform\n";
}
```

### 高级用法：批量查询

```cpp
std::vector<std::pair<std::string, std::string>> pairs = {
    {"camera", "base_link"},
    {"imu", "base_link"},
    {"lidar", "world"}
};

auto results = tree.parallel_lookup_transforms(pairs, timestamp_ns);

for (std::size_t i = 0; i < results.size(); ++i) {
    if (results[i]) {
        std::cout << "Transform " << i << " found\n";
    }
}
```

## 编译要求

- **编译器**：GCC 13.2+ 或 Clang 17+（需要 C++23 支持）
- **依赖**：
  - Eigen3（线性代数）
  - Intel TBB（并行计算）
  - GTest（测试，可选）
  - Google Benchmark（性能测试，可选）

## 构建说明

```bash
# 创建构建目录
mkdir build && cd build

# 配置和构建
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release

# 运行示例
./bin/fast_tf_example

# 运行测试（如果有GTest）
ctest
```

## 性能基准

### 查询性能（单线程）

- **精确查询**：~0.1-0.5 μs（取决于缓冲区大小）
- **插值查询**：~0.5-2.0 μs
- **矩阵乘法**：~0.2-0.5 μs（使用 SIMD）

### 缓冲区操作

- **Push 操作**：~0.05 μs（固定大小分配）
- **内存使用**：固定，编译期确定

### 并行化

- **N 个查询的并行化**：使用 TBB 自动负载均衡
- **无锁读取**：多个读者可以并发访问

## 常见编译错误

### 错误：树结构无效

```
error: 'validate()' is not a constant expression
```

**原因**：树结构存在问题（循环、不连接等）
**解决**：检查树节点的父子关系

### 错误：TransformInput 不匹配

```
error: no matching function for call to 'set_transform_input'
```

**原因**：输入不符合 RPY 或四元数格式
**解决**：确保结构包含必需的字段

### 错误：概念约束违反

```
error: concept constraint violated
```

**原因**：类型不满足要求的概念
**解决**：检查类型是否实现了所需的接口

## 设计原则

1. **零开销抽象**：所有编译期验证都有零运行时成本
2. **类型安全**：坐标系名称和变换格式通过类型系统保证
3. **可组合性**：模块化设计允许灵活组合
4. **性能优先**：使用现代 C++ 特性实现最优性能
5. **简单接口**：复杂性隐藏在编译期

## 许可证

见项目根目录的 LICENSE 文件。

## 相关资源

- [Eigen 文档](https://eigen.tuxfamily.org/)
- [TBB 文档](https://www.intel.com/content/www/en/en/develop/documentation/oneapi-programming-guide/top/programming-with-tbb.html)
- [C++23 标准](https://en.cppreference.com/)
- [ROS2 tf2 文档](https://docs.ros.org/en/humble/Concepts/Advanced/TF2/Overview.html)
