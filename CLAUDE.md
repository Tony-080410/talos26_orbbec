# 项目编译

```
cmake -G Ninja -B build -S .
cmake --build build -- talos -j64
```

# C++ 开发规范

## 元原则

1. 能用 RAII 的，绝不手动管理资源
2. 能用类型系统的，绝不用运行时检查
3. 能用编译时检查的，绝不用文档约定
4. 能用成熟库的，绝不自造轮子

## 类型与内存

1. 禁止运行时 RTTI（dynamic_cast、typeid），允许编译期类型检查（if constexpr、concepts）
2. 禁止 NULL，优先 std::optional 表达值缺失，智能指针表达所有权
3. 禁止裸 new/delete，使用 std::make_unique / std::make_shared
4. 指针语义：
    - 拥有：unique_ptr（独占）、shared_ptr（共享）
    - 非拥有：引用（保证有效）、weak_ptr（需检测失效）、裸指针（生命周期明确时，需注释说明）
5. 禁止 cstring / carray，使用 std::string / std::array
6. 禁止 T* + size_t，使用 std::span<T>

## 错误处理

1. 禁止 exception，使用 std::expected
2. 构造函数必须 noexcept，复杂构造使用工厂函数
3. 第三方库异常在模块边界 try-catch 转换为 expected
4. 禁止 bool 表达多义性错误

## 控制流

1. 枚举 switch：开启 -Wswitch-enum，禁止 default 兜底
2. 类型分发：强制 std::variant + std::visit 穷尽匹配
3. 递归必须满足之一：
    - 编译期深度限制
    - 运行时深度参数 + 超限处理
    - 可证明深度有界

## 类型安全

1. 单参数构造函数强制 explicit
2. 强类型参数优于弱类型（Position 而非 double x, y, z）
3. 限制 std::function，优先模板；必须使用时注释说明原因
4. 禁止 std::any

## 其他

1. 全局可变状态谨慎使用，需注释说明必要性
