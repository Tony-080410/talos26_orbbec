# C++ 项目全面测试开发 Prompt

## 项目概述
为高性能 C++ 项目开发完整的测试套件，包含单元测试、集成测试、性能基准测试和边界条件验证。

## 项目结构
```
├── CMakeLists.txt
├── include/
│   ├── detector_state.hpp
│   ├── performance_probe.hpp
│   ├── spdlog_hook.hpp
│   ├── tracker_soa.hpp
│   └── triple_buffer.hpp
├── src/
│   └── main.cpp
└── test/
    ├── latency_benchmark.cpp
    ├── tracker_soa_test.cpp
    └── triple_buffer_test.cpp
```

## 任务要求

### 1. 测试框架设置
- [ ] 分析现有 `CMakeLists.txt`，集成 Google Test 或 Catch2 测试框架
- [ ] 配置测试目标编译，确保与主项目分离
- [ ] 设置代码覆盖率工具（gcov/lcov）
- [ ] 添加 CTest 集成以支持 `ctest` 命令

### 2. 补全缺失的测试文件
为每个头文件创建对应的测试：

#### detector_state_test.cpp
- 状态转换正确性测试
- 并发状态访问安全性测试
- 状态持久化/序列化测试（如适用）
- 边界条件：空状态、无效转换

#### performance_probe_test.cpp
- 时间测量精度验证
- 高频调用性能开销测试
- 多线程探针安全性测试
- 统计数据准确性验证

### 3. 增强现有测试

#### tracker_soa_test.cpp 增强
- **数据布局验证**：确认 SoA（Structure of Arrays）内存对齐
- **容量扩展测试**：动态增长、收缩行为
- **迭代器有效性**：增删操作后的迭代器失效测试
- **异常安全**：RAII、strong exception guarantee

#### triple_buffer_test.cpp 增强
- **无锁读写**：验证 lock-free 语义
- **内存序测试**：memory_order 正确性（acquire/release）
- **生产者-消费者**：多线程场景压力测试
- **数据竞争检测**：使用 ThreadSanitizer 验证
- **极端频率**：高频切换缓冲区的稳定性

#### latency_benchmark.cpp 增强
- **百分位数统计**：P50/P95/P99/P99.9 延迟
- **预热阶段**：排除 JIT/缓存影响
- **CPU 亲和性**：绑定核心以减少抖动
- **对比基准**：与标准库/竞品组件对比
- **可视化输出**：生成延迟分布图（可选）

### 4. 集成测试
创建 `integration_test.cpp`：
- 多组件协同工作场景
- 端到端工作流测试
- 真实使用场景模拟
- 资源泄漏检测（Valgrind/ASan）

## 技术要求

### 代码质量
- 遵循现代 C++ 最佳实践（C++17/20）
- 使用智能指针，避免裸指针
- RAII 模式管理资源
- constexpr/noexcept 合理使用

### 性能考量
- 基准测试使用 Google Benchmark 或类似库
- 避免测试代码中的不必要分配
- 使用 `DoNotOptimize()` 防止编译器过度优化

### 可维护性
- 清晰的测试命名（`TEST(ComponentName, BehaviorUnderTest)`）
- 每个测试一个断言主题
- 使用 fixture 共享测试设置
- 添加注释说明复杂测试意图

## 输出清单
1. ✅ 更新的 `CMakeLists.txt` 含完整测试配置
2. ✅ 5 个新/增强的测试文件（覆盖所有头文件）
3. ✅ 1 个集成测试文件
6. ✅ 测试覆盖率报告（>90% 目标）

## 验收标准
- [ ] 所有测试通过（`ctest` 零失败）
- [ ] 代码覆盖率 >90%
- [ ] 无内存泄漏（Valgrind 清洁）
- [ ] 无数据竞争（TSan 清洁）
- [ ] 基准测试性能达标（根据项目需求）
- [ ] CI 管道全绿

## 参考资源
- [Google Test 文档](https://google.github.io/googletest/)
- [Google Benchmark 指南](https://github.com/google/benchmark)
- [Modern C++ Testing](https://www.amazon.com/Modern-Testing-Catch2-Gregory-Law/dp/1803242000)

---

**开始开发前请**：
1. 审查现有代码，理解组件职责
2. 确认编译器版本和依赖库
3. 询问性能基准目标（如延迟阈值）