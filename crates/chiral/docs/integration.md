# Chiral 开发者对接手册（仅三重缓冲）

本文档面向外部对接开发者。当前 `chiral` **只维护一套实现**：`talos_triple_buffer_shm`。

## 1. 协议与入口

- 唯一头文件：`chiral/talos_triple_buffer_shm.hpp`
- 共享内存对象名：`/talos_data_triple`
- 协议版本：`TALOS_SHM_VERSION = 2`
- 数据结构：`talos::chiral::TalosData`（定义在 `chiral/chiral.hpp`）

Talos 发布侧入口：

- 系统注册：`src/main.cpp`
- 发布系统：`src/fcs/chiral/chiral_collector_system.cpp`

## 2. 数据内容

`TalosData` 包含：

- 目标类型与状态：`state_kind`, `state.status`, `state.color`, `state.name`
- 目标运动学：`state.robot.*`, `state.outpost.*`
- 关键变换：`gimbal_link`, `muzzle_link`, `camera_link`

单位约定：

- 长度：米
- 角度：弧度
- 角速度：弧度/秒

## 3. 接入要求

- C++23（建议 `-std=c++2b`）
- 编译时包含头文件目录：`crates/chiral/src`
- 并发模型：`SPSC`（单 Writer + 单 Reader）

## 4. 最小 Reader 示例

```cpp
#include "chiral/talos_triple_buffer_shm.hpp"
#include <chrono>
#include <iostream>
#include <thread>

int main() {
    using namespace talos::chiral::ipc;

    auto reader = TalosDataReader::open();
    if (!reader) {
        std::cerr << "open failed, err=" << static_cast<int>(reader.error()) << "\n";
        return 1;
    }

    while (true) {
        if (auto data = reader->read_new()) {
            std::cout << "status=" << static_cast<int>(data->state.status)
                      << " x=" << data->state.robot.position.x << "\n";
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}
```

## 5. API 语义

- `read_new()`: 有新数据返回 `std::optional<TalosData>`；无新数据返回 `std::nullopt`。
- `read_latest()`: 返回当前最新稳定快照的 `TalosData` 拷贝（即使不是新版本）。

TOCTOU 约束（禁止级别）：

- 对接 API **不提供** `has_new()`，禁止 check-then-use。
- 原因是“先检查再读取”没有事务绑定，状态可在两次调用间变化，存在 TOCTOU 风险。
- 必须直接调用 `read_new()`，并以 optional 结果判定是否消费。

禁止模式（示意）：

```cpp
if (some_has_new_flag) {
    // 任何 check-then-read 都存在 TOCTOU 风险
    auto data = reader->read_new();
}
```

唯一允许模式：

```cpp
if (auto data = reader->read_new()) {
    // 消费 data
}
```

## 6. 无撕裂保证

三重缓冲每个槽位都有 `slot_seq[i]`（序列锁）：

1. Writer 写入槽位前把 `slot_seq` 置为奇数（写入中）
2. Writer 完成槽位写入后把 `slot_seq` 置为偶数（稳定）
3. Reader 读取 ready 槽位时做前后两次 `slot_seq` 一致性校验
4. 仅在一致时返回快照

因此 Reader 不会拿到“半写入数据”。

## 7. 错误码与排障

常见错误码：

- `OpenFailed`: Writer 创建失败（常见为对象已存在或权限问题）
- `NotFound`: Reader 打开失败（Talos 侧尚未发布）
- `InvalidMagic`: 共享内存对象不是 chiral 协议对象
- `VersionMismatch`: 协议版本不匹配

排障建议：

1. 确认 Talos 已运行并注册 `chiral_collector`
2. 确认读写双方使用同一版本头文件
3. 清理残留对象后重试（`/talos_data_triple`）
4. 若改过 `TalosData` 定义，读写端必须一起重新编译

## 8. 验收命令

在仓库根目录执行：

```bash
cmake --build build --target triple_buffer_simple_test -j 8
./build/bin/triple_buffer_simple_test
```

通过标准：

- 测试通过
- 无 `Data tearing detected` 报告

## 9. 兼容性策略

- 当前尚未发布 v1，接口仍处于 pre-v1 阶段，允许破坏性调整。
- `TalosData` 视为协议结构体，字段变更属于协议变更。
- 对接方建议固定版本（commit/tag）以避免 ABI 偏差。
