# Chiral 对接文档

本目录用于说明 Talos 与外部开发者通过 `chiral` 进行状态共享时的接入方式与约束。

- 核心文档（对接方必读）: [integration.md](./integration.md)
- 数据类型定义: `crates/chiral/src/chiral/chiral.hpp`
- 三重缓冲实现（唯一维护方案）: `crates/chiral/src/chiral/talos_triple_buffer_shm.hpp`

如果你只看一篇文档，直接看 `integration.md`。
