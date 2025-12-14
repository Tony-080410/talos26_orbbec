# 🤖 Talos Tracker

> _如同守卫克里特岛的青铜巨人，Talos 追踪器作为唯一的、永不疲倦的哨兵，_

[![C++20](https://img.shields.io/badge/C++-20-00599C.svg)](https://isocpp.org/)
[![Status](https://img.shields.io/badge/Status-Phase_1_Prototype-green.svg)](#)

## 📖 项目简介

Talos 是一款专为 **RoboMaster 自瞄系统**设计的高性能单例追踪器，通过无锁原子同步和缓存友好的数据结构，彻底消除传统互斥锁瓶颈，实现 **< 1 微秒**的端到端延迟。

### ⚡ 核心特性

- **🎯 亚微秒延迟** - 平均 450ns，P99 < 1µs 的目标追踪响应
- **🔓 无锁并发** - 基于三缓冲区 (Triple-Buffer) 的零阻塞原子同步
- **💾 缓存优化** - 64 字节对齐的单实例追踪器，完整驻留 L1 缓存
- **🤖 自主运行** - 自包含的单例守护者，处理多目标追踪
- **📊 SoA 架构** - 固定 10 槽装甲数组，位掩码验证，零动态分配

## 🚀 快速开始

### 环境要求

- **编译器**: GCC 11+ / Clang 13+ (C++20)
- **依赖**: Eigen3, OpenCV 4.x, fmt
- **构建**: CMake 3.20+
- **测试**: GoogleTest, Google Benchmark

### 编译安装

```bash
# 克隆仓库
git clone https://github.com/your-org/talos-tracker.git
cd talos-tracker

# 构建
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-O3 -march=native" ..
make -j$(nproc)

# 运行测试
ctest --output-on-failure

# 运行性能基准
./bin/latency_benchmark
```

<div align="center">

**🤖 Built with precision · Engineered for speed · Guarded by Talos**

[⭐ Star](https://github.com/your-org/talos-tracker) · [🐛 Report Bug](https://github.com/your-org/talos-tracker/issues) · [💡 Request Feature](https://github.com/your-org/talos-tracker/issues)

</div>
