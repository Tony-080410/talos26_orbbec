#!/bin/bash
set -e

cmake -G Ninja -B build -S . \
  -DENABLE_CALIBRATION=OFF \
  -DCMAKE_CXX_COMPILER=clang++-21 \
  -DCMAKE_C_COMPILER=clang-21
cmake --build build -- talos -j$(nproc)

# Set library path for HIK SDK and run
export LD_LIBRARY_PATH="$(pwd)/3dparty/hik_sdk/lib/arm64:$LD_LIBRARY_PATH"
exec ./build/bin/talos "$@"
