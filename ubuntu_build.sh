cmake -G Ninja -B build -S . \
  -DCMAKE_CXX_COMPILER=clang++-21 \
  -DCMAKE_C_COMPILER=clang-21
cmake --build build -- -j$(nproc)