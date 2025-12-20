cmake -G Ninja -B build -S . \
  -DCMAKE_CXX_COMPILER=clang++-20 \
  -DCMAKE_C_COMPILER=clang-20
cmake --build build -- -j$(nproc)