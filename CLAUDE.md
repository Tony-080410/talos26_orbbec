1. **能用 RAII 的，绝不手动管理资源**
2. **能用类型系统的，绝不用运行时检查**
3. **能用编译时检查的，绝不用文档约定**
4. **能用成熟库的，绝不自造轮子**
5. 禁止nullptr, NULL，强制使用std::optional
6. 禁止exception，强制使用std::expected
7. 禁止switch类型匹配，强制使用variant，强制穷尽访问
8. 禁用cstring，carray，强制使用std::string和std::array
9. 禁用T* + size_t，强制使用std::span<T>
10. 一般情况下禁止裸指针，使用unique_ptr, shared_ptr, weak_ptr, 语义标明是否可能为空
11. 禁止使用 bool 表示存在多义性的错误信息
12. 禁止隐式类型擦除（std::any、std::function）
13. 禁止运行时未指定最大深度的递归
