#!/usr/bin/env bash
set -euo pipefail

CLANG_FORMAT=${CLANG_FORMAT:-clang-format}

# 只检查暂存区里的 C/C++ 文件
FILES=$(git diff --cached --name-only --diff-filter=ACM \
    | grep -E '\.(c|cc|cpp|cxx|h|hpp)$' || true)

if [ -z "$FILES" ]; then
    exit 0
fi

FAILED=0

for file in $FILES; do
    # 用 clang-format 生成格式化结果
    formatted=$($CLANG_FORMAT "$file")

    # 取暂存区版本
    staged=$(git show ":$file")

    if [ "$formatted" != "$staged" ]; then
        echo "❌ clang-format mismatch: $file"
        FAILED=1
    fi
done

if [ "$FAILED" -ne 0 ]; then
    echo
    echo "✋ clang-format check failed."
    echo "👉 请运行: clang-format -i <files> 并重新 git add"
    exit 1
fi
