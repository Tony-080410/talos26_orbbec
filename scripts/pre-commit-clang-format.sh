#!/usr/bin/env bash
set -euo pipefail

CLANG_FORMAT=${CLANG_FORMAT:-clang-format}

FILES=$(git diff --cached --name-only --diff-filter=ACM \
    | grep -E '\.(c|cc|cpp|cxx|h|hpp)$' || true)

[ -z "$FILES" ] && exit 0

FAILED=0

for file in $FILES; do
    [ ! -f "$file" ] && continue

    staged=$(git show ":$file")
    formatted=$(echo "$staged" | $CLANG_FORMAT --assume-filename="$file")

    if [ "$staged" != "$formatted" ]; then
        echo "❌ clang-format mismatch: $file"
        echo
        # 使用 diff 的彩色输出
        diff -u <(echo "$staged") <(echo "$formatted") \
            | tail -n +3 \
            | head -50 \
            | sed "s/^-/$(tput setaf 1)-/;s/^+/$(tput setaf 2)+/;s/^@/$(tput setaf 6)@/" \
            || true
        echo "$(tput sgr0)"
        FAILED=1
    fi
done

if [ "$FAILED" -ne 0 ]; then
    echo "✋ clang-format check failed."
    echo "👉 请运行: clang-format -i <files> 并重新 git add"
    exit 1
fi