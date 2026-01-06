#!/usr/bin/env bash
set -euo pipefail
FAILED=0

for file in $FILES; do
    matches=$(grep -nE 'spdlog::(log|trace|debug|info|warn|error|critical)\s*\(' "$file" 2>/dev/null)

    if [ -n "$matches" ]; then
        echo "$matches" | while IFS=: read -r line_num content; do
            echo "❌ spdlog function call detected: $file:$line_num: $(echo "$content" | xargs)"
        done
        FAILED=1
    fi
done

if [ "$FAILED" -ne 0 ]; then
    echo
    echo "✋ 禁止使用 spdlog::XXXX()，请改用 SPDLOG_XXXX() 宏"
    exit 1
fi