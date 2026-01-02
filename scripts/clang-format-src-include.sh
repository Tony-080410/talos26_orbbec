#!/usr/bin/env bash
set -euo pipefail

CLANG_FORMAT=${CLANG_FORMAT:-clang-format}

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

FILES=()
while IFS= read -r -d '' file; do
    FILES+=("$file")
done < <(find "$ROOT_DIR/src" "$ROOT_DIR/include" -type f \( \
    -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' -o \
    -name '*.h' -o -name '*.hpp' \) -print0)

if [ "${#FILES[@]}" -eq 0 ]; then
    echo "No C/C++ files found under src/ or include/."
    exit 0
fi

"$CLANG_FORMAT" -i "${FILES[@]}"

echo "Formatted ${#FILES[@]} files under src/ and include/."
