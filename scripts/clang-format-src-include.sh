#!/usr/bin/env bash
set -euo pipefail

CLANG_FORMAT=${CLANG_FORMAT:-clang-format}
DRY_RUN=false
VERBOSE=false
CHECK_ONLY=false

show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Format all C/C++ files under src/ and include/ with clang-format.

OPTIONS:
    -n, --dry-run       Show which files need formatting without modifying
    -c, --check         Check formatting and exit with error if any file needs formatting
    -v, --verbose       Show all files (including already formatted ones)
    -h, --help          Show this help message

EXAMPLES:
    $0                  # Format all files
    $0 --dry-run        # Preview which files need formatting
    $0 --check          # Check formatting (useful for CI)
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -c|--check)
            CHECK_ONLY=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# 收集文件
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

echo "Checking ${#FILES[@]} files..."
echo

NEEDS_FORMAT=0
FAILED_FILES=()

for file in "${FILES[@]}"; do
    rel_path="${file#$ROOT_DIR/}"

    # 检查是否需要格式化
    formatted=$("$CLANG_FORMAT" "$file")
    original=$(cat "$file")

    if [ "$formatted" != "$original" ]; then
        if [ "$CHECK_ONLY" = true ] || [ "$DRY_RUN" = true ]; then
            echo "❌ Needs formatting: $rel_path"
            FAILED_FILES+=("$rel_path")
        else
            echo "✅ Formatting: $rel_path"
            "$CLANG_FORMAT" -i "$file"
        fi
        ((NEEDS_FORMAT++))
    elif [ "$VERBOSE" = true ]; then
        echo "✨ Already formatted: $rel_path"
    fi
done

echo
if [ "$NEEDS_FORMAT" -eq 0 ]; then
    echo "✨ All ${#FILES[@]} files are properly formatted."
    exit 0
else
    if [ "$CHECK_ONLY" = true ]; then
        echo "❌ $NEEDS_FORMAT out of ${#FILES[@]} files need formatting:"
        printf '   - %s\n' "${FAILED_FILES[@]}"
        echo
        echo "👉 Run: $0"
        exit 1
    elif [ "$DRY_RUN" = true ]; then
        echo "⚠️  $NEEDS_FORMAT out of ${#FILES[@]} files need formatting."
        echo "👉 Run without --dry-run to format them."
        exit 1
    else
        echo "✅ Formatted $NEEDS_FORMAT out of ${#FILES[@]} files."
    fi
fi