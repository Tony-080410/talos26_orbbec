#!/usr/bin/env bash
set -euo pipefail

# 保证在仓库根目录执行
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

HOOK_SRC="scripts/pre-commit-clang-format.sh"
HOOK_DST=".git/hooks/pre-commit"

echo "🔧 Installing git pre-commit hook..."

# 1. 基本检查
if [ ! -d ".git" ]; then
    echo "❌ Not a git repository"
    exit 1
fi

if [ ! -f "$HOOK_SRC" ]; then
    echo "❌ Hook source not found: $HOOK_SRC"
    exit 1
fi

# 2. 确保脚本可执行
chmod +x "$HOOK_SRC"

# 3. 创建 hooks 目录（极端情况下）
mkdir -p .git/hooks

# 4. 备份已有 hook（不覆盖用户已有逻辑）
if [ -e "$HOOK_DST" ] && [ ! -L "$HOOK_DST" ]; then
    echo "⚠️  Existing pre-commit hook found, backing up to pre-commit.bak"
    mv "$HOOK_DST" "$HOOK_DST.bak"
fi

# 5. 创建软链接（相对路径，方便仓库移动）
ln -sf "../../$HOOK_SRC" "$HOOK_DST"

echo "✅ pre-commit hook installed successfully"
