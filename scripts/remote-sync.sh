#!/usr/bin/env bash
set -euo pipefail

# ==================== 配置区 ====================
SRC=$(pwd)
DST="nuc@nuc.local:/home/nuc/talos"
PORT="22"
DEBOUNCE_SEC=0.5

# 需要忽略的目录/文件模式
IGNORE_PATTERNS=(
    "build"
    ".cache"
    ".git"
    "_deps"
    ".idea"
    ".claude/settings.local.json"
    "logs"
    "old"
    "dataset"
    "*.swp"
    "*.tmp"
    "*.~"
)

# rsync 排除规则
RSYNC_EXCLUDES=(
    "dataset/"
    "build/"
    ".cache/"
    ".git/"
    "_deps/"
    ".claude/settings.local.json"
    ".idea"
    "logs/"
    "*.swp"
    "*.tmp"
    "*~"
)

# ==================== 颜色定义 ====================
if [[ -t 1 ]]; then
    GREEN='\033[92m'
    YELLOW='\033[93m'
    RED='\033[91m'
    RESET='\033[0m'
else
    GREEN='' YELLOW='' RED='' RESET=''
fi

# ==================== 依赖检查 ====================
check_deps() {
    local missing=()
    for cmd in rsync ssh fswatch; do
        if ! command -v "$cmd" &>/dev/null; then
            missing+=("$cmd")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        echo -e "${RED}[ERROR]${RESET} Missing dependencies: ${missing[*]}"
        echo ""
        echo "Installation guide:"
        echo "  macOS:   brew install fswatch rsync"
        echo "  Ubuntu:  sudo apt install fswatch rsync openssh-client"
        echo "  Arch:    sudo pacman -S fswatch rsync openssh"
        exit 1
    fi
}

# ==================== 同步函数 ====================
LAST_SYNC=0

do_sync() {
    local reason="$1"
    local now
    now=$(date +%s)

    # Debounce: 防抖动
    if (( now - LAST_SYNC < 1 )); then
        return
    fi
    LAST_SYNC=$now

    echo -e "${GREEN}[SYNC]${RESET} Triggered by: $reason"

    # 构建 rsync 命令
    local cmd=(rsync -az --delete -e "ssh -p $PORT")
    for pattern in "${RSYNC_EXCLUDES[@]}"; do
        cmd+=(--exclude "$pattern")
    done
    cmd+=("$SRC/" "$DST")

    if "${cmd[@]}"; then
        echo -e "${YELLOW}[DEBUG]${RESET} rsync done at $(date +'%H:%M:%S')"
    else
        echo -e "${RED}[ERROR]${RESET} rsync failed with exit code $?"
    fi
}

# ==================== 信号处理 ====================
cleanup() {
    echo -e "\n${RED}[EXIT]${RESET} Caught signal, cleaning up..."
    jobs -p | xargs -r kill 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# ==================== 主逻辑 ====================
main() {
    check_deps

    echo -e "${YELLOW}[INFO]${RESET} Monitoring: $SRC"
    echo -e "${YELLOW}[INFO]${RESET} Destination: $DST"
    echo -e "${YELLOW}[INFO]${RESET} Ignored patterns: ${IGNORE_PATTERNS[*]}"
    echo ""

    # 初始同步
    do_sync "INITIAL"

    # 构建 fswatch 排除参数
    local fswatch_args=(
        --recursive
        --event Created
        --event Updated
        --event Removed
        --event Renamed
        --latency "$DEBOUNCE_SEC"
    )

    for pattern in "${IGNORE_PATTERNS[@]}"; do
        fswatch_args+=(--exclude "$pattern")
    done

    fswatch_args+=("$SRC")

    # 启动文件监控
    fswatch "${fswatch_args[@]}" | while read -r event; do
        do_sync "FILE CHANGE: $(basename "$event")"
    done
}

main "$@"