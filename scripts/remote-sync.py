#!/usr/bin/env python3
import ctypes
import errno
import os
import select
import shutil
import signal
import subprocess
import sys
import time

IN_MODIFY = 0x00000002
IN_CREATE = 0x00000100
IN_DELETE = 0x00000200
EVENT_MASK = IN_MODIFY | IN_CREATE | IN_DELETE

libc = ctypes.CDLL("libc.so.6")
SRC = "/home/praysky/talos"
DST = "sentry@192.168.10.10:/home/sentry/vision/ws"
PORT = "22"

# 需要忽略的目录名（按“目录名”匹配，任何层级出现都忽略）
IGNORE_DIRS = {"build", ".cache", ".git", "_deps", ".claude", "logs"}

# rsync 排除规则（相对 SRC 的路径模式）
RSYNC_EXCLUDES = ["build/", ".cache/", ".git/", "_deps/", ".claude/", "logs/"]

RSYNC = shutil.which("rsync") or "rsync"

fd = libc.inotify_init()


def add_watch_recursive(root):
    for d, dirs, _ in os.walk(root):
        # 关键：剪枝，不进入被忽略目录
        dirs[:] = [x for x in dirs if x not in IGNORE_DIRS]

        # 保险：当前目录本身也不加 watch
        if os.path.basename(d) in IGNORE_DIRS:
            continue

        libc.inotify_add_watch(fd, d.encode(), EVENT_MASK)


add_watch_recursive(SRC)

GREEN, YELLOW, RED, RESET = "\033[92m", "\033[93m", "\033[91m", "\033[0m"
last_sync = 0
running = True


def remote_sync(reason):
    global last_sync
    if time.time() - last_sync < 0.5:
        return
    last_sync = time.time()
    print(f"{GREEN}[SYNC]{RESET} Triggered by {reason}")

    cmd = [RSYNC, "-az", "-e", f"ssh -p {PORT}", "--delete"]
    for ex in RSYNC_EXCLUDES:
        cmd += ["--exclude", ex]
    cmd += [SRC + "/", DST]

    subprocess.run(cmd)
    print(f"{YELLOW}[DEBUG]{RESET} rsync done {time.strftime('%H:%M:%S')}")


def handle_exit(signum, frame):
    global running
    print(f"\n{RED}[EXIT]{RESET} Caught signal {signum}, cleaning up...")
    os.close(fd)
    running = False


signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

print(f"{YELLOW}[DEBUG]{RESET} Monitoring {SRC} ...")

while running:
    try:
        r, _, _ = select.select([fd], [], [], 1)
        if fd in r:
            os.read(fd, 4096)
            remote_sync("FILE CHANGE")
    except OSError as e:
        if e.errno == errno.EINTR:
            continue
        if e.errno == errno.EBADF:
            break
        print(f"{RED}[ERROR]{RESET} OSError: {e} (errno={e.errno})")
        break

print(f"{GREEN}[INFO]{RESET} Remote sync watcher exited.")
sys.exit(0)
