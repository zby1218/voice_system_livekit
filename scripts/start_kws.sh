#!/usr/bin/env bash
# 一键启动 KWS 流程：kws_server + kws_wake_listener + kws_trigger
#
# 目录结构（kws 与 voice_system_livekit 同级）：
#   project/
#   ├── kws/                    <- kws_trigger.py（此处）
#   │   └── kws_trigger.py
#   └── voice_system_livekit/   <- kws_server + kws_wake_listener
#       ├── kws/kws_server.py
#       └── client/kws_wake_listener.py
#
# 1. conda voice_system -> voice_system_livekit/kws/kws_server.py
# 2. conda video        -> project/kws/kws_trigger.py（同级目录）
# 3. uv run             -> voice_system_livekit/client/kws_wake_listener.py
# Ctrl+C 会先停掉 kws_trigger，再清理后台的 server 和 listener。

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOICE_SYSTEM_LIVEKIT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(dirname "$VOICE_SYSTEM_LIVEKIT")"

BG_PIDS=()

cleanup() {
  echo ""
  echo "正在停止后台进程..."
  for pid in "${BG_PIDS[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  wait "${BG_PIDS[@]}" 2>/dev/null || true
  echo "已退出。"
  exit 0
}

trap cleanup SIGINT SIGTERM

# 1. KWS 服务（voice_system）
(cd "$VOICE_SYSTEM_LIVEKIT" && conda run -n voice_system python kws/kws_server.py) &
BG_PIDS+=($!)
echo "[1/3] kws_server 已启动 (conda: voice_system) PID=$!"

sleep 2

# 3. KWS 唤醒 Listener（uv）
(cd "$VOICE_SYSTEM_LIVEKIT" && uv run client/kws_wake_listener.py) &
BG_PIDS+=($!)
echo "[2/3] kws_wake_listener 已启动 (uv) PID=$!"

sleep 1

# 2. KWS 触发器（video）
echo "[3/3] 启动 kws_trigger (conda: video)，按 Ctrl+C 停止全部"
(cd "$PROJECT_ROOT" && conda run -n video python kws/kws_trigger.py) || true

cleanup
