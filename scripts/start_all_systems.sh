#!/usr/bin/env bash
# 一键启动整条语音 / LiveKit 链路（后台进程 + 日志）。
#
# 1. livekit-server --dev --bind 0.0.0.0
# 2. conda voice_system -> kws/kws_server.py
# 3. conda voice_system -> stt/stt_server_novad.py
# 4. conda voice_system -> tts/tts_server.py
# 5. conda fawbot-agent -> fawbot_0410/fawtd_robot/scripts/start_services.py --all
#    （含 vLLM、MCP、API Server，就绪后自动退出）
# 6. conda livekit -> server/stt_llm_agent.py start
#
# 覆盖 Fawbot 目录（若实际路径不同）:
#   export FAWBOT_AGENT_DIR=/path/to/your/fawbot_0410/fawtd_robot
#
# 停止：Ctrl+C（会尝试优雅结束本脚本拉起的子进程）。
#
# 仅写文件、不在终端跟日志：
#   QUIET_TERMINAL_LOGS=1 bash start_all_systems.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOICE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FAWBOT_AGENT_DIR="${FAWBOT_AGENT_DIR:-/home/zhangchi/project/fawbot_0410/fawtd_robot}"

LOG_DIR="$VOICE_ROOT/log/startup"
mkdir -p "$LOG_DIR"

# Python 尽快按行刷日志，避免管道里长时间看不到输出
export PYTHONUNBUFFERED=1

BG_PIDS=()

_cleanup() {
  trap - EXIT SIGINT SIGTERM
  echo ""
  echo "正在停止后台进程..."
  for pid in "${BG_PIDS[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  wait "${BG_PIDS[@]}" 2>/dev/null || true

  echo "正在停止 Fawbot (start_services.py --stop)..."
  (cd "$FAWBOT_AGENT_DIR" && conda run --no-capture-output -n fawbot-agent python scripts/start_services.py --stop) || true

  echo "已退出。"
}

trap _cleanup SIGINT SIGTERM EXIT

if ! command -v livekit-server >/dev/null 2>&1; then
  echo "错误: 未找到 livekit-server，请先安装并加入 PATH。" >&2
  exit 1
fi

if [[ ! -d "$FAWBOT_AGENT_DIR" ]]; then
  echo "错误: Fawbot 目录不存在: $FAWBOT_AGENT_DIR" >&2
  echo "请创建该目录或设置环境变量 FAWBOT_AGENT_DIR 指向实际路径。" >&2
  exit 1
fi

_launch() {
  local name="$1"
  shift
  local logfile="$LOG_DIR/${name}.log"
  echo "[$name] 启动中 -> 终端（带前缀）+ $logfile（原始）"
  if [[ "${QUIET_TERMINAL_LOGS:-0}" == "1" ]]; then
    ("$@") >>"$logfile" 2>&1 &
  else
    # 文件里保留原始行；终端每行前加 [name]，多模块并行时便于分辨
    (
      "$@" 2>&1 | stdbuf -oL tee -a "$logfile" | stdbuf -oL sed "s/^/[${name}] /"
    ) &
  fi
  BG_PIDS+=($!)
  echo "[$name] PID=${BG_PIDS[-1]}"
}

# ── 第一步：start_services.py --all（阻塞至就绪后自动退出）再起其余服务 ──────
# 注意：不加 --daemon，脚本在所有服务就绪后自动退出，wait 才能返回
_launch fawbot-start-services bash -c "cd \"$FAWBOT_AGENT_DIR\" && conda run --no-capture-output -n fawbot-agent python scripts/start_services.py --all"

echo "[fawbot-start-services] 等待 start_services --all 完成..."
wait "${BG_PIDS[-1]}" 2>/dev/null || true
echo "[fawbot-start-services] 就绪，继续启动其余服务。"

# ── 第二步：LiveKit / KWS / STT / TTS ───────────────────────────────────────
_launch livekit-server livekit-server --dev --bind 0.0.0.0
sleep 1

_launch kws-server bash -c "cd \"$VOICE_ROOT\" && conda run --no-capture-output -n voice_system python kws/kws_server.py"
sleep 1

_launch stt-server bash -c "cd \"$VOICE_ROOT\" && conda run --no-capture-output -n voice_system python stt/stt_server_novad.py"
sleep 1

_launch tts-server bash -c "cd \"$VOICE_ROOT\" && conda run --no-capture-output -n voice_system python tts/tts_server.py"
sleep 1

# ── 第三步：STT/LLM Agent ─────────────────────────────────────────────────────
# fawbot-robot-api 已由 start_services.py --all 启动，无需重复拉起
_launch stt-llm-agent bash -c "cd \"$VOICE_ROOT\" && conda run --no-capture-output -n livekit python server/stt_llm_agent.py start"

echo ""
echo "全部已在后台运行；日志目录: $LOG_DIR（各模块原始输出）"
if [[ "${QUIET_TERMINAL_LOGS:-0}" != "1" ]]; then
  echo "当前终端会混排打印各模块日志（行首 [模块名]）；按 Ctrl+C 结束并尝试停止子进程。"
else
  echo "终端不跟日志，请 tail -f $LOG_DIR/*.log；按 Ctrl+C 结束并尝试停止子进程。"
fi
wait
