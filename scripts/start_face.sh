#!/usr/bin/env bash
# 开机自启：Face 人脸识别唤醒系统
#
# 环境对应：
#   摄像头检测 + face_ratio_detect → conda: video
#   face_wake_listener（含 client_face）→ conda: livekit
#
# 启动顺序：
#   1. 发现 conda base（支持 anaconda3 / miniconda3 / miniforge3 / /opt/conda 等）
#   2. 摄像头检测（conda video + camera_check.py）
#      无摄像头 → exit 0（干净退出，systemd 不重启）
#      检测脚本出错 → exit 1（systemd 按策略重启）
#   3. face_wake_listener（conda livekit，监听 present 端口并连接 LiveKit）
#   4. face_ratio_detect（conda video，人脸检测，检到人后发 present）
#
# Ctrl+C / SIGTERM → cleanup 停掉所有子进程
#
# 目录结构（voice_system_livekit 与 face 同级）：
#   project/
#   ├── face/                        ← face_ratio_detect.py、camera_check.py
#   └── voice_system_livekit/
#       ├── client/face_wake_listener.py
#       └── scripts/start_face.sh    ← 本脚本
#
# 开机自启时 udev 可能尚未就绪，脚本会先等待 CAMERA_WAIT_SEC 秒再检测，
# 若无摄像头则重试 CAMERA_RETRIES 次（每次间隔 CAMERA_RETRY_SEC），避免启动过早导致误判。

CAMERA_WAIT_SEC=3
CAMERA_RETRIES=3
CAMERA_RETRY_SEC=5

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOICE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(dirname "$VOICE_DIR")"
FACE_DIR="$PROJECT_ROOT/face"

# ---------------------------------------------------------------------------
# conda base 发现：无硬编码路径，支持多种安装位置
# ---------------------------------------------------------------------------
find_conda_base() {
    # 方式 1：conda 命令已在 PATH 中（交互式 shell / 已激活 base）
    if command -v conda &>/dev/null; then
        local base
        base=$(conda info --base 2>/dev/null)
        [ -n "$base" ] && echo "$base" && return 0
    fi
    # 方式 2：常见安装路径（按优先级）
    local candidate
    for candidate in \
        "$HOME/anaconda3" \
        "$HOME/miniconda3" \
        "$HOME/miniforge3" \
        "$HOME/mambaforge" \
        "/opt/conda" \
        "/opt/anaconda3" \
        "/opt/miniconda3"; do
        [ -f "$candidate/bin/conda" ] && echo "$candidate" && return 0
    done
    return 1
}

CONDA_BASE=$(find_conda_base) || {
    echo "[face] 错误：未找到 conda 安装，请确认 conda 已安装并加入 PATH"
    exit 1
}

VIDEO_PYTHON="$CONDA_BASE/envs/video/bin/python"
LIVEKIT_PYTHON="$CONDA_BASE/envs/livekit/bin/python"

# 校验两个 conda 环境的 Python 可执行文件存在
for _py_check in "$VIDEO_PYTHON" "$LIVEKIT_PYTHON"; do
    if [ ! -x "$_py_check" ]; then
        echo "[face] 错误：未找到 $_py_check，请确认对应 conda 环境已创建"
        exit 1
    fi
done

BG_PIDS=()

# ---------------------------------------------------------------------------
# cleanup：终止所有后台子进程
# ---------------------------------------------------------------------------
cleanup() {
    echo ""
    echo "[face] 正在停止 face 系统进程..."
    for pid in "${BG_PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    wait "${BG_PIDS[@]}" 2>/dev/null || true
    echo "[face] 已退出。"
    exit 0
}

trap cleanup SIGINT SIGTERM

# ---------------------------------------------------------------------------
# Step 1: 摄像头检测（conda video），开机时先等待设备就绪，失败则重试
# ---------------------------------------------------------------------------
echo "[face] 等待 ${CAMERA_WAIT_SEC}s 以便 USB/RealSense 设备就绪..."
sleep "$CAMERA_WAIT_SEC"

do_camera_check() {
    "$VIDEO_PYTHON" -c "
import sys, os
sys.path.insert(0, '$FACE_DIR')
from camera_check import CameraDiscovery
cameras = CameraDiscovery().discover(include_usb=True, include_realsense=True)
if not cameras:
    print('[face] 未检测到摄像头', flush=True)
    sys.exit(2)
print('[face] 检测到摄像头：', flush=True)
for c in cameras:
    suffix = f' (serial={c.serial})' if c.serial else ''
    print(f'  - [{c.type}] {c.name} (index={c.index}){suffix}', flush=True)
sys.exit(0)
"
}

CAMERA_EXIT=0
attempt=1
while [ "$attempt" -le "$CAMERA_RETRIES" ]; do
    echo "[face] 摄像头检测 (conda: video) 第 $attempt/$CAMERA_RETRIES 次..."
    if do_camera_check; then
        CAMERA_EXIT=0
        break
    fi
    CAMERA_EXIT=$?
    if [ "$CAMERA_EXIT" -eq 2 ]; then
        if [ "$attempt" -lt "$CAMERA_RETRIES" ]; then
            echo "[face] 未检测到摄像头，${CAMERA_RETRY_SEC}s 后重试..."
            sleep "$CAMERA_RETRY_SEC"
        fi
        attempt=$((attempt + 1))
    else
        echo "[face] 摄像头检测脚本出错 (exit=$CAMERA_EXIT)，退出"
        exit 1
    fi
done

if [ "$CAMERA_EXIT" -eq 2 ] || [ "$attempt" -gt "$CAMERA_RETRIES" ]; then
    echo "[face] 无摄像头，退出（systemd 不会因此重启）"
    exit 0   # 干净退出 → systemd Restart=on-failure 不触发
fi

# ---------------------------------------------------------------------------
# Step 2: 启动 face_wake_listener（conda livekit，先启动确保端口就绪）
# ---------------------------------------------------------------------------
echo "[face] 启动 face_wake_listener (conda: livekit)..."
(
    cd "$VOICE_DIR/client"
    "$LIVEKIT_PYTHON" face_wake_listener.py \
        --present-port 9999 \
        --face-host 127.0.0.1 \
        --resume-port 9998 \
        --livekit-retry-delay 30 \
        --no-check-camera
) &
LISTENER_PID=$!
BG_PIDS+=($LISTENER_PID)
echo "[face] face_wake_listener 已启动 (conda: livekit) PID=$LISTENER_PID"

sleep 1

# 确认 listener 进程仍在运行（绑定失败会立即退出）
if ! kill -0 "$LISTENER_PID" 2>/dev/null; then
    echo "[face] face_wake_listener 启动失败（可能端口已占用），退出"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 3: 启动 face_ratio_detect（conda video，前台运行）
# ---------------------------------------------------------------------------
echo "[face] 启动 face_ratio_detect (conda: video)，按 Ctrl+C 停止全部"
(
    cd "$FACE_DIR"
    "$VIDEO_PYTHON" face_ratio_detect.py \
        --notify-host 127.0.0.1 \
        --notify-port 9999 \
        --resume-port 9998 \
        --headless
) || true

cleanup
