#!/usr/bin/env bash
# ==============================================================
# build_and_export.sh —— 开发机打包镜像，传给演示机
#
# 用法（在 voice_system_livekit/ 目录下执行）：
#   bash scripts/build_and_export.sh
#
# 输出：
#   /tmp/voice-system.tar.gz   （约 15~20 GB，含三个 conda 环境）
#
# 完成后将该文件 scp 到演示机，再执行：
#   bash scripts/deploy.sh --load /path/to/voice-system.tar.gz
# ==============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOICE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(dirname "$VOICE_ROOT")"

IMAGE_NAME="voice-system:latest"
OUTPUT="${OUTPUT:-/tmp/voice-system.tar.gz}"

echo "[build] Build context: $PROJECT_ROOT"
echo "[build] 开始构建镜像，这可能需要 30-60 分钟..."

docker build \
    -f "$VOICE_ROOT/Dockerfile" \
    -t "$IMAGE_NAME" \
    "$PROJECT_ROOT"

echo "[build] 构建完成，正在导出镜像到 $OUTPUT ..."
docker save "$IMAGE_NAME" | gzip > "$OUTPUT"

SIZE=$(du -sh "$OUTPUT" | cut -f1)
echo "[build] 打包完成：$OUTPUT ($SIZE)"
echo ""
echo "传输到演示机："
echo "  scp $OUTPUT user@demo-machine:/tmp/"
echo ""
echo "演示机上执行："
echo "  bash deploy.sh --load /tmp/voice-system.tar.gz"
