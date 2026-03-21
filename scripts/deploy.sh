#!/usr/bin/env bash
# ==============================================================
# deploy.sh —— 目标机（5090 演示机）一键部署脚本
#
# 用法：
#   # 第一次部署（从开发机传来镜像）
#   bash deploy.sh --load path/to/voice-system.tar.gz
#
#   # 后续更新（开发机重新打包后）
#   bash deploy.sh --load path/to/voice-system.tar.gz
#
#   # 直接从 registry pull（需提前 push）
#   REGISTRY=192.168.x.x:5000 bash deploy.sh --pull
#
# ==============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="voice-system:latest"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"

# ---------- 模型目录（可用环境变量覆盖）----------
export TTS_MODEL_DIR="${TTS_MODEL_DIR:-/data/models/Fun-CosyVoice3-0.5B}"
export STT_MODEL_DIR="${STT_MODEL_DIR:-/data/models/stt/models}"
export STT_AUDIO_DIR="${STT_AUDIO_DIR:-/data/models/stt/FunAudioLLM}"
export STT_QWEN_DIR="${STT_QWEN_DIR:-/data/models/stt/Qwen}"
export FAWBOT_MODELS_DIR="${FAWBOT_MODELS_DIR:-/data/models/hub}"
export VLLM_MODEL_DIR="${VLLM_MODEL_DIR:-/data/models/Qwen3-8B}"
export TTS_ASSETS_DIR="${TTS_ASSETS_DIR:-$SCRIPT_DIR/tts/assets}"
export LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/log}"
export FAWBOT_DATA_DIR="${FAWBOT_DATA_DIR:-$SCRIPT_DIR/fawbot_data}"

# ---------- 辅助函数 ----------
_info()  { echo -e "\033[32m[INFO]\033[0m  $*"; }
_warn()  { echo -e "\033[33m[WARN]\033[0m  $*"; }
_error() { echo -e "\033[31m[ERROR]\033[0m $*" >&2; exit 1; }

_check_prereqs() {
    command -v docker &>/dev/null || _error "Docker 未安装，请先安装 Docker Engine"
    docker compose version &>/dev/null 2>&1 || _error "docker compose 插件未找到，请安装 docker-compose-plugin"
    # NVIDIA Container Toolkit
    if ! docker run --rm --gpus all nvcr.io/nvidia/cuda:12.8.1-base-ubuntu24.04 \
            nvidia-smi -L &>/dev/null 2>&1; then
        _warn "NVIDIA Container Toolkit 未就绪，GPU 将不可用"
        _warn "安装方法：https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    fi
}

_create_model_dirs() {
    _info "创建模型挂载目录（如果不存在）..."
    local dirs=(
        "$TTS_MODEL_DIR"
        "$STT_MODEL_DIR"
        "$STT_AUDIO_DIR"
        "$STT_QWEN_DIR"
        "$FAWBOT_MODELS_DIR"
        "$VLLM_MODEL_DIR"
        "$TTS_ASSETS_DIR"
        "$LOG_DIR"
        "$FAWBOT_DATA_DIR"
    )
    for d in "${dirs[@]}"; do
        mkdir -p "$d"
    done
    _info "目录检查完毕。"
}

_check_models() {
    _info "检查关键模型文件..."
    local ok=1

    if [ -z "$(ls -A "$TTS_MODEL_DIR" 2>/dev/null)" ]; then
        _warn "TTS 模型目录为空：$TTS_MODEL_DIR"
        _warn "  请将 Fun-CosyVoice3-0.5B 模型文件复制到该目录。"
        ok=0
    fi
    if [ -z "$(ls -A "$STT_MODEL_DIR" 2>/dev/null)" ]; then
        _warn "STT 模型目录为空：$STT_MODEL_DIR"
        _warn "  请将 SenseVoice/FunASR 模型文件复制到该目录。"
        ok=0
    fi
    if [ -z "$(ls -A "$VLLM_MODEL_DIR" 2>/dev/null)" ]; then
        _warn "vLLM 模型目录为空：$VLLM_MODEL_DIR"
        _warn "  请将 Qwen3-8B 权重复制到该目录。"
        ok=0
    fi
    if [ -z "$(ls -A "$FAWBOT_MODELS_DIR" 2>/dev/null)" ]; then
        _warn "Embedding 模型目录为空：$FAWBOT_MODELS_DIR"
        _warn "  请将 BAAI/bge-small-zh-v1.5 缓存复制到该目录。"
        ok=0
    fi

    if [ "$ok" -eq 0 ]; then
        _warn "部分模型目录为空，服务可能无法正常启动。"
        read -r -p "是否仍然继续启动？[y/N] " ans
        [[ "$ans" =~ ^[Yy]$ ]] || exit 1
    fi
}

# ==============================================================
# 主流程
# ==============================================================
MODE="${1:-}"

_check_prereqs
_create_model_dirs

case "$MODE" in
    --load)
        TAR="${2:-}"
        [ -f "$TAR" ] || _error "找不到镜像包：$TAR"
        _info "正在导入镜像：$TAR ..."
        docker load -i "$TAR"
        ;;
    --pull)
        REGISTRY="${REGISTRY:-}"
        if [ -n "$REGISTRY" ]; then
            REMOTE_IMAGE="$REGISTRY/voice-system:latest"
            _info "正在从 $REGISTRY 拉取镜像..."
            docker pull "$REMOTE_IMAGE"
            docker tag "$REMOTE_IMAGE" "$IMAGE_NAME"
        else
            _error "请先设置 REGISTRY 环境变量，例如：REGISTRY=192.168.1.100:5000 bash deploy.sh --pull"
        fi
        ;;
    --build)
        _info "在本机重新构建镜像（需要完整源码）..."
        BUILD_CONTEXT="$(dirname "$SCRIPT_DIR")"  # project/ 目录
        docker build -f "$SCRIPT_DIR/Dockerfile" -t "$IMAGE_NAME" "$BUILD_CONTEXT"
        ;;
    "")
        _info "未指定 --load / --pull / --build，假设镜像已存在，直接启动容器。"
        ;;
    *)
        echo "用法: $0 [--load <image.tar.gz>] [--pull] [--build]"
        exit 1
        ;;
esac

_check_models

_info "启动所有服务..."
docker compose -f "$COMPOSE_FILE" up -d

_info "=================================================="
_info "服务已在后台启动。"
_info ""
_info "  查看实时日志："
_info "    docker logs -f voice-system"
_info ""
_info "  进入容器 shell："
_info "    docker exec -it voice-system bash"
_info ""
_info "  停止所有服务："
_info "    docker compose -f $COMPOSE_FILE down"
_info "=================================================="
