#!/bin/bash
# ===============================================================
# 语音系统 统一环境安装脚本
# 包含：KWS (唤醒) + ASR (识别) + TTS (合成)
# ===============================================================

ENV_NAME="voice_system"
PYTHON_VER="3.12"

# 获取脚本所在目录
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

echo "=============================================="
echo "  语音系统 统一环境安装"
echo "  环境名: $ENV_NAME (Python $PYTHON_VER)"
echo "=============================================="

# ==================== 1. 检查 Conda ====================
if ! command -v conda &> /dev/null; then
    echo "❌ 未找到 conda！请先安装 Anaconda 或 Miniconda"
    exit 1
fi

# 初始化 Conda
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ==================== 2. 创建/检查环境 ====================
if conda info --envs | grep -q "^$ENV_NAME "; then
    echo "✅ 环境 '$ENV_NAME' 已存在"
else
    echo "创建 Conda 环境: $ENV_NAME (Python $PYTHON_VER)..."
    conda create -n "$ENV_NAME" python="$PYTHON_VER" -y
fi

# 激活环境
echo "🔄 激活环境..."
conda activate "$ENV_NAME"

# 验证 Python 版本
CURRENT_PY=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "当前 Python 版本: $CURRENT_PY"

# ==================== 3. 升级 pip ====================
echo "升级 pip..."
pip install --upgrade pip

# ==================== 4. 安装 PyTorch (本地 whl) ====================
echo ""
echo "安装 PyTorch 2.7.0 (CUDA 12.8)..."

LOCAL_TORCH="$SCRIPT_DIR/python_packages/torch-2.7.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl"
LOCAL_AUDIO="$SCRIPT_DIR/python_packages/torchaudio-2.7.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl"
LOCAL_VISION="$SCRIPT_DIR/python_packages/torchvision-0.22.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl"

if [ -f "$LOCAL_TORCH" ]; then
    echo "   使用本地 whl 文件..."
    pip install "$LOCAL_TORCH" "$LOCAL_AUDIO" "$LOCAL_VISION"
else
    echo "   从 PyTorch 官方源下载..."
    pip install torch==2.7.0+cu128 torchaudio==2.7.0+cu128 torchvision==0.22.0+cu128 \
        --index-url https://download.pytorch.org/whl/cu128
fi

# ==================== 5. 安装 onnxruntime-gpu (本地 whl) ====================
echo ""
echo "安装 onnxruntime-gpu..."

LOCAL_ONNX="$SCRIPT_DIR/python_packages/onnxruntime_gpu-1.18.0-cp312-cp312-manylinux_2_28_x86_64.whl"
if [ -f "$LOCAL_ONNX" ]; then
    pip install "$LOCAL_ONNX"
else
    pip install onnxruntime-gpu==1.18.0 \
        --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
fi

# ==================== 6. 安装 requirements.txt ====================
echo ""
echo "安装 requirements.txt 中的所有依赖..."
pip install -r "$SCRIPT_DIR/requirements.txt" \
    -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# ==================== 7. 验证安装 ====================
echo ""
echo "🔍 验证安装..."

python << 'EOF'
import sys
print(f"Python: {sys.version}")

checks = [
    ("torch", "PyTorch"),
    ("numpy", "NumPy"),
    ("onnxruntime", "ONNXRuntime"),
    ("funasr", "FunASR"),
    ("librosa", "Librosa"),
    ("websockets", "WebSockets"),
    ("whisper", "Whisper"),
    ("hyperpyyaml", "HyperPyYAML"),
    ("rich", "Rich"),
    ("ruamel.yaml", "ruamel.yaml"),
]

for module, name in checks:
    try:
        m = __import__(module)
        ver = getattr(m, '__version__', 'OK')
        print(f"✅ {name}: {ver}")
    except ImportError as e:
        print(f"❌ {name}: {e}")

# 检查 CUDA
import torch
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
EOF

# ==================== 完成 ====================
echo ""
echo "=============================================="
echo "✅ 安装完成！"
echo ""
echo "使用方法："
echo "  1. 激活环境: conda activate $ENV_NAME"
echo "=============================================="
