# ==============================================================
# voice_system_livekit - 生产部署镜像
#
# 包含：
#   - conda 环境 voice_system  (KWS / STT / TTS)
#   - conda 环境 livekit        (stt_llm_agent)
#   - conda 环境 fawbot-agent   (fawtd_agent / vLLM)
#   - livekit-server 二进制
#
# 模型文件（TTS / STT / Qwen 权重）通过 volume 挂载，不打进镜像。
#
# Build context 必须是 project/ 父目录（docker-compose 已正确配置）：
#   cd /home/zhangchi/project
#   docker build -f voice_system_livekit/Dockerfile -t voice-system .
# ==============================================================
FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CONDA_DIR=/opt/conda \
    PATH="/opt/conda/bin:$PATH"

# ----------------------------------------------------------
# 1. 系统依赖
# ----------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget curl git git-lfs \
        build-essential libssl-dev libffi-dev \
        libsndfile1 libportaudio2 portaudio19-dev \
        ffmpeg sox \
        ca-certificates \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------
# 2. Miniconda
# ----------------------------------------------------------
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh \
    && conda clean -afy

# ----------------------------------------------------------
# 3. livekit-server 二进制（离线环境可提前下载后 COPY）
# ----------------------------------------------------------
ARG LIVEKIT_VER=1.8.3
RUN wget -q "https://github.com/livekit/livekit/releases/download/v${LIVEKIT_VER}/livekit_linux_amd64.tar.gz" \
        -O /tmp/lk.tar.gz \
    && tar -xzf /tmp/lk.tar.gz -C /usr/local/bin livekit-server \
    && rm /tmp/lk.tar.gz

# ----------------------------------------------------------
# 4. 先复制 wheel（利用 layer 缓存：依赖不变则不重新安装）
# ----------------------------------------------------------
COPY voice_system_livekit/python_packages/ /opt/python_packages/

# ----------------------------------------------------------
# 5. 复制代码（模型目录由 .dockerignore 排除）
# ----------------------------------------------------------
WORKDIR /app
COPY voice_system_livekit/ /app/voice_system_livekit/
COPY answerAgent/fawtd_agent/ /app/fawtd_agent/

# ----------------------------------------------------------
# 6. conda 环境 voice_system（KWS / STT / TTS）
# ----------------------------------------------------------
RUN conda create -n voice_system python=3.12 -y \
    && conda run -n voice_system pip install --upgrade pip \
    && conda run -n voice_system pip install --no-index \
        /opt/python_packages/torch-2.7.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl \
        /opt/python_packages/torchaudio-2.7.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl \
        /opt/python_packages/torchvision-0.22.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl \
        /opt/python_packages/onnxruntime_gpu-1.18.0-cp312-cp312-manylinux_2_28_x86_64.whl \
    && conda run -n voice_system pip install \
        -r /app/voice_system_livekit/requirements.txt \
        -i https://mirrors.aliyun.com/pypi/simple/ \
        --trusted-host mirrors.aliyun.com

# ----------------------------------------------------------
# 7. conda 环境 livekit（stt_llm_agent）
# ----------------------------------------------------------
RUN conda create -n livekit python=3.12 -y \
    && conda run -n livekit pip install --upgrade pip \
    && conda run -n livekit pip install --no-index \
        /opt/python_packages/torch-2.7.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl \
    && conda run -n livekit pip install \
        -e /app/voice_system_livekit/livekit-agents[all] \
        -e /app/voice_system_livekit/livekit-plugins/livekit-plugins-openai \
        -e /app/voice_system_livekit/livekit-plugins/livekit-plugins-silero \
        websockets httpx loguru \
        -i https://mirrors.aliyun.com/pypi/simple/ \
        --trusted-host mirrors.aliyun.com

# ----------------------------------------------------------
# 8. conda 环境 fawbot-agent（fawtd_agent / vLLM）
# ----------------------------------------------------------
RUN conda create -n fawbot-agent python=3.12 -y \
    && conda run -n fawbot-agent pip install --upgrade pip \
    && conda run -n fawbot-agent pip install --no-index \
        /opt/python_packages/torch-2.7.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl \
        /opt/python_packages/torchaudio-2.7.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl \
        /opt/python_packages/torchvision-0.22.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl \
    && conda run -n fawbot-agent pip install \
        -r /app/fawtd_agent/deploy.txt \
        -i https://mirrors.aliyun.com/pypi/simple/ \
        --trusted-host mirrors.aliyun.com \
        --ignore-installed

# ----------------------------------------------------------
# 9. 运行时环境变量默认值（可在 docker run / compose 里覆盖）
# ----------------------------------------------------------
ENV FAWBOT_AGENT_DIR=/app/fawtd_agent \
    LIVEKIT_URL=ws://localhost:7880 \
    LIVEKIT_API_KEY=devkey \
    LIVEKIT_API_SECRET=secret

WORKDIR /app/voice_system_livekit
CMD ["bash", "scripts/start_all_systems.sh"]
