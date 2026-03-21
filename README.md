# voice_system_livekit

基于 LiveKit 的语音对话系统：STT、LLM、TTS 与房间调度。

## 启动顺序

在项目根目录下执行。

| 服务 | 命令 |
|------|------|
| **LiveKit Server** | `livekit-server --dev` |
| **LiveKit Server（局域网）** | `livekit-server --dev --bind 0.0.0.0` |
| **Agent** | `uv run server/stt_llm_agent.py start` |
| **STT** | `conda activate voice_system` → `python stt/stt_server_novad.py` |
| **TTS** | `conda activate cosyvoice-vllm` → `python tts/tts_server.py` |

## 目录结构

```
├── client/           # 客户端（人脸/唤醒/连房等）
├── server/            # Agent 服务（STT+LLM+TTS 编排）
├── stt/               # 语音识别服务与配置
├── tts/               # 语音合成服务与配置
├── kws/               # 关键词唤醒
├── livekit-agents/    # Agent 逻辑、会话、唤醒策略等
├── livekit-plugins/   # 插件（Silero VAD、OpenAI 等）
├── docs/              # 文档
├── scripts/           # 脚本
└── awake_result/      # 唤醒结果等
```

## 注意事项（克隆与 Git LFS）

本仓库部分大文件通过 **Git LFS** 存储，克隆后若未拉取 LFS 对象，本地只会看到很小的指针文件，**无法**正常安装或使用离线 wheel、STT 辅助模型权重等。

**新机器上建议按顺序执行：**

```bash
git lfs install
git clone <本仓库 URL>
cd voice_system_livekit
git lfs pull
```

**当前走 LFS 的典型路径（见 `.gitattributes`）：**

- `python_packages/*.whl` — 离线 PyTorch / ONNX Runtime GPU 等 wheel（体积约 1GB+）
- `stt/model/models/**/*.pt` — FunASR 依赖的 VAD / 标点等 `.pt` 权重

若 `git lfs pull` 失败，请确认已安装 [Git LFS](https://git-lfs.com/)，且对仓库有读权限；LFS 带宽/配额受 Git 托管平台限制。

**环境与依赖：** 根目录 `install.sh` 会创建/使用 `voice_system` conda 环境，并在存在完整 wheel 时**优先**从 `python_packages/` 安装 PyTorch 与 ONNX Runtime GPU，再 `pip install -r requirements.txt`。

## Docker 部署（快速迁移到新机器）

对于展示机或 5090 机器，推荐用 Docker 镜像实现环境零配置迁移。

### 工作流

```
开发机                          演示机
------                          ------
1. git clone + 准备好所有模型
2. bash scripts/build_and_export.sh
   → /tmp/voice-system.tar.gz
3. scp /tmp/voice-system.tar.gz  →  演示机:/tmp/
                                 4. bash scripts/deploy.sh --load /tmp/voice-system.tar.gz
                                    → 导入镜像 + 检查模型目录 + docker compose up -d
```

### 第一步：开发机打包镜像

```bash
# 在 voice_system_livekit/ 目录下
bash scripts/build_and_export.sh
# 输出：/tmp/voice-system.tar.gz（约 15~20 GB）
```

### 第二步：传输模型文件

模型文件体积较大，不打进镜像，需要单独传输。建议用 `rsync`：

```bash
# TTS 模型（~2GB）
rsync -avz --progress tts/model/Fun-CosyVoice3-0.5B/ user@demo:/data/models/Fun-CosyVoice3-0.5B/
# STT 模型
rsync -avz --progress stt/model/ user@demo:/data/models/stt/
# vLLM 模型（~16GB，Qwen3-8B）
rsync -avz --progress /path/to/Qwen3-8B/ user@demo:/data/models/Qwen3-8B/
# Embedding 模型（BAAI/bge-small-zh-v1.5）
rsync -avz --progress ~/.cache/huggingface/hub/ user@demo:/data/models/hub/
# TTS 音色 WAV
rsync -avz --progress tts/assets/ user@demo:/data/tts_assets/
```

### 第三步：演示机启动

```bash
# 克隆代码（只需代码，不需要模型）
git clone git@github.com:zby1218/voice_system_livekit.git
cd voice_system_livekit

# 导入并启动（自动检查模型目录）
bash scripts/deploy.sh --load /tmp/voice-system.tar.gz

# 查看日志
docker logs -f voice-system
```

如果模型不在默认的 `/data/models/` 路径下，可通过环境变量覆盖：

```bash
TTS_MODEL_DIR=/your/path/Fun-CosyVoice3-0.5B \
VLLM_MODEL_DIR=/your/path/Qwen3-8B \
bash scripts/deploy.sh --load /tmp/voice-system.tar.gz
```

### 演示机前置条件

| 依赖 | 安装方法 |
|------|---------|
| Docker Engine | `curl -fsSL https://get.docker.com \| sh` |
| docker-compose-plugin | `apt install docker-compose-plugin` |
| NVIDIA Container Toolkit | [官方文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) |
