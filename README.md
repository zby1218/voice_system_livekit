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

**代码是否已经在镜像里？** 是的。`Dockerfile` 里已通过 `COPY` 把 `voice_system_livekit/` 与 `answerAgent/fawtd_agent/` 打进镜像；演示机**不需要**再挂一份源码目录来跑服务。之前单独挂载模型，只是为了**减小镜像体积**、方便只更新权重。

构建上下文必须是 **`voice_system_livekit` 的上一级目录**（例如 `project/`，其中并列存在 `voice_system_livekit` 与 `answerAgent/fawtd_agent`）。Docker 只认该目录下的 `.dockerignore`：本仓库用 `scripts/ensure_dockerignore.sh` 在构建前把 `docker/context-project*.dockerignore` 复制为 `project/.dockerignore`，避免误用子目录里的占位说明。

### 两种部署方式

| 方式 | 构建命令 | 演示机启动 | 说明 |
|------|----------|------------|------|
| **精简镜像 + 宿主机挂模型** | `bash scripts/build_and_export.sh` | `bash scripts/deploy.sh --load …/voice-system.tar.gz` | 镜像约 15～20GB；演示机需准备 `/data/models/…`（见 `docker-compose.yml`） |
| **全量镜像（最省事）** | `BUNDLE_MODELS=1 bash scripts/build_and_export.sh` | `bash scripts/deploy.sh --embedded --load …/voice-system.tar.gz` | 镜像含本地已有的大权重，体积可达数十 GB；演示机**不必**再 rsync 模型 |

全量模式会把你开发机 `project/` 里**已经存在**的模型目录一并 `COPY` 进镜像（前提是未被 `docker/context-project-bundle.dockerignore` 排除）。若某路径在本机为空，镜像里同样为空。

### 工作流（精简镜像）

```
开发机                          演示机
------                          ------
1. 并列放好 voice_system_livekit + answerAgent/fawtd_agent
2. bash scripts/build_and_export.sh → /tmp/voice-system.tar.gz
3. rsync 模型到演示机 /data/models/…
4. scp 镜像 tar 到演示机
                                 5. 浅克隆仓库（只要 compose + 脚本）
                                 6. bash scripts/deploy.sh --load /tmp/voice-system.tar.gz
```

### 工作流（全量镜像）

```
开发机：BUNDLE_MODELS=1 bash scripts/build_and_export.sh
演示机：scp 大 tar → bash scripts/deploy.sh --embedded --load …
```

演示机仍需本仓库中的 `docker-compose.embedded.yml` 与 `scripts/deploy.sh`（可用 `git clone --depth 1`）；**不必**再在宿主机准备模型目录。

### 演示机前置条件

| 依赖 | 安装方法 |
|------|---------|
| Docker Engine | `curl -fsSL https://get.docker.com \| sh` |
| docker-compose-plugin | `apt install docker-compose-plugin` |
| NVIDIA Container Toolkit | [官方文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) |

### 宿主机模型路径（仅精简模式）

若使用默认 `docker-compose.yml`，可用 `rsync` 将模型同步到演示机，或通过环境变量覆盖挂载路径，例如：

```bash
TTS_MODEL_DIR=/your/path/Fun-CosyVoice3-0.5B \
VLLM_MODEL_DIR=/your/path/Qwen3-8B \
bash scripts/deploy.sh --load /tmp/voice-system.tar.gz
```
