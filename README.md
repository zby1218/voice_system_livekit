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
