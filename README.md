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
