#!/usr/bin/env python3
"""
STT + LLM Agent (使用 AgentServer) - 语音识别后调用 LLM 并打印结果
启动: python stt_llm_agent.py dev
"""
import os
import sys
import time
import numpy as np
import asyncio
import logging
from typing import Optional

from pipeline_logger import (
    init_pipeline_logging,
    log_stt,
    log_llm,
    log_tts,
    log_module,
    log_e2e,
    set_e2e_start,
    record_stt_result_time,
    record_e2e_end,
    record_llm_first_chunk_time,
    record_tts_request_time,
    log_tts_request_to_first_frame,
    get_e2e_stt_result_at,
)

# 项目根目录
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_log_dir = os.path.join(_project_root, "log", "agent")
os.makedirs(_log_dir, exist_ok=True)

# 流水线日志：控制台 + log/agent/pipeline.log，带时间戳
init_pipeline_logging(_log_dir, "pipeline.log")

# 播放边界打点：log/agent/playback_boundary.log，用于分析残留播放位置（agent 侧）
try:
    from livekit.agents.voice.playback_boundary_log import init_playback_boundary_file_logging
    init_playback_boundary_file_logging(_log_dir, "playback_boundary.log")
    logging.getLogger("playback_boundary").info("PlaybackBoundary 日志将写入: %s", os.path.join(_log_dir, "playback_boundary.log"))
except Exception as e:
    logging.warning("PlaybackBoundary 文件日志未初始化: %s", e)

# 抑制所有 livekit/agents 的 INFO，只保留本进程的 pipeline 流水线日志，避免刷屏
for _name in (
    "livekit",
    "livekit.agents",
    "livekit.agents.voice.audio_recognition",
    "livekit.agents.voice.agent_activity",
    "my_stt",
    "ASRServer",
    "custom_tts",
):
    logging.getLogger(_name).setLevel(logging.WARNING)

# 添加 stt 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stt"))
sys.path.insert(1, os.path.join(os.path.dirname(__file__), "..", "tts"))
from livekit import rtc
from livekit.agents import (
    Agent,
    AutoSubscribe,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
)
from livekit.agents.voice.room_io import RoomOptions, AudioInputOptions
from livekit.agents.voice.events import (
    UserInputTranscribedEvent,
    ConversationItemAddedEvent,
    MetricsCollectedEvent,
    E2ETimingEvent,
)
from livekit.agents.metrics import LLMMetrics, STTMetrics, TTSMetrics





from livekit.plugins import silero, openai

# 导入自定义 STT
from custom_stt import MySTT
from qwen_stt import QwenSTT
from custom_tts import CosyVoiceTTS

current_dir = os.path.dirname(os.path.abspath(__file__))
PROMPT_WAV_PATH = os.path.join(current_dir, "..", "tts", "assets", "zero_shot_prompt.wav")

# LiveKit 本地开发环境
os.environ.setdefault("LIVEKIT_URL", "ws://localhost:7880")
os.environ.setdefault("LIVEKIT_API_KEY", "devkey")
os.environ.setdefault("LIVEKIT_API_SECRET", "secret")

# logger = logging.getLogger("stt-llm-agent")


# ========== STT 实例 ==========
# on_segment_submitted: 音频段提交给 STT 时回调，用作 E2E 起点（从给 STT 到 TTS 首帧）
my_stt = MySTT(
    host="localhost",
    port=10095,
    ssl=False,
    mode="2pass",
    chunk_size=[5, 10, 5],
    chunk_interval=10,
    encoder_chunk_look_back=4,
    decoder_chunk_look_back=0,
    itn=True,
    hotwords="",
    on_segment_submitted=lambda: set_e2e_start(time.time()),
)

qwen_stt = QwenSTT(
    host="localhost",
    port=10096,
    ssl=False,
    context="",
    language=None,
    chunk_interval_ms=60
)

# ========== TTS 实例 (CosyVoice) ==========
my_tts = CosyVoiceTTS(
    base_url="http://localhost:50000",
    endpoint="zero_shot",
    prompt_wav_path=PROMPT_WAV_PATH,
    prompt_text="You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
    sample_rate=24000,
    num_channels=1,
    max_chars=140,
    min_chars=25,
    first_audio_deadline_s=60.0,
    segment_deadline_s=180.0,
    total_timeout_s=600.0,
    add_silence_ms=80,
    on_first_frame_pushed=lambda: record_e2e_end(time.time()),  # TTS 首帧 push 时记 E2E 终点
    on_tts_request_sent=lambda: record_tts_request_time(time.time()),  # 向 TTS 发请求时打点，与首帧相减=请求→首帧
)

# ========== LLM 实例 (阿里云 Dashscope Qwen) ==========
my_llm = openai.LLM(
    model="qwen-plus-2025-12-01",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # base_url="http://192.168.68.65:8001/v1",
    api_key="sk-4ece3e68f5654afc8646e2fe6aabcfdd"
    # api_key="",

)


class MyAgent(Agent):
    """简单的 Agent，只做 STT + LLM，不做 TTS"""
    
    def __init__(self) -> None:
        super().__init__(
            instructions="你是一个友好的助手，用简洁的中文回答问题。保持回复简短。",
        )


# ---------- 流水线事件：只打 STT/LLM/TTS/耗时/E2E/VAD ----------
# E2E 起点在 custom_stt 发送 is_speaking=False 时由 on_segment_submitted 设置
_last_stt_duration: Optional[float] = None
_last_llm_ttft: Optional[float] = None  # STT结果→首chunk 耗时，用于 E2E 阶段和
_llm_ttft_logged: bool = False


def _on_user_input_transcribed(ev: UserInputTranscribedEvent) -> None:
    global _llm_ttft_logged
    if ev.is_final and (ev.transcript or "").strip():
        _llm_ttft_logged = False
        record_stt_result_time(time.time())
        log_stt(ev.transcript)


def _on_e2e_timing(ev: E2ETimingEvent) -> None:
    """LLM 流式首 chunk 时刻：记录并打 [耗时-LLM首包] = 首chunk时刻 - STT结果时刻。"""
    global _last_llm_ttft, _llm_ttft_logged
    if ev.llm_first_token is None or _llm_ttft_logged:
        return
    _llm_ttft_logged = True
    record_llm_first_chunk_time(ev.llm_first_token)
    stt_at = get_e2e_stt_result_at()
    if stt_at is not None:
        duration = max(0.0, ev.llm_first_token - stt_at)
        _last_llm_ttft = duration
        log_module("LLM首包", duration)


def _on_conversation_item_added(ev: ConversationItemAddedEvent) -> None:
    item = ev.item
    if getattr(item, "role", None) == "assistant" and hasattr(item, "text_content"):
        text = (getattr(item, "text_content") or "").strip()
        if text:
            log_llm(text)


def _on_metrics_collected(ev: MetricsCollectedEvent) -> None:
    global _last_stt_duration, _last_llm_ttft, _llm_ttft_logged
    m = ev.metrics
    if isinstance(m, STTMetrics):
        _last_stt_duration = m.duration
        log_module("STT推理", m.duration)
    elif isinstance(m, LLMMetrics):
        if not _llm_ttft_logged:
            _last_llm_ttft = m.ttft
            _llm_ttft_logged = True
            record_llm_first_chunk_time(time.time())
            log_module("LLM首包", m.ttft)
    elif isinstance(m, TTSMetrics):
        log_tts_request_to_first_frame()  # 请求时刻→首帧时刻 的墙钟耗时
        log_tts("", duration_s=m.ttfb, chars=m.characters_count)
        log_e2e(
            stage_stt=_last_stt_duration,
            stage_llm=_last_llm_ttft,
            stage_tts=m.ttfb,
        )


# ========== AgentServer 方式 (和 myagent.py 一致) ==========
server = AgentServer()


def prewarm(proc: JobProcess):
    """预热：加载 VAD 模型"""
    proc.userdata["vad"] = silero.VAD.load(
        activation_threshold=0.5, # 超过0.5判断在说话
        deactivation_threshold=0.3, # 超过0.3判断说话结束
        min_speech_duration=0.2, # 超过0.2s判断在说话
        min_silence_duration=0.5, # 超过0.5s判断说话结束，进行识别
    )


server.setup_fnc = prewarm



@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # 1. 连接房间
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    # 2. 等待用户 (这一步其实 session.start 内部也会做，但写在这里更稳妥)
    participant = await ctx.wait_for_participant()
    attributes = participant.attributes or {}
    # 人脸唤醒 (wake_source=face) 时不走 KWS，直接播「我在呢」并进入 STT/TTS
    kws_enabled = attributes.get("wake_source") != "face"

    # 3. 初始化 Session
    session = AgentSession(
        stt=my_stt,
        llm=my_llm,
        tts=my_tts,
        vad=ctx.proc.userdata["vad"],
        allow_interruptions=True,
        kws_enabled=kws_enabled,
    )

    session.on("user_input_transcribed", _on_user_input_transcribed)
    session.on("conversation_item_added", _on_conversation_item_added)
    session.on("metrics_collected", _on_metrics_collected)
    session.on("e2e_timing", _on_e2e_timing)

    # @session.on("user_state_changed")
    # def on_user_state_changed(ev: UserStateChangedEvent):
    #     if ev.new_state == "away":
    #         print("用户状态转为离开")
    #         session.say("我先休息了, 有事情再叫我吧", allow_interruptions=True)
    #         # asyncio.create_task(
    #         #     session.say("我先休息了", allow_interruptions=True)
    #         # )


        
    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_options=RoomOptions(
            audio_input=AudioInputOptions(sample_rate=16000, frame_size_ms=150),
        ),
    )



    





if __name__ == "__main__":
    cli.run_app(server)
