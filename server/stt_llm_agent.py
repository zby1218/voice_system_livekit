#!/usr/bin/env python3
"""
STT + LLM Agent (使用 AgentServer) - 语音识别后调用 LLM 并打印结果
启动: python stt_llm_agent.py dev
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

# 避免 socks 代理导致 httpx 报错 (Unknown scheme for proxy URL socks://...)
for _p in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_p, None)

import websockets

from pipeline_logger import (
    get_e2e_stt_result_at,
    init_pipeline_logging,
    log_e2e,
    log_llm,
    log_module,
    log_stt,
    log_tts,
    log_tts_request_to_first_frame,
    record_e2e_end,
    record_llm_first_chunk_time,
    record_stt_result_time,
    record_tts_request_time,
    set_e2e_start,
)

# 项目根目录与日志目录
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_log_dir = os.path.join(_project_root, "log", "agent")
os.makedirs(_log_dir, exist_ok=True)

init_pipeline_logging(_log_dir, "pipeline.log")

for _name in (
    "livekit",
    "livekit.agents",
    "livekit.agents.voice.audio_recognition",
    "livekit.agents.voice.agent_activity",
    "my_stt",
    "ASRServer",
    "custom_tts",
    "playback_boundary",
):
    logging.getLogger(_name).setLevel(logging.WARNING)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stt"))
sys.path.insert(1, os.path.join(os.path.dirname(__file__), "..", "tts"))

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    cli,
)
from livekit.agents.metrics import LLMMetrics, STTMetrics, TTSMetrics
from livekit.agents.voice.events import (
    ConversationItemAddedEvent,
    MetricsCollectedEvent,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
)
from livekit.agents.voice.room_io import AudioInputOptions, RoomOptions
from livekit.plugins import openai, silero

from custom_stt import MySTT
from custom_tts import CosyVoiceTTS
from qwen_stt import QwenSTT


# ---------------------------------------------------------------------------
# 配置：集中管理路径与 KWS 参数，便于扩展与测试
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AgentConfig:
    """Agent 运行所需配置（路径、KWS、LiveKit 环境等）。"""

    project_root: str
    log_dir: str
    kws_ws_uri: str
    kws_sample_rate: int
    kws_frame_size_ms: int


def _load_config() -> AgentConfig:
    return AgentConfig(
        project_root=_project_root,
        log_dir=_log_dir,
        kws_ws_uri="ws://localhost:8765",
        kws_sample_rate=16000,
        kws_frame_size_ms=150,
    )


# ---------------------------------------------------------------------------
# 常驻 KWS 监听器：独立模块，仅依赖 session/room/track，便于单测与复用
# ---------------------------------------------------------------------------
class AlwaysOnKWSListener:
    """从同一路麦克风轨道创建独立 AudioStream，持续送 KWS；检测到唤醒词后打断并说「我在听」。"""

    def __init__(self, config: AgentConfig) -> None:
        self._config = config

    async def run(
        self,
        room: rtc.Room,
        participant_identity: str,
        track: rtc.RemoteTrack,
        session: AgentSession,
    ) -> None:
        stream = rtc.AudioStream.from_track(
            track=track,
            sample_rate=self._config.kws_sample_rate,
            num_channels=1,
            frame_size_ms=self._config.kws_frame_size_ms,
        )
        try:
            async with websockets.connect(self._config.kws_ws_uri) as ws:
                await asyncio.gather(
                    self._send_loop(stream, ws),
                    self._recv_loop(ws, session),
                )
        except Exception as e:
            logging.warning("[KWS always-on] connection/listener ended: %s", e)

    async def _send_loop(self, stream: rtc.AudioStream, ws: websockets.WebSocketClientProtocol) -> None:
        try:
            async for event in stream:
                await ws.send(event.frame.data.tobytes())
        except Exception as e:
            logging.warning("[KWS always-on] send_loop ended: %s", e)

    async def _recv_loop(self, ws: websockets.WebSocketClientProtocol, session: AgentSession) -> None:
        try:
            while True:
                msg = await ws.recv()
                data = json.loads(msg)
                if data.get("type") == "wake_detected" and data.get("success"):
                    logging.info("[KWS always-on] 唤醒词检测到，执行打断")
                    activity = getattr(session, "_activity", None)
                    if activity is None:
                        logging.warning("[KWS always-on] session._activity 未就绪，跳过")
                        continue
                    try:
                        session.clear_user_turn()
                    except Exception as e:
                        logging.warning("[KWS always-on] clear_user_turn 失败: %s", e)
                    try:
                        activity.interrupt(force=True)
                        session.say("我在听", allow_interruptions=False)
                    except Exception as e:
                        logging.warning("[KWS always-on] 打断失败: %s", e)
        except Exception as e:
            logging.warning("[KWS always-on] recv_loop ended: %s", e)


# ---------------------------------------------------------------------------
# 流水线事件处理：封装 E2E 打点与日志状态，避免模块级全局变量
# ---------------------------------------------------------------------------
class PipelineEventHandler:
    """处理 session 的 user_input_transcribed / conversation_item_added / metrics_collected，打流水线 log。"""

    def __init__(self) -> None:
        self._last_stt_duration: Optional[float] = None
        self._last_llm_ttft: Optional[float] = None
        self._llm_ttft_logged: bool = False

    def on_user_input_transcribed(self, ev: UserInputTranscribedEvent) -> None:
        if ev.is_final and (ev.transcript or "").strip():
            self._llm_ttft_logged = False
            record_stt_result_time(time.time())
            log_stt(ev.transcript)

    def on_conversation_item_added(self, ev: ConversationItemAddedEvent) -> None:
        item = ev.item
        if getattr(item, "role", None) == "assistant" and hasattr(item, "text_content"):
            text = (getattr(item, "text_content") or "").strip()
            if text:
                log_llm(text)

    def on_metrics_collected(self, ev: MetricsCollectedEvent) -> None:
        m = ev.metrics
        if isinstance(m, STTMetrics):
            self._last_stt_duration = m.duration
            log_module("STT推理", m.duration)
        elif isinstance(m, LLMMetrics):
            if not self._llm_ttft_logged:
                self._llm_ttft_logged = True
                self._last_llm_ttft = m.ttft
                record_llm_first_chunk_time(time.time())
                log_module("LLM首包", m.ttft)
                stt_at = get_e2e_stt_result_at()
                if stt_at is not None:
                    log_module("STT结果 → LLM首包(墙钟)", max(0.0, time.time() - stt_at))
        elif isinstance(m, TTSMetrics):
            log_tts_request_to_first_frame()
            log_tts("", duration_s=m.ttfb, chars=m.characters_count)
            log_e2e(
                stage_stt=self._last_stt_duration,
                stage_llm=self._last_llm_ttft,
                stage_tts=m.ttfb,
            )


# ---------------------------------------------------------------------------
# 流水线组件工厂：STT / TTS / LLM 构建集中在一处，便于配置与替换
# ---------------------------------------------------------------------------
def build_pipeline_components(config: AgentConfig) -> tuple[MySTT, CosyVoiceTTS, openai.LLM]:
    """根据配置构建 STT、TTS、LLM 实例（含 E2E 打点回调）。"""
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
    QwenSTT(
        host="localhost",
        port=10096,
        ssl=False,
        context="",
        language=None,
        chunk_interval_ms=60,
    )
    my_tts = CosyVoiceTTS(
        base_url="http://localhost:50000",
        endpoint="sft",
        voice="default",
        sample_rate=24000,
        num_channels=1,
        max_chars=140,
        min_chars=25,
        first_audio_deadline_s=60.0,
        segment_deadline_s=180.0,
        total_timeout_s=600.0,
        add_silence_ms=80,
        on_first_frame_pushed=lambda: record_e2e_end(time.time()),
        on_tts_request_sent=lambda: record_tts_request_time(time.time()),
    )
    my_llm = openai.LLM(
        model="qwen-plus-2025-12-01",
        base_url="http://localhost:8001/v1",
        # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="sk-4ece3e68f5654afc8646e2fe6aabcfdd",
        # api_key="1",
    )
    return my_stt, my_tts, my_llm


# ---------------------------------------------------------------------------
# LiveKit 环境与全局组件（由配置与工厂生成，供 entrypoint 使用）
# ---------------------------------------------------------------------------
os.environ.setdefault("LIVEKIT_URL", "ws://localhost:7880")
os.environ.setdefault("LIVEKIT_API_KEY", "devkey")
os.environ.setdefault("LIVEKIT_API_SECRET", "secret")

CONFIG = _load_config()
MY_STT, MY_TTS, MY_LLM = build_pipeline_components(CONFIG)


# ---------------------------------------------------------------------------
# Agent 定义与 Server
# ---------------------------------------------------------------------------
class MyAgent(Agent):
    """简单 Agent：STT + LLM + TTS，用简洁中文回答。"""

    def __init__(self) -> None:
        super().__init__(
            instructions="你是一个友好的助手，用简洁的中文回答问题。保持回复简短。",
        )


server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    """预热：加载 VAD 模型。"""
    proc.userdata["vad"] = silero.VAD.load(
        activation_threshold=0.5,
        deactivation_threshold=0.3,
        min_speech_duration=0.2,
        min_silence_duration=0.5,
    )


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()
    participant_identity = participant.identity

    session = AgentSession(
        stt=MY_STT,
        llm=MY_LLM,
        tts=MY_TTS,
        vad=ctx.proc.userdata["vad"],
        allow_interruptions=False,
        # TTS 播放期间仍保留用户音频送 STT，避免用户开口时机被丢失
        discard_audio_if_uninterruptible=False,
    )

    pipeline_handler = PipelineEventHandler()
    session.on("user_input_transcribed", pipeline_handler.on_user_input_transcribed)
    session.on("conversation_item_added", pipeline_handler.on_conversation_item_added)
    session.on("metrics_collected", pipeline_handler.on_metrics_collected)

    # 休眠逻辑（pip 版无内置）：用户 away 时播休眠语并通知 client，不修改源码
    room = ctx.room

    def _on_user_state_changed(ev: UserStateChangedEvent) -> None:
        if ev.new_state != "away":
            return
        logging.info("用户状态变为 away，进入休眠")
        session.say("我先休息了，有事再叫我吧", allow_interruptions=False)
        asyncio.create_task(_notify_client_sleep(room))

    async def _notify_client_sleep(r: rtc.Room) -> None:
        try:
            await r.local_participant.publish_data(b"session_end", reliable=True)
            logging.debug("已通知 client 进入休眠")
        except Exception as e:
            logging.warning("通知 client 休眠失败: %s", e)

    session.on("user_state_changed", _on_user_state_changed)

    kws_listener = AlwaysOnKWSListener(CONFIG)
    kws_task: Optional[asyncio.Task[None]] = None

    @ctx.room.on("track_subscribed")
    def _on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        pub_participant: rtc.RemoteParticipant,
    ) -> None:
        nonlocal kws_task
        if pub_participant.identity != participant_identity:
            return
        if publication.source != rtc.TrackSource.SOURCE_MICROPHONE:
            return
        if not publication.track:
            return
        if kws_task is not None and not kws_task.done():
            return
        session.say("你好~", allow_interruptions=False)
        kws_task = asyncio.create_task(
            kws_listener.run(ctx.room, participant_identity, publication.track, session)
        )
        
    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_options=RoomOptions(
            audio_input=AudioInputOptions(sample_rate=16000, frame_size_ms=150),
        ),
    )

    # 连接建立后播一句「我在呢」
    # session.say("你好~", allow_interruptions=False)


if __name__ == "__main__":
    cli.run_app(server)
