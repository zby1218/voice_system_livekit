#!/usr/bin/env python3
"""
STT + LLM Agent (使用 AgentServer) - 语音识别后调用 LLM 并打印结果
启动: python stt_llm_agent.py dev
"""
from __future__ import annotations

import asyncio
import inspect
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
    get_llm_request_at,
    init_pipeline_logging,
    log_llm_raw_chunk,
    log_module,
    log_stt,
    on_tts_first_frame_pushed,
    record_tts_segment_sent_time,
    record_llm_request_time,
    record_llm_first_chunk_time,
    record_stt_result_time,
    record_tts_request_time,
    set_e2e_start,
    set_rtc_room,
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
from time_greeting import ruibo_welcome_message


# ---------------------------------------------------------------------------
# 功能开关
# ---------------------------------------------------------------------------

# KWS（关键词唤醒）开关：True 时在 track_subscribed 回调中启动 AlwaysOnKWSListener；
# False 时跳过 KWS，适用于全程人脸唤醒、不需要唤醒词的场景。
ENABLE_KWS: bool = True

# TTS 音色选择：对应 tts_server.py VOICE_CONFIGS 中的 id 字段。
# 可选值: "default" | "robot_bazong" | "robot" | "longanhuan"
TTS_VOICE: str = "default"


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
                        # session.say("我在听", allow_interruptions=False)
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
        role = getattr(item, "role", None)
        if role == "user":
            # 尽量贴近 on_end_of_turn：用户话轮提交到对话上下文后，视为“送往 LLM”起点。
            record_llm_request_time(time.time())
            return
        if role == "assistant" and hasattr(item, "text_content"):
            text = (getattr(item, "text_content") or "").strip()
            if text:
                return

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
                llm_request_at = get_llm_request_at()
                if llm_request_at is not None:
                    log_module("送往LLM → LLM首包(墙钟)", max(0.0, time.time() - llm_request_at))
                else:
                    # 兜底兼容：若未拿到“送往LLM”时刻，退回到 STT 最终结果时刻。
                    stt_at = get_e2e_stt_result_at()
                    if stt_at is not None:
                        log_module("STT结果 → LLM首包(墙钟,兜底)", max(0.0, time.time() - stt_at))
        elif isinstance(m, TTSMetrics):
            # TTSMetrics 往往在整段/整轮结束后才到达，不用于首帧关键时延日志，避免“晚打”。
            pass


def _extract_stream_text(chunk: object) -> str:
    """尽量从 LLM stream chunk 提取文本内容（兼容 str / ChatChunk）。"""
    if isinstance(chunk, str):
        return chunk
    delta = getattr(chunk, "delta", None)
    if delta is not None:
        content = getattr(delta, "content", None)
        if isinstance(content, str):
            return content
    return ""


class _LLMStreamProbe:
    """包装底层 LLM stream，打印前三个原始文本块。"""

    def __init__(self, inner_stream: object) -> None:
        self._inner_stream = inner_stream
        self._aiter = inner_stream.__aiter__()
        self._count = 0

    def __aiter__(self) -> "_LLMStreamProbe":
        return self

    async def __anext__(self) -> object:
        chunk = await self._aiter.__anext__()
        text = (_extract_stream_text(chunk) or "").strip()
        if text:
            self._count += 1
            if self._count <= 3:
                llm_request_at = get_llm_request_at()
                elapsed = None
                if llm_request_at is not None:
                    elapsed = max(0.0, time.time() - llm_request_at)
                log_llm_raw_chunk(self._count, text, elapsed)
        return chunk

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner_stream, name)


class _LLMChatContextProbe:
    """包装 LLM chat 的异步上下文管理器，进入后再包 stream。"""

    def __init__(self, inner_ctx: object) -> None:
        self._inner_ctx = inner_ctx

    async def __aenter__(self) -> _LLMStreamProbe:
        stream = await self._inner_ctx.__aenter__()
        return _LLMStreamProbe(stream)

    async def __aexit__(self, exc_type, exc, tb) -> object:
        return await self._inner_ctx.__aexit__(exc_type, exc, tb)

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner_ctx, name)


def _install_llm_stream_probe(llm_obj: openai.LLM) -> openai.LLM:
    """给 LLM.chat 安装原始流块探针，不改变原有行为。"""
    original_chat = llm_obj.chat

    def chat_with_probe(*args, **kwargs):
        chat_ret = original_chat(*args, **kwargs)
        # LiveKit 当前路径使用 async with activity_llm.chat(...)
        if hasattr(chat_ret, "__aenter__") and hasattr(chat_ret, "__aexit__"):
            return _LLMChatContextProbe(chat_ret)
        # 兼容：若直接返回可异步迭代 stream
        if hasattr(chat_ret, "__aiter__"):
            return _LLMStreamProbe(chat_ret)
        # 兼容：若返回 awaitable，等待后再按类型包裹
        if inspect.isawaitable(chat_ret):
            async def _await_and_wrap():
                resolved = await chat_ret
                if hasattr(resolved, "__aenter__") and hasattr(resolved, "__aexit__"):
                    return _LLMChatContextProbe(resolved)
                return _LLMStreamProbe(resolved)
            return _await_and_wrap()
        return chat_ret

    llm_obj.chat = chat_with_probe  # type: ignore[method-assign]
    return llm_obj


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

    my_tts = _build_tts(TTS_VOICE)
    my_llm = openai.LLM(
        model="qwen-plus-2025-12-01",
        base_url="http://localhost:8001/v1",
        # base_url="http://192.168.68.113:8001/v1",
        # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        # api_key="sk-4ece3e68f5654afc8646e2fe6aabcfdd",
        api_key="1"
    )
    my_llm = _install_llm_stream_probe(my_llm)
    return my_stt, my_tts, my_llm


def _build_tts(voice: str) -> CosyVoiceTTS:
    """构造一个 CosyVoiceTTS 实例。voice 为空时回退到模块级默认值 TTS_VOICE。"""
    return CosyVoiceTTS(
        base_url="http://localhost:50000",
        endpoint="sft",
        voice=voice or TTS_VOICE,
        sample_rate=24000,
        num_channels=1,
        max_chars=140,
        min_chars=25,
        first_audio_deadline_s=60.0,
        segment_deadline_s=180.0,
        total_timeout_s=600.0,
        add_silence_ms=80,
        on_first_frame_pushed=lambda: on_tts_first_frame_pushed(time.time()),
        on_tts_request_sent=lambda: record_tts_request_time(time.time()),
        on_first_segment_ready=lambda: record_tts_segment_sent_time(time.time()),
    )


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
        # activation_threshold和deactivation_threshold都是通过silero的语音概率p值，判断是否存在语音。
        # activation_threshold: 激活阈值，高于该值认为有语音，打断的阈值也基于此，大于activation_threshold且持续0.5s才会打断
        activation_threshold=0.7,
        # deactivation_threshold: 非激活阈值，低于该值认为没有语音
        deactivation_threshold=0.3,
        # min_speech_duration: 最小语音持续时间，低于该值认为没有语音
        min_speech_duration=0.2,
        # min_silence_duration: 最小静默持续时间，低于该值认为有语音
        min_silence_duration=0.3,
    )


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    set_rtc_room(ctx.room)
    participant = await ctx.wait_for_participant()
    participant_identity = participant.identity

    # 按 client 声明的 tts_voice 属性构造本次会话专用的 TTS 实例；
    # 未携带该属性时使用服务端默认音色（TTS_VOICE）。
    _tts_voice = participant.attributes.get("tts_voice", "").strip()
    if _tts_voice:
        logging.info("[TTS] 使用客户端指定音色: %s (participant=%s)", _tts_voice, participant_identity)
    session_tts = _build_tts(_tts_voice)

    session = AgentSession(
        stt=MY_STT,
        llm=MY_LLM,
        tts=session_tts,
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

        session.say(ruibo_welcome_message(), allow_interruptions=False)

        # 双重判断：server 常量 ENABLE_KWS 且 client 属性 kws_enabled=true 时才启动
        client_kws_enabled = pub_participant.attributes.get("kws_enabled", "true") == "true"
        if ENABLE_KWS and client_kws_enabled:
            kws_task = asyncio.create_task(
                kws_listener.run(ctx.room, participant_identity, publication.track, session)
            )
        else:
            logging.info("[KWS] 已跳过（ENABLE_KWS=%s, client kws_enabled=%s）", ENABLE_KWS, client_kws_enabled)
        
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
