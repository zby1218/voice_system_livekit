#!/usr/bin/env python3
"""
STT + LLM Agent (使用 AgentServer) - 语音识别后调用 LLM 并打印结果
启动: python stt_llm_agent.py dev
"""
import os
import sys
import time
import json
import websockets
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

# 抑制所有 livekit/agents 的 INFO，只保留本进程的 pipeline 流水线日志，避免刷屏
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

KWS_WS_URI = "ws://localhost:8765"
SAMPLE_RATE = 16000
FRAME_SIZE_MS = 150

async def _run_always_on_kws_listener(
    room: rtc.Room,
    participant_identity: str,
    track: rtc.RemoteTrack,
    session: AgentSession,
) -> None:
    """从同一路麦克风轨道创建独立 AudioStream，持续送 KWS；检测到唤醒词后打断当前播放并说「我在听」。"""
    stream = rtc.AudioStream.from_track(
        track=track,
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        frame_size_ms=FRAME_SIZE_MS,
    )
    try:
        async with websockets.connect(KWS_WS_URI) as ws:
            async def send_loop():
                try:
                    async for event in stream:
                        await ws.send(event.frame.data.tobytes())
                except Exception as e:
                    logging.warning("[KWS always-on] send_loop ended: %s", e)

            async def recv_loop():
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

                            # ① 无论打断是否成功，都必须清空 STT 状态
                            #    防止唤醒词这一句被 STT 识别后提交给 LLM 产生错位回答
                            try:
                                session.clear_user_turn()
                            except Exception as e:
                                logging.warning("[KWS always-on] clear_user_turn 失败: %s", e)

                            # ② 强制打断当前语音并播「我在听」
                            #    force=True：allow_interruptions=False 时也能打断
                            #    唤醒词场景下，用户主动唤醒优先级最高，应无条件打断
                            try:
                                activity.interrupt(force=True)
                                session.say("我在听", allow_interruptions=False)
                            except Exception as e:
                                logging.warning("[KWS always-on] 打断失败: %s", e)
                except Exception as e:
                    logging.warning("[KWS always-on] recv_loop ended: %s", e)

            await asyncio.gather(send_loop(), recv_loop())
    except Exception as e:
        logging.warning("[KWS always-on] connection/listener ended: %s", e)


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
            # pip 无 E2ETimingEvent，用墙钟补算「STT 结果 → 收到 LLM 首包」间隔
            stt_at = get_e2e_stt_result_at()
            if stt_at is not None:
                log_module("STT结果 → LLM首包(墙钟)", max(0.0, time.time() - stt_at))
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
    participant_identity = participant.identity  # 供常驻 KWS 的 track_subscribed 回调使用

    # 3. 初始化 Session（pip 版无 kws_enabled，唤醒与打断由下方常驻 KWS 负责）
    session = AgentSession(
        stt=my_stt,
        llm=my_llm,
        tts=my_tts,
        vad=ctx.proc.userdata["vad"],
        # allow_interruptions=False：agent 播放时用户说话不送 STT，防止误触发 LLM
        # 唤醒词打断通过 always-on KWS + force=True interrupt 实现
        allow_interruptions=False,
    )

    session.on("user_input_transcribed", _on_user_input_transcribed)
    session.on("conversation_item_added", _on_conversation_item_added)
    session.on("metrics_collected", _on_metrics_collected)

    # @session.on("user_state_changed")
    # def on_user_state_changed(ev: UserStateChangedEvent):
    #     if ev.new_state == "away":
    #         print("用户状态转为离开")
    #         session.say("我先休息了, 有事情再叫我吧", allow_interruptions=True)
    #         # asyncio.create_task(
    #         #     session.say("我先休息了", allow_interruptions=True)
    #         # )
    _kws_listener_task: asyncio.Task | None = None
    @ctx.room.on("track_subscribed")
    def _on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        nonlocal _kws_listener_task
        if participant.identity != participant_identity:
            return
        if publication.source != rtc.TrackSource.SOURCE_MICROPHONE:
            return
        if not publication.track:
            return
        # 只起一个常驻 KWS 任务，避免重复
        if _kws_listener_task is not None and not _kws_listener_task.done():
            return
        _kws_listener_task = asyncio.create_task(
            _run_always_on_kws_listener(ctx.room, participant_identity, publication.track, session)
        )
        
    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_options=RoomOptions(
            audio_input=AudioInputOptions(sample_rate=16000, frame_size_ms=150),
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
