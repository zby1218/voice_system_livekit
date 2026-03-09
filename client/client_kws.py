#!/usr/bin/env python3
"""
KWS 唤醒用 LiveKit 客户端：由 project/kws 触发器经 kws_wake_listener 拉起，带 wake_source=kws。
供 kws_wake_listener 调用；Agent 可根据 participant.attributes 识别并走 kws_enabled=False。
与 client_face 对称，无 face resume 逻辑（会话结束即断开）。
"""
import asyncio
import json
import os
import time
import threading
from typing import Optional

import numpy as np
import sounddevice as sd
from livekit import rtc, api

# ========== 配置（与 client_face 一致）==========
MIC_SAMPLE_RATE = 16000
MIC_CHANNELS = 1
MIC_FRAME_SAMPLES = 160
MIC_BLOCK_SIZE = 1600

PLAYBACK_SAMPLE_RATE = 48000
PLAYBACK_CHANNELS = 1
PLAYBACK_BLOCK_SIZE = 4800

ROOM_NAME_PREFIX = "kws_room"

LIVEKIT_URL = "ws://127.0.0.1:7880"
LIVEKIT_API_KEY = "devkey"
LIVEKIT_API_SECRET = "secret"

os.environ.setdefault("LIVEKIT_URL", LIVEKIT_URL)
os.environ.setdefault("LIVEKIT_API_KEY", LIVEKIT_API_KEY)
os.environ.setdefault("LIVEKIT_API_SECRET", LIVEKIT_API_SECRET)

# ========== 播放缓冲区 ==========
playback_buffer = bytearray()
playback_lock = threading.Lock()

def playback_callback(outdata, frames, time_info, status):
    if status:
        print("Playback status: %s" % status)
    bytes_needed = frames * 2
    with playback_lock:
        if len(playback_buffer) >= bytes_needed:
            chunk = bytes(playback_buffer[:bytes_needed])
            del playback_buffer[:bytes_needed]
            outdata[:, 0] = np.frombuffer(chunk, dtype=np.int16)
        else:
            available = len(playback_buffer)
            if available > 0:
                chunk = bytes(playback_buffer[:available])
                playback_buffer.clear()
                samples = np.frombuffer(chunk, dtype=np.int16)
                outdata[:len(samples), 0] = samples
                outdata[len(samples):, 0] = 0
            else:
                outdata[:] = 0


def _ts() -> str:
    t = time.time()
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) + ".%03d" % int((t % 1) * 1000)


SESSION_END_AUDIO_IDLE_SEC = 1.0
SESSION_END_VOICE_THRESHOLD = 200
SESSION_END_AUDIO_WAIT_TIMEOUT = 30.0


async def receive_audio(
    stream: rtc.AudioStream,
    on_stream_ended: Optional[asyncio.Event] = None,
    last_frame_time: Optional[list] = None,
    sleep_speech_state: Optional[dict] = None,
):
    SILENCE_AMP = 1
    try:
        async for event in stream:
            frame = event.frame
            audio_bytes = bytes(frame.data)
            samples = np.frombuffer(audio_bytes, dtype=np.int16)
            max_amplitude = int(np.max(np.abs(samples)))
            if last_frame_time is not None and max_amplitude > SESSION_END_VOICE_THRESHOLD:
                last_frame_time[0] = time.monotonic()
            if sleep_speech_state is not None:
                if sleep_speech_state.get("session_end") and max_amplitude > SESSION_END_VOICE_THRESHOLD:
                    if not sleep_speech_state.get("sleep_speech_started"):
                        print("%s 收到休眠语音" % _ts())
                        sleep_speech_state["sleep_speech_started"] = True
                if sleep_speech_state.get("sleep_speech_started") and not sleep_speech_state.get("sleep_speech_ended_printed") and max_amplitude <= SILENCE_AMP:
                    print("%s 休眠语音结束" % _ts())
                    sleep_speech_state["sleep_speech_ended_printed"] = True
            with playback_lock:
                playback_buffer.extend(audio_bytes)
    finally:
        if on_stream_ended is not None:
            on_stream_ended.set()


async def run_client(
    keyword: str = "",
    *,
    room_name: Optional[str] = None,
):
    """连接 LiveKit 房间并收发音频；wake_source=kws，无 face resume。"""
    room_name = room_name or ("%s_%d" % (ROOM_NAME_PREFIX, int(time.time() * 1000)))
    apm = rtc.AudioProcessingModule(
        echo_cancellation=False,
        noise_suppression=False,
        high_pass_filter=False,
        auto_gain_control=False,
    )
    attrs = {"wake_source": "kws"}
    if keyword:
        attrs["keyword"] = keyword
    token = api.AccessToken(
        os.getenv("LIVEKIT_API_KEY"),
        os.getenv("LIVEKIT_API_SECRET"),
    ).with_identity("kws-mic-user") \
     .with_name("KWS Mic") \
     .with_grants(api.VideoGrants(room_join=True, room=room_name)) \
     .with_attributes(attrs) \
     .to_jwt()

    room = rtc.Room()

    @room.on("disconnected")
    def on_disconnected(reason):
        print("%s 🔌 [Event] Room disconnected: %s" % (_ts(), reason))

    @room.on("connection_state_changed")
    def on_connection_state_changed(state):
        print("%s 🔄 [Event] Connection state changed: %s" % (_ts(), state))

    audio_stream_ended_event = asyncio.Event()
    last_audio_frame_time: list = [0.0]
    sleep_speech_state: dict = {"session_end": False, "sleep_speech_started": False, "sleep_speech_ended_printed": False}
    disconnect_after_audio = False
    session_end_time: Optional[float] = None

    @room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        print("%s 📡 订阅到轨道: %s from %s" % (_ts(), track.kind, participant.identity))
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            audio_stream = rtc.AudioStream(track)
            asyncio.create_task(receive_audio(audio_stream, audio_stream_ended_event, last_audio_frame_time, sleep_speech_state))

    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        print("%s 👤 参与者加入: %s" % (_ts(), participant.identity))

    @room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        print("%s 👋 参与者离开: %s" % (_ts(), participant.identity))

    @room.on("data_received")
    def on_data_received(data: rtc.DataPacket):
        nonlocal disconnect_after_audio, session_end_time
        if data.data == b"session_end":
            print("%s 收到休眠信号" % _ts())
            sleep_speech_state["session_end"] = True
            disconnect_after_audio = True
            session_end_time = time.monotonic()

    print("🔗 Connecting to room: %s ..." % room_name)
    await room.connect(os.getenv("LIVEKIT_URL"), token)
    print("✅ Connected to room: %s" % room.name)

    source = rtc.AudioSource(MIC_SAMPLE_RATE, MIC_CHANNELS)
    track = rtc.LocalAudioTrack.create_audio_track("mic-track", source)
    track_opts = rtc.TrackPublishOptions()
    track_opts.source = rtc.TrackSource.SOURCE_MICROPHONE
    await room.local_participant.publish_track(track, track_opts)
    print("🎤 Microphone track published!")

    loop = asyncio.get_event_loop()

    def mic_callback(indata, frames, time_info, status):
        if status:
            print("Mic status: %s" % status)
        num_frames = frames // MIC_FRAME_SAMPLES
        for i in range(num_frames):
            start = i * MIC_FRAME_SAMPLES
            end = start + MIC_FRAME_SAMPLES
            capture_chunk = indata[start:end]
            frame = rtc.AudioFrame.create(MIC_SAMPLE_RATE, MIC_CHANNELS, MIC_FRAME_SAMPLES)
            np.copyto(np.frombuffer(frame.data, dtype=np.int16), capture_chunk.flatten())
            apm.process_stream(frame)
            asyncio.run_coroutine_threadsafe(source.capture_frame(frame), loop)

    print("\n" + "=" * 50)
    print("🎤 麦克风输入: 已启动")
    print("🔊 扬声器输出: 已启动")
    print("⌨️  按 Ctrl+C 退出")
    print("=" * 50 + "\n")

    try:
        with sd.InputStream(
            channels=MIC_CHANNELS,
            samplerate=MIC_SAMPLE_RATE,
            callback=mic_callback,
            blocksize=MIC_BLOCK_SIZE,
            dtype="int16",
        ), sd.OutputStream(
            channels=PLAYBACK_CHANNELS,
            samplerate=PLAYBACK_SAMPLE_RATE,
            callback=playback_callback,
            blocksize=PLAYBACK_BLOCK_SIZE,
            dtype="int16",
        ):
            while True:
                if disconnect_after_audio:
                    if audio_stream_ended_event.is_set():
                        print("%s 休眠语音已播完，断开连接" % _ts())
                        break
                    if sleep_speech_state.get("sleep_speech_started") and last_audio_frame_time[0] > 0 and (time.monotonic() - last_audio_frame_time[0]) >= SESSION_END_AUDIO_IDLE_SEC:
                        print("%s 休眠语音已发完（%.1fs 无新帧），断开连接" % (_ts(), SESSION_END_AUDIO_IDLE_SEC))
                        break
                    if session_end_time is not None and (time.monotonic() - session_end_time) >= SESSION_END_AUDIO_WAIT_TIMEOUT:
                        print("%s 等待音频流超时 (%.0fs)，断开连接" % (_ts(), SESSION_END_AUDIO_WAIT_TIMEOUT))
                        break
                await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        print("%s 🛑 正在关闭..." % _ts())
    finally:
        print("%s 🔌 断开房间连接..." % _ts())
        await room.disconnect()
        print("%s ✅ 已断开连接" % _ts())


if __name__ == "__main__":
    try:
        asyncio.run(run_client())
    except KeyboardInterrupt:
        print("\n👋 程序已停止")
