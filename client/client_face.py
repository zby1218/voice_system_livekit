#!/usr/bin/env python3
"""
Face 唤醒用 LiveKit 客户端：由人脸检测触发，不传唤醒词，仅传 wake_source=face 与 gender。
供 face_wake_listener 调用；Agent 可根据 participant.attributes 识别人脸唤醒并走 kws_enabled=False。

支持可选 AEC：开启 echo_cancellation 时会在播放回调中喂 process_reverse_stream，在麦克风回调中设置 set_stream_delay_ms。
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np
import sounddevice as sd
from livekit import api, rtc


# ========== 功能开关 ==========

# KWS（关键词唤醒）开关：True 时在连接属性中声明 kws_enabled=true，Server 端可据此启动 KWS 监听；
# False 时声明 kws_enabled=false，Server 端跳过 KWS，适用于人脸唤醒等不需要唤醒词的场景。
ENABLE_KWS: bool = True

# TTS 音色：对应 tts_server.py VOICE_CONFIGS 中的 id 字段。
# 留空时由服务端使用其自身默认值。
# 可选值: "default" | "robot_bazong" | "robot" | "longanhuan"
TTS_VOICE: str = "longanyun"


# ========== 配置 ==========


@dataclass
class AudioConfig:
    """麦克风与播放的采样参数。"""

    mic_sample_rate: int = 16000
    mic_channels: int = 1
    mic_frame_samples: int = 160  # 10ms @ 16kHz
    mic_block_size: int = 1600

    playback_sample_rate: int = 48000
    playback_channels: int = 1
    playback_frame_samples: int = 480  # 10ms @ 48kHz
    playback_block_size: int = 4800


@dataclass
class APMConfig:
    """AudioProcessingModule 开关。开启 echo_cancellation 时需在播放回调中喂 process_reverse_stream（与 mic 同采样率）并设置 stream_delay_ms。
    这里是唯一默认值定义处，修改 APM 行为只改这里。"""

    echo_cancellation: bool = True
    noise_suppression: bool = False
    high_pass_filter: bool = False
    auto_gain_control: bool = False


@dataclass
class SessionConfig:
    """会话与断连相关常量。"""

    room_name_prefix: str = "test_room"
    face_resume_host: str = "127.0.0.1"
    face_resume_port: int = 9998
    session_end_audio_idle_sec: float = 1.0
    session_end_voice_threshold: int = 200
    session_end_audio_wait_timeout: float = 30.0


@dataclass
class NetworkMonitorConfig:
    """网络连通性监控配置。"""

    enabled: bool = True
    ping_interval_sec: float = 0.5    # 每隔多少秒检测一次
    max_fail_count: int = 4           # 连续失败多少次后触发主动断开
    connect_timeout_sec: float = 1.0  # 单次 TCP 探测超时


@dataclass
class FaceClientConfig:
    """Face 客户端总配置。"""

    audio: AudioConfig = field(default_factory=AudioConfig)
    apm: APMConfig = field(default_factory=APMConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    network_monitor: NetworkMonitorConfig = field(default_factory=NetworkMonitorConfig)
    livekit_url: str = "ws://127.0.0.1:7880"
    livekit_api_key: str = "devkey"
    livekit_api_secret: str = "secret"
    tts_voice: str = TTS_VOICE


# ========== 播放缓冲区 ==========


class PlaybackBuffer:
    """线程安全的播放缓冲区，供接收端写入、播放回调读出。"""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._lock = threading.Lock()

    def extend(self, data: bytes) -> None:
        with self._lock:
            self._buffer.extend(data)

    def consume(self, size: int) -> tuple[bytes, int]:
        """取出最多 size 字节。返回 (实际取出的 bytes, 取出长度)。"""
        with self._lock:
            available = len(self._buffer)
            if available >= size:
                chunk = bytes(self._buffer[:size])
                del self._buffer[:size]
                return chunk, size
            if available > 0:
                chunk = bytes(self._buffer[:available])
                self._buffer.clear()
                return chunk, available
            return b"", 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)


# ========== 工具 ==========


def _ts() -> str:
    """当前时间戳 HH:MM:SS.mmm，与服务端 [TIMING] 格式一致。"""
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


async def _log_client_rtc_stats(room: rtc.Room) -> None:
    """查询 WebRTC subscriber stats，打印抖动缓冲延迟和到 SFU 的 RTT。"""
    try:
        stats = await room.get_rtc_stats()
    except Exception:
        return
    rtt_ms: Optional[float] = None
    jitter_buf_ms: Optional[float] = None
    jitter_ms: Optional[float] = None
    for item in stats.subscriber_stats:
        which = item.WhichOneof("stats")
        if which == "candidate_pair":
            rtt = item.candidate_pair.candidate_pair.current_round_trip_time
            if rtt > 0:
                rtt_ms = rtt * 1000
        elif which == "inbound_rtp":
            inbound = item.inbound_rtp.inbound
            emitted = inbound.jitter_buffer_emitted_count
            if emitted > 0:
                jitter_buf_ms = inbound.jitter_buffer_delay / emitted * 1000
            jitter_ms = item.inbound_rtp.received.jitter * 1000
    parts = []
    if rtt_ms is not None:
        parts.append(f"RTT(到SFU)={rtt_ms:.1f}ms  下行单向≈{rtt_ms/2:.1f}ms")
    if jitter_buf_ms is not None:
        parts.append(f"抖动缓冲={jitter_buf_ms:.1f}ms")
    if jitter_ms is not None:
        parts.append(f"网络抖动={jitter_ms:.1f}ms")
    msg = "  ".join(parts) if parts else "stats 暂无数据"
    print(f"[TIMING] {_ts()} 🌐 [CLIENT-TRANSPORT] {msg}", flush=True)


def _send_resume_to_face(host: str, port: int) -> None:
    """给 Face 发一条 resume（TCP 一发一收），Face 收到后会再启动检测。"""
    try:
        with socket.create_connection((host, port), timeout=5.0) as s:
            s.sendall(
                json.dumps({"event": "resume"}, ensure_ascii=False).encode("utf-8") + b"\n"
            )
    except (socket.error, OSError):
        pass


# ========== Face LiveKit 客户端 ==========


class FaceLiveKitClient:
    """
    Face 唤醒的 LiveKit 客户端：发布麦克风、订阅 Agent 音频并播放。
    支持可选 AEC：当 apm_config.echo_cancellation 为 True 时，在播放回调中喂 process_reverse_stream，
    在麦克风回调中根据 time_info 设置 set_stream_delay_ms。
    """

    def __init__(
        self,
        config: Optional[FaceClientConfig] = None,
        *,
        gender: Optional[str] = None,
        face_resume_host: Optional[str] = None,
        face_resume_port: Optional[int] = None,
        room_name: Optional[str] = None,
    ) -> None:
        self._config = config or FaceClientConfig()
        self._gender = gender
        self._face_resume_host = face_resume_host or self._config.session.face_resume_host
        self._face_resume_port = (
            face_resume_port
            if face_resume_port is not None
            else self._config.session.face_resume_port
        )
        self._room_name = room_name or (
            "%s_%d" % (self._config.session.room_name_prefix, int(time.time() * 1000))
        )

        self._audio = self._config.audio
        self._apm_config = self._config.apm
        self._session_config = self._config.session

        self._playback_buffer = PlaybackBuffer()

        # 每轮 TTS 追踪状态（由 tts_turn_start data 信号重置）
        self._turn_server_ts_ms: Optional[int] = None  # 服务端推首帧的时间戳
        self._turn_first_rtp_logged: bool = False       # 本轮首帧从 RTC 到达
        self._turn_first_real_logged: bool = False      # 本轮首个有效音频帧
        self._turn_first_buf_logged: bool = False       # 本轮首帧写入缓冲区
        self._turn_first_play_logged: bool = False      # 本轮首次写入扬声器
        self._turn_id: int = 0                          # 轮次计数

        self._apm = rtc.AudioProcessingModule(
            echo_cancellation=self._apm_config.echo_cancellation,
            noise_suppression=self._apm_config.noise_suppression,
            high_pass_filter=self._apm_config.high_pass_filter,
            auto_gain_control=self._apm_config.auto_gain_control,
        )

        # AEC 用：播放/采集延迟（秒），由回调更新，用于 set_stream_delay_ms
        self._output_delay: float = 0.0
        self._input_delay: float = 0.0
        self._delay_lock = threading.Lock()

        self._room = rtc.Room()
        self._source: Optional[rtc.AudioSource] = None
        self._audio_stream_ended = asyncio.Event()
        self._last_audio_frame_time: list[float] = [0.0]
        self._sleep_speech_state: dict = {
            "session_end": False,
            "sleep_speech_started": False,
            "sleep_speech_ended_printed": False,
        }
        self._session_end_event = asyncio.Event()
        self._disconnect_after_audio = False
        self._session_end_time: Optional[float] = None

        # 意外断连标志：True 表示非正常 session_end 流程断开，需要重试
        self._unexpected_disconnect = False
        self._closing = False  # Ctrl+C 等主动关闭时设为 True，避免触发重试
        self._room_disconnected_event = asyncio.Event()
        self._network_fail_event = asyncio.Event()  # 网络监控检测到不可达时设置
        self._monitor_task: Optional[asyncio.Task] = None

        # SDK 重连超时：进入 RECONNECTING 超过此秒数后主动 disconnect，让上层重试接管
        self.reconnect_timeout_sec: float = 2.0

        # 从 livekit_url 预解析监控目标，避免在 run() 里混入解析逻辑
        _parsed = urllib.parse.urlparse(self._config.livekit_url)
        self._monitor_host: str = _parsed.hostname or "127.0.0.1"
        self._monitor_port: int = _parsed.port or 7880

    def _mic_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Any,
        source: rtc.AudioSource,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        if status:
            print("Mic status: %s" % status)
        ac = self._audio
        num_frames = frames // ac.mic_frame_samples
        if num_frames == 0:
            return

        # 更新采集延迟，供 set_stream_delay_ms 使用（仅 AEC 开启时需要）
        if self._apm_config.echo_cancellation and time_info is not None:
            current = getattr(time_info, "currentTime", None)
            adc = getattr(time_info, "inputBufferAdcTime", None)
            if current is not None and adc is not None:
                input_delay = current - adc
                with self._delay_lock:
                    self._input_delay = input_delay
                    total_delay_ms = int((self._output_delay + input_delay) * 1000)
                try:
                    self._apm.set_stream_delay_ms(total_delay_ms)
                except RuntimeError:
                    pass

        for i in range(num_frames):
            start = i * ac.mic_frame_samples
            end = start + ac.mic_frame_samples
            capture_chunk = indata[start:end]
            frame = rtc.AudioFrame.create(
                ac.mic_sample_rate, ac.mic_channels, ac.mic_frame_samples
            )
            np.copyto(
                np.frombuffer(frame.data, dtype=np.int16), capture_chunk.flatten()
            )
            self._apm.process_stream(frame)
            asyncio.run_coroutine_threadsafe(source.capture_frame(frame), loop)

    def _playback_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        if status:
            print("Playback status: %s" % status)
        ac = self._audio
        bytes_needed = frames * 2
        chunk, n = self._playback_buffer.consume(bytes_needed)
        if n >= bytes_needed:
            outdata[:, 0] = np.frombuffer(chunk, dtype=np.int16)
            # ④ 本轮首次有效音频写入扬声器
            if not self._turn_first_play_logged:
                _amp = int(np.max(np.abs(np.frombuffer(chunk, dtype=np.int16))))
                if _amp > self._session_config.session_end_voice_threshold:
                    self._turn_first_play_logged = True
                    lag_str = ""
                    if self._turn_server_ts_ms is not None:
                        lag = int(time.time() * 1000) - self._turn_server_ts_ms
                        lag_str = f"  服务端推帧→扬声器出声={lag}ms"
                    print(
                        f"[TIMING] {_ts()} 🔈 [CLIENT] 首次有效音频写入扬声器 turn={self._turn_id} amp={_amp}{lag_str}",
                        flush=True,
                    )
        elif n > 0:
            samples = np.frombuffer(chunk, dtype=np.int16)
            outdata[: len(samples), 0] = samples
            outdata[len(samples) :, 0] = 0
        else:
            outdata[:] = 0

        # AEC：更新播放延迟并喂远端参考流（10ms 一帧）
        if self._apm_config.echo_cancellation and time_info is not None:
            current = getattr(time_info, "currentTime", None)
            dac = getattr(time_info, "outputBufferDacTime", None)
            if current is not None and dac is not None:
                with self._delay_lock:
                    self._output_delay = dac - current
            n_chunks = frames // ac.playback_frame_samples
            for i in range(n_chunks):
                start = i * ac.playback_frame_samples
                end = start + ac.playback_frame_samples
                render_chunk = outdata[start:end, 0]
                render_frame = rtc.AudioFrame(
                    data=render_chunk.tobytes(),
                    samples_per_channel=ac.playback_frame_samples,
                    sample_rate=ac.playback_sample_rate,
                    num_channels=ac.playback_channels,
                )
                try:
                    self._apm.process_reverse_stream(render_frame)
                except RuntimeError:
                    pass

    async def _receive_audio(
        self,
        stream: rtc.AudioStream,
        on_stream_ended: Optional[asyncio.Event] = None,
        room: Optional[rtc.Room] = None,
    ) -> None:
        sc = self._session_config
        try:
            async for event in stream:
                frame = event.frame
                audio_bytes = bytes(frame.data)
                samples = np.frombuffer(audio_bytes, dtype=np.int16)
                max_amplitude = int(np.max(np.abs(samples)))

                # ① 本轮首个音频帧到达 RTC（任意振幅）
                if not self._turn_first_rtp_logged:
                    self._turn_first_rtp_logged = True
                    lag_str = ""
                    if self._turn_server_ts_ms is not None:
                        lag = int(time.time() * 1000) - self._turn_server_ts_ms
                        lag_str = f"  服务端推帧→客端RTC传输={lag}ms"
                    print(
                        f"[TIMING] {_ts()} 📥 [CLIENT] 首帧从RTC到达 turn={self._turn_id} "
                        f"samples={frame.samples_per_channel} sr={frame.sample_rate} amp={max_amplitude}{lag_str}",
                        flush=True,
                    )
                    if room is not None:
                        asyncio.create_task(_log_client_rtc_stats(room))

                if max_amplitude > sc.session_end_voice_threshold:
                    self._last_audio_frame_time[0] = time.monotonic()

                # ② 本轮首个有效音频帧（振幅过阈值，确认是真实语音）
                if not self._turn_first_real_logged and max_amplitude > sc.session_end_voice_threshold:
                    self._turn_first_real_logged = True
                    lag_str = ""
                    if self._turn_server_ts_ms is not None:
                        lag = int(time.time() * 1000) - self._turn_server_ts_ms
                        lag_str = f"  服务端推帧→客端有效音频={lag}ms"
                    print(
                        f"[TIMING] {_ts()} 🎵 [CLIENT] 首个有效音频帧 turn={self._turn_id} amp={max_amplitude}{lag_str}",
                        flush=True,
                    )

                if self._sleep_speech_state.get("session_end") and max_amplitude > sc.session_end_voice_threshold:
                    if not self._sleep_speech_state.get("sleep_speech_started"):
                        print(f"[TIMING] {_ts()} 💤 [CLIENT] 收到休眠语音", flush=True)
                        self._sleep_speech_state["sleep_speech_started"] = True
                if (
                    self._sleep_speech_state.get("sleep_speech_started")
                    and not self._sleep_speech_state.get("sleep_speech_ended_printed")
                    and max_amplitude <= 1
                ):
                    print(f"[TIMING] {_ts()} 💤 [CLIENT] 休眠语音结束", flush=True)
                    self._sleep_speech_state["sleep_speech_ended_printed"] = True

                self._playback_buffer.extend(audio_bytes)

                # ③ 本轮首帧写入播放缓冲区
                if not self._turn_first_buf_logged and max_amplitude > sc.session_end_voice_threshold:
                    self._turn_first_buf_logged = True
                    print(
                        f"[TIMING] {_ts()} 📦 [CLIENT] 首帧写入播放缓冲区 turn={self._turn_id} buf_size={len(self._playback_buffer)}B",
                        flush=True,
                    )
        finally:
            if on_stream_ended is not None:
                on_stream_ended.set()

    async def _network_monitor_loop(self) -> None:
        """后台 TCP 探测循环：只负责检测，连续失败超过阈值后设置 _network_fail_event 通知主循环处理。"""
        cfg = self._config.network_monitor
        host, port = self._monitor_host, self._monitor_port
        fail_count = 0
        while not self._room_disconnected_event.is_set():
            await asyncio.sleep(cfg.ping_interval_sec)
            if self._room_disconnected_event.is_set():
                break
            try:
                _, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port),
                    timeout=cfg.connect_timeout_sec,
                )
                writer.close()
                await writer.wait_closed()
                fail_count = 0
            except Exception:
                fail_count += 1
                if fail_count >= cfg.max_fail_count:
                    print(
                        "%s 🌐 [NetMonitor] 连续 %d 次探测失败，通知主循环断开"
                        % (_ts(), fail_count)
                    )
                    self._network_fail_event.set()
                    break

    async def run(self) -> None:
        """连接房间、发布麦克风、订阅远端音频并运行主循环，直到断开或 session_end 后收完休眠语音。"""
        os.environ.setdefault("LIVEKIT_URL", self._config.livekit_url)
        os.environ.setdefault("LIVEKIT_API_KEY", self._config.livekit_api_key)
        os.environ.setdefault("LIVEKIT_API_SECRET", self._config.livekit_api_secret)

        attrs = {"wake_source": "face", "kws_enabled": "true" if ENABLE_KWS else "false"}
        if self._gender is not None:
            attrs["gender"] = self._gender
        if self._config.tts_voice:
            attrs["tts_voice"] = self._config.tts_voice
        token = (
            api.AccessToken(
                os.getenv("LIVEKIT_API_KEY"),
                os.getenv("LIVEKIT_API_SECRET"),
            )
            .with_identity("python-mic-user")
            .with_name("Mic Publisher")
            .with_grants(api.VideoGrants(room_join=True, room=self._room_name))
            .with_attributes(attrs)
            .to_jwt()
        )

        room = self._room
        ac = self._audio

        @room.on("disconnected")
        def on_disconnected(reason: str) -> None:
            print("%s 🔌 [Event] Room disconnected: %s" % (_ts(), reason))
            # 主动关闭（Ctrl+C）或正常 session_end 流程，不触发重试
            if (
                not self._closing
                and not self._disconnect_after_audio
                and not self._session_end_event.is_set()
            ):
                self._unexpected_disconnect = True
            self._room_disconnected_event.set()

        _reconnect_watchdog_task: list[asyncio.Task] = []

        @room.on("connection_state_changed")
        def on_connection_state_changed(state: str) -> None:
            print("%s 🔄 [Event] Connection state changed: %s" % (_ts(), state))
            # 取消上一个 watchdog（如果有）
            for t in _reconnect_watchdog_task:
                t.cancel()
            _reconnect_watchdog_task.clear()
            # 进入 RECONNECTING（state==2）时启动超时 watchdog
            if str(state) == "2":
                async def _watchdog() -> None:
                    await asyncio.sleep(self.reconnect_timeout_sec)
                    print(
                        "%s ⏱️ SDK 重连超时 (%.0fs)，主动断开让上层重试"
                        % (_ts(), self.reconnect_timeout_sec)
                    )
                    await room.disconnect()
                _reconnect_watchdog_task.append(asyncio.create_task(_watchdog()))

        @room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            print("%s 📡 订阅到轨道: %s from %s" % (_ts(), track.kind, participant.identity))
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                audio_stream = rtc.AudioStream(track)
                asyncio.create_task(
                    self._receive_audio(audio_stream, self._audio_stream_ended, room)
                )

        @room.on("participant_connected")
        def on_participant_connected(participant: rtc.RemoteParticipant) -> None:
            print("%s 👤 参与者加入: %s" % (_ts(), participant.identity))

        @room.on("participant_disconnected")
        def on_participant_disconnected(participant: rtc.RemoteParticipant) -> None:
            print("%s 👋 参与者离开: %s" % (_ts(), participant.identity))

        @room.on("data_received")
        def on_data_received(data: rtc.DataPacket) -> None:
            payload = data.data
            if payload == b"session_end":
                print("%s 收到休眠信号" % _ts())
                self._sleep_speech_state["session_end"] = True
                asyncio.create_task(
                    asyncio.to_thread(
                        _send_resume_to_face,
                        self._face_resume_host,
                        self._face_resume_port,
                    )
                )
                self._disconnect_after_audio = True
                self._session_end_time = time.monotonic()
            elif isinstance(payload, (bytes, bytearray)) and payload.startswith(b"tts_turn_start:"):
                # 服务端通知：本轮 TTS 首帧已推入 AudioSource，重置追踪状态
                try:
                    server_ts_ms = int(payload.split(b":")[1])
                except Exception:
                    server_ts_ms = None
                self._turn_id += 1
                self._turn_server_ts_ms = server_ts_ms
                self._turn_first_rtp_logged = False
                self._turn_first_real_logged = False
                self._turn_first_buf_logged = False
                self._turn_first_play_logged = False
                now_ms = int(time.time() * 1000)
                lag = (now_ms - server_ts_ms) if server_ts_ms else None
                lag_str = f"  信号传输延迟={lag}ms" if lag is not None else ""
                print(
                    f"[TIMING] {_ts()} 📨 [CLIENT] tts_turn_start 收到 turn={self._turn_id}{lag_str}",
                    flush=True,
                )

        print("🔗 Connecting to room: %s ..." % self._room_name)
        await room.connect(os.getenv("LIVEKIT_URL"), token)
        print("✅ Connected to room: %s" % room.name)

        # 连接成功后启动网络监控
        if self._config.network_monitor.enabled:
            self._monitor_task = asyncio.create_task(self._network_monitor_loop())
            print("%s 🌐 [NetMonitor] 已启动，探测 %s:%d" % (_ts(), self._monitor_host, self._monitor_port))

        source = rtc.AudioSource(ac.mic_sample_rate, ac.mic_channels)
        self._source = source
        track = rtc.LocalAudioTrack.create_audio_track("mic-track", source)
        track_opts = rtc.TrackPublishOptions()
        track_opts.source = rtc.TrackSource.SOURCE_MICROPHONE
        await room.local_participant.publish_track(track, track_opts)
        print("🎤 Microphone track published!")

        loop = asyncio.get_event_loop()

        def mic_callback(
            indata: np.ndarray,
            frames: int,
            time_info: sd.CallbackFlags,
            status: sd.CallbackFlags,
        ) -> None:
            self._mic_callback(indata, frames, time_info, status, source, loop)

        def playback_callback(
            outdata: np.ndarray,
            frames: int,
            time_info: sd.CallbackFlags,
            status: sd.CallbackFlags,
        ) -> None:
            self._playback_callback(outdata, frames, time_info, status)

        apm = self._apm_config
        _on = lambda b: "开启" if b else "关闭"
        print("\n" + "=" * 50)
        print("🎤 麦克风输入: 已启动")
        print("🔊 扬声器输出: 已启动")
        print("🔧 APM: 回声消除=%s, 降噪=%s, 高通滤波=%s, 自动增益=%s" % (
            _on(apm.echo_cancellation),
            _on(apm.noise_suppression),
            _on(apm.high_pass_filter),
            _on(apm.auto_gain_control),
        ))
        print("⌨️  按 Ctrl+C 退出")
        print("=" * 50 + "\n")

        sc = self._session_config
        try:
            with sd.InputStream(
                channels=ac.mic_channels,
                samplerate=ac.mic_sample_rate,
                callback=mic_callback,
                blocksize=ac.mic_block_size,
                dtype="int16",
            ), sd.OutputStream(
                channels=ac.playback_channels,
                samplerate=ac.playback_sample_rate,
                callback=playback_callback,
                blocksize=ac.playback_block_size,
                dtype="int16",
            ):
                while True:
                    if self._session_end_event.is_set():
                        break
                    if self._room_disconnected_event.is_set():
                        break
                    if self._network_fail_event.is_set():
                        print("%s 🌐 [NetMonitor] 网络不可达，主动断开" % _ts())
                        await room.disconnect()
                        break
                    if self._disconnect_after_audio:
                        if self._audio_stream_ended.is_set():
                            print("%s 休眠语音已播完，断开连接" % _ts())
                            break
                        if (
                            self._sleep_speech_state.get("sleep_speech_started")
                            and self._last_audio_frame_time[0] > 0
                            and (time.monotonic() - self._last_audio_frame_time[0])
                            >= sc.session_end_audio_idle_sec
                        ):
                            print(
                                "%s 休眠语音已发完（%.1fs 无新帧），断开连接"
                                % (_ts(), sc.session_end_audio_idle_sec)
                            )
                            break
                        if (
                            self._session_end_time is not None
                            and (time.monotonic() - self._session_end_time)
                            >= sc.session_end_audio_wait_timeout
                        ):
                            print(
                                "%s 等待音频流超时 (%.0fs)，断开连接"
                                % (_ts(), sc.session_end_audio_wait_timeout)
                            )
                            break
                    await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            self._closing = True
            print("%s 🛑 正在关闭..." % _ts())
        finally:
            if self._monitor_task is not None and not self._monitor_task.done():
                self._monitor_task.cancel()
            print("%s 🔌 断开房间连接..." % _ts())
            await room.disconnect()
            print("%s ✅ 已断开连接" % _ts())


# ========== 入口 ==========


async def run_client(
    gender: Optional[str] = None,
    *,
    face_resume_host: Optional[str] = None,
    face_resume_port: Optional[int] = None,
    room_name: Optional[str] = None,
    config: Optional[FaceClientConfig] = None,
    max_retries: int = 5,
    retry_delay: float = 2.0,
) -> None:
    """
    便捷入口：创建 FaceLiveKitClient 并运行。
    不传 config 时使用 FaceClientConfig / APMConfig 的默认值；
    需要覆盖 APM 等参数时，构造 FaceClientConfig 传入即可。

    意外断连时（非 session_end 流程）自动用新 room name 重试，最多 max_retries 次。
    """
    cfg = config or FaceClientConfig()
    attempt = 0
    while True:
        # 每次重试生成新的 room name（避免 StateMismatch）
        current_room = room_name if (room_name and attempt == 0) else None
        client = FaceLiveKitClient(
            cfg,
            gender=gender,
            face_resume_host=face_resume_host,
            face_resume_port=face_resume_port,
            room_name=current_room,
        )
        try:
            await client.run()
        except Exception as e:
            print("%s ❌ run() 异常退出: %s" % (_ts(), e))
            client._unexpected_disconnect = True

        if not client._unexpected_disconnect:
            # 正常 session_end 流程结束，不重试
            break

        attempt += 1
        if attempt > max_retries:
            print("%s ⚠️ 重试次数已达上限 (%d)，停止重试" % (_ts(), max_retries))
            break

        print(
            "%s 🔁 检测到意外断连，%gs 后发起第 %d/%d 次重试..."
            % (_ts(), retry_delay, attempt, max_retries)
        )
        await asyncio.sleep(retry_delay)


if __name__ == "__main__":
    try:
        asyncio.run(run_client())
    except KeyboardInterrupt:
        print("\n👋 程序已停止")
