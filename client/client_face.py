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
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import sounddevice as sd
from livekit import api, rtc


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
    """AudioProcessingModule 开关。开启 echo_cancellation 时需在播放回调中喂 process_reverse_stream（与 mic 同采样率）并设置 stream_delay_ms。"""

    echo_cancellation: bool = True
    noise_suppression: bool = True
    high_pass_filter: bool = True
    auto_gain_control: bool = True


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
class FaceClientConfig:
    """Face 客户端总配置。"""

    audio: AudioConfig = field(default_factory=AudioConfig)
    apm: APMConfig = field(default_factory=APMConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    livekit_url: str = "ws://127.0.0.1:7880"
    livekit_api_key: str = "devkey"
    livekit_api_secret: str = "secret"


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
    """当前时间戳，用于休眠语音日志。"""
    t = time.time()
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) + ".%03d" % int((t % 1) * 1000)


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
    ) -> None:
        sc = self._session_config
        try:
            async for event in stream:
                frame = event.frame
                audio_bytes = bytes(frame.data)
                samples = np.frombuffer(audio_bytes, dtype=np.int16)
                max_amplitude = int(np.max(np.abs(samples)))
                if max_amplitude > sc.session_end_voice_threshold:
                    self._last_audio_frame_time[0] = time.monotonic()
                if self._sleep_speech_state.get("session_end") and max_amplitude > sc.session_end_voice_threshold:
                    if not self._sleep_speech_state.get("sleep_speech_started"):
                        print("%s 收到休眠语音" % _ts())
                        self._sleep_speech_state["sleep_speech_started"] = True
                if (
                    self._sleep_speech_state.get("sleep_speech_started")
                    and not self._sleep_speech_state.get("sleep_speech_ended_printed")
                    and max_amplitude <= 1
                ):
                    print("%s 休眠语音结束" % _ts())
                    self._sleep_speech_state["sleep_speech_ended_printed"] = True
                self._playback_buffer.extend(audio_bytes)
        finally:
            if on_stream_ended is not None:
                on_stream_ended.set()

    async def run(self) -> None:
        """连接房间、发布麦克风、订阅远端音频并运行主循环，直到断开或 session_end 后收完休眠语音。"""
        os.environ.setdefault("LIVEKIT_URL", self._config.livekit_url)
        os.environ.setdefault("LIVEKIT_API_KEY", self._config.livekit_api_key)
        os.environ.setdefault("LIVEKIT_API_SECRET", self._config.livekit_api_secret)

        attrs = {"wake_source": "face"}
        if self._gender is not None:
            attrs["gender"] = self._gender
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

        @room.on("connection_state_changed")
        def on_connection_state_changed(state: str) -> None:
            print("%s 🔄 [Event] Connection state changed: %s" % (_ts(), state))

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
                    self._receive_audio(audio_stream, self._audio_stream_ended)
                )

        @room.on("participant_connected")
        def on_participant_connected(participant: rtc.RemoteParticipant) -> None:
            print("%s 👤 参与者加入: %s" % (_ts(), participant.identity))

        @room.on("participant_disconnected")
        def on_participant_disconnected(participant: rtc.RemoteParticipant) -> None:
            print("%s 👋 参与者离开: %s" % (_ts(), participant.identity))

        @room.on("data_received")
        def on_data_received(data: rtc.DataPacket) -> None:
            if data.data == b"session_end":
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

        print("🔗 Connecting to room: %s ..." % self._room_name)
        await room.connect(os.getenv("LIVEKIT_URL"), token)
        print("✅ Connected to room: %s" % room.name)

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
            print("%s 🛑 正在关闭..." % _ts())
        finally:
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
    apm_echo_cancellation: bool = True,
    apm_noise_suppression: bool = False,
    apm_high_pass_filter: bool = False,
    apm_auto_gain_control: bool = False,
) -> None:
    """
    便捷入口：创建 FaceLiveKitClient 并运行。
    若需 AEC，设置 apm_echo_cancellation=True；会同时在播放回调中喂 process_reverse_stream 并设置 stream_delay_ms。
    """
    config = FaceClientConfig(
        apm=APMConfig(
            echo_cancellation=apm_echo_cancellation,
            noise_suppression=apm_noise_suppression,
            high_pass_filter=apm_high_pass_filter,
            auto_gain_control=apm_auto_gain_control,
        )
    )
    client = FaceLiveKitClient(
        config,
        gender=gender,
        face_resume_host=face_resume_host,
        face_resume_port=face_resume_port,
        room_name=room_name,
    )
    await client.run()


if __name__ == "__main__":
    try:
        asyncio.run(run_client(apm_echo_cancellation=True))
    except KeyboardInterrupt:
        print("\n👋 程序已停止")
