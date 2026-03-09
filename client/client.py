#!/usr/bin/env python3
"""
LiveKit 客户端 - 麦克风输入 + 音频播放
支持与 Agent 双向音频通信
"""
import asyncio
import numpy as np
import sounddevice as sd
from livekit import rtc, api
import os
import queue
import threading

# ========== 配置 ==========
# 麦克风采样率 (发送给 Agent)
MIC_SAMPLE_RATE = 16000
MIC_CHANNELS = 1
MIC_FRAME_SAMPLES = 160   # 10ms @ 16kHz
MIC_BLOCK_SIZE = 1600     # 100ms @ 16kHz

# 播放采样率 (接收 Agent TTS) - LiveKit WebRTC 使用 48kHz
PLAYBACK_SAMPLE_RATE = 48000
PLAYBACK_CHANNELS = 1
PLAYBACK_BLOCK_SIZE = 4800   # 100ms @ 48kHz

ROOM_NAME = "dog_room"

# LIVEKIT_URL = "ws://localhost:7880"
# LIVEKIT_URL = "ws://43.143.225.193:7880"
LIVEKIT_URL = "ws://43.143.240.89:7880"
LIVEKIT_API_KEY = "devkey"
# LIVEKIT_API_SECRET = "secret"
LIVEKIT_API_SECRET = "secret12345678909876543211234567890987654321"

os.environ['LIVEKIT_URL'] = LIVEKIT_URL
os.environ['LIVEKIT_API_KEY'] = LIVEKIT_API_KEY
os.environ['LIVEKIT_API_SECRET'] = LIVEKIT_API_SECRET

# ========== 播放缓冲区 ==========
playback_buffer = bytearray()
playback_lock = threading.Lock()


def playback_callback(outdata, frames, time_info, status):
    """音频播放回调 - 从缓冲区读取数据输出到扬声器"""
    if status:
        print(f"Playback status: {status}")
    
    bytes_needed = frames * 2  # int16 = 2 bytes per sample
    
    with playback_lock:
        if len(playback_buffer) >= bytes_needed:
            # 有足够数据，正常播放
            chunk = bytes(playback_buffer[:bytes_needed])
            del playback_buffer[:bytes_needed]
            outdata[:, 0] = np.frombuffer(chunk, dtype=np.int16)
        else:
            # 数据不足，播放已有数据 + 静音填充
            available = len(playback_buffer)
            if available > 0:
                chunk = bytes(playback_buffer[:available])
                playback_buffer.clear()
                samples = np.frombuffer(chunk, dtype=np.int16)
                outdata[:len(samples), 0] = samples
                outdata[len(samples):, 0] = 0
            else:
                outdata[:] = 0


async def receive_audio(stream: rtc.AudioStream):
    """接收 Agent 音频流并放入播放缓冲区"""
    print("🔊 开始接收 Agent 音频流...")
    frame_count = 0
    first_audio_time = None
    import time
    
    async for event in stream:
        frame = event.frame
        audio_bytes = bytes(frame.data)
        frame_count += 1
        
        # 检测有声音内容的首帧 (振幅超过阈值)
        if first_audio_time is None:
            samples = np.frombuffer(audio_bytes, dtype=np.int16)
            max_amplitude = np.max(np.abs(samples))
            if max_amplitude > 500:  # 阈值，过滤静音帧
                first_audio_time = time.time()
                h_time = time.strftime('%H:%M:%S', time.localtime(first_audio_time)) + f".{int((first_audio_time % 1) * 1000):03d}"
                print(f"\n📥 [Client] {h_time} 收到TTS首帧 | {len(audio_bytes)} 字节 | SR: {frame.sample_rate} | Amp: {max_amplitude}")
        
        with playback_lock:
            playback_buffer.extend(audio_bytes)
    
    print(f"🔇 Agent 音频流结束，共收到 {frame_count} 帧")
    first_audio_time = None  # 重置，准备下一次


async def run_client():
    # 音频处理模块 (回声消除等)
    apm = rtc.AudioProcessingModule(
        echo_cancellation=False,
        noise_suppression=False,
        high_pass_filter=False,
        auto_gain_control=False,
    )
    
    # 1. 创建 Token
    token = api.AccessToken(
        os.getenv("LIVEKIT_API_KEY"), 
        os.getenv("LIVEKIT_API_SECRET")
    ).with_identity("python-mic-user") \
     .with_name("Mic Publisher") \
     .with_grants(api.VideoGrants(room_join=True, room=ROOM_NAME)) \
     .with_attributes({"keyword": "xiaomoxiaomo"}) \
     .to_jwt()

    # 2. 连接房间
    room = rtc.Room()
    
    @room.on("disconnected")
    def on_disconnected(reason):
        print(f"🔌 [Event] Room disconnected: {reason}")
        
    @room.on("connection_state_changed")
    def on_connection_state_changed(state):
        print(f"🔄 [Event] Connection state changed: {state}")

    # ========== 订阅 Agent 音频轨道 ==========
    @room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        print(f"📡 订阅到轨道: {track.kind} from {participant.identity}")
        
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            # print(f"🎵 检测到 Agent 音频轨道，开始接收...")
            audio_stream = rtc.AudioStream(track)
            asyncio.create_task(receive_audio(audio_stream))
    
    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        print(f"👤 参与者加入: {participant.identity}")
    
    @room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        print(f"👋 参与者离开: {participant.identity}")

    @room.on("data_received")
    def on_data_received(data: rtc.DataPacket):
        payload = data.data
        print(f"\n触发！！\n")
        if payload == b"session_end":
            print("收到休眠信号！")
    
    print("🔗 Connecting to room...")
    await room.connect(os.getenv("LIVEKIT_URL"), token)
    print(f"✅ Connected to room: {room.name}")

    # 3. 创建 LiveKit 音频源 (麦克风)
    source = rtc.AudioSource(MIC_SAMPLE_RATE, MIC_CHANNELS)
    track = rtc.LocalAudioTrack.create_audio_track("mic-track", source)
    
    # 4. 发布 Track
    track_opts = rtc.TrackPublishOptions()
    track_opts.source = rtc.TrackSource.SOURCE_MICROPHONE
    
    await room.local_participant.publish_track(track, track_opts)
    print("🎤 Microphone track published!")

    # 5. 麦克风回调
    loop = asyncio.get_event_loop()
    
    def mic_callback(indata, frames, time_info, status):
        if status:
            print(f"Mic status: {status}")
        
        num_frames = frames // MIC_FRAME_SAMPLES
        for i in range(num_frames):
            start = i * MIC_FRAME_SAMPLES
            end = start + MIC_FRAME_SAMPLES
            capture_chunk = indata[start:end]

            frame = rtc.AudioFrame.create(MIC_SAMPLE_RATE, MIC_CHANNELS, MIC_FRAME_SAMPLES)
            np.copyto(np.frombuffer(frame.data, dtype=np.int16), capture_chunk.flatten())
            apm.process_stream(frame)
            asyncio.run_coroutine_threadsafe(source.capture_frame(frame), loop)

    # 6. 启动音频流
    print("\n" + "=" * 50)
    print("🎤 麦克风输入: 已启动")
    print("🔊 扬声器输出: 已启动")
    print("⌨️  按 Ctrl+C 退出")
    print("=" * 50 + "\n")
    
    # 同时启动麦克风输入和扬声器输出
    try:
        with sd.InputStream(
            channels=MIC_CHANNELS,
            samplerate=MIC_SAMPLE_RATE,
            callback=mic_callback,
            blocksize=MIC_BLOCK_SIZE,
            dtype="int16"
        ), sd.OutputStream(
            channels=PLAYBACK_CHANNELS,
            samplerate=PLAYBACK_SAMPLE_RATE,
            callback=playback_callback,
            blocksize=PLAYBACK_BLOCK_SIZE,
            dtype="int16"
        ):
            # 保持程序运行
            while True:
                await asyncio.sleep(1)
    except asyncio.CancelledError:
        print("🛑 正在关闭...")
    finally:
        print("🔌 断开房间连接...")
        await room.disconnect()
        print("✅ 已断开连接")


if __name__ == "__main__":
    try:
        asyncio.run(run_client())
    except KeyboardInterrupt:
        print("\n👋 程序已停止")