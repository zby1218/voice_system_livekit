import asyncio
import os
import logging
import numpy as np
from livekit import rtc, api  # 修正：使用 api 模块生成 token
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(".env.local")

URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secret")
ROOM_NAME = "test_room"  # ⚠️ 确保这里和你的 client/test.py 进的是同一个房间

async def main():
    print(f"🔌 正在连接到 {URL} (房间: {ROOM_NAME})...")
    room = rtc.Room()
    
    # 监听 Track 订阅事件
    @room.on("track_subscribed")
    def on_track_subscribed(track: rtc.RemoteTrack, publication, participant):
        print(f"✅ [事件] 成功订阅轨道: {track.sid} (类型: {track.kind}) 来自: {participant.identity}")
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            print(f"🎧 启动音频读取任务...")
            asyncio.create_task(read_audio(track))

    try:
        # 1. 创建 Token (修正部分)
        token = api.AccessToken(API_KEY, API_SECRET) \
            .with_identity("debug-receiver") \
            .with_name("Debug Receiver") \
            .with_grants(api.VideoGrants(room_join=True, room=ROOM_NAME)) \
            .to_jwt()

        # 2. 连接房间 (修正部分: AutoSubscribe 在 rtc 模块下)
        await room.connect(URL, token, auto_subscribe=rtc.AutoSubscribe.AUDIO_ONLY)
        print("🚀 已连接！正在等待音频流...")
        
        # 保持运行
        await asyncio.Event().wait()
    except Exception as e:
        print(f"❌ 连接错误: {e}")

async def read_audio(track: rtc.RemoteTrack):
    # 创建音频流读取器
    stream = rtc.AudioStream(track)
    print(f"🌊 流读取器已创建，开始接收数据...")
    
    count = 0
    async for event in stream:
        count += 1
        # 每 20 帧 (约0.2秒) 打印一次，证明数据活着
        if count % 20 == 0:
            data = np.frombuffer(event.frame.data, dtype=np.int16)
            # 计算 RMS (均方根) 音量
            rms = np.sqrt(np.mean(data.astype(np.float32)**2))
            
            # 只有音量 > 0 才打印，或者你可以去掉 if 强制打印所有
            print(f"🔊 [收到音频] 帧数: {count} | 音量RMS: {rms:.2f}")

if __name__ == "__main__":
    asyncio.run(main())