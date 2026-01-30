import asyncio
import numpy as np
from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli
import os

os.environ.setdefault("LIVEKIT_URL", "ws://localhost:7880")
os.environ.setdefault("LIVEKIT_API_KEY", "devkey")
os.environ.setdefault("LIVEKIT_API_SECRET", "secret")

async def entrypoint(ctx: JobContext):
    # 这里的 ctx 就相当于你想要的 "Session"，它包含了当前房间连接的所有上下文
    print(f"🔗 Agent connected to room: {ctx.room.name}")

    # 连接房间
    await ctx.connect()

    # --- 核心逻辑：监听 Track ---
    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track: rtc.RemoteTrack, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            print(f"✅ Subscribed to Audio Track: {track.sid} from {participant.identity}")
            # 启动一个任务来处理这个音频流
            asyncio.create_task(process_audio(track))

    # 处理已经在房间里的 Track (防止 Agent 后加入漏掉)
    for participant in ctx.room.remote_participants.values():
        for publication in participant.track_publications.values():
            if publication.track:
                on_track_subscribed(publication.track, publication, participant)

async def process_audio(track: rtc.RemoteTrack):
    # 创建流读取器
    stream = rtc.AudioStream(track)
    
    print("👂 Started listening to audio stream...")
    
    i = 0
    async for event in stream:
        # event.frame 是接收到的音频帧
        data = np.frombuffer(event.frame.data, dtype=np.int16)
        
        # 为了不刷屏太快，我们每接收 50 帧 (约 0.5秒) 打印一次统计信息
        if i % 50 == 0:
            # 计算音量 (RMS)
            rms = np.sqrt(np.mean(data.astype(np.float32)**2))
            
            print(f"📊 [Stream Info] "
                  f"Format: {event.frame.sample_rate}Hz/{event.frame.num_channels}ch | "
                  f"Samples: {len(data)} | "
                  f"Volume: {rms:.2f}")
        i += 1

if __name__ == "__main__":
    # 启动 Worker
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))