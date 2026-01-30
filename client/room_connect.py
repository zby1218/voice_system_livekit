import os
import logging
import asyncio
from livekit import rtc

# os.environ["LIVEKIT_TOKEN"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE4MDUxMzc1ODAsImlzcyI6IkFQSTZ5dU5pSGdpNDZpcCIsIm5hbWUiOiJUZXN0IFVzZXIiLCJuYmYiOjE3NjkxMzc1ODAsInN1YiI6InRlc3QtdXNlciIsInZpZGVvIjp7InJvb20iOiJteS1maXJzdC1yb29tIiwicm9vbUpvaW4iOnRydWV9fQ.wc4TyyM1fufIVcwGJKAELnNj4qXBP6hzVNbnwedLJx0"
# os.environ["LIVEKIT_URL"] = "http://localhost:7880"

os.environ["LIVEKIT_TOKEN"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJuYW1lIjoiUHl0aG9uIEJvdCIsInZpZGVvIjp7InJvb21Kb2luIjp0cnVlLCJyb29tIjoibXktcm9vbSIsImNhblB1Ymxpc2giOnRydWUsImNhblN1YnNjcmliZSI6dHJ1ZSwiY2FuUHVibGlzaERhdGEiOnRydWV9LCJzdWIiOiJweXRob24tYm90IiwiaXNzIjoiZGV2a2V5IiwibmJmIjoxNzY5MTU1MjQ1LCJleHAiOjE3NjkxNzY4NDV9.e2L1fjQY5oFvYAXSvzrcS6Yn275SSvAM14dEepYK2R4"
os.environ["LIVEKIT_URL"] = "http://localhost:7880"
TOKEN = os.environ.get("LIVEKIT_TOKEN")
URL = os.environ.get("LIVEKIT_URL")

async def main():
    room = rtc.Room()

    # --- 1. 定义事件监听 (接收端逻辑) ---
    
    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        logging.info("有人加入: %s %s", participant.sid, participant.identity)

    # 异步处理视频帧
    async def receive_frames(stream: rtc.VideoStream):
        async for frame in stream:
            # 这里处理收到的视频画面 (例如: pass 或 cv2.imshow)
            pass

    # 异步处理音频帧 (如果你想把听到的声音存下来或给AI听)
    async def receive_audio(stream: rtc.AudioStream):
        async for frame in stream:
            # 这里处理收到的音频数据
            pass

    # 【关键修正】：将视频和音频的监听逻辑合并到一个函数里
    @room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        logging.info(f"订阅成功: {publication.sid} 类型: {track.kind}")
        
        #如果是视频
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            print(f"收到来自 {participant.identity} 的视频流")
            video_stream = rtc.VideoStream(track)
            asyncio.ensure_future(receive_frames(video_stream))
            
        # 如果是音频
        elif track.kind == rtc.TrackKind.KIND_AUDIO:
            print(f"收到来自 {participant.identity} 的音频流")
            audio_stream = rtc.AudioStream(track)
            asyncio.ensure_future(receive_audio(audio_stream))

    # --- 2. 准备硬件 (发送端准备) ---
    print("正在打开麦克风...")
    devices = rtc.MediaDevices()
    mic = devices.open_input(
        enable_aec=True,          
        noise_suppression=True,   
        high_pass_filter=True,    
        auto_gain_control=True    
    )
    # 创建本地音频轨道
    local_audio_track = rtc.LocalAudioTrack.create_audio_track("microphone", mic.source)

    # --- 3. 连接房间 ---
    print(f"正在连接到 {URL} ...")
    await room.connect(URL, TOKEN)
    logging.info("已连接到房间: %s", room.name)

    # --- 4. 发布轨道 (关键动作：把麦克风推流出去) ---
    # 只有连接成功后，才能把自己发布出去
    print("正在发布麦克风音频...")
    await room.local_participant.publish_track(local_audio_track)
    print("麦克风发布成功！别人应该能听到你了。")

    # --- 5. 打印房间现状 ---
    for identity, participant in room.remote_participants.items():
        print(f"在房间里的用户: {identity}")

    # --- 6. 永久等待 ---
    print("🤖 机器人运行中 - 正在接收视音频，同时广播麦克风...")
    await asyncio.Event().wait()
    
    # 退出前的清理 (虽然强制退出时不一定执行，但写上是好习惯)
    await mic.aclose()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) # 开启日志打印
    if not TOKEN or not URL:
        print("TOKEN and URL are required environment variables")
        exit(1)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序停止")