#!/usr/bin/env python3
"""
STT 测试客户端 - 连接房间并发布麦克风音频
"""
import os
import asyncio
import logging
from livekit import rtc, api
import jwt
import time

# LiveKit 本地开发环境
LIVEKIT_URL = "ws://localhost:7880"
LIVEKIT_API_KEY = "devkey"
LIVEKIT_API_SECRET = "secret"
ROOM_NAME = "my-room"
USER_IDENTITY = "test-user"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stt-client")


def generate_token() -> str:
    """生成 LiveKit 访问 token"""
    claims = {
        "iss": LIVEKIT_API_KEY,
        "sub": USER_IDENTITY,
        "name": "Test User",
        "nbf": int(time.time()),
        "exp": int(time.time()) + 86400,  # 24小时有效
        "video": {
            "room": ROOM_NAME,
            "roomJoin": True,
            "canPublish": True,
            "canSubscribe": True,
            "canPublishData": True,
        }
    }
    return jwt.encode(claims, LIVEKIT_API_SECRET, algorithm="HS256")


async def ensure_room_exists():
    """确保房间存在"""
    try:
        lkapi = api.LiveKitAPI(
            url="http://localhost:7880",
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
        )
        await lkapi.room.create_room(api.CreateRoomRequest(name=ROOM_NAME))
        logger.info(f"✅ 房间 '{ROOM_NAME}' 已创建/确认存在")
        await lkapi.aclose()
    except Exception as e:
        logger.warning(f"创建房间时出错 (可能已存在): {e}")


async def main():
    # 1. 确保房间存在
    await ensure_room_exists()
    
    # 2. 生成 token
    token = generate_token()
    logger.info(f"🔑 Token 已生成")
    
    # 3. 创建房间连接
    room = rtc.Room()
    
    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        logger.info(f"👤 Agent 加入: {participant.identity}")
    
    @room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication, participant):
        logger.info(f"📡 订阅到轨道: {track.kind} from {participant.identity}")
    
    # 4. 打开麦克风
    logger.info("🎤 正在打开麦克风...")
    devices = rtc.MediaDevices()
    mic = devices.open_input(
        enable_aec=True,
        noise_suppression=True,
        high_pass_filter=True,
        auto_gain_control=True,
    )
    local_audio_track = rtc.LocalAudioTrack.create_audio_track("microphone", mic.source)
    
    # 5. 连接房间
    logger.info(f"🔗 正在连接到 {LIVEKIT_URL} ...")
    await room.connect(LIVEKIT_URL, token)
    logger.info(f"✅ 已连接到房间: {room.name}")
    
    # 6. 发布音频轨道
    logger.info("📤 正在发布麦克风音频...")
    await room.local_participant.publish_track(local_audio_track)
    logger.info("✅ 麦克风发布成功！Agent 应该能听到你了。")
    
    # 7. 打印使用说明
    print("\n" + "=" * 50)
    print("🎙️  客户端运行中 - 请对着麦克风说话")
    print("🤖  Agent 会识别你的语音并打印结果")
    print("⌨️  按 Ctrl+C 退出")
    print("=" * 50 + "\n")
    
    # 8. 保持运行
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        await mic.aclose()
        await room.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 程序已停止")
