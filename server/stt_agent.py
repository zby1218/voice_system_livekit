#!/usr/bin/env python3
"""
最简 STT Agent - 只做语音识别并打印结果
启动: python stt_agent.py dev
"""
import os
import sys
import asyncio
import logging

# 添加 stt 目录到路径，以便导入 custom_stt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stt"))

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    stt,
)

from livekit.plugins import openai

# 导入你的自定义 STT
from custom_stt import MySTT

# LiveKit 本地开发环境
os.environ.setdefault("LIVEKIT_URL", "ws://localhost:7880")
os.environ.setdefault("LIVEKIT_API_KEY", "devkey")
os.environ.setdefault("LIVEKIT_API_SECRET", "secret")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stt-agent")


async def entrypoint(ctx: JobContext):
    """Agent 入口: 订阅房间音频 -> STT -> 打印结果"""
    
    logger.info(f"🎤 Agent 加入房间: {ctx.room.name}")
    
    # 1. 连接房间，自动订阅音频轨道
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    # 2. 创建 STT 实例 (连接到 stt_server.py)
    my_stt = MySTT(
        host="localhost",
        port=10095,
        mode="2pass",
    )
    
    # 3. 等待第一个参与者发布音频
    participant = await ctx.wait_for_participant()
    logger.info(f"✅ 检测到参与者: {participant.identity}")
    
    # 4. 获取音频轨道
    audio_track = None
    for pub in participant.track_publications.values():
        if pub.track and pub.track.kind == rtc.TrackKind.KIND_AUDIO:
            audio_track = pub.track
            break
    
    if not audio_track:
        # 等待音频轨道发布
        @ctx.room.on("track_subscribed")
        def on_track_subscribed(track: rtc.Track, publication, remote_participant):
            nonlocal audio_track
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                audio_track = track
        
        # 等待音频轨道
        while not audio_track:
            await asyncio.sleep(0.1)
    
    logger.info(f"🎧 开始监听音频轨道: {audio_track.sid}")
    
    # 5. 创建音频流并进行 STT
    audio_stream = rtc.AudioStream(audio_track)
    
    # 6. 创建 STT 流
    stt_stream = my_stt.stream()
    
    # 7. 启动两个任务：音频转发 + 结果打印
    async def forward_audio():
        """将音频帧转发到 STT"""
        async for event in audio_stream:
            stt_stream.push_frame(event.frame)
    
    async def print_results():
        """打印 STT 结果"""
        async for event in stt_stream:
            if event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
                text = event.alternatives[0].text if event.alternatives else ""
                # if text:
                #     print(f"[中间结果] {text}", flush=True)
            
            elif event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                text = event.alternatives[0].text if event.alternatives else ""
                if text:
                    print(f"✅ [最终结果] {text}", flush=True)
    
    # 8. 并行运行
    await asyncio.gather(
        forward_audio(),
        print_results(),
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        ),
    )
