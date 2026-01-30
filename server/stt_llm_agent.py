#!/usr/bin/env python3
"""
STT + LLM Agent (使用 AgentServer) - 语音识别后调用 LLM 并打印结果
启动: python stt_llm_agent.py dev
"""
import os
import sys
import numpy as np
import asyncio
import logging

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
    metrics,
    MetricsCollectedEvent,
    UserStateChangedEvent,
    ConversationItemAddedEvent,
)
from livekit.agents.voice.room_io import RoomOptions, AudioInputOptions





from livekit.plugins import silero, openai

# 导入自定义 STT
from custom_stt import MySTT
from custom_tts import CosyVoiceTTS

current_dir = os.path.dirname(os.path.abspath(__file__))
PROMPT_WAV_PATH = os.path.join(current_dir, "..", "tts", "assets", "zero_shot_prompt.wav")

# LiveKit 本地开发环境
os.environ.setdefault("LIVEKIT_URL", "ws://localhost:7880")
os.environ.setdefault("LIVEKIT_API_KEY", "devkey")
os.environ.setdefault("LIVEKIT_API_SECRET", "secret")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
# logger = logging.getLogger("stt-llm-agent")


# ========== STT 实例 ==========
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
)

# ========== LLM 实例 (阿里云 Dashscope Qwen) ==========
my_llm = openai.LLM(
    model="qwen-plus-2025-12-01",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-3a9ca0038afe434abf0c703000ebb694",
)


class MyAgent(Agent):
    """简单的 Agent，只做 STT + LLM，不做 TTS"""
    
    def __init__(self) -> None:
        super().__init__(
            instructions="你是一个友好的助手，用简洁的中文回答问题。保持回复简短。",
        )


# ========== AgentServer 方式 (和 myagent.py 一致) ==========
server = AgentServer()


def prewarm(proc: JobProcess):
    """预热：加载 VAD 模型"""
    proc.userdata["vad"] = silero.VAD.load(
        activation_threshold=0.7,
        deactivation_threshold=0.5,
        min_speech_duration=0.1,
        min_silence_duration=0.5,
    )


server.setup_fnc = prewarm

# async def do_something(track: rtc.RemoteAudioTrack):
#     audio_stream = rtc.AudioStream(track)
#     async for event in audio_stream:
#         # Do something here to process event.frame
#         pass
#     await audio_stream.aclose()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # 1. 连接房间
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    # 2. 等待用户 (这一步其实 session.start 内部也会做，但写在这里更稳妥)
    participant = await ctx.wait_for_participant()
    
    # 3. 初始化 Session
    # attributes = participant.attributes
    # logger.info(f"Participant attributes: {attributes}")
    
    session = AgentSession(
        stt=my_stt,
        llm=my_llm, # 你现在不需要 LLM，先设为 None 排除干扰
        tts=my_tts,
        vad=ctx.proc.userdata["vad"],
    )

    # @session.on("user_state_changed")
    # def on_user_state_changed(ev: UserStateChangedEvent):
    #     if ev.new_state == "away":
    #         print("用户状态转为离开")
    #         session.say("我先休息了, 有事情再叫我吧", allow_interruptions=True)
    #         # asyncio.create_task(
    #         #     session.say("我先休息了", allow_interruptions=True)
    #         # )

    # @session.on("conversation_item_added")
    # def on_item_added(event: ConversationItemAddedEvent):
    #     # event.item 是一个 ChatMessage 对象
    #     item = event.item
    #     print(1122)
    #     # 我们只关心 AI (assistant) 说的话，不打印用户 (user) 说的话
    #     if item.role == "assistant":
    #         # item.content 可能是字符串，也可能是内容列表，处理一下更稳健
    #         text_content = ""
    #         if isinstance(item.content, str):
    #             text_content = item.content
    #         elif isinstance(item.content, list):
    #             # 提取列表中的文本部分
    #             text_content = "".join([str(c) for c in item.content])
            
    #         print(f"\n🤖 [LLM 回复]: {text_content}\n")
        
    # 4. 启动！(不要在 start 之前做任何关于 audio_track 的操作)
    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_options=RoomOptions(
            audio_input=AudioInputOptions(sample_rate=16000, frame_size_ms=150),
        ),
    )



    # await session.say("你好啊，很高兴见到你！", allow_interruptions=True)


    





if __name__ == "__main__":
    cli.run_app(server)
