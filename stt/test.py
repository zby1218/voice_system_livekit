#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import asyncio
import websockets
import json
import os
import sys
import argparse

# === 配置项 ===
HOST = "localhost"
PORT = 10095
DEFAULT_AUDIO_FILE = "output_custom_voice.wav"  # 默认音频文件
CHUNK_SIZE = [5, 10, 5]  # [5, 10, 5] 对应 60ms 一包
CHUNK_INTERVAL = 10
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION_MS = 60  # 每个音频块的时长(ms)
BYTES_PER_CHUNK = int(SAMPLE_RATE * CHANNELS * 2 * CHUNK_DURATION_MS / 1000)  # 1920 bytes


async def recv_task(ws):
    """后台接收服务端消息"""
    try:
        async for msg in ws:
            msg_json = json.loads(msg)
            text = msg_json.get("text", "")
            mode = msg_json.get("mode", "")
            is_final = msg_json.get("is_final", False)
            
            if mode == "2pass-online":
                print(f"\r[流式] {text}", end="", flush=True)
            elif mode == "2pass-offline":
                print(f"\n✅ [最终] {text}")
    except websockets.ConnectionClosed:
        print("\n连接关闭")
    except asyncio.CancelledError:
        pass


async def test_file(ws, audio_file, fast_mode=False):
    """从文件发送音频"""
    # 检查音频文件
    if not os.path.exists(audio_file):
        print(f"❌ 错误: 找不到文件 '{audio_file}'")
        print("请指定一个存在的 16000Hz 单声道 wav 文件")
        return

    # 读取音频数据
    print(f"🎵 使用音频文件: {audio_file}")
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()

    # 发送音频流
    print("▶️ 开始发送音频流...")
    chunk_num = (len(audio_bytes) - 1) // BYTES_PER_CHUNK + 1
    
    for i in range(chunk_num):
        beg = i * BYTES_PER_CHUNK
        end = beg + BYTES_PER_CHUNK
        data = audio_bytes[beg:end]
        await ws.send(data)
        
        # 模拟真实语速 (fast_mode 时跳过)
        if not fast_mode:
            await asyncio.sleep(CHUNK_DURATION_MS / 1000)
        else:
            await asyncio.sleep(0.005)

    # 发送结束信号
    print("\n⏹️ 音频发送完毕，发送结束信号...")
    await ws.send(json.dumps({"is_speaking": False}))
    
    # 等待最终结果
    await asyncio.sleep(2)


async def test_mic(ws):
    """从麦克风发送实时音频"""
    try:
        import pyaudio
    except ImportError:
        print("❌ 错误: 未安装 pyaudio")
        print("请运行: pip install pyaudio")
        print("如果安装失败，请先安装 portaudio:")
        print("  Ubuntu/Debian: sudo apt-get install portaudio19-dev")
        print("  macOS: brew install portaudio")
        return

    # 初始化 PyAudio
    p = pyaudio.PyAudio()
    
    # 打开麦克风流
    stream = p.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=BYTES_PER_CHUNK // 2  # 16-bit = 2 bytes per sample
    )
    
    print("🎤 麦克风已开启，开始录音...")
    print("💡 提示: 按 Ctrl+C 停止录音")
    print("-" * 40)
    
    try:
        while True:
            # 读取麦克风数据
            data = stream.read(BYTES_PER_CHUNK // 2, exception_on_overflow=False)
            
            # 发送到服务端
            await ws.send(data)
            
            # 让出控制权
            await asyncio.sleep(0.001)
            
    except KeyboardInterrupt:
        print("\n\n⏹️ 停止录音...")
    finally:
        # 发送结束信号
        await ws.send(json.dumps({"is_speaking": False}))
        
        # 等待最终结果
        await asyncio.sleep(2)
        
        # 关闭流
        stream.stop_stream()
        stream.close()
        p.terminate()


async def main(args):
    """主函数"""
    uri = f"ws://{HOST}:{PORT}"
    print(f"正在连接到服务端: {uri} ...")

    async with websockets.connect(uri, subprotocols=["binary"], ping_interval=None) as ws:
        print("✅ 连接成功！")

        # 发送握手配置
        config_msg = json.dumps({
            "mode": "2pass",
            "chunk_size": CHUNK_SIZE,
            "chunk_interval": CHUNK_INTERVAL,
            "wav_name": "mic" if args.mic else "file",
            "is_speaking": True,
            "itn": True
        })
        await ws.send(config_msg)
        print(f"📤 发送配置: {config_msg}")

        # 启动接收任务
        receiver = asyncio.create_task(recv_task(ws))

        # 根据模式选择输入源
        if args.mic:
            await test_mic(ws)
        else:
            await test_file(ws, args.file, args.fast)

        receiver.cancel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STT 测试客户端")
    parser.add_argument("-f", "--file", type=str, default=DEFAULT_AUDIO_FILE,
                        help=f"要检测的 wav 文件路径 (默认: {DEFAULT_AUDIO_FILE})")
    parser.add_argument("--fast", action="store_true",
                        help="快速模式，不模拟实时语速（用于快速测试）")
    parser.add_argument("--mic", action="store_true",
                        help="使用麦克风实时录音模式")
    args = parser.parse_args()
    
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\n停止测试")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")