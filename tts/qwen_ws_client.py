#!/usr/bin/env python3
"""
连接 qwen_tts_server，发送一段话，接收流式 PCM 并用 PyAudio 播放。

用法: python qwen_ws_client.py [--url ws://localhost:50001/ws/tts] [--text "要合成的内容"]
"""
import argparse
import asyncio
import json
import queue
import threading

import pyaudio
import websockets

SAMPLE_RATE = 24000
CHANNELS = 1
PLAY_CHUNK = 1024


def play_worker(audio_queue: queue.Queue):
    """从队列取 PCM 字节并写入 PyAudio 播放。"""
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        output=True,
        frames_per_buffer=PLAY_CHUNK,
    )
    while True:
        data = audio_queue.get()
        if data is None:
            break
        stream.write(data)
    stream.stop_stream()
    stream.close()
    pa.terminate()


async def run(url: str, text: str):
    audio_queue = queue.Queue()
    player = threading.Thread(target=play_worker, args=(audio_queue,))
    player.start()

    try:
        async with websockets.connect(url, open_timeout=10) as ws:
            await ws.send(json.dumps({
                "text": text,
                "speaker": "Vivian",
                "language": "Chinese",
                "instruct": "用亲切的语气说",
            }))
            while True:
                msg = await ws.recv()
                if isinstance(msg, bytes):
                    audio_queue.put(msg)
                else:
                    obj = json.loads(msg)
                    if obj.get("done"):
                        break
                    if obj.get("error"):
                        print("服务端错误:", obj["error"])
                        break
    finally:
        audio_queue.put(None)
        player.join()


def main():
    parser = argparse.ArgumentParser(description="Qwen TTS WebSocket 客户端，发送文本并播放")
    parser.add_argument("--url", type=str, default="ws://localhost:50001/ws/tts", help="WebSocket 地址")
    parser.add_argument("--text", type=str, default="赏花灯、猜灯谜、包汤圆等元宵民俗贯穿晚会始终。开场歌舞《元宵正好》将街巷欢腾的元宵景致搬上舞台，一场团圆喜庆的元宵游园会拉开帷幕。歌曲《千灯万户》里鱼灯、龙灯、兔儿灯、走马灯、花篮灯等精彩亮相，千家万户闹花灯的热气腾腾尽收眼底。", help="要合成的文本")
    args = parser.parse_args()

    asyncio.run(run(args.url, args.text))


if __name__ == "__main__":
    main()
