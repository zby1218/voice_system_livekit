#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import httpx
import sounddevice as sd
import numpy as np

TTS_URL = "http://127.0.0.1:50000/tts/stream"
TEXT = "一汽红旗以46万年销量、14.9万新能源车的成绩完成华丽转身，全固态电池200℃不起火、产线OTA两小时升级等硬核技术，证明其从央企责任到市场突围的实力。2026年7款新车将携华为全栈方案出击，技术+体系创新正改写中国高端汽车格局。"

def main():
    # 1) 发起流式请求
    with httpx.Client(timeout=None) as client:
        with client.stream(
            "POST",
            TTS_URL,
            data={
                "text": TEXT,
                # 可选参数
                "voice_id": "default",
                "session_id": "demo",
                "interrupt_mode": "normal",
            },
        ) as resp:
            resp.raise_for_status()

            # 2) 从响应头拿采样率（服务端会返回 X-Sample-Rate）
            sr = int(resp.headers.get("X-Sample-Rate", "24000"))
            print(f"sample_rate={sr}, start playing...")

            # 3) 打开播放流，边接收边播放（16-bit PCM mono）
            with sd.RawOutputStream(
                samplerate=sr,
                channels=1,
                dtype="int16",
                blocksize=0,  # 让底层自适应
            ) as stream:
                for chunk in resp.iter_bytes():
                    if chunk:
                        stream.write(chunk)

    print("done.")

if __name__ == "__main__":
    main()