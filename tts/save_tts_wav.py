#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import wave
import io

import httpx
import numpy as np

TTS_URL = "http://127.0.0.1:50000/tts/stream"
OUTPUT_WAV_ORIGINAL = "power_low.wav"
OUTPUT_WAV_AMPLIFIED = "power_no_service.wav"
GAIN = 2.0  # 增益倍数
TEXT = "未识别到电量通知服务，请联系技术人员！"


# """
# 红旗天工"Sconcept"以东方美学理念和新锐运动轿跑姿态呈现，自带潮流气场，满足用户彰显个性的审美需求。高度集成的Onebox舱驾一体设计，和以AI为原生能力的座舱体验，从人车交互进化至情感共生，加上L3级智能驾驶，让它成为有温度的“智能生命体”。
# 红旗天工"Sconcept"还将AI与前沿底盘技术融合，将底盘从被动支撑升级为会预判，懂感知，秒响应的“智慧老司机”，无论城市穿梭、高速巡航，还是山路驰骋，都能自适应路况、自适应风格、自适应心境，带来随心而动的驾驭享受。
# 此外，它还是国内首款在制造环节采用开箱工艺的产品，从源头开始重构，整车品质再提升。红旗天工 "Sconcept" 计划于2027年下半年上市，相信它将成为先锋人群彰显个性,享受科技与品质的理想之选，敬请各位期待。  

# """


def amplify(pcm_bytes: bytes, gain: float) -> bytes:
    """软限幅增益：放大后用 tanh 平滑压限，避免硬截断失真。"""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    samples /= 32768.0
    samples = np.tanh(samples * gain)
    return (samples * 32768).clip(-32768, 32767).astype(np.int16).tobytes()


def save_wav(path: str, pcm_bytes: bytes, sample_rate: int):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    duration = len(pcm_bytes) / 2 / sample_rate
    print(f"  saved → {path}  ({duration:.2f}s, {len(pcm_bytes)} bytes)")


def main():
    pcm_buffer = io.BytesIO()

    # 1) 发起流式请求，接收所有 PCM 数据
    with httpx.Client(timeout=None, trust_env=False) as client:
        with client.stream(
            "POST",
            TTS_URL,
            data={
                "text": TEXT,
                "voice_id": "longanyun",
                "session_id": "demo",
                "interrupt_mode": "normal",
            },
        ) as resp:
            resp.raise_for_status()

            sr = int(resp.headers.get("X-Sample-Rate", "24000"))
            print(f"sample_rate={sr}, receiving...")

            chunk_count = 0
            for chunk in resp.iter_bytes():
                if chunk:
                    pcm_buffer.write(chunk)
                    chunk_count += 1
                    print(f"received chunk #{chunk_count}, size={len(chunk)}")

    # 2) 保存原始版本 & 增强响度版本，方便对比
    pcm_data = pcm_buffer.getvalue()
    print(f"\n[原始]")
    save_wav(OUTPUT_WAV_ORIGINAL, pcm_data, sr)

    print(f"\n[增益 x{GAIN}，软限幅]")
    save_wav(OUTPUT_WAV_AMPLIFIED, amplify(pcm_data, GAIN), sr)


if __name__ == "__main__":
    main()
