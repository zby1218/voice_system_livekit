#!/usr/bin/env python3
"""
直接测试 TTS Server - 发送请求并实时播放音频
"""
import requests
import sounddevice as sd
import numpy as np
import io

# TTS Server 配置
TTS_SERVER_URL = "http://localhost:50000"
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_WAV_PATH = os.path.join(SCRIPT_DIR, "assets", "zero_shot_prompt.wav")
PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
TTS_TEXT = "今天天气怎么样？"

# 音频配置
SAMPLE_RATE = 24000
CHANNELS = 1


def test_tts_streaming():
    """测试 TTS 流式输出"""
    print(f"🎵 TTS Server: {TTS_SERVER_URL}")
    print(f"📝 Text: {TTS_TEXT}")
    print(f"🗣️ Prompt: {PROMPT_TEXT[:50]}...")
    print()
    
    # 读取 prompt wav
    with open(PROMPT_WAV_PATH, "rb") as f:
        prompt_wav_bytes = f.read()
    
    # 发送请求
    url = f"{TTS_SERVER_URL}/inference_zero_shot"
    data = {
        "tts_text": TTS_TEXT,
        "prompt_text": PROMPT_TEXT,
    }
    files = {
        "prompt_wav": ("prompt.wav", prompt_wav_bytes, "audio/wav"),
    }
    
    print("📤 发送请求...")
    
    try:
        response = requests.post(url, data=data, files=files, stream=True)
        response.raise_for_status()
        
        print("✅ 收到响应，开始播放...")
        
        # 收集所有音频数据
        audio_chunks = []
        total_bytes = 0
        
        for chunk in response.iter_content(chunk_size=4096):
            if chunk:
                audio_chunks.append(chunk)
                total_bytes += len(chunk)
                print(f"📦 收到 {len(chunk)} 字节 (累计: {total_bytes})", end="\r")
        
        print(f"\n✅ 共收到 {total_bytes} 字节")
        
        # 合并音频数据
        audio_data = b"".join(audio_chunks)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # 分析音频
        max_val = np.max(np.abs(audio_array))
        rms = np.sqrt(np.mean(audio_array.astype(np.float32)**2))
        print(f"� 音频分析: max={max_val}, RMS={rms:.1f}, 样本数={len(audio_array)}")
        
        if max_val < 100:
            print("⚠️ 警告: 音频幅度很低，可能是静音!")
        
        # 显示设备信息
        print(f"🔊 输出设备: {sd.query_devices(kind='output')['name']}")
        print(f"�🔊 播放 {len(audio_array)} 样本 @ {SAMPLE_RATE}Hz...")
        
        # 归一化到 float32 [-1, 1] 范围
        audio_float = audio_array.astype(np.float32) / 32768.0
        
        # 播放
        sd.play(audio_float, samplerate=SAMPLE_RATE)
        sd.wait()
        
        print("✅ 播放完成!")
        
    except requests.exceptions.ConnectionError:
        print("❌ 连接失败！请确保 TTS Server 正在运行")
    except Exception as e:
        print(f"❌ 错误: {e}")


if __name__ == "__main__":
    test_tts_streaming()
