#!/usr/bin/env python3
"""
直接测试 TTS Server - 先保存 WAV，再尝试播放
"""
import requests
import numpy as np
import wave
import os

# TTS Server 配置
TTS_SERVER_URL = "http://localhost:50000"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_WAV_PATH = os.path.join(SCRIPT_DIR, "assets", "zero_shot_prompt.wav")
PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
TTS_TEXT = "见到你很高兴！"
OUTPUT_WAV = os.path.join(SCRIPT_DIR, "tts_output.wav")

# 音频配置
SAMPLE_RATE = 24000
CHANNELS = 1


def test_tts():
    """测试 TTS - 先保存 WAV，再尝试播放"""
    print(f"🎵 TTS Server: {TTS_SERVER_URL}")
    print(f"📝 Text: {TTS_TEXT}")
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
        response = requests.post(url, data=data, files=files, stream=True, timeout=30)
        response.raise_for_status()
        
        print("✅ 收到响应，接收音频数据...")
        
        # 收集所有音频数据
        audio_chunks = []
        total_bytes = 0
        
        for chunk in response.iter_content(chunk_size=4096):
            if chunk:
                total_bytes += len(chunk)
                audio_chunks.append(chunk)
                print(f"📦 收到 {len(chunk)} 字节 (累计: {total_bytes})", end="\r")
        
        print(f"\n✅ 共收到 {total_bytes} 字节")
        
        # 合并音频数据
        audio_data = b"".join(audio_chunks)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # 分析音频
        max_val = np.max(np.abs(audio_array))
        rms = np.sqrt(np.mean(audio_array.astype(np.float32)**2))
        duration = len(audio_array) / SAMPLE_RATE
        print(f"📊 音频分析: max={max_val}, RMS={rms:.1f}, 样本数={len(audio_array)}, 时长={duration:.2f}s")
        
        # 保存为 WAV 文件
        with wave.open(OUTPUT_WAV, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data)
        
        print(f"💾 已保存到: {OUTPUT_WAV}")
        
        # 尝试播放
        print("\n🔊 尝试播放...")
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            
            stream = p.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                output=True,
                frames_per_buffer=2400,
            )
            
            # 分块写入，每块 100ms
            chunk_size = SAMPLE_RATE // 10 * 2  # 100ms of int16 data
            for i in range(0, len(audio_data), chunk_size):
                stream.write(audio_data[i:i+chunk_size])
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            print("✅ 播放完成!")
            
        except Exception as e:
            print(f"⚠️ 播放失败: {e}")
            print(f"   请手动播放: {OUTPUT_WAV}")
        
    except requests.exceptions.ConnectionError:
        print("❌ 连接失败！请确保 TTS Server 正在运行")
    except requests.exceptions.Timeout:
        print("❌ 请求超时！")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_tts()
