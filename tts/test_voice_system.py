#!/usr/bin/env python3
"""
测试 voice_system 的 TTS Server - 使用 /tts/stream 接口
"""
import requests
import numpy as np
import wave
import os

# TTS Server 配置 - voice_system 版本
TTS_SERVER_URL = "http://localhost:50000"
TTS_TEXT = "见到你很高兴！"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_WAV = os.path.join(SCRIPT_DIR, "tts_output_voice_system.wav")

# 音频配置 - voice_system 输出 16kHz
SAMPLE_RATE = 16000
CHANNELS = 1


def test_voice_system_tts():
    """测试 voice_system 的 /tts/stream 接口"""
    print(f"🎵 TTS Server (voice_system): {TTS_SERVER_URL}")
    print(f"📝 Text: {TTS_TEXT}")
    print()
    
    # 发送请求到 /tts/stream
    url = f"{TTS_SERVER_URL}/tts/stream"
    data = {
        "text": TTS_TEXT,
        "voice_id": "default",
    }
    
    print("📤 发送请求到 /tts/stream ...")
    
    try:
        response = requests.post(url, data=data, stream=True, timeout=30)
        response.raise_for_status()
        
        # 从响应头获取采样率
        sample_rate = int(response.headers.get("X-Sample-Rate", SAMPLE_RATE))
        print(f"✅ 收到响应，采样率: {sample_rate} Hz")
        
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
        duration = len(audio_array) / sample_rate
        print(f"📊 音频分析: max={max_val}, RMS={rms:.1f}, 样本数={len(audio_array)}, 时长={duration:.2f}s")
        
        # 保存为 WAV 文件
        with wave.open(OUTPUT_WAV, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
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
                rate=sample_rate,
                output=True,
                frames_per_buffer=1600,
            )
            
            chunk_size = sample_rate // 10 * 2
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
    test_voice_system_tts()
