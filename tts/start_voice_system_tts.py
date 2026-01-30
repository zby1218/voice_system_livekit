#!/usr/bin/env python3
"""
直接启动 voice_system TTS Server（不需要 ROS2）
"""
import os
import sys

# 设置路径
TTS_PKG_DIR = "/home/zhangchi/project/faw_application/voice_system/src/tts"
sys.path.insert(0, TTS_PKG_DIR)
sys.path.insert(0, os.path.join(TTS_PKG_DIR, 'third_party', 'Matcha-TTS'))

MODEL_DIR = os.path.join(TTS_PKG_DIR, 'model', 'Fun-CosyVoice3-0.5B')
ASSET_DIR = os.path.join(TTS_PKG_DIR, 'asset')

import uvicorn
from tts.tts_server import app, load_model

if __name__ == "__main__":
    print(f"📁 模型: {MODEL_DIR}")
    print(f"📁 音色: {ASSET_DIR}")
    print()
    
    print("🔄 加载模型...")
    load_model(MODEL_DIR, ASSET_DIR, "cuda", True, False)
    print("✅ 模型加载完成")
    print()
    
    print("🚀 启动服务器: http://localhost:50000")
    uvicorn.run(app, host="0.0.0.0", port=50000)
