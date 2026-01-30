import asyncio
import json
import pyaudio
import websockets
import numpy as np
import argparse
from contextlib import contextmanager
from ctypes import *

# ==========================================
# ALSA 错误屏蔽器 (防止 PyAudio 初始化刷屏)
# ==========================================
ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
def py_error_handler(filename, line, function, err, fmt): pass
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

@contextmanager
def no_alsa_error():
    try:
        asound = cdll.LoadLibrary('libasound.so')
        asound.snd_lib_error_set_handler(c_error_handler)
        yield
        asound.snd_lib_error_set_handler(None)
    except OSError:
        yield


class KWSWebSocketClient:
    """WebSocket 唤醒词检测客户端"""
    
    def __init__(self, server_url="ws://localhost:8765"):
        """
        初始化 WebSocket 客户端
        
        :param server_url: WebSocket 服务器地址
        """
        self.server_url = server_url
        
        # 音频参数 (需要与服务器保持一致)
        self.CHUNK = 2400  # 150ms per chunk at 16kHz
        self.RATE = 16000
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        
        self.running = False
        self.websocket = None
        
    async def connect_and_stream(self):
        """连接到服务器并开始音频流传输"""
        try:
            async with websockets.connect(self.server_url) as websocket:
                self.websocket = websocket
                print(f"✅ 已连接到服务器: {self.server_url}")
                
                # 接收欢迎消息
                welcome = await websocket.recv()
                welcome_data = json.loads(welcome)
                print(f"📨 服务器消息: {welcome_data.get('message')}")
                print(f"   唤醒词: {welcome_data.get('keywords')}")
                print(f"   阈值: {welcome_data.get('threshold')}")
                
                # 初始化音频流
                with no_alsa_error():
                    p = pyaudio.PyAudio()
                
                stream = p.open(
                    format=self.FORMAT,
                    channels=self.CHANNELS,
                    rate=self.RATE,
                    input=True,
                    frames_per_buffer=self.CHUNK
                )
                
                print("\n🎤 开始采集音频，请说出唤醒词...\n")
                self.running = True
                
                # 创建两个并发任务：发送音频 和 接收响应
                send_task = asyncio.create_task(self._send_audio(stream, websocket))
                recv_task = asyncio.create_task(self._receive_responses(websocket))
                
                # 等待任务完成
                await asyncio.gather(send_task, recv_task)
                
                # 清理资源
                stream.stop_stream()
                stream.close()
                p.terminate()
                
        except websockets.exceptions.ConnectionClosed:
            print("\n❌ 与服务器的连接已断开")
        except Exception as e:
            print(f"\n❌ 连接错误: {e}")
        finally:
            self.running = False
    
    async def _send_audio(self, stream, websocket):
        """持续发送音频数据到服务器"""
        try:
            while self.running:
                # 读取音频数据
                audio_data = stream.read(self.CHUNK, exception_on_overflow=False)
                
                # 发送二进制数据到服务器
                await websocket.send(audio_data)
                
                # 短暂延迟，避免过快发送
                await asyncio.sleep(0.01)
                
        except Exception as e:
            print(f"发送音频时出错: {e}")
            self.running = False
    
    async def _receive_responses(self, websocket):
        """接收服务器响应"""
        try:
            while self.running:
                response = await websocket.recv()
                
                # 解析 JSON 响应
                try:
                    data = json.loads(response)
                    msg_type = data.get("type")
                    
                    if msg_type == "wake_detected":
                        score = data.get("score", 0)
                        keyword = data.get("keyword", "")
                        print(f"\n🎉 ===============================")
                        print(f"   唤醒成功!")
                        print(f"   关键词: {keyword}")
                        print(f"   置信度: {score:.3f}")
                        print(f"   ===============================\n")
                        
                        # 可选：检测成功后停止
                        # self.running = False
                        # break
                        
                    elif msg_type == "error":
                        print(f"⚠️ 服务器错误: {data.get('message')}")
                    
                except json.JSONDecodeError:
                    print(f"收到非 JSON 响应: {response}")
                    
        except websockets.exceptions.ConnectionClosed:
            print("连接已关闭")
            self.running = False
        except Exception as e:
            print(f"接收响应时出错: {e}")
            self.running = False
    
    async def send_command(self, command_type, **kwargs):
        """发送控制命令到服务器"""
        if self.websocket:
            cmd = {"type": command_type, **kwargs}
            await self.websocket.send(json.dumps(cmd))


async def main():
    parser = argparse.ArgumentParser(description="WebSocket 唤醒词检测客户端")
    parser.add_argument("--server", type=str, default="ws://localhost:8765",
                        help="WebSocket 服务器地址")
    
    args = parser.parse_args()
    
    client = KWSWebSocketClient(server_url=args.server)
    
    try:
        await client.connect_and_stream()
    except KeyboardInterrupt:
        print("\n\n⚠️ 手动停止客户端")
        client.running = False


if __name__ == "__main__":
    asyncio.run(main())