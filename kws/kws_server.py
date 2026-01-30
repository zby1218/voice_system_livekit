import asyncio
import json
import logging
import os
import sys
import argparse
import numpy as np
import torch
import time
from funasr import AutoModel
import websockets
from websockets.server import serve

# 诊断工具
from audio_diagnostics import get_diagnostics

# 确保能找到 kws_engine.py
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class KWSWebSocketServer:
    """基于 WebSocket 的唤醒词检测服务"""
    
    def __init__(self, 
                 model_dir='./model',
                 keywords="小莫小莫,你好小莫",
                 threshold=0.4,
                 host='0.0.0.0',
                 port=8765):
        """
        初始化 WebSocket 唤醒词服务
        
        :param model_dir: 模型目录
        :param keywords: 唤醒词列表
        :param threshold: 唤醒置信度阈值 (0.0 ~ 1.0)
        :param host: WebSocket 服务器地址
        :param port: WebSocket 服务器端口
        """
        self.model_dir = model_dir
        self.keywords = keywords
        self.threshold = threshold
        self.host = host
        self.port = port
        
        # 诊断模式
        self.diagnostic_mode = False
        self.diagnostic_duration = 5.0  # 录制时长 (秒)
        self._diagnostics = None
        
        # 路径配置
        self.weight_path = os.path.join(model_dir, 'model_weight/model.pt.avg10')
        self.token_list = os.path.join(model_dir, 'tokens_2599.txt')
        self.lexicon_list = os.path.join(model_dir, 'lexicon.txt')
        self.cmvn_file = os.path.join(model_dir, 'am.mvn.dim80_l2r2')
        
        # 音频参数
        self.CHUNK = 2400  # 150ms per chunk at 16kHz
        self.RATE = 16000
        self.CHANNELS = 1
        # 加载模型
        self._load_model()
        
    def _load_model(self):
        """加载唤醒词模型"""
        logger.info(f"正在加载唤醒模型...")
        logging.getLogger('funasr').setLevel(logging.ERROR)
        # logger.info(self.model_dir)
        print(self.model_dir)
        try:
            self.model = AutoModel(
                model=self.model_dir,
                init_param=self.weight_path,
                tokenizer_conf={
                    "token_list": self.token_list, 
                    "seg_dict": self.lexicon_list
                },
                frontend_conf={"cmvn_file": self.cmvn_file},
                device="cuda:0" if torch.cuda.is_available() else "cpu",
                keywords=self.keywords,
                disable_update=True,
                disable_log=True,
                log_level="ERROR"
            )
            device = "GPU" if torch.cuda.is_available() else "CPU"
            print(f"✅ 唤醒模型加载成功 (设备: {device}, 阈值: {self.threshold}, 唤醒词: {self.keywords})")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise e
    
    def _detect_keyword(self, audio_data):
        """
        检测音频中是否包含唤醒词
        
        :param audio_data: numpy array 格式的音频数据
        :return: (is_detected: bool, score: float, keyword: str)
        """
        try:
            # print(f"检测音频数据: {audio_data.shape}")
            # 确保音频数据是 float32 格式
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # 推理
            with torch.no_grad():
                res = self.model.generate(
                    input=audio_data, 
                    cache={}, 
                    disable_pbar=True
                )
            
            # 解析结果
            if res and len(res) > 0:
                # print(res)
                text_output = res[0].get('text', '')
                if "detected" in text_output:
                    try:
                        # 解析分数: "detected 0.85" -> 0.85
                        score = float(text_output.split()[-1])
                        print(f"检测到唤醒词: {self.keywords}, 置信度: {score}")
                        if score > self.threshold:
                            return True, score, self.keywords
                    except ValueError:
                        pass
            
            return False, 0.0, ""
            
        except Exception as e:
            logger.error(f"检测过程出错: {e}")
            return False, 0.0, ""
    
    async def handle_client(self, websocket, path):
        """
        处理单个 WebSocket 客户端连接
        
        :param websocket: WebSocket 连接对象
        :param path: 请求路径
        """
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        print(f"🔌 新客户端连接: {client_id}")
        
        # 滑动窗口参数 (基于字节流，更健壮)
        self.RATE = 16000
        # 16kHz 采样率: 16000 samples = 1 秒
        self.MIN_SAMPLES_FOR_INFERENCE = 16000 // 2  # 至少 500ms 才开始推理
        self.MAX_SAMPLES_IN_BUFFER = 16000 * 2       # 最多保留 2s
        
        audio_buffer = b''
        
        # 诊断录制初始化
        diag_start_time = None
        diag_recording = False
        if self.diagnostic_mode:
            self._diagnostics = get_diagnostics(output_dir="./diagnostics")
            self._diagnostics.start_recording(source="kws_server")
            diag_start_time = time.time()
            diag_recording = True
            print(f"🔬 诊断模式: 录制 {self.diagnostic_duration}s 音频...")
        
        try:
            # 发送欢迎消息
            welcome_msg = {
                "type": "connected",
                "message": "已连接到唤醒词检测服务",
                "keywords": self.keywords,
                "threshold": self.threshold,
                "audio_format": {
                    "sample_rate": self.RATE,
                    "channels": self.CHANNELS,
                    "chunk_size": self.CHUNK
                }
            }
            await websocket.send(json.dumps(welcome_msg))
            
            # 持续接收音频数据
            async for message in websocket:
                try:
                    # 接收二进制音频数据
                    if isinstance(message, bytes):
                        # 诊断录制
                        if diag_recording and self._diagnostics:
                            self._diagnostics.push_audio(message)
                            elapsed = time.time() - diag_start_time
                            if elapsed >= self.diagnostic_duration:
                                print(f"🔬 诊断录制完成，正在分析...")
                                self._diagnostics.stop_recording()
                                diag_recording = False
                        
                        # 添加到缓冲区
                        audio_buffer += message
                        
                        # 限制缓冲区最大长度（保留最新的数据）
                        max_bytes = self.MAX_SAMPLES_IN_BUFFER * 2  # int16 = 2 bytes
                        if len(audio_buffer) > max_bytes:
                            audio_buffer = audio_buffer[-max_bytes:]
                        
                        # 检查是否积累足够数据
                        current_samples = len(audio_buffer) // 2
                        if current_samples < self.MIN_SAMPLES_FOR_INFERENCE:
                            continue
                        
                        # 全部做推理
                        data_np = np.frombuffer(audio_buffer, dtype=np.int16)
                        data_input = data_np.astype(np.float32) / 32768.0
                        
                        # 执行唤醒词检测
                        is_detected, score, keyword = self._detect_keyword(data_input)
                        
                        if is_detected:
                            print(f"✅ [{client_id}] 唤醒成功! Score: {score:.3f}")
                            
                            # 发送检测成功响应
                            response = {
                                "type": "wake_detected",
                                "success": True,
                                "keyword": keyword,
                                "score": float(score),
                                "timestamp": asyncio.get_event_loop().time()
                            }
                            await websocket.send(json.dumps(response))
                            
                            # 清空缓冲区 (防止一次说话多次触发)
                            audio_buffer = b''
                    
                    # 接收 JSON 控制命令
                    elif isinstance(message, str):
                        try:
                            cmd = json.loads(message)
                            cmd_type = cmd.get("type")
                            
                            if cmd_type == "ping":
                                await websocket.send(json.dumps({"type": "pong"}))
                            elif cmd_type == "reset":
                                audio_buffer = b''
                                await websocket.send(json.dumps({
                                    "type": "reset_ack",
                                    "message": "缓冲区已清空"
                                }))
                            elif cmd_type == "close":
                                print(f"📴 [{client_id}] 客户端请求关闭连接")
                                break
                            else:
                                print(f"未知命令类型: {cmd_type}")
                                
                        except json.JSONDecodeError:
                            print(f"收到无效的 JSON 数据: {message[:100]}")
                
                except Exception as e:
                    print(f"处理消息时出错: {e}")
                    error_msg = {
                        "type": "error",
                        "message": str(e)
                    }
                    await websocket.send(json.dumps(error_msg))
        
        except websockets.exceptions.ConnectionClosed:
            print(f"🔌 [{client_id}] 连接已关闭")
        except Exception as e:
            print(f"❌ [{client_id}] 连接处理异常: {e}")
        finally:
            print(f"📴 [{client_id}] 客户端断开连接")
    
    async def start_server(self):
        """启动 WebSocket 服务器"""
        print(f"🚀 启动 WebSocket 唤醒词服务...")
        print(f"   监听地址: ws://{self.host}:{self.port}")
        
        async with serve(self.handle_client, self.host, self.port):
            print(f"✅ 服务器已启动，等待客户端连接...")
            await asyncio.Future()  # 永久运行


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_dir = os.path.join(base_dir, "model")

    parser = argparse.ArgumentParser(description="WebSocket 唤醒词检测服务")
    parser.add_argument("--model_dir", type=str, default=default_model_dir, 
                        help="模型路径")
    parser.add_argument("--keywords", type=str, default="小莫小莫,你好小莫", 
                        help="唤醒词,用逗号分隔")
    parser.add_argument("--threshold", type=float, default=0.4, 
                        help="唤醒阈值 (0.0-1.0)")
    parser.add_argument("--host", type=str, default="0.0.0.0", 
                        help="服务器地址")
    parser.add_argument("--port", type=int, default=8765, 
                        help="服务器端口")
    parser.add_argument("--diagnose", action="store_true",
                        help="启用诊断模式，录制并分析音频")
    parser.add_argument("--diagnose-duration", type=float, default=5.0,
                        help="诊断录制时长 (秒)")
    
    args = parser.parse_args()
    
    # 检查模型路径
    if not os.path.exists(args.model_dir):
        print(f"❌ 模型路径不存在: {args.model_dir}")
        sys.exit(1)
    
    try:
        # 创建并启动服务器
        server = KWSWebSocketServer(
            model_dir=args.model_dir,
            keywords=args.keywords,
            threshold=args.threshold,
            host=args.host,
            port=args.port
        )
        
        # 应用诊断模式
        if args.diagnose:
            server.diagnostic_mode = True
            server.diagnostic_duration = args.diagnose_duration
            print(f"🔬 诊断模式已启用，将录制 {args.diagnose_duration}s 音频")
        
        # 运行服务器
        asyncio.run(server.start_server())
        
    except KeyboardInterrupt:
        print("\n⚠️ 收到中断信号，正在关闭服务器...")
    except Exception as e:
        print(f"❌ 服务器运行异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()