import asyncio
import json
import logging
import os
import sys
import argparse
import numpy as np
import torch
import time
from concurrent.futures import ThreadPoolExecutor
from funasr import AutoModel
import websockets
from websockets.server import serve


def _setup_logging(log_subdir: str, log_filename: str) -> None:
    """配置日志：同时输出到控制台和文件，每次启动清空日志文件"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(project_root, "log", log_subdir)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


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
        # 推理并发控制：KWS 模型非线程安全，用单线程 executor + asyncio Lock 串行化调用
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._infer_lock: asyncio.Lock | None = None  # 在事件循环内创建
        # 加载模型
        self._load_model()
        
    def _load_model(self):
        """加载唤醒词模型"""
        logger.info(f"正在加载唤醒模型...")
        logging.getLogger('funasr').setLevel(logging.ERROR)
        logger.info(f"模型目录: {self.model_dir}")
        try:
            self.model = AutoModel(
                model=self.model_dir,
                init_param=self.weight_path,
                tokenizer_conf={
                    "token_list": self.token_list, 
                    "seg_dict": self.lexicon_list
                },
                frontend_conf={"cmvn_file": self.cmvn_file},
                # device="cuda:0" if torch.cuda.is_available() else "cpu",
                device="cpu",
                keywords=self.keywords,
                disable_update=True,
                disable_log=True,
                log_level="ERROR"
            )
            # device = "GPU" if torch.cuda.is_available() else "CPU"
            device = "CPU"
            logger.info(f"✅ 唤醒模型加载成功 (设备: {device}, 阈值: {self.threshold}, 唤醒词: {self.keywords})")
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise e
    
    def _detect_keyword(self, audio_data, client_logger: logging.Logger | None = None):
        """
        检测音频中是否包含唤醒词

        :param audio_data: numpy array 格式的音频数据
        :param client_logger: per-client logger，传入后日志自动携带客户端标识
        :return: (is_detected: bool, score: float, keyword: str, infer_time: float)
        """
        log = client_logger or logger
        try:
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            infer_start = time.time()
            with torch.no_grad():
                res = self.model.generate(
                    input=audio_data,
                    cache={},
                    disable_pbar=True
                )
                infer_time = (time.time() - infer_start) * 1000

            if res and len(res) > 0:
                text_output = res[0].get('text', '')
                if "detected" in text_output:
                    try:
                        score = float(text_output.split()[-1])
                        log.info("[TIMING] 检测到唤醒词: %s, 置信度: %.3f, 推理耗时: %.1fms",
                                 self.keywords, score, infer_time)
                        if score > self.threshold:
                            return True, score, self.keywords, infer_time
                    except ValueError:
                        pass

            return False, 0.0, "", infer_time

        except Exception as e:
            log.error("检测过程出错: %s", e)
            return False, 0.0, "", 0.0
    
    async def handle_client(self, websocket, path):
        """
        处理单个 WebSocket 客户端连接

        :param websocket: WebSocket 连接对象
        :param path: 请求路径
        """
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        clog = logging.getLogger(f"KWSServer.{client_id}")

        # 滑动窗口参数 (基于字节流，更健壮)
        min_samples = self.RATE // 2       # 至少 500ms 才开始推理
        max_samples = self.RATE * 2        # 最多保留 2s
        max_bytes   = max_samples * 2      # int16 = 2 bytes

        audio_buffer = b''

        clog.info("🔌 新客户端连接 (当前阈值: %.2f, 唤醒词: %s)", self.threshold, self.keywords)

        try:
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

            receive_count = 0
            last_log_time = time.time()
            first_audio_time = None

            async for message in websocket:
                try:
                    if isinstance(message, bytes):
                        recv_time = time.time()
                        receive_count += 1

                        if first_audio_time is None:
                            first_audio_time = recv_time
                            clog.info("[TIMING] 首次接收音频 @ %.3f", recv_time)

                        if recv_time - last_log_time >= 10.0:
                            clog.info("[TIMING] 音频接收统计: 过去10秒收到 %d 个数据包, 每包 %d 字节",
                                      receive_count, len(message))
                            receive_count = 0
                            last_log_time = recv_time

                        audio_buffer += message
                        if len(audio_buffer) > max_bytes:
                            audio_buffer = audio_buffer[-max_bytes:]

                        current_samples = len(audio_buffer) // 2
                        buffer_duration_ms = current_samples / self.RATE * 1000

                        if current_samples < min_samples:
                            continue

                        preprocess_start = time.time()
                        data_np = np.frombuffer(audio_buffer, dtype=np.int16)
                        data_input = data_np.astype(np.float32) / 32768.0
                        preprocess_time = (time.time() - preprocess_start) * 1000

                        # 推理（executor 中运行，避免阻塞事件循环；传入 clog 使推理日志同样带 client_id）
                        detect_start = time.time()
                        loop = asyncio.get_running_loop()
                        async with self._infer_lock:  # type: ignore[union-attr]
                            is_detected, score, keyword, infer_time = await loop.run_in_executor(
                                self._executor,
                                lambda d=data_input: self._detect_keyword(d, clog),
                            )
                        total_detect_time = (time.time() - detect_start) * 1000

                        if is_detected:
                            clog.info("[TIMING] ========== 唤醒成功 ==========")
                            clog.info("[TIMING] 置信度: %.3f", score)
                            clog.info("[TIMING] 缓冲区: %.0fms", buffer_duration_ms)
                            clog.info("[TIMING] 预处理: %.1fms", preprocess_time)
                            clog.info("[TIMING] 推理耗时: %.1fms", infer_time)
                            clog.info("[TIMING] 检测总耗时: %.1fms", total_detect_time)
                            clog.info("[TIMING] ================================")

                            send_start = time.time()
                            response = {
                                "type": "wake_detected",
                                "success": True,
                                "keyword": keyword,
                                "score": float(score),
                                "timestamp": asyncio.get_running_loop().time()
                            }
                            await websocket.send(json.dumps(response))
                            clog.info("[TIMING] 发送响应耗时: %.1fms", (time.time() - send_start) * 1000)

                            audio_buffer = b''
                            first_audio_time = None

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
                                clog.info("📴 客户端请求关闭连接")
                                break
                            else:
                                clog.warning("未知命令类型: %s", cmd_type)

                        except json.JSONDecodeError:
                            clog.warning("收到无效的 JSON 数据: %s", message[:100])

                except Exception as e:
                    clog.error("处理消息时出错: %s", e)
                    try:
                        await websocket.send(json.dumps({"type": "error", "message": str(e)}))
                    except Exception:
                        pass

        except websockets.exceptions.ConnectionClosed:
            clog.info("🔌 连接已关闭")
        except Exception as e:
            clog.error("❌ 连接处理异常: %s", e)
        finally:
            clog.info("📴 客户端断开连接")
    
    async def start_server(self):
        """启动 WebSocket 服务器"""
        self._infer_lock = asyncio.Lock()  # 在事件循环内创建，避免跨循环绑定问题
        logger.info(f"🚀 启动 WebSocket 唤醒词服务...")
        logger.info(f"   监听地址: ws://{self.host}:{self.port}")
        
        async with serve(self.handle_client, self.host, self.port):
            logger.info(f"✅ 服务器已启动，等待客户端连接...")
            await asyncio.Future()  # 永久运行


def main():
    _setup_logging("kws", "kws.log")

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

    
    args = parser.parse_args()
    
    # 检查模型路径
    if not os.path.exists(args.model_dir):
        logger.error(f"❌ 模型路径不存在: {args.model_dir}")
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
        
        # 运行服务器
        asyncio.run(server.start_server())
        
    except KeyboardInterrupt:
        logger.info("⚠️ 收到中断信号，正在关闭服务器...")
    except Exception as e:
        logger.error(f"❌ 服务器运行异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()