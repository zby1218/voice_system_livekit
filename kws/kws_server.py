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
from dataclasses import dataclass
from typing import Any

from funasr import AutoModel
import websockets
from websockets.server import serve


# ---------------------------------------------------------------------------
# 配置与常量
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KWSConfig:
    """KWS 服务配置：路径、音频参数、网络与推理常量。"""
    model_dir: str
    keywords: str = "小莫小莫,你好小莫"
    threshold: float = 0.4
    host: str = "0.0.0.0"
    port: int = 8765
    # 音频
    sample_rate: int = 16000
    channels: int = 1
    chunk_samples: int = 2400  # 150ms @ 16kHz
    min_samples_for_inference: int = 8000   # 500ms
    max_samples_in_buffer: int = 32000       # 2s
    # 日志统计间隔（秒）
    stats_interval_sec: float = 10.0

    @property
    def weight_path(self) -> str:
        return os.path.join(self.model_dir, "model_weight", "model.pt.avg10")

    @property
    def token_list(self) -> str:
        return os.path.join(self.model_dir, "tokens_2599.txt")

    @property
    def lexicon_list(self) -> str:
        return os.path.join(self.model_dir, "lexicon.txt")

    @property
    def cmvn_file(self) -> str:
        return os.path.join(self.model_dir, "am.mvn.dim80_l2r2")


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


# ---------------------------------------------------------------------------
# 协议消息构建（便于单测与复用）
# ---------------------------------------------------------------------------

def _make_welcome_message(config: KWSConfig) -> dict[str, Any]:
    """构建连接成功时的欢迎消息。"""
    return {
        "type": "connected",
        "message": "已连接到唤醒词检测服务",
        "keywords": config.keywords,
        "threshold": config.threshold,
        "audio_format": {
            "sample_rate": config.sample_rate,
            "channels": config.channels,
            "chunk_size": config.chunk_samples,
        },
    }


def _make_wake_detected_response(keyword: str, score: float) -> dict[str, Any]:
    """构建唤醒检测成功的响应。"""
    return {
        "type": "wake_detected",
        "success": True,
        "keyword": keyword,
        "score": float(score),
        "timestamp": asyncio.get_running_loop().time(),
    }


class KWSWebSocketServer:
    """基于 WebSocket 的唤醒词检测服务。职责：配置、模型加载、连接处理、推理调度。"""

    def __init__(
        self,
        model_dir: str = "./model",
        keywords: str = "小莫小莫,你好小莫",
        threshold: float = 0.4,
        host: str = "0.0.0.0",
        port: int = 8765,
        config: KWSConfig | None = None,
    ):
        self._config = config or KWSConfig(model_dir=model_dir, keywords=keywords, threshold=threshold, host=host, port=port)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._infer_lock: asyncio.Lock | None = None
        self._model: Any = None
        self._load_model()

    @property
    def config(self) -> KWSConfig:
        return self._config

    def _load_model(self) -> None:
        """加载唤醒词模型（仅读配置与模型文件，无副作用）。"""
        cfg = self._config
        logger.info("正在加载唤醒模型...")
        logging.getLogger("funasr").setLevel(logging.ERROR)
        logger.info("模型目录: %s", cfg.model_dir)
        try:
            self._model = AutoModel(
                model=cfg.model_dir,
                init_param=cfg.weight_path,
                tokenizer_conf={"token_list": cfg.token_list, "seg_dict": cfg.lexicon_list},
                frontend_conf={"cmvn_file": cfg.cmvn_file},
                device="cpu",
                keywords=cfg.keywords,
                disable_update=True,
                disable_log=True,
                log_level="ERROR",
            )
            logger.info("✅ 唤醒模型加载成功 (设备: CPU, 阈值: %s, 唤醒词: %s)", cfg.threshold, cfg.keywords)
        except Exception as e:
            logger.error("❌ 模型加载失败: %s", e)
            raise

    def _detect_keyword(self, audio_data: np.ndarray, client_logger: logging.Logger | None = None):
        """
        检测音频中是否包含唤醒词。

        :param audio_data: numpy array 格式的音频数据
        :param client_logger: per-client logger，传入后日志自动携带客户端标识
        :return: (is_detected: bool, score: float, keyword: str, infer_time: float)
        """
        cfg = self._config
        log = client_logger or logger
        try:
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            infer_start = time.time()
            with torch.no_grad():
                res = self._model.generate(
                    input=audio_data,
                    cache={},
                    disable_pbar=True,
                )
                infer_time = (time.time() - infer_start) * 1000

            if res and len(res) > 0:
                text_output = res[0].get("text", "")
                if "detected" in text_output:
                    try:
                        score = float(text_output.split()[-1])
                        log.info(
                            "[TIMING] 检测到唤醒词: %s, 置信度: %.3f, 推理耗时: %.1fms",
                            cfg.keywords,
                            score,
                            infer_time,
                        )
                        if score > cfg.threshold:
                            return True, score, cfg.keywords, infer_time
                    except ValueError:
                        pass

            return False, 0.0, "", infer_time

        except Exception as e:
            log.error("检测过程出错: %s", e)
            return False, 0.0, "", 0.0

    async def _handle_audio_frame(
        self,
        websocket: Any,
        chunk: bytes,
        audio_buffer: bytearray,
        first_audio_time: float | None,
        receive_count: int,
        last_log_time: float,
        clog: logging.Logger,
    ) -> tuple[float | None, int, float, bool]:
        """
        处理一帧二进制音频：追加、截断、满足长度则推理；若检测到唤醒则发响应。
        返回 (新 first_audio_time, 新 receive_count, 新 last_log_time, 是否已清空缓冲需重置 first_audio_time)。
        """
        cfg = self._config
        recv_time = time.time()
        receive_count += 1
        if first_audio_time is None:
            first_audio_time = recv_time
            clog.info("[TIMING] 首次接收音频 @ %.3f", recv_time)

        if recv_time - last_log_time >= cfg.stats_interval_sec:
            clog.info(
                "[TIMING] 音频接收统计: 过去%.0fs 收到 %d 个数据包, 每包 %d 字节",
                cfg.stats_interval_sec,
                receive_count,
                len(chunk),
            )
            receive_count = 0
            last_log_time = recv_time

        audio_buffer.extend(chunk)
        max_bytes = cfg.max_samples_in_buffer * 2
        if len(audio_buffer) > max_bytes:
            del audio_buffer[: len(audio_buffer) - max_bytes]

        current_samples = len(audio_buffer) // 2
        buffer_duration_ms = current_samples / cfg.sample_rate * 1000
        if current_samples < cfg.min_samples_for_inference:
            return first_audio_time, receive_count, last_log_time, False

        data_np = np.frombuffer(bytes(audio_buffer), dtype=np.int16)
        data_input = (data_np.astype(np.float32) / 32768.0).copy()

        loop = asyncio.get_running_loop()
        async with self._infer_lock:  # type: ignore[union-attr]
            is_detected, score, keyword, infer_time = await loop.run_in_executor(
                self._executor,
                lambda: self._detect_keyword(data_input, clog),
            )

        if is_detected:
            total_latency = (time.time() - first_audio_time) * 1000 if first_audio_time else 0
            clog.info("[TIMING] ========== 唤醒成功 ==========")
            clog.info("[TIMING] 置信度: %.3f", score)
            clog.info("[TIMING] 缓冲区: %.0fms", buffer_duration_ms)
            clog.info("[TIMING] 推理耗时: %.1fms", infer_time)
            clog.info("[TIMING] 从首次接收到唤醒: %.0fms", total_latency)
            clog.info("[TIMING] ================================")
            send_start = time.time()
            await websocket.send(json.dumps(_make_wake_detected_response(keyword, score)))
            clog.info("[TIMING] 发送响应耗时: %.1fms", (time.time() - send_start) * 1000)
            audio_buffer.clear()
            return None, receive_count, last_log_time, True

        return first_audio_time, receive_count, last_log_time, False

    async def _handle_text_command(
        self,
        websocket: Any,
        message: str,
        audio_buffer: bytearray,
        clog: logging.Logger,
    ) -> bool:
        """处理文本命令（ping/reset/close）。返回是否应关闭连接。"""
        try:
            cmd = json.loads(message)
            cmd_type = cmd.get("type")
            if cmd_type == "ping":
                await websocket.send(json.dumps({"type": "pong"}))
                return False
            if cmd_type == "reset":
                audio_buffer.clear()
                await websocket.send(json.dumps({"type": "reset_ack", "message": "缓冲区已清空"}))
                return False
            if cmd_type == "close":
                clog.info("📴 客户端请求关闭连接")
                return True
            clog.warning("未知命令类型: %s", cmd_type)
            return False
        except json.JSONDecodeError:
            clog.warning("收到无效的 JSON 数据: %s", message[:100])
            return False

    async def handle_client(self, websocket: Any, path: str) -> None:
        """处理单个 WebSocket 客户端连接：欢迎消息、按类型分发二进制/文本消息。"""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        clog = logging.getLogger(f"KWSServer.{client_id}")
        cfg = self._config

        audio_buffer = bytearray()
        receive_count = 0
        last_log_time = time.time()
        first_audio_time: float | None = None

        clog.info("🔌 新客户端连接 (当前阈值: %.2f, 唤醒词: %s)", cfg.threshold, cfg.keywords)

        try:
            await websocket.send(json.dumps(_make_welcome_message(cfg)))

            async for message in websocket:
                try:
                    if isinstance(message, bytes):
                        first_audio_time, receive_count, last_log_time, cleared = await self._handle_audio_frame(
                            websocket,
                            message,
                            audio_buffer,
                            first_audio_time,
                            receive_count,
                            last_log_time,
                            clog,
                        )
                        if cleared:
                            first_audio_time = None
                    elif isinstance(message, str):
                        should_close = await self._handle_text_command(websocket, message, audio_buffer, clog)
                        if should_close:
                            break
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

    async def start_server(self) -> None:
        """启动 WebSocket 服务器（在事件循环内创建 Lock）。"""
        self._infer_lock = asyncio.Lock()
        cfg = self._config
        logger.info("🚀 启动 WebSocket 唤醒词服务...")
        logger.info("   监听地址: ws://%s:%s", cfg.host, cfg.port)

        async with serve(self.handle_client, cfg.host, cfg.port):
            logger.info("✅ 服务器已启动，等待客户端连接...")
            await asyncio.Future()


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