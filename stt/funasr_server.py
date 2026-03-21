#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import os
import sys
import json
import asyncio
import logging
import websockets
import torch
import numpy as np
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

# 忽略警告
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 配置与常量
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ASRServerConfig:
    """ASR 服务配置，集中管理常量和可调参数。"""
    sample_rate: int = 16000
    min_duration_sec: float = 0.5
    energy_threshold: float = 50.0
    inference_workers: int = 5
    default_hotwords: list = field(default_factory=lambda: ["天工"])
    model_subdir: str = "FunAudioLLM/Fun-ASR-Nano-2512"


# ---------------------------------------------------------------------------
# 音频过滤（防幻觉）
# ---------------------------------------------------------------------------

class AudioFilter:
    """对即将送入 ASR 的音频做前置过滤，过短或过静则跳过推理。"""

    def __init__(self, config: ASRServerConfig, logger: logging.Logger):
        self._config = config
        self._logger = logger
        # 低于0.5秒的音频不进行推理
        self._min_bytes = int(config.sample_rate * config.min_duration_sec) * 2

    def should_skip(
        self, audio_bytes: bytes, logger: logging.Logger | None = None
    ) -> tuple[bool, str]:
        """
        判断是否应跳过本次推理（以全量音频为准）。
        :return: (是否跳过, 原因描述)，不跳过时原因为空字符串。
        """
        log = logger or self._logger
        if len(audio_bytes) == 0:
            return True, "空音频"

        if len(audio_bytes) < self._min_bytes:
            log.info(
                "音频过短 (%s bytes < %s bytes)，跳过推理",
                len(audio_bytes), self._min_bytes,
            )
            return True, "过短"

        data_int16 = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(data_int16 ** 2)))
        if rms < self._config.energy_threshold:
            log.info(
                "音频能量过低 (RMS=%.1f < %s)，跳过推理",
                rms, self._config.energy_threshold,
            )
            return True, "能量过低"

        return False, ""

    def should_skip_with_prebuffer(
        self, audio_bytes: bytes, prebuffer_bytes: int, logger: logging.Logger | None = None
    ) -> tuple[bool, str]:
        """
        基于真实语音部分（去除预缓冲帧后）进行过滤。

        VAD 门控预缓冲在音频流最前端，不代表真实人声，用它来满足最短时长
        和能量门限会让原本该过滤的幻觉音频漏进推理。
        此方法对真实语音部分独立运行 should_skip()，预缓冲部分不参与判断。

        :param audio_bytes:    本轮累积的全量 PCM bytes
        :param prebuffer_bytes: 其中前几字节属于预缓冲（0 表示不区分）
        :param logger:         可选的 per-client logger，未传则使用全局 logger
        :return: (是否跳过, 原因描述)
        """
        log = logger or self._logger
        if prebuffer_bytes <= 0:
            return self.should_skip(audio_bytes, log)

        real_speech = audio_bytes[prebuffer_bytes:]
        skip, reason = self.should_skip(real_speech, log)
        if skip:
            log.info(
                "真实语音部分过滤命中（预缓冲=%d bytes，真实=%d bytes）: %s",
                prebuffer_bytes, len(real_speech), reason,
            )
        return skip, reason


# ---------------------------------------------------------------------------
# WebSocket 会话状态
# ---------------------------------------------------------------------------

@dataclass
class ClientSession:
    """单条 WebSocket 连接对应的会话状态。"""
    status_dict_asr_online: dict = field(default_factory=lambda: {"cache": {}, "is_final": False})
    chunk_interval: int = 10
    mode: str = "2pass"
    is_speaking: bool = True
    itn: bool = True
    # 预缓冲字节数：客户端在 is_speaking=True 的配置包里携带，
    # 表示本轮累积音频的前 N 字节是 VAD 门控预缓冲，并非真实人声。
    # 服务端在过滤时只看 [prebuffer_bytes:] 之后的真实语音部分。
    prebuffer_bytes: int = 0


# ---------------------------------------------------------------------------
# 依赖与入口
# ---------------------------------------------------------------------------

def _ensure_funasr():
    """确保 funasr 可用，不可用时退出进程。"""
    try:
        from funasr.models.fun_asr_nano.model import FunASRNano  # noqa: F401
        from funasr import AutoModel  # noqa: F401
    except ImportError:
        print("❌ 错误: 未安装 funasr。请运行: pip install funasr torch websockets numpy")
        sys.exit(1)


def _suppress_transformers_logs():
    try:
        import transformers
        transformers.logging.set_verbosity_error()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# ASR 服务
# ---------------------------------------------------------------------------

class ASRServer:
    """基于 FunASR 的 WebSocket ASR 服务：接收 PCM 流，按 is_speaking 切句并做离线识别。"""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 10095,
        device: str = "cuda",
        config: ASRServerConfig | None = None,
    ):
        self._host = host
        self._port = port
        self._device = device if torch.cuda.is_available() else "cpu"
        self._config = config or ASRServerConfig()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_root = os.path.join(current_dir, "model")
        self._path_asr = os.path.join(model_root, self._config.model_subdir)

        self._logger = logging.getLogger("ASRServer")
        self._logger.info("锁定模型根目录: %s", model_root)
        self._check_path(self._path_asr, "ASR")

        self._model_asr = None
        self._websocket_users: set = set()
        self._server = None
        self._executor = ThreadPoolExecutor(max_workers=self._config.inference_workers)
        self._audio_filter = AudioFilter(self._config, self._logger)
        self._infer_lock: asyncio.Lock | None = None

    def _check_path(self, path: str, name: str) -> None:
        if not os.path.exists(path):
            self._logger.error("❌ 找不到 %s 模型路径: %s", name, path)
            sys.exit(1)
        self._logger.info("✅ %s 路径确认: %s", name, path)

    def load_models(self) -> None:
        from funasr import AutoModel

        self._logger.info("开始加载模型 (Device: %s)...", self._device)
        try:
            self._model_asr = AutoModel(
                model=self._path_asr,
                device=self._device,
                disable_pbar=True,
                disable_log=True,
                disable_update=True,
                local_files_only=True,
            )
            self._logger.info("🎉 模型加载成功！服务准备就绪。")
        except Exception as e:
            self._logger.error("❌ 模型加载崩溃: %s", e)
            sys.exit(1)

    async def start(self) -> None:
        self._infer_lock = asyncio.Lock()  # 在事件循环内创建，避免跨循环绑定问题
        self.load_models()
        self._server = await websockets.serve(
            self._ws_serve,
            self._host,
            self._port,
            subprotocols=None,
            ping_interval=None,
        )
        self._logger.info("🚀 服务已启动，监听地址: ws://%s:%s", self._host, self._port)
        await asyncio.Future()

    @staticmethod
    def _decode_audio_chunk(chunk_bytes: bytes, sample_rate: int = 16000) -> torch.Tensor:
        """PCM s16le -> float32 tensor（假设 16kHz）。"""
        data_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
        data_float32 = data_int16.astype(np.float32) / 32768.0
        return torch.from_numpy(data_float32)

    async def _run_model_inference(self, model, input_tensor, **kwargs):
        loop = asyncio.get_running_loop()
        async with self._infer_lock:  # type: ignore[union-attr]
            return await loop.run_in_executor(
                self._executor,
                lambda: model.generate(input=input_tensor, **kwargs),
            )

    def _get_or_create_session(self, websocket) -> ClientSession:
        if not hasattr(websocket, "_asr_session"):
            websocket._asr_session = ClientSession()
        return websocket._asr_session

    async def _ws_serve(self, websocket, path=None) -> None:
        self._websocket_users.add(websocket)
        session = self._get_or_create_session(websocket)
        frames_asr: list = []

        addr = websocket.remote_address
        client_id = f"{addr[0]}:{addr[1]}" if addr else "unknown"
        logger = logging.getLogger(f"ASRServer.{client_id}")
        logger.info("新客户端连接 (当前连接数: %d)", len(self._websocket_users))

        try:
            async for message in websocket:
                if isinstance(message, str):
                    await self._handle_text_message(websocket, session, message, frames_asr, logger)
                else:
                    frames_asr.append(message)
        except websockets.ConnectionClosed:
            logger.info("客户端断开连接 (剩余连接数: %d)", len(self._websocket_users) - 1)
        except Exception as e:
            logger.error("处理异常: %s", e)
            import traceback
            traceback.print_exc()
        finally:
            self._websocket_users.discard(websocket)

    async def _handle_text_message(
        self,
        websocket,
        session: ClientSession,
        message: str,
        frames_asr: list,
        logger: logging.Logger,
    ) -> None:
        logger.info("📨 Received text message: %s", message[:200])
        try:
            msg = json.loads(message)
        except Exception as e:
            logger.error("JSON处理错误: %s", e)
            return

        if "chunk_size" in msg:
            chunk = msg["chunk_size"]
            if isinstance(chunk, str):
                chunk = [int(x) for x in chunk.split(",")]
            session.status_dict_asr_online["chunk_size"] = chunk

        if "chunk_interval" in msg:
            session.chunk_interval = int(msg["chunk_interval"])

        if "mode" in msg:
            session.mode = msg["mode"]

        if "itn" in msg:
            session.itn = bool(msg["itn"])

        # 客户端携带的预缓冲字节数（由 VadGatedActivity 填充）
        if "prebuffer_bytes" in msg:
            session.prebuffer_bytes = max(0, int(msg["prebuffer_bytes"]))

        if "is_speaking" not in msg:
            return

        session.is_speaking = msg["is_speaking"]
        session.status_dict_asr_online["is_final"] = not session.is_speaking

        if not session.is_speaking:
            if len(frames_asr) > 0:
                audio_in = b"".join(frames_asr)
                prebuffer = session.prebuffer_bytes
                session.prebuffer_bytes = 0  # 消费后立即清零，避免污染下一轮
                await self._async_asr_offline(websocket, session, audio_in, prebuffer, logger)
            frames_asr.clear()
            session.status_dict_asr_online["cache"] = {}

    async def _send_empty_final(self, websocket) -> None:
        """跳过推理时仍返回空 final，让客户端正常结束本轮等待，避免挂住。"""
        try:
            await websocket.send(json.dumps({
                "mode": "2pass-offline", "text": "", "is_final": True
            }))
        except Exception as e:
            self._logger.warning("发送空 final 失败: %s", e)

    async def _async_asr_offline(
        self,
        websocket,
        session: ClientSession,
        audio_in: bytes,
        prebuffer_bytes: int = 0,
        logger: logging.Logger | None = None,
    ) -> None:
        logger = logger or self._logger
        if len(audio_in) == 0:
            await self._send_empty_final(websocket)
            return

        # 使用感知预缓冲的过滤：只对真实语音部分判断时长/能量
        skip, _ = self._audio_filter.should_skip_with_prebuffer(audio_in, prebuffer_bytes, logger)
        if skip:
            await self._send_empty_final(websocket)
            return

        try:
            logger.info("开始推理 (音频 %.2f 秒，等待锁...)", len(audio_in) / (self._config.sample_rate * 2))
            audio_tensor = self._decode_audio_chunk(audio_in, self._config.sample_rate)
            res = await self._run_model_inference(
                self._model_asr,
                [audio_tensor],
                language="zh",
                hotwords=list(self._config.default_hotwords),
                itn=session.itn,
            )
            text = res[0]["text"] if res else ""

            logger.info("Final Result: %s", text)

            await websocket.send(json.dumps({
                "mode": "2pass-offline", "text": text, "is_final": True
            }))
        except Exception as e:
            logger.error("离线识别错误: %s", e)

    async def async_asr_online(self, websocket, audio_in: bytes) -> None:
        """在线流式识别（当前协议下由客户端决定是否使用）。"""
        if len(audio_in) == 0:
            return
        session = self._get_or_create_session(websocket)
        try:
            audio_tensor = self._decode_audio_chunk(audio_in, self._config.sample_rate)
            res = await self._run_model_inference(
                self._model_asr,
                [audio_tensor],
                **session.status_dict_asr_online,
            )
            text = res[0]["text"] if res else ""
            if text:
                await websocket.send(json.dumps({
                    "mode": "2pass-online", "text": text, "is_final": False
                }))
        except Exception as e:
            self._logger.warning("Online 推理异常: %s", e)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _setup_logging("stt", "stt.log")
    _ensure_funasr()
    _suppress_transformers_logs()

    server = ASRServer()
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logging.getLogger("ASRServer").info("停止服务")
