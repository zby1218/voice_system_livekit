#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import os
import re
import sys
import json
import time
import asyncio
import logging
import traceback
import websockets
import torch
import numpy as np
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

# 忽略警告
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------

def _setup_logging(log_subdir: str) -> None:
    """配置日志：仅输出到控制台。各客户端连接时按端口创建独立日志文件 log/<subdir>/stt_<port>.log。"""
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)


# ---------------------------------------------------------------------------
# 配置与常量
# ---------------------------------------------------------------------------

# 说话人分离开关：改为 True 即启用 MossFormer2_SS_16K 前处理（走独立 speaker 服务），False 则直通
ENABLE_SPEAKER_SELECT: bool = False
# speaker 服务地址（需先在本机启动：python -m speaker_select.server）
SPEAKER_SERVICE_URL: str = "http://127.0.0.1:1999/separate"


@dataclass(frozen=True)
class ASRServerConfig:
    """ASR 服务配置，集中管理常量和可调参数。"""
    sample_rate: int = 16000
    min_duration_sec: float = 0.5
    energy_threshold: float = 50.0
    inference_workers: int = 5
    default_hotwords: list = field(default_factory=lambda: ["天工", "小莫", "一汽"])
    model_subdir: str = "FunAudioLLM/Fun-ASR-Nano-2512"
    # 文本过滤：单字符重复次数阈值（>=N 视为幻觉）
    single_char_repeat_min: int = 4
    # 文本过滤：短语（>=2字）重复次数阈值（>=N 视为幻觉）
    phrase_repeat_min: int = 3


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
        self,
        audio_bytes: bytes,
        logger: Optional[logging.Logger] = None,
        client_id: str = "",
    ) -> tuple[bool, str]:
        """
        判断是否应跳过本次推理。
        :param logger: 若提供则用其写跳过原因（便于按连接落盘）；否则用 self._logger。
        :param client_id: 连接标识，写入日志前缀，便于多客户端追踪。
        :return: (是否跳过, 原因描述)，不跳过时原因为空字符串。
        """
        log = logger if logger is not None else self._logger
        prefix = f"[{client_id}] " if client_id else ""

        if len(audio_bytes) == 0:
            return True, "空音频"

        if len(audio_bytes) < self._min_bytes:
            log.info(
                "%s音频过短 (%s bytes < %s bytes)，跳过推理",
                prefix, len(audio_bytes), self._min_bytes,
            )
            return True, "过短"

        data_int16 = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(data_int16 ** 2)))
        if rms < self._config.energy_threshold:
            log.info(
                "%s音频能量过低 (RMS=%.1f < %s)，跳过推理",
                prefix, rms, self._config.energy_threshold,
            )
            return True, "能量过低"

        return False, ""


# ---------------------------------------------------------------------------
# 文本过滤（防幻觉）
# ---------------------------------------------------------------------------

# 常见填充音/语气词，模型容易把背景噪声识别成这些单字
_DEFAULT_FILLER_WORDS: frozenset = frozenset({
    "嗯", 
})

# 过滤前先去掉标点和空白，避免"嗯。"漏网
_PUNCT_RE = re.compile(r'[\s，。！？、,\.!?；;…—\-～·《》【】「」『』\(\)\[\]]')


class TextFilter:
    """对 ASR 离线识别结果做后处理，过滤常见幻觉（填充音、重复词组）。

    过滤规则：
    1. 填充音精确匹配（嗯/啊/哦 等），"好"/"行"/"你好" 不在黑名单，不受影响。
    2. 单字符大量重复（如"嗯嗯嗯嗯嗯"），阈值由 single_char_repeat_min 控制。
    3. 短语大量重复（如"你好你好你好你好"），阈值由 phrase_repeat_min 控制。
    """

    def __init__(
        self,
        filler_words: Optional[frozenset] = None,
        single_char_repeat_min: int = 4,
        phrase_repeat_min: int = 3,
    ) -> None:
        self._filler_words = filler_words if filler_words is not None else _DEFAULT_FILLER_WORDS
        self._single_char_repeat_min = single_char_repeat_min
        self._phrase_repeat_min = phrase_repeat_min

    def should_filter(self, text: str) -> tuple[bool, str]:
        """
        判断 ASR 识别结果是否应被过滤。
        :return: (是否过滤, 原因描述)，不过滤时原因为空字符串。
        """
        clean = _PUNCT_RE.sub("", text).strip()

        if not clean:
            return True, "空文本"

        # 1. 填充音精确匹配
        if clean in self._filler_words:
            return True, f"填充音 {clean!r}"

        n = len(clean)

        # 2. 单字符大量重复（如"嗯嗯嗯嗯嗯"）
        if n >= self._single_char_repeat_min and len(set(clean)) == 1:
            return True, f"单字符重复 ×{n}: {clean[0]!r}"

        # 3. 短语大量重复（如"你好你好你好你好"）—— 整串恰好是某单元的整倍数
        max_unit_len = n // self._phrase_repeat_min
        for unit_len in range(2, max_unit_len + 1):
            repeats, remainder = divmod(n, unit_len)
            if remainder != 0:
                continue
            if repeats >= self._phrase_repeat_min and clean == clean[:unit_len] * repeats:
                return True, f"短语重复 ×{repeats}: {clean[:unit_len]!r}"

        # 4. 高频短语主导文本（带任意前缀/后缀，如"嗯天工天工天工..."）
        #    取靠前若干位置的候选单元，统计其在全串的非重叠出现次数；
        #    若某单元出现次数 × 单元长度 ≥ 全串 70%，则视为重复幻觉。
        max_unit_len = min(6, n // self._phrase_repeat_min)
        for unit_len in range(2, max_unit_len + 1):
            checked: set[str] = set()
            for start in range(min(n - unit_len + 1, unit_len * 2)):
                unit = clean[start:start + unit_len]
                if unit in checked:
                    continue
                checked.add(unit)
                count = clean.count(unit)
                if count >= self._phrase_repeat_min and count * unit_len * 10 >= n * 7:
                    return True, f"高频短语重复 ×{count}: {unit!r}"

        return False, ""


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
        project_root = os.path.dirname(current_dir)
        model_root = os.path.join(current_dir, "model")
        self._path_asr = os.path.join(model_root, self._config.model_subdir)
        self._stt_log_dir = os.path.join(project_root, "log", "stt")

        self._logger = logging.getLogger("ASRServer")
        self._logger.info("锁定模型根目录: %s", model_root)
        self._check_path(self._path_asr, "ASR")

        self._model_asr = None
        self._websocket_users: set = set()
        self._server = None
        self._executor = ThreadPoolExecutor(max_workers=self._config.inference_workers)
        self._audio_filter = AudioFilter(self._config, self._logger)
        self._text_filter = TextFilter(
            single_char_repeat_min=self._config.single_char_repeat_min,
            phrase_repeat_min=self._config.phrase_repeat_min,
        )

    def _check_path(self, path: str, name: str) -> None:
        if not os.path.exists(path):
            self._logger.error("❌ 找不到 %s 模型路径: %s", name, path)
            sys.exit(1)
        self._logger.info("✅ %s 路径确认: %s", name, path)

    def load_models(self) -> None:
        import funasr.models.fun_asr_nano.model  # noqa: F401 — 注册 FunASRNano（FunASR 1.3.x）
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

        if ENABLE_SPEAKER_SELECT:
            self._logger.info("已启用说话人分离，将请求独立服务: %s", SPEAKER_SERVICE_URL)

    async def start(self) -> None:
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
    def _decode_audio_chunk(chunk_bytes: bytes) -> torch.Tensor:
        """PCM s16le -> float32 tensor（16kHz mono）。"""
        data_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
        return torch.from_numpy(data_int16.astype(np.float32) / 32768.0)

    async def _run_model_inference(self, model, input_tensor, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: model.generate(input=input_tensor, **kwargs),
        )

    def _get_or_create_session(self, websocket) -> ClientSession:
        if not hasattr(websocket, "_asr_session"):
            websocket._asr_session = ClientSession()
        return websocket._asr_session

    def _create_client_logger(self, port: int) -> tuple[logging.Logger, logging.FileHandler]:
        """为当前连接创建独立日志：log/stt/stt_<port>.log，返回 (logger, file_handler)。"""
        os.makedirs(self._stt_log_dir, exist_ok=True)
        log_path = os.path.join(self._stt_log_dir, f"stt_{port}.log")
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger = logging.getLogger(f"ASRServer.client_{port}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        for old_fh in list(logger.handlers):
            try:
                old_fh.close()
            except Exception:
                pass
        logger.handlers.clear()
        logger.addHandler(fh)
        return logger, fh

    def _close_client_logger(self, websocket) -> None:
        """断开时移除并关闭该连接的文件 handler，避免句柄泄漏。"""
        fh = getattr(websocket, "_client_log_file_handler", None)
        if fh is None:
            return
        logger = getattr(websocket, "_client_logger", None)
        if logger is not None:
            logger.removeHandler(fh)
        try:
            fh.close()
        except Exception:
            pass

    async def _ws_serve(self, websocket, path=None) -> None:
        self._websocket_users.add(websocket)
        session = self._get_or_create_session(websocket)
        frames_asr: list = []
        port = websocket.remote_address[1]
        client_id = str(websocket.remote_address)
        client_logger, file_handler = self._create_client_logger(port)
        websocket._client_logger = client_logger
        websocket._client_log_file_handler = file_handler

        client_logger.info("[%s] 新客户端连接", client_id)
        self._logger.info("[%s] 新客户端连接", client_id)

        try:
            async for message in websocket:
                if isinstance(message, str):
                    await self._handle_text_message(websocket, session, message, frames_asr, client_id)
                else:
                    frames_asr.append(message)
                    if len(frames_asr) == 1:
                        self._logger.info("[%s] 开始接收音频帧", client_id)
        except websockets.ConnectionClosed:
            client_logger.info("[%s] 客户端断开连接", client_id)
            self._logger.info("[%s] 客户端断开连接", client_id)
        except Exception as e:
            client_logger.error("[%s] 处理异常: %s", client_id, e, exc_info=True)
            self._logger.error("[%s] 处理异常: %s", client_id, e)
        finally:
            self._close_client_logger(websocket)
            self._websocket_users.discard(websocket)

    def _client_logger(self, websocket) -> logging.Logger:
        """当前连接对应的日志（写入 stt_<port>.log），无则回退到服务 logger。"""
        return getattr(websocket, "_client_logger", self._logger)

    async def _handle_text_message(
        self,
        websocket,
        session: ClientSession,
        message: str,
        frames_asr: list,
        client_id: str = "",
    ) -> None:
        log = self._client_logger(websocket)
        log.info("[%s] 📨 Received text message: %s", client_id, message[:200])
        self._logger.info("[%s] 📨 Received text message: %s", client_id, message[:200])
        try:
            msg = json.loads(message)
        except Exception as e:
            log.error("[%s] JSON处理错误: %s", client_id, e)
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

        if "is_speaking" not in msg:
            return

        session.is_speaking = msg["is_speaking"]
        session.status_dict_asr_online["is_final"] = not session.is_speaking
        print("is_speaking", session.is_speaking)
        if not session.is_speaking:
            if len(frames_asr) > 0:
                audio_in = b"".join(frames_asr)
                self._logger.info("[%s] is_speaking=False，收到 %d 帧 (%d bytes)，开始推理",
                                  client_id, len(frames_asr), len(audio_in))
                if ENABLE_SPEAKER_SELECT:
                    try:
                        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        if project_root not in sys.path:
                            sys.path.insert(0, project_root)
                        from speaker_select.client import separate_via_service
                        speaker_select_start_time = time.time()
                        first_speaker = await asyncio.get_running_loop().run_in_executor(
                            self._executor,
                            separate_via_service,
                            audio_in,
                            SPEAKER_SERVICE_URL,
                        )
                        speaker_select_time = time.time() - speaker_select_start_time
                        self._logger.info(
                            "[%s] 说话人分离完成，耗时 %.3f 秒", client_id, speaker_select_time
                        )
                        if first_speaker:
                            audio_in = first_speaker
                            self._logger.info(
                                "[%s] 说话人分离完成，取第一路 (%d bytes)", client_id, len(audio_in)
                            )
                    except Exception as e:
                        self._logger.warning(
                            "[%s] 说话人分离失败，使用原始音频: %s", client_id, e
                        )
                await self._async_asr_offline(websocket, session, audio_in, client_id)
            else:
                self._logger.info("[%s] is_speaking=False 但无音频帧，跳过推理", client_id)
                await self._send_empty_final(websocket)
            frames_asr.clear()
            session.status_dict_asr_online["cache"] = {}

    async def _send_empty_final(self, websocket) -> None:
        """跳过推理时仍返回空 final，让客户端正常结束本轮等待，避免挂住。"""
        try:
            await websocket.send(json.dumps({
                "mode": "2pass-offline", "text": "", "is_final": True
            }))
        except Exception as e:
            self._client_logger(websocket).warning("发送空 final 失败: %s", e)

    async def _async_asr_offline(
        self,
        websocket,
        session: ClientSession,
        audio_in: bytes,
        client_id: str = "",
    ) -> None:
        print("有语音进行推理")
        log = self._client_logger(websocket)
        if len(audio_in) == 0:
            await self._send_empty_final(websocket)
            return

        skip, _ = self._audio_filter.should_skip(audio_in, logger=log, client_id=client_id)
        if skip:
            await self._send_empty_final(websocket)
            return


        try:
            audio_tensor = self._decode_audio_chunk(audio_in)
            res = await self._run_model_inference(
                self._model_asr,
                [audio_tensor],
                language="zh",
                hotwords=list(self._config.default_hotwords),
                itn=session.itn,
            )
            print("res", res)
            text = res[0]["text"] if res else ""

            filter_out, reason = self._text_filter.should_filter(text)
            if filter_out:
                log.info("[%s] 过滤幻觉文本 (%s): %r", client_id, reason, text)
                self._logger.info("[%s] 过滤幻觉文本 (%s): %r", client_id, reason, text)
                await self._send_empty_final(websocket)
                return

            log.info("[%s] Final Result: %s", client_id, text)
            self._logger.info("[%s] Final Result: %s", client_id, text)

            await websocket.send(json.dumps({
                "mode": "2pass-offline", "text": text, "is_final": True
            }))
        except Exception as e:
            log.error("[%s] 离线识别错误: %s", client_id, e)
            # 如果识别错误，不发送结果，则client侧会持续等待直到超时。此处直接发送空 final 让 client 结束等待。
            await self._send_empty_final(websocket)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _setup_logging("stt")
    _ensure_funasr()
    _suppress_transformers_logs()

    server = ASRServer()
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logging.getLogger("ASRServer").info("停止服务")
