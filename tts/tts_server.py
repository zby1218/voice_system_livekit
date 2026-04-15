#!/usr/bin/env python3
"""TTS HTTP 服务器 - CosyVoice3 封装

启动方式:
    python tts_server.py --model-dir ./model/Fun-CosyVoice3-0.5B --asset-dir ./assets --port 50000
"""

import os
import re
import sys
import time
import logging
import argparse
import threading
import asyncio
from datetime import datetime
from typing import Generator, Dict, Any, Optional, Literal

# 路径设置（必须在导入同目录模块之前）
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from tts_request_metrics import RequestMetricsRecorder
from tts_request_control import SessionRequestController, RequestTicket

# 添加 Matcha-TTS 路径（如需要）
matcha_path = os.path.join(SCRIPT_DIR, 'third_party', 'Matcha-TTS')
if os.path.exists(matcha_path) and matcha_path not in sys.path:
    sys.path.insert(0, matcha_path)

from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


def _setup_logging() -> None:
    """配置日志：同时输出到控制台和文件。日志写在项目根目录的 log/tts/ 下，与 agent、stt 等一致。每次启动覆盖旧文件。"""
    # 与 stt 一致：用 abspath(__file__) 解析项目根，避免 cwd 或 uvicorn 加载方式导致路径错误
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(project_root, "log", "tts")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "tts.log")

    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)

    # 只配置 tts_server 的 logger，不依赖 root，避免 uvicorn 等后续清空 root.handlers 导致文件无内容
    tts_logger = logging.getLogger("tts_server")
    tts_logger.setLevel(logging.INFO)
    tts_logger.propagate = False
    tts_logger.handlers.clear()
    tts_logger.addHandler(file_handler)
    tts_logger.addHandler(console_handler)
    tts_logger.info("TTS 日志文件: %s", os.path.abspath(log_path))
    file_handler.flush()

    # 屏蔽 python-multipart 的 DEBUG 日志（FastAPI multipart/form-data 解析噪音）
    logging.getLogger("multipart").setLevel(logging.WARNING)
    logging.getLogger("multipart.multipart").setLevel(logging.WARNING)


# 使用固定名称，避免以脚本运行时显示为 __main__
logger = logging.getLogger("tts_server")

# ========== 音色配置 ==========
VOICE_CONFIGS = [
    {"id": "default", "file": "robot_bazong_cold.wav",
    "prompt_text": "You are a helpful assistant.用平静沉稳的语气，正常的语速来回答<|endofprompt|>一个人呆久了，突然和你聊聊天，倒也不觉得那么无聊了"},
    {"id": "longanyun", "file": "longanyun.wav",
    "prompt_text": "You are a helpful assistant.用平静沉稳的语气，正常的语速来回答<|endofprompt|>您已经辛苦一天了，我已经为您调整好室内温度，还准备了您喜欢的饮品，您可以放松休息一下，有任何需要，都可以叫我哦。"},
    {"id": "longtian", "file": "longtian.wav",
    "prompt_text": "You are a helpful assistant.用平静沉稳的语气，正常的语速来回答<|endofprompt|>没必要为不值得的人费时间，想开点，生活还得继续，先做再说，别光想着做，等啥时候。"},
    {"id": "longhao", "file": "longhao.wav",
    "prompt_text": "You are a helpful assistant.用平静沉稳的语气，正常的语速来回答<|endofprompt|>唉，时间过的可真快，一晃眼，曾经说好要互相照顾的人，都像是被风吹走的蒲公英，一个不留意就走丢了。"},
    {"id": "muju", "file": "muju.wav",
    "prompt_text": "You are a helpful assistant.用平静沉稳的语气，正常的语速来回答<|endofprompt|>首发的一汽世界模型，是认知的觉醒，也是行动的序章"},
    {"id": "robot_bazong", "file": "robot_bazong.wav",
    "prompt_text": "You are a helpful assistant.<|endofprompt|>别老盯着我看，有话直说，我可不喜欢猜别人心思"},
    {"id": "robot", "file": "robot.wav",
    "prompt_text": "You are a helpful assistant.用最大的音量来回答<|endofprompt|>大家好，我来自一汽机器人"},
    {"id": "longanhuan", "file": "zero_shot_prompt.wav",
     "prompt_text": "You are a helpful assistant.用正常的语速和语调以及情绪来回答<|endofprompt|>希望你以后能够做的比我还好呦。"},
    
]

# 用稍快的语速，平静的语气来回答
class RequestCancelledError(RuntimeError):
    """请求在排队或推理阶段被取消。"""


InterruptMode = Literal["normal", "latest_only"]


def _normalize_interrupt_mode(mode: Optional[str]) -> InterruptMode:
    v = (mode or "normal").strip().lower()
    if v in ("latest", "latest_only", "interrupt", "interrupting"):
        return "latest_only"
    return "normal"


def _resolve_session_id(session_id: Optional[str], request: Request) -> str:
    # 推荐由调用方显式透传 session_id；未透传时回退到客户端地址，避免全局冲突。
    if session_id and session_id.strip():
        return session_id.strip()
    host = request.client.host if request.client else "unknown"
    return f"client:{host}"


def _configure_cuda_environment(device: str) -> str:
    """将 cuda:N 映射为单卡可见环境，避免库内部出现 cuda:0/cuda:N 混用。"""
    normalized = (device or "cuda").strip().lower()
    if not normalized.startswith("cuda:"):
        return device

    index_text = normalized.split(":", 1)[1].strip()
    if not index_text.isdigit():
        raise ValueError(f"非法设备参数: {device}，期望格式如 cuda:1")

    os.environ["CUDA_VISIBLE_DEVICES"] = index_text
    return "cuda"


# 送入合成模型前：去掉夹在 ASCII 字母/数字之间的连字符（如 E-H7、A-B-12 -> EH7、AB12），避免 TTS 把「杠」读出来。
_TTS_ALNUM_BOUNDED_HYPHEN = re.compile(r"(?<=[A-Za-z0-9])-(?=[A-Za-z0-9])")


def filter_tts_synth_text(text: str) -> str:
    """对整段待合成文本做规整，可安全多次调用。"""
    if not text:
        return text
    return _TTS_ALNUM_BOUNDED_HYPHEN.sub("", text)


# ========== 音频后处理 ==========

_AUDIO_GAIN: float = 2.0  # 响度增益倍数，1.0 = 不变

# 将输出的音频流进行响度增益处理，增益倍数为2倍
def amplify_pcm(pcm_bytes: bytes, gain: float = _AUDIO_GAIN) -> bytes:
    """软限幅增益：将 16-bit PCM 放大 gain 倍，用 tanh 平滑压限防止截断失真。
    逐帧调用安全，无跨帧依赖，几乎不增加耗时（numpy SIMD，<0.1ms/帧）。
    """
    import numpy as _np
    samples = _np.frombuffer(pcm_bytes, dtype=_np.int16).astype(_np.float32)
    samples /= 32768.0
    samples = _np.tanh(samples * gain)
    return (samples * 32768).clip(-32768, 32767).astype(_np.int16).tobytes()


class TTSEngine:
    """CosyVoice3 TTS 引擎 - 支持多用户并发"""
    
    def __init__(self, model_dir: str, asset_dir: str, device: str = "cuda",
                 fp16: bool = True, use_vllm: bool = False,
                 output_sample_rate: int = 24000, max_concurrent: int = 2,
                 default_voice: str = "default", logger=None):
        self.model_dir = model_dir
        self.asset_dir = asset_dir
        self.device = device
        self.fp16 = fp16
        self.use_vllm = use_vllm
        self.output_sample_rate = output_sample_rate
        self.max_concurrent = max_concurrent
        self.default_voice = default_voice
        self.logger = logger or logging.getLogger(__name__)
        
        self.cosyvoice = None
        self.model_sample_rate = 24000
        self.voice_cache: Dict[str, Dict[str, Any]] = {}
        
        # ========== 并发控制 ==========
        # 使用信号量控制最大并发数（替代原来的串行锁）
        self._semaphore = threading.Semaphore(max_concurrent)
        self._active_count = 0
        self._count_lock = threading.Lock()
        self._request_control = SessionRequestController()
        self._request_metrics: Dict[int, RequestMetricsRecorder] = {}  # request_id -> 单次请求指标记录器
        self._request_ticket: Dict[int, RequestTicket] = {}  # request_id -> ticket(session/event)
        self._total_requests = 0
        self.loaded = False
    
    def load_model(self):
        if self.loaded:
            return
        
        import torch
        import torchaudio
        self.torch = torch
        self.torchaudio = torchaudio
        
        self.logger.info(f"加载模型: {self.model_dir}")
        self.logger.info(f"设备: {self.device}, FP16: {self.fp16}")
        
        from cosyvoice.cli.cosyvoice import AutoModel
        self.cosyvoice = AutoModel(model_dir=self.model_dir, fp16=self.fp16, load_vllm=self.use_vllm)
        self.model_sample_rate = self.cosyvoice.sample_rate
        
        # 加载音色
        for cfg in VOICE_CONFIGS:
            path = os.path.join(self.asset_dir, cfg["file"])
            if not os.path.exists(path):
                self.logger.warning(f"音色文件不存在: {path}")
                continue
            try:
                self.cosyvoice.add_zero_shot_spk(cfg["prompt_text"], path, cfg["id"])
                self.voice_cache[cfg["id"]] = {"file": path, "prompt_text": cfg["prompt_text"]}
                self.logger.info(f"✅ 音色: {cfg['id']}")
            except Exception as e:
                self.logger.warning(f"音色加载失败 {cfg['id']}: {e}")

        # 确定实际默认音色：优先命令行指定，不在缓存则回退到第一个可用
        if self.default_voice in self.voice_cache:
            effective_default = self.default_voice
        elif self.voice_cache:
            effective_default = list(self.voice_cache.keys())[0]
            self.logger.warning(
                f"⚠️ 指定音色 '{self.default_voice}' 未加载，实际默认音色回退为: '{effective_default}'"
            )
        else:
            effective_default = None

        if effective_default:
            self.logger.info(f"🎙️ 当前默认音色: {effective_default}")

        # 仅用默认音色预热一次（激活 CUDA kernel 和显存分配，与具体音色无关）
        if effective_default:
            vid = effective_default
            try:
                self.logger.info(f"预热中（音色: {vid}）...")
                for _ in self.cosyvoice.inference_zero_shot(
                    "预热", self.voice_cache[vid]["prompt_text"],
                    self.voice_cache[vid]["file"], stream=False, zero_shot_spk_id=vid
                ):
                    pass
                self.logger.info(f"✅ 预热完成（音色: {vid}）")
            except Exception as e:
                self.logger.warning(f"预热失败（音色: {vid}）: {e}")
        
        self.loaded = True
    
    def get_available_voices(self) -> list:
        return list(self.voice_cache.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """获取并发状态统计"""
        with self._count_lock:
            return {
                "active_requests": self._active_count,
                "max_concurrent": self.max_concurrent,
                "total_requests": self._total_requests,
                "available_slots": self.max_concurrent - self._active_count,
            }

    def _register_request(self, session_id: str, interrupt_mode: InterruptMode) -> int:
        ticket, cancelled = self._request_control.create_request(session_id, interrupt_mode)
        request_id = ticket.request_id
        with self._count_lock:
            self._total_requests = max(self._total_requests, request_id)
            self._request_ticket[request_id] = ticket
            recorder = RequestMetricsRecorder(self.logger)
            recorder.start_acquire()
            self._request_metrics[request_id] = recorder

        self.logger.info(
            "📨 [Arrive] 请求 #%s 到达 | session=%s mode=%s",
            request_id,
            session_id,
            interrupt_mode,
        )
        if cancelled:
            self.logger.info(
                "🛑 [Cancel] session=%s latest_only 取消旧请求: %s",
                session_id,
                cancelled,
            )
            for rid in cancelled:
                rec = self._request_metrics.get(rid)
                if rec is not None:
                    rec.record_cancelled("queued_or_running")
        return request_id

    def _acquire_slot(self, request_id: int) -> None:
        """等待并占用推理槽位；若排队阶段已被取消则抛 RequestCancelledError。"""
        while True:
            if self._request_control.is_cancelled(request_id):
                raise RequestCancelledError(f"request #{request_id} cancelled before acquire")
            if self._semaphore.acquire(timeout=0.1):
                break

        with self._count_lock:
            self._active_count += 1
            recorder = self._request_metrics.get(request_id)
            if recorder is not None:
                recorder.finish_acquire(request_id, self._active_count)
            ticket = self._request_ticket.get(request_id)
            session = ticket.session_id if ticket else "unknown"
            self.logger.info(
                "📥 [Slot] 请求 #%s 开始 | session=%s 当前并发: %s/%s",
                request_id,
                session,
                self._active_count,
                self.max_concurrent,
            )

    def _release_slot(self, request_id: int, *, cancelled: bool = False) -> None:
        """释放推理槽位并清理请求状态。"""
        recorder = None
        session = self._request_control.remove_request(request_id) or "unknown"
        with self._count_lock:
            if self._active_count > 0:
                self._active_count -= 1
            recorder = self._request_metrics.pop(request_id, None)
            self._request_ticket.pop(request_id, None)
            self.logger.info(
                "📤 [Slot] 请求 #%s 结束 | session=%s status=%s 当前并发: %s/%s",
                request_id,
                session,
                "cancelled" if cancelled else "completed",
                self._active_count,
                self.max_concurrent,
            )
        self._semaphore.release()
        if recorder is not None:
            if cancelled:
                recorder.record_cancelled("running")
            else:
                recorder.finish_request()

    def _drop_request_without_slot(self, request_id: int, reason: str) -> None:
        """未拿到槽位时清理请求（例如排队阶段被取消）。"""
        session = self._request_control.remove_request(request_id) or "unknown"
        with self._count_lock:
            recorder = self._request_metrics.pop(request_id, None)
            self._request_ticket.pop(request_id, None)
        if recorder is not None:
            recorder.record_cancelled("queued")
        self.logger.info(
            "🧹 [Drop] 请求 #%s 已丢弃 | session=%s reason=%s",
            request_id,
            session,
            reason,
        )
    
    def synthesize(
        self,
        text: str,
        voice_id: str = None,
        *,
        session_id: str = "default",
        interrupt_mode: InterruptMode = "normal",
    ) -> Generator[bytes, None, None]:
        """流式合成 - 返回 PCM 字节流（支持并发）"""
        text = filter_tts_synth_text(text)
        if not self.loaded or not text.strip():
            return
        
        requested = voice_id or "default"
        if requested in self.voice_cache:
            vid = requested
        elif "default" in self.voice_cache:
            vid = "default"
            if requested != "default":
                self.logger.warning(f"音色 '{requested}' 不存在，回退为 'default'")
        elif self.voice_cache:
            vid = list(self.voice_cache.keys())[0]
            self.logger.warning(f"音色 '{requested}' 及 'default' 均不存在，回退为 '{vid}'")
        else:
            raise ValueError("无可用音色")
        self.logger.debug(f"🎙️ 使用音色: {vid}")
        
        voice = self.voice_cache[vid]
        request_id = self._register_request(session_id, interrupt_mode)
        acquired = False
        cancelled = False

        try:
            self._acquire_slot(request_id)
            acquired = True
            self.logger.info(
                "[TIMING] %s 🎙️ 开始生成音频流 request_id=%s text='%s'",
                datetime.now().strftime("%H:%M:%S.%f")[:-3],
                request_id,
                text,
            )
            first_chunk = True
            for result in self.cosyvoice.inference_zero_shot(
                text, voice["prompt_text"], voice["file"], stream=True, zero_shot_spk_id=vid
            ):
                if self._request_control.is_cancelled(request_id):
                    cancelled = True
                    self.logger.info("⚠️ 请求 #%s 在推理阶段被取消", request_id)
                    break

                if first_chunk:
                    rec = self._request_metrics.get(request_id)
                    if rec is not None:
                        rec.record_first_byte()
                    first_chunk = False
                audio = result['tts_speech']
                
                if self.output_sample_rate != self.model_sample_rate:
                    audio = self.torchaudio.functional.resample(
                        audio, self.model_sample_rate, self.output_sample_rate
                    )
                yield amplify_pcm((audio * 32768).to(self.torch.int16).cpu().numpy().tobytes())
        
        except RequestCancelledError:
            cancelled = True
            self.logger.info("⚠️ 请求 #%s 在排队阶段被取消", request_id)
        finally:
            if acquired:
                self._release_slot(request_id, cancelled=cancelled)
            else:
                self._drop_request_without_slot(request_id, "cancelled_before_acquire")
    
    
    def synthesize_zero_shot(
        self,
        text: str,
        prompt_text: str,
        prompt_wav_bytes: bytes,
        *,
        session_id: str = "default",
        interrupt_mode: InterruptMode = "normal",
    ) -> Generator[bytes, None, None]:
        """零样本合成 - 优先使用预注册音色（支持并发）"""
        text = filter_tts_synth_text(text)
        if not self.loaded or not text.strip():
            return
        
        # 检查是否有匹配的预注册音色
        matched_voice_id = None
        for vid, voice_info in self.voice_cache.items():
            if voice_info["prompt_text"] == prompt_text:
                matched_voice_id = vid
                self.logger.info(f"✅ 匹配到预注册音色: {vid}")
                break
        
        if matched_voice_id:
            # 使用预注册音色（与 voice_system 一致）
            voice = self.voice_cache[matched_voice_id]
            request_id = self._register_request(session_id, interrupt_mode)
            acquired = False
            cancelled = False
            try:
                self._acquire_slot(request_id)
                acquired = True
                self.logger.info(
                    "[TIMING] %s 🎙️ 开始生成音频流 request_id=%s text='%s'",
                    datetime.now().strftime("%H:%M:%S.%f")[:-3],
                    request_id,
                    text,
                )
                first_chunk = True
                for result in self.cosyvoice.inference_zero_shot(
                    text, voice["prompt_text"], voice["file"], stream=True, zero_shot_spk_id=matched_voice_id
                ):
                    if self._request_control.is_cancelled(request_id):
                        cancelled = True
                        self.logger.info("⚠️ 请求 #%s 在推理阶段被取消", request_id)
                        break
                    if first_chunk:
                        rec = self._request_metrics.get(request_id)
                        if rec is not None:
                            rec.record_first_byte()
                        first_chunk = False
                    audio = result['tts_speech']
                    if self.output_sample_rate != self.model_sample_rate:
                        audio = self.torchaudio.functional.resample(audio, self.model_sample_rate, self.output_sample_rate)
                    yield amplify_pcm((audio * 32768).to(self.torch.int16).cpu().numpy().tobytes())
            except RequestCancelledError:
                cancelled = True
                self.logger.info("⚠️ 请求 #%s 在排队阶段被取消", request_id)
            finally:
                if acquired:
                    self._release_slot(request_id, cancelled=cancelled)
                else:
                    self._drop_request_without_slot(request_id, "cancelled_before_acquire")
        else:
            # 未匹配到预注册音色，使用临时文件（冷推理）
            self.logger.warning(f"⚠️ 未匹配到预注册音色，使用冷推理")
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(prompt_wav_bytes)
                temp_path = f.name
            
            request_id = self._register_request(session_id, interrupt_mode)
            acquired = False
            cancelled = False
            try:
                self._acquire_slot(request_id)
                acquired = True
                self.logger.info(
                    "[TIMING] %s 🎙️ 开始生成音频流 request_id=%s text='%s'",
                    datetime.now().strftime("%H:%M:%S.%f")[:-3],
                    request_id,
                    text,
                )
                first_chunk = True
                for result in self.cosyvoice.inference_zero_shot(text, prompt_text, temp_path, stream=True):
                    if self._request_control.is_cancelled(request_id):
                        cancelled = True
                        self.logger.info("⚠️ 请求 #%s 在推理阶段被取消", request_id)
                        break
                    if first_chunk:
                        rec = self._request_metrics.get(request_id)
                        if rec is not None:
                            rec.record_first_byte()
                        first_chunk = False
                    audio = result['tts_speech']
                    if self.output_sample_rate != self.model_sample_rate:
                        audio = self.torchaudio.functional.resample(audio, self.model_sample_rate, self.output_sample_rate)
                    yield amplify_pcm((audio * 32768).to(self.torch.int16).cpu().numpy().tobytes())
            except RequestCancelledError:
                cancelled = True
                self.logger.info("⚠️ 请求 #%s 在排队阶段被取消", request_id)
            finally:
                if acquired:
                    self._release_slot(request_id, cancelled=cancelled)
                else:
                    self._drop_request_without_slot(request_id, "cancelled_before_acquire")
                os.unlink(temp_path)
    
    def interrupt_all(self):
        """中断所有正在进行的请求"""
        cancelled = self._request_control.cancel_all()
        self.logger.info("⚠️ 中断所有请求，共 %s 个: %s", len(cancelled), cancelled)


# ========== FastAPI 应用 ==========
app = FastAPI(title="TTS Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

tts_engine: Optional[TTSEngine] = None


@app.get("/health")
async def health():
    """健康检查"""
    import torch
    stats = tts_engine.get_stats() if tts_engine and tts_engine.loaded else {}
    return JSONResponse({
        "status": "ok",
        "loaded": tts_engine is not None and tts_engine.loaded,
        "voices": tts_engine.get_available_voices() if tts_engine and tts_engine.loaded else [],
        "sample_rate": tts_engine.output_sample_rate if tts_engine else 24000,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        **stats  # 包含 active_requests, max_concurrent, total_requests, available_slots
    })


@app.post("/inference_sft")
async def inference_sft(
    request: Request,
    tts_text: str = Form(...),
    spk_id: str = Form(default="default"),
    session_id: Optional[str] = Form(default=None),
    interrupt_mode: str = Form(default="normal"),
):
    """SFT 推理 (预训练音色)"""
    if not tts_engine or not tts_engine.loaded:
        raise HTTPException(503, "模型未加载")
    if not tts_text.strip():
        raise HTTPException(400, "文本为空")
    
    session = _resolve_session_id(session_id, request)
    mode = _normalize_interrupt_mode(interrupt_mode)
    logger.info(f"[SFT] session={session} mode={mode} text='{tts_text[:30]}...' voice={spk_id}")
    start_time = time.time()
    
    def gen():
        first = True
        chunk_idx = 0
        t0 = time.time()
        for chunk in tts_engine.synthesize(
            tts_text,
            voice_id=spk_id,
            session_id=session,
            interrupt_mode=mode,
        ):
            now = time.time()
            if first:
                logger.info(f"[SFT] ⏱️ 首包延迟: {now - start_time:.3f}s")
                first = False
            logger.info(f"[SFT-CHUNK] #{chunk_idx} size={len(chunk)}B elapsed={now - t0:.3f}s ts={now:.3f}")
            chunk_idx += 1
            yield chunk
    
    return StreamingResponse(gen(), media_type="application/octet-stream",
                            headers={"X-Sample-Rate": str(tts_engine.output_sample_rate)})

@app.post("/inference_zero_shot")
async def inference_zero_shot(
    request: Request,
    tts_text: str = Form(...),
    prompt_text: str = Form(...),
    prompt_wav: UploadFile = File(...),
    session_id: Optional[str] = Form(default=None),
    interrupt_mode: str = Form(default="normal"),
):
    """零样本推理 (上传参考音频和文本)"""
    if not tts_engine or not tts_engine.loaded:
        raise HTTPException(503, "模型未加载")
    if not tts_text.strip():
        raise HTTPException(400, "文本为空")
    if not prompt_text.strip():
        raise HTTPException(400, "参考文本为空")
    
    prompt_bytes = await prompt_wav.read()
    session = _resolve_session_id(session_id, request)
    mode = _normalize_interrupt_mode(interrupt_mode)
    logger.info(
        f"[ZeroShot] session={session} mode={mode} text='{tts_text}' prompt='{prompt_text[:20]}...'"
    )
    start_time = time.time()
    
    def gen():
        first = True
        chunk_idx = 0
        t0 = time.time()
        for chunk in tts_engine.synthesize_zero_shot(
            tts_text,
            prompt_text,
            prompt_bytes,
            session_id=session,
            interrupt_mode=mode,
        ):
            now = time.time()
            if first:
                logger.info(f"[ZeroShot] ⏱️ 首包延迟: {now - start_time:.3f}s")
                first = False
            logger.info(f"[ZeroShot-CHUNK] #{chunk_idx} size={len(chunk)}B elapsed={now - t0:.3f}s ts={now:.3f}")
            chunk_idx += 1
            yield chunk
    
    return StreamingResponse(gen(), media_type="application/octet-stream",
                            headers={"X-Sample-Rate": str(tts_engine.output_sample_rate)})


@app.post("/tts/stream")
async def tts_stream(
    request: Request,
    text: str = Form(...),
    voice_id: Optional[str] = Form(default=None),
    session_id: Optional[str] = Form(default=None),
    interrupt_mode: str = Form(default="normal"),
):
    """简化版 TTS 接口 (兼容 voice_system)"""
    if not tts_engine or not tts_engine.loaded:
        raise HTTPException(503, "模型未加载")
    if not text.strip():
        raise HTTPException(400, "文本为空")
    
    session = _resolve_session_id(session_id, request)
    mode = _normalize_interrupt_mode(interrupt_mode)
    logger.info(f"[TTS] session={session} mode={mode} '{text[:30]}' voice={voice_id or 'default'}")
    start_time = time.time()
    
    def gen():
        first = True
        chunk_idx = 0
        t0 = time.time()
        for chunk in tts_engine.synthesize(
            text,
            voice_id=voice_id,
            session_id=session,
            interrupt_mode=mode,
        ):
            now = time.time()
            if first:
                logger.info(f"[TTS] ⏱️ 首包延迟: {now - start_time:.3f}s")
                first = False
            logger.info(f"[TTS-CHUNK] #{chunk_idx} size={len(chunk)}B elapsed={now - t0:.3f}s ts={now:.3f}")
            chunk_idx += 1
            yield chunk
    
    return StreamingResponse(gen(), media_type="application/octet-stream",
                            headers={"X-Sample-Rate": str(tts_engine.output_sample_rate)})


@app.post("/interrupt")
async def interrupt():
    """中断所有正在进行的合成"""
    if tts_engine:
        stats = tts_engine.get_stats()
        if stats["active_requests"] > 0:
            tts_engine.interrupt_all()
            return {"success": True, "interrupted": stats["active_requests"]}
    return {"success": False, "interrupted": 0}


def main():
    global tts_engine
    # 日志已在模块加载时配置（见文件末尾），保证 python tts_server.py 与 uvicorn tts_server:app 均有 log 文件

    # 默认路径
    default_model_dir = os.path.join(SCRIPT_DIR, "model", "Fun-CosyVoice3-0.5B")
    default_asset_dir = os.path.join(SCRIPT_DIR, "assets")
    
    parser = argparse.ArgumentParser(description="TTS Server")
    parser.add_argument("--model-dir", type=str, default=default_model_dir,
                        help=f"模型目录路径 (默认: {default_model_dir})")
    parser.add_argument("--asset-dir", type=str, default=default_asset_dir,
                        help=f"音色资源目录 (默认: {default_asset_dir})")
    parser.add_argument("--device", type=str, default="cuda", help="设备 (cuda/cpu)")
    parser.add_argument("--fp16", action="store_true", default=True, help="使用 FP16")
    parser.add_argument("--no-fp16", action="store_false", dest="fp16", help="不使用 FP16")
    parser.add_argument("--vllm", action="store_true", default=False, help="使用 vLLM")
    parser.add_argument("--sample-rate", type=int, default=24000, help="输出采样率")
    parser.add_argument("--max-concurrent", type=int, default=2, help="最大并发请求数")
    parser.add_argument("--voice", type=str, default="default",
                        help="默认音色 id，对应 VOICE_CONFIGS（默认: default）")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=50000, help="监听端口")
    
    args = parser.parse_args()
    try:
        original_device = args.device
        args.device = _configure_cuda_environment(args.device)
        if original_device != args.device:
            logger.info(
                "检测到多卡参数 %s，已设置 CUDA_VISIBLE_DEVICES=%s，进程内设备统一为 %s",
                original_device,
                os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                args.device,
            )
    except ValueError as exc:
        parser.error(str(exc))
    
    logger.info("=" * 60)
    logger.info("启动 TTS Server")
    logger.info(f"  模型目录: {args.model_dir}")
    logger.info(f"  资源目录: {args.asset_dir}")
    logger.info(f"  设备: {args.device}, FP16: {args.fp16}")
    logger.info(f"  采样率: {args.sample_rate}")
    logger.info(f"  最大并发: {args.max_concurrent}")
    logger.info(f"  默认音色: {args.voice}")
    logger.info(f"  监听: {args.host}:{args.port}")
    logger.info("=" * 60)
    
    # 初始化引擎
    tts_engine = TTSEngine(
        model_dir=args.model_dir,
        asset_dir=args.asset_dir,
        device=args.device,
        fp16=args.fp16,
        use_vllm=args.vllm,
        output_sample_rate=args.sample_rate,
        max_concurrent=args.max_concurrent,
        default_voice=args.voice,
        logger=logger
    )
    tts_engine.load_model()
    
    # 启动服务
    uvicorn.run(app, host=args.host, port=args.port, log_config=None)


# 模块加载时即配置日志（python tts_server.py 或 uvicorn tts_server:app 都会执行），日志文件在项目根 log/tts/tts.log
_setup_logging()

if __name__ == "__main__":
    main()
