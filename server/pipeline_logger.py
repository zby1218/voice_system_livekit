# -*- coding: utf-8 -*-
"""
流水线日志：STT/LLM/TTS 结果、各模块耗时、端到端耗时、VAD 打断。
统一带时间戳，输出到控制台和 .log 文件，格式简洁可读。
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

# 当前轮次 E2E 打点（墙钟）：4 个时刻即可算 3 段
_e2e_start: Optional[float] = None  # 音频给到 STT
_e2e_stt_result_at: Optional[float] = None  # STT 出最终结果
_e2e_llm_first_chunk_at: Optional[float] = None  # LLM 流式首 chunk 时刻
_e2e_end_timestamp: Optional[float] = None  # TTS 首帧推送
_tts_request_at: Optional[float] = None  # 向 TTS server 发送请求的时刻（用于「请求→首帧」耗时）
_tts_request_to_first_frame_logged: bool = False  # 本轮是否已打过「TTS请求→首帧」，避免多段重复

# 流水线专用 logger，由 init 配置
_logger: Optional[logging.Logger] = None


class _MilliFormatter(logging.Formatter):
    """时间格式带毫秒：HH:MM:SS.mmm"""

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        ct = time.localtime(record.created)
        if datefmt:
            s = time.strftime(datefmt, ct)
        else:
            s = time.strftime("%H:%M:%S", ct)
        return s + f".{int(record.created % 1 * 1000):03d}"


def init_pipeline_logging(log_dir: str, log_filename: str = "pipeline.log") -> None:
    """初始化流水线日志：控制台 + 文件，仅本 logger 使用，不影响 root。"""
    global _logger
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    _logger = logging.getLogger("pipeline")
    _logger.setLevel(logging.INFO)
    _logger.propagate = False
    _logger.handlers.clear()

    fmt = _MilliFormatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)
    _logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    _logger.addHandler(console_handler)


def _log(msg: str) -> None:
    if _logger:
        _logger.info(msg)


def _snippet(text: str, max_len: int = 60) -> str:
    t = (text or "").strip() or "(空)"
    return (t[:max_len] + "..") if len(t) > max_len else t


def set_e2e_start(t: float) -> None:
    global _e2e_start, _tts_request_to_first_frame_logged
    _e2e_start = t
    _tts_request_to_first_frame_logged = False


def get_e2e_start() -> Optional[float]:
    return _e2e_start


def get_e2e_stt_result_at() -> Optional[float]:
    return _e2e_stt_result_at


def record_stt_result_time(t: float) -> None:
    """STT 出最终结果时调用，用于 E2E 分段。"""
    global _e2e_stt_result_at
    _e2e_stt_result_at = t


def record_e2e_end(t: float) -> None:
    """TTS 首帧推送给客户端时调用，记录该时刻作为 E2E 终点。"""
    global _e2e_end_timestamp
    _e2e_end_timestamp = t


def record_llm_first_chunk_time(t: float) -> None:
    """LLM 流式发出第一个 chunk 的时刻（用于 E2E 分段）。"""
    global _e2e_llm_first_chunk_at
    _e2e_llm_first_chunk_at = t


def record_tts_request_time(t: float) -> None:
    """向 TTS server 发送请求的时刻（与首帧时刻相减 = 请求→首帧 耗时）。"""
    global _tts_request_at
    _tts_request_at = t


def log_tts_request_to_first_frame() -> None:
    """仅在第一段：若有「请求时刻」和「首帧时刻」，打 [耗时-TTS请求→首帧] 一次，避免多段重复。"""
    global _tts_request_at, _tts_request_to_first_frame_logged
    if _tts_request_to_first_frame_logged:
        _tts_request_at = None
        return
    if _tts_request_at is not None and _e2e_end_timestamp is not None:
        duration = max(0.0, _e2e_end_timestamp - _tts_request_at)
        _log(f"[耗时-TTS请求→首帧] {duration:.2f}s")
        _tts_request_to_first_frame_logged = True
    _tts_request_at = None


def log_stt(text: str) -> None:
    _log(f"[STT结果] {_snippet(text)}")


def log_llm(text: str, duration_s: Optional[float] = None) -> None:
    msg = f"[LLM输出结果] {_snippet(text)}"
    if duration_s is not None:
        msg += f"  ({duration_s:.2f}s)"
    _log(msg)


def log_tts(desc: str = "合成", duration_s: Optional[float] = None, chars: Optional[int] = None) -> None:
    msg = f"[TTS首帧] {desc}".rstrip() if desc else "[TTS首帧]"
    if chars is not None:
        msg += f" {chars}字"
    if duration_s is not None:
        msg += f" ({duration_s:.2f}s)"
    _log(msg)


def log_module(module: str, duration_s: float) -> None:
    _log(f"[耗时-{module}] {duration_s:.2f}s")


def log_e2e(
    stage_stt: Optional[float] = None,
    stage_llm: Optional[float] = None,
    stage_tts: Optional[float] = None,
) -> None:
    """端到端耗时 + 三段：音频→STT结果、STT结果→首chunk、首chunk→TTS首帧。"""
    global _e2e_start, _e2e_stt_result_at, _e2e_llm_first_chunk_at, _e2e_end_timestamp, _tts_request_to_first_frame_logged
    if _e2e_start is None:
        return
    end = _e2e_end_timestamp if _e2e_end_timestamp is not None else time.time()
    d = end - _e2e_start

    msg = f"[E2E] {d:.2f}s (从音频给STT到TTS首帧)"
    if stage_stt is not None and stage_llm is not None and stage_tts is not None:
        msg += f"，阶段和≈{stage_stt + stage_llm + stage_tts:.2f}s"
    _log(msg)

    if _e2e_stt_result_at is not None:
        if (
            _e2e_llm_first_chunk_at is not None
            and _e2e_end_timestamp is not None
        ):
            s1 = _e2e_stt_result_at - _e2e_start
            s2 = _e2e_llm_first_chunk_at - _e2e_stt_result_at
            s3 = _e2e_end_timestamp - _e2e_llm_first_chunk_at
            _log(f"[E2E-分段] 1. 音频 → STT: {s1:.2f}s  2. STT结果 → 首chunk: {s2:.2f}s  3. 首chunk → TTS首帧: {s3:.2f}s")
        else:
            seg1 = _e2e_stt_result_at - _e2e_start
            seg2 = end - _e2e_stt_result_at
            _log(f"[E2E-分段] 1. 音频 → STT结果: {seg1:.2f}s  2. STT结果 → TTS首帧: {seg2:.2f}s")

    _e2e_start = None
    _e2e_stt_result_at = None
    _e2e_llm_first_chunk_at = None
    _e2e_end_timestamp = None
    _tts_request_to_first_frame_logged = False


def log_vad_interrupt(reason: str = "用户打断") -> None:
    _log(f"[VAD打断] {reason}")
