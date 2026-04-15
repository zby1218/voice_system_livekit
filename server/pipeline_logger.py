# -*- coding: utf-8 -*-
"""
流水线日志：STT/LLM/TTS 结果、各模块耗时、端到端耗时、VAD 打断。
统一带时间戳，输出到控制台和 .log 文件，格式简洁可读。
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from livekit import rtc as _rtc

# 当前轮次 E2E 打点（墙钟）：4 个时刻即可算 3 段
_e2e_start: Optional[float] = None  # 音频给到 STT
_e2e_stt_result_at: Optional[float] = None  # STT 出最终结果
_e2e_llm_first_chunk_at: Optional[float] = None  # LLM 流式首 chunk 时刻
_e2e_tts_segment_sent_at: Optional[float] = None  # 首句段发送到 TTS 时刻（句段就绪并发请求）
_e2e_end_timestamp: Optional[float] = None  # TTS 首帧推送
_tts_request_at: Optional[float] = None  # 向 TTS server 发送请求的时刻（用于「请求→首帧」耗时）
_tts_request_to_first_frame_logged: bool = False  # 本轮是否已打过「TTS请求→首帧」，避免多段重复
_llm_request_at: Optional[float] = None  # 送往 LLM 的时刻（建议在 on_end_of_turn 附近记录）
_rtc_room: Optional[object] = None  # livekit.rtc.Room，用于查询 WebRTC stats

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


def record_tts_segment_sent_time(t: float) -> None:
    """首句段发送到 TTS 的时刻（用于 on_end_of_turn→句段送TTS 分段统计）。"""
    global _e2e_tts_segment_sent_at
    _e2e_tts_segment_sent_at = t


def record_tts_request_time(t: float) -> None:
    """向 TTS server 发送请求的时刻（与首帧时刻相减 = 请求→首帧 耗时）。"""
    global _tts_request_at
    _tts_request_at = t


def set_rtc_room(room: object) -> None:
    """注入 livekit.rtc.Room，用于首帧时查询 WebRTC transport stats。"""
    global _rtc_room
    _rtc_room = room


async def _publish_tts_turn_start() -> None:
    """通过 data channel 通知客户端本轮 TTS 首帧即将推入，携带服务端时间戳（毫秒）。"""
    if _rtc_room is None:
        return
    try:
        ts_ms = int(time.time() * 1000)
        payload = f"tts_turn_start:{ts_ms}".encode()
        await _rtc_room.local_participant.publish_data(payload, reliable=True)  # type: ignore[union-attr]
        _log(f"[TIMING] {_hms_now()} 📤 [SERVER] tts_turn_start 已发送 ts={ts_ms}")
    except Exception:
        pass


async def _log_server_transport_stats() -> None:
    """查询 WebRTC publisher stats，打印到 SFU 的 RTT 和传输层参数。"""
    if _rtc_room is None:
        return
    try:
        stats = await _rtc_room.get_rtc_stats()  # type: ignore[union-attr]
    except Exception:
        return
    from livekit.rtc._proto import stats_pb2 as _pb
    rtt_ms: Optional[float] = None
    remote_rtt_ms: Optional[float] = None
    for item in stats.publisher_stats:
        which = item.WhichOneof("stats")
        if which == "candidate_pair":
            rtt = item.candidate_pair.candidate_pair.current_round_trip_time
            if rtt > 0:
                rtt_ms = rtt * 1000
        elif which == "remote_inbound_rtp":
            rtt = item.remote_inbound_rtp.remote_inbound.round_trip_time
            if rtt > 0:
                remote_rtt_ms = rtt * 1000
    parts = []
    if rtt_ms is not None:
        parts.append(f"RTT(候选对)={rtt_ms:.1f}ms  →SFU单向≈{rtt_ms/2:.1f}ms")
    if remote_rtt_ms is not None:
        parts.append(f"RTT(remote-inbound)={remote_rtt_ms:.1f}ms")
    if parts:
        _log(f"[TIMING] {_hms_now()} 📡 [SERVER-TRANSPORT] " + "  ".join(parts))
    else:
        _log(f"[TIMING] {_hms_now()} 📡 [SERVER-TRANSPORT] stats 暂无数据")


def _hms_now() -> str:
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def record_llm_request_time(t: float) -> None:
    """送往 LLM 的时刻（用于 on_end_of_turn/送往LLM → LLM首包 墙钟耗时）。"""
    global _llm_request_at
    _llm_request_at = t


def get_llm_request_at() -> Optional[float]:
    return _llm_request_at


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


def on_tts_first_frame_pushed(t: float) -> None:
    """在首帧 push 当下立即输出关键耗时，并触发一次 WebRTC stats 查询。"""
    record_e2e_end(t)
    log_tts_request_to_first_frame()
    log_e2e()
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(_publish_tts_turn_start())
            loop.create_task(_log_server_transport_stats())
    except Exception:
        pass


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


def log_llm_first_chunk_text(text: str) -> None:
    """打印 LLM 首包文本（完整文本，不截断）。"""
    _log(f"[LLM首包文本] {text}")


def log_llm_chunk(index: int, text: str, elapsed_s: Optional[float] = None) -> None:
    """打印 LLM 流式文本块（完整文本）。"""
    msg = f"[LLM文本块#{index}] {text}"
    if elapsed_s is not None:
        msg += f"  (+{elapsed_s:.3f}s)"
    _log(msg)


def log_llm_raw_chunk(index: int, text: str, elapsed_s: Optional[float] = None) -> None:
    """打印 LLM 底层原始流块（直接来自 LLM stream）。"""
    msg = f"[LLM原始流块#{index}] {text}"
    if elapsed_s is not None:
        msg += f"  (+{elapsed_s:.3f}s)"
    _log(msg)


def log_e2e(
    stage_stt: Optional[float] = None,
    stage_llm: Optional[float] = None,
    stage_tts: Optional[float] = None,
) -> None:
    """端到端耗时 + 三段：音频→STT结果、STT结果→首chunk、首chunk→TTS首帧。"""
    global _e2e_start, _e2e_stt_result_at, _e2e_llm_first_chunk_at, _e2e_tts_segment_sent_at, _e2e_end_timestamp, _tts_request_to_first_frame_logged, _llm_request_at
    if _e2e_start is None:
        return
    end = _e2e_end_timestamp if _e2e_end_timestamp is not None else time.time()
    d = end - _e2e_start

    msg = f"[E2E] {d:.2f}s (从音频给STT到TTS首帧)"
    if stage_stt is not None and stage_llm is not None and stage_tts is not None:
        msg += f"，阶段和≈{stage_stt + stage_llm + stage_tts:.2f}s"
    _log(msg)

    if _e2e_stt_result_at is not None:
        if _llm_request_at is not None and _e2e_tts_segment_sent_at is not None and _e2e_end_timestamp is not None:
            s1 = _e2e_stt_result_at - _e2e_start
            s2 = _e2e_tts_segment_sent_at - _llm_request_at
            s3 = _e2e_end_timestamp - _e2e_tts_segment_sent_at
            _log(f"[E2E-分段] 1. 音频 → STT: {s1:.2f}s  2. 送往LLM → 句段送TTS: {s2:.2f}s  3. 句段送TTS → TTS首帧: {s3:.2f}s")
        elif _e2e_llm_first_chunk_at is not None and _e2e_end_timestamp is not None:
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
    _e2e_tts_segment_sent_at = None
    _e2e_end_timestamp = None
    _tts_request_to_first_frame_logged = False
    _llm_request_at = None


def log_vad_interrupt(reason: str = "用户打断") -> None:
    _log(f"[VAD打断] {reason}")
