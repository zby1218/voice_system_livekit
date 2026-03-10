"""
TTS 请求指标记录模块（面向单次请求）

用于在 tts_server 侧记录 queue_wait_s、active_count、ttfb_s、total_s 等，
便于判断是否存在请求堆积（参见 docs/VAD_INTERRUPT_AND_TTS_FLOW.md 第 8 节）。
"""

import logging
import time
from typing import Optional


class RequestMetricsRecorder:
    """单次 TTS 请求的指标记录器。一次请求创建一个实例，按顺序调用各 record 方法。"""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._t_before_acquire: Optional[float] = None
        self._t_acquire: Optional[float] = None
        self._request_id: Optional[int] = None
        self._first_byte_recorded = False

    def start_acquire(self) -> None:
        """在尝试获取槽位之前调用（在 semaphore.acquire() 之前）。"""
        self._t_before_acquire = time.perf_counter()

    def finish_acquire(self, request_id: int, active_count: int) -> None:
        """获取槽位之后调用，记录排队等待时间和当前并发数。"""
        self._t_acquire = time.perf_counter()
        self._request_id = request_id
        queue_wait_s = self._t_acquire - self._t_before_acquire if self._t_before_acquire is not None else 0.0
        self._logger.info(
            "[TTS-Metrics] acquire | request_id=%s queue_wait_s=%.3f active_count_at_acquire=%s",
            request_id,
            queue_wait_s,
            active_count,
        )

    def record_first_byte(self) -> None:
        """在第一次产出音频字节（首包）时调用，仅首次有效。"""
        if self._first_byte_recorded or self._t_acquire is None or self._request_id is None:
            return
        self._first_byte_recorded = True
        ttfb_s = time.perf_counter() - self._t_acquire
        self._logger.info(
            "[TTS-Metrics] first_byte | request_id=%s ttfb_s=%.3f",
            self._request_id,
            ttfb_s,
        )

    def finish_request(self) -> None:
        """请求结束、释放槽位时调用，记录总占用时长。"""
        if self._t_acquire is None or self._request_id is None:
            return
        total_s = time.perf_counter() - self._t_acquire
        self._logger.info(
            "[TTS-Metrics] release | request_id=%s total_s=%.3f",
            self._request_id,
            total_s,
        )

    def record_cancelled(self, stage: str) -> None:
        """请求被取消时调用，stage 如 queued/running。"""
        if self._request_id is None:
            return
        now = time.perf_counter()
        queue_wait_s = (
            now - self._t_before_acquire
            if self._t_before_acquire is not None and self._t_acquire is None
            else 0.0
        )
        run_s = (
            now - self._t_acquire
            if self._t_acquire is not None
            else 0.0
        )
        self._logger.info(
            "[TTS-Metrics] cancelled | request_id=%s stage=%s queue_wait_s=%.3f run_s=%.3f",
            self._request_id,
            stage,
            queue_wait_s,
            run_s,
        )
