"""
播放边界打点日志：用于分析「残留播放」发生在 pipeline / output / track 的哪一阶段。

四个边界（不打 ① 等符号，便于控制台与 grep）：
  - pipeline_to_output: pipeline 将帧交给 session 的 audio output 前
  - output_received: Room output 收到帧（进入 _audio_buf 前）
  - track_submit: 帧即将写入 RTC 轨道（_audio_source.capture_frame 前，脱离 agent 掌控的起点）
  - clear: 发生 clear_buffer / clear_queue

日志为 agent 侧（服务端）输出，非 client 侧。若需落盘，在 agent 启动时调用
init_playback_boundary_file_logging(log_dir, log_filename)。
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional


def init_playback_boundary_file_logging(
    log_dir: str,
    log_filename: str = "playback_boundary.log",
) -> None:
    """
    为 playback_boundary 增加文件输出，便于分析残留播放。
    在 agent 进程启动时调用一次即可；不调用则仅走 root/控制台。
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)
    log = logging.getLogger("playback_boundary")
    log.setLevel(logging.INFO)
    if not any(h for h in log.handlers if getattr(h, "baseFilename", None) == log_path):
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        log.addHandler(fh)


class PlaybackBoundaryLogger:
    """面向对象：单一职责，仅负责在播放链路的四个边界打点，便于分析残留播放位置。"""

    _LOG_PREFIX = "[PlaybackBoundary]"

    def __init__(self, log: Optional[logging.Logger] = None) -> None:
        self._log = log or logging.getLogger("playback_boundary")

    def log_pipeline_to_output(self, frame_index: Optional[int] = None) -> None:
        """边界 1：pipeline 即将把帧交给 audio_output.capture_frame。"""
        _ensure_file_handler()
        ts = time.perf_counter()
        self._log.info(
            "%s stage=pipeline_to_output ts=%.6f frame_index=%s",
            self._LOG_PREFIX,
            ts,
            frame_index,
        )

    def log_output_received(self, frame_index: Optional[int] = None) -> None:
        """边界 2：Room output 收到帧（capture_frame 入口，进入 _audio_buf 前）。"""
        _ensure_file_handler()
        ts = time.perf_counter()
        self._log.info(
            "%s stage=output_received ts=%.6f frame_index=%s",
            self._LOG_PREFIX,
            ts,
            frame_index,
        )

    def log_track_submit(self, frame_index: Optional[int] = None) -> None:
        """边界 3：帧即将写入 RTC 轨道（_audio_source.capture_frame 前，可能脱离 agent 掌控）。"""
        _ensure_file_handler()
        ts = time.perf_counter()
        self._log.info(
            "%s stage=track_submit ts=%.6f frame_index=%s",
            self._LOG_PREFIX,
            ts,
            frame_index,
        )

    def log_clear(self, reason: str = "clear_buffer") -> None:
        """边界 4：发生清空（clear_buffer 或 clear_queue）。"""
        _ensure_file_handler()
        ts = time.perf_counter()
        self._log.info(
            "%s stage=clear ts=%.6f reason=%s",
            self._LOG_PREFIX,
            ts,
            reason,
        )


def _ensure_file_handler() -> None:
    """首次打 log 时若尚无文件 handler，则按 cwd/log/agent 兜底落盘（避免 worker 未执行 init）。"""
    log = logging.getLogger("playback_boundary")
    if any(isinstance(h, logging.FileHandler) for h in log.handlers):
        return
    try:
        log_dir = os.path.join(os.getcwd(), "log", "agent")
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, "playback_boundary.log")
        fh = logging.FileHandler(path, mode="w", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        log.addHandler(fh)
    except Exception:
        pass


# 模块级单例，供 generation 与 room_io 使用，避免到处传参、保持低耦合
default_logger: PlaybackBoundaryLogger = PlaybackBoundaryLogger()
