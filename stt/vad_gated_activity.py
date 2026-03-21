"""
vad_gated_activity.py
======================
VAD 门控音频识别：仅在 VAD 检测到人声后，才将音频推送给 STT。

核心改进：
  - 维护滑动窗口预缓冲区（pre-buffer），在 START_OF_SPEECH 触发时将缓冲帧
    补发给 STT，弥补 VAD 检测延迟（通常 100~400ms）导致的首字截断问题。
  - 音频帧始终送入 VAD（不影响打断检测、用户状态机等逻辑）。

使用方式（无需修改任何 livekit-agents 源文件）：

    # 在创建 AgentSession 之前调用一次 install()
    from vad_gated_activity import install
    install(pre_buffer_duration=0.5)

    # 之后正常使用 AgentSession，框架会自动使用 VadGatedActivity
    session = AgentSession(stt=..., vad=..., ...)

设计说明：
  - install() 通过替换 livekit.agents.voice.agent_session 模块中的
    AgentActivity 名字来生效，属于标准 monkey-patch 手法，不修改任何文件。
  - 兼容 pip install livekit-agents 后直接使用。
"""

from __future__ import annotations

import collections
import logging
from typing import TYPE_CHECKING

from livekit import rtc
from livekit.agents import vad as vad_module
from livekit.agents.voice.agent_activity import AgentActivity

if TYPE_CHECKING:
    from livekit.agents.voice.agent import Agent
    from livekit.agents.voice.agent_session import AgentSession

logger = logging.getLogger(__name__)


class VadGatedActivity(AgentActivity):
    """
    在 AgentActivity 基础上增加 VAD 门控 + 预缓冲：

    静音期间：
      push_audio → VAD only（skip_stt=True）
      同时将帧写入滑动窗口预缓冲区

    START_OF_SPEECH 触发时：
      将预缓冲区所有帧直接写入 STT 通道（补全 VAD 延迟丢失的开头）
      之后的帧正常送 VAD + STT

    END_OF_SPEECH 触发时：
      停止发 STT，清空预缓冲区，等待下一轮

    fallback 保险：
      若 self.vad 为 None（未配置 VAD），完全退化为父类行为，不影响原有逻辑。
    """

    # 类级别配置，install() 会在替换前修改此值
    PRE_BUFFER_DURATION: float = 0.5

    def __init__(self, agent: "Agent", sess: "AgentSession") -> None:
        super().__init__(agent, sess)
        self._vad_speaking: bool = False
        # 滑动窗口预缓冲区（deque 保证 O(1) 两端操作）
        self._pre_buffer: collections.deque[rtc.AudioFrame] = collections.deque()
        self._pre_buffer_secs: float = 0.0

    # ------------------------------------------------------------------ #
    # RecognitionHooks 回调（由 AudioRecognition 内部 VAD 事件触发）
    # ------------------------------------------------------------------ #

    def on_start_of_speech(self, ev: vad_module.VADEvent | None) -> None:
        print("\n on_start_of_speech \n")
        super().on_start_of_speech(ev)
        self._vad_speaking = True
        self._flush_pre_buffer_to_stt()

    def on_end_of_speech(self, ev: vad_module.VADEvent | None) -> None:
        super().on_end_of_speech(ev)
        self._vad_speaking = False
        self._pre_buffer.clear()
        self._pre_buffer_secs = 0.0

    # ------------------------------------------------------------------ #
    # 音频推送（完整 override，加入 VAD 门控逻辑）
    # ------------------------------------------------------------------ #

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        # 未配置 VAD 时，完全退化为父类行为（保险 fallback）
        if self.vad is None:
            super().push_audio(frame)
            return

        # ---------- 以下复刻父类 push_audio 逻辑，仅改动 skip_stt 计算 ----------

        if not self._started:
            return

        # KWS 唤醒前：音频送 KWS 队列（父类逻辑保留）
        if not self.is_awake:
            self._kws_queue.put_nowait(frame.data.tobytes())
            return

        should_discard = bool(
            self._current_speech
            and not self._current_speech.allow_interruptions
            and self._session.options.discard_audio_if_uninterruptible
        )

        # RealtimeModel 走独立通道，不受 VAD 门控影响
        if not should_discard and self._rt_session is not None:
            self._rt_session.push_audio(frame)

        if self._audio_recognition is None:
            return

        if self._vad_speaking:
            # 说话期间：VAD + STT 均正常接收
            self._audio_recognition.push_audio(frame, skip_stt=should_discard)
        else:
            # 静音期间：仅 VAD 接收；同时维护预缓冲区
            self._audio_recognition.push_audio(frame, skip_stt=True)
            if not should_discard:
                self._enqueue_pre_buffer(frame)

    # ------------------------------------------------------------------ #
    # 预缓冲区管理
    # ------------------------------------------------------------------ #

    def _enqueue_pre_buffer(self, frame: rtc.AudioFrame) -> None:
        """将帧写入滑动窗口，超出 PRE_BUFFER_DURATION 时自动淘汰最旧帧。"""
        self._pre_buffer.append(frame)
        self._pre_buffer_secs += frame.samples_per_channel / max(frame.sample_rate, 1)

        while self._pre_buffer_secs > self.PRE_BUFFER_DURATION and self._pre_buffer:
            old = self._pre_buffer.popleft()
            self._pre_buffer_secs -= old.samples_per_channel / max(old.sample_rate, 1)

    def _flush_pre_buffer_to_stt(self) -> None:
        """
        将预缓冲区帧直接写入 STT 通道，绕过 VAD 通道（避免重复检测）。

        同时将预缓冲字节数通知给 STT 实例（若支持 set_prebuffer_bytes 接口），
        使服务端过滤时能区分预缓冲帧和真实语音帧，避免幻觉漏网。
        """
        if not self._pre_buffer or self._audio_recognition is None:
            return

        stt_ch = self._audio_recognition._stt_ch
        if stt_ch is None:
            logger.warning(
                "[VadGated] stt_ch is None，丢弃预缓冲 %d 帧 (%.3fs)——第一句话可能漏识别",
                len(self._pre_buffer),
                self._pre_buffer_secs,
            )
            self._pre_buffer.clear()
            self._pre_buffer_secs = 0.0
            return

        # 计算预缓冲字节数（int16 PCM，mono）并通知 STT 实例
        # 必须在发帧之前调用，确保 SpeechStream.ensure_started() 读到正确值
        prebuffer_samples = sum(f.samples_per_channel for f in self._pre_buffer)
        prebuffer_bytes = prebuffer_samples * 2  # int16 = 2 bytes/sample
        stt = self.stt
        if prebuffer_bytes > 0 and stt is not None and hasattr(stt, "set_prebuffer_bytes"):
            stt.set_prebuffer_bytes(prebuffer_bytes)

        logger.debug(
            "[VadGated] flushing pre-buffer to STT: %.3fs, %d frames, %d bytes",
            self._pre_buffer_secs,
            len(self._pre_buffer),
            prebuffer_bytes,
        )
        for buffered_frame in self._pre_buffer:
            try:
                stt_ch.send_nowait(buffered_frame)
            except Exception:
                pass  # 通道满或已关闭，安全忽略

        self._pre_buffer.clear()
        self._pre_buffer_secs = 0.0


# ------------------------------------------------------------------ #
# 安装函数：monkey-patch AgentSession，不修改任何源文件
# ------------------------------------------------------------------ #

def install(pre_buffer_duration: float = 0.5) -> None:
    """
    将 AgentSession 内部创建 activity 时使用的 AgentActivity 替换为
    VadGatedActivity，从而在不修改 livekit-agents 任何源文件的情况下
    注入 VAD 门控逻辑。

    原理：
        agent_session.py 第 1040 行：
            self._next_activity = AgentActivity(agent, self)
        此处 AgentActivity 是模块级名字，替换该名字即可让 Session
        在运行时使用我们的子类。

    必须在首次调用 AgentSession() 之前执行此函数。

    Args:
        pre_buffer_duration: 预缓冲时长（秒）。
            较大值能捕获更完整的话语开头，但内存开销略增。
            默认 0.5s，可覆盖绝大多数 VAD 检测延迟（100~400ms）。
    """
    import livekit.agents.voice.agent_session as _mod

    if getattr(_mod, "_vad_gated_installed", False):
        logger.debug("[VadGated] already installed, skipping")
        return

    VadGatedActivity.PRE_BUFFER_DURATION = pre_buffer_duration
    _mod.AgentActivity = VadGatedActivity
    _mod._vad_gated_installed = True

    logger.info(
        "[VadGated] VadGatedActivity installed (pre_buffer=%.2fs)", pre_buffer_duration
    )
