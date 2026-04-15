#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频尾部能量过滤模块

【核心原理】
  基于端点检测截取的对话音频，结构固定为：
    [ 前段静音（时长不定）] + [ 触发端点的语音（尾部）]

  过滤问题等价于：「触发端点的那段语音，能量够不够响？」
  → 直接分析音频尾部 N 秒，完全绕开静音稀释问题。

【三类误触场景均可覆盖】
  ① 长静音 + 响亮语音  → 尾部 RMS 高  → 放行
  ② 全程低音量语音     → 尾部 RMS 低  → 过滤
  ③ 长静音 + 低音量语音（误触发）→ 尾部 RMS 低  → 过滤

【双指标 OR 联合判断】
  主指标  tail_rms  ：尾部全局 RMS ≥ tail_rms_threshold
  副指标  tail_top5 ：尾部帧 Top5% 均值 ≥ tail_top5_threshold
  任一通过 → 放行；两者均不通过 → 能量过低，跳过推理

  副指标的作用：应对尾部末尾存在短暂 hangover 静音的情况
  （端点检测 hangover 期约 200-500ms，会将 RMS 轻微拉低，
    Top5% 只取帧中最响的 5%，对尾端短暂静音不敏感）

【配套调参工具】
  record_analysis/compare_voice_filter.py
  用法：python compare_voice_filter.py low_voice.wav norm_voice.wav
  可在新音频样本上验证阈值，输出各指标对比和建议阈值。

【实测数据（int16 scale，16kHz）】
  low_voice  (需过滤) 尾2s RMS= 327  Top5%= 558
  low_voice1 (需过滤) 尾2s RMS= 605  Top5%=1789
  norm_voice1(需保留) 尾2s RMS=4261  Top5%=8103
  norm_voice2(需保留) 尾2s RMS=2985  Top5%=7012
  阈值 tail_rms=1500 / tail_top5=2500 → 全部样本正确分类
"""

from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TailEnergyConfig:
    """
    尾部能量过滤参数。

    所有阈值均使用 int16 scale（0 ~ 32768），与 PCM s16le 原始数据单位一致，
    无需额外归一化，便于直接与录音工具测量值对比。

    调参流程：
      1. 录制若干需过滤和需保留的音频样本
      2. 运行 record_analysis/compare_voice_filter.py 查看各样本的尾部指标
      3. 根据输出的「建议阈值」和间隙倍数调整下方参数
    """
    # 采样率（Hz），须与音频流一致
    sample_rate: int = 16000

    # 尾部分析窗口（秒）
    #   - 取音频末尾 tail_sec 秒做能量分析
    #   - 建议值：2.0s（即使 hangover 有 500ms 静音，仍有 1.5s 语音参与计算）
    #   - 若端点检测 hangover 较长，可适当增大（如 3.0s）
    tail_sec: float = 2.0

    # 主指标阈值：尾部全局 RMS（int16 scale）
    #   - 直接衡量触发端点的语音段平均响度
    #   - 实测间隙：低音量样本 ≤ 605，正常样本 ≥ 2985，建议阈值 1500
    tail_rms_threshold: float = 1500.0

    # 副指标阈值：尾部帧 Top5% 均值（int16 scale）
    #   - 只取尾部最响的 5% 帧求均值，对尾端短暂静音不敏感
    #   - 实测间隙：低音量样本 ≤ 1789，正常样本 ≥ 7012，建议阈值 2500
    tail_top5_threshold: float = 2500.0


# ---------------------------------------------------------------------------
# 计算函数
# ---------------------------------------------------------------------------

def compute_tail_energy(audio_bytes: bytes, config: TailEnergyConfig) -> dict:
    """
    计算音频尾部的能量指标，返回字典：
      tail_rms        : 尾部全局 RMS（主指标）
      tail_top5       : 尾部帧 Top5% 均值（副指标）
      tail_sec_actual : 实际分析的尾部时长（秒）

    :param audio_bytes: PCM s16le 字节流（单声道，16kHz）
    :param config:      TailEnergyConfig 实例
    """
    data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)

    # 尾部截取：取末尾 tail_sec 秒；若音频本身较短，最多取总长的一半
    tail_samples = min(len(data), int(config.tail_sec * config.sample_rate))
    tail_samples = min(tail_samples, max(1, len(data) // 2))
    tail = data[-tail_samples:]

    # 主指标：尾部全局 RMS
    tail_rms = float(np.sqrt(np.mean(tail ** 2))) if len(tail) > 0 else 0.0

    # 副指标：尾部帧 Top5% 均值（20ms 帧，取最响的前 5% 帧）
    frame_size = int(config.sample_rate * 0.02)          # 20ms = 320 采样点
    n_frames = max(1, len(tail) // frame_size)
    frames = tail[: n_frames * frame_size].reshape(n_frames, frame_size)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
    n_top = max(1, n_frames // 20)                       # 取前 5% 帧
    tail_top5 = float(np.sort(frame_rms)[::-1][:n_top].mean())

    return {
        "tail_rms":        tail_rms,
        "tail_top5":       tail_top5,
        "tail_sec_actual": tail_samples / config.sample_rate,
    }


# ---------------------------------------------------------------------------
# 过滤判断
# ---------------------------------------------------------------------------

def is_low_energy(audio_bytes: bytes, config: TailEnergyConfig) -> tuple[bool, str]:
    """
    判断音频尾部能量是否过低（双指标 OR 联合）。

    返回：(能量是否过低, 原因描述)
      True  → 能量过低，应跳过 STT 推理
      False → 能量充足，可进行推理（原因描述为空字符串）

    判断逻辑：
      pass_rms  = tail_rms  >= tail_rms_threshold   (主指标)
      pass_top5 = tail_top5 >= tail_top5_threshold  (副指标)
      任一通过 → 放行；两者均不通过 → 能量过低

    :param audio_bytes: PCM s16le 字节流（单声道，16kHz）
    :param config:      TailEnergyConfig 实例
    """
    metrics = compute_tail_energy(audio_bytes, config)
    tail_rms  = metrics["tail_rms"]
    tail_top5 = metrics["tail_top5"]

    pass_rms  = tail_rms  >= config.tail_rms_threshold
    pass_top5 = tail_top5 >= config.tail_top5_threshold

    if pass_rms or pass_top5:
        return False, ""

    return True, (
        f"尾部RMS={tail_rms:.1f}<{config.tail_rms_threshold:.0f} "
        f"且Top5%={tail_top5:.1f}<{config.tail_top5_threshold:.0f}"
    )
