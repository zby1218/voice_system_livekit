#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("SpeakerSeparator")

# ClearerVoice-Studio 与本项目同级：../ClearerVoice-Studio/clearvoice
_DEFAULT_CLEARVOICE_PKG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../ClearerVoice-Studio/clearvoice")
)


class SpeakerSeparator:
    """持久化说话人分离模块。

    模型在构造时加载一次，之后常驻内存等待音频输入，不会反复加载。
    异步接口通过单线程 ThreadPoolExecutor 在后台串行执行推理，不阻塞
    asyncio 事件循环，也避免模型并发访问问题。

    输入/输出格式均为 PCM s16le 单声道 16kHz 字节流，与 stt_server_novad
    的 frames_asr 格式直接兼容。
    """

    def __init__(
        self,
        clearvoice_pkg: str = _DEFAULT_CLEARVOICE_PKG,
        device: str = "cuda",
    ) -> None:
        """
        Parameters
        ----------
        clearvoice_pkg : str
            ClearerVoice-Studio 中 clearvoice Python 包所在目录，即包含
            clearvoice/__init__.py 的那一层目录（默认自动定位兄弟项目）。
        device : str
            推理设备，"cuda" 或 "cpu"。
        """
        self._clearvoice_pkg = clearvoice_pkg
        # ClearerVoice-Studio 项目根（checkpoints/ 在这里）
        self._clearervoice_root = os.path.dirname(clearvoice_pkg)
        self._device = device
        self._cv = None
        # 单线程执行器：保证模型推理串行（MossFormer2 不支持并发）
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="speaker_sep"
        )
        self._load_model()

    # ------------------------------------------------------------------
    # 内部：模型加载
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """加载 ClearVoice MossFormer2_SS_16K，只在构造时执行一次。"""
        if not os.path.isdir(self._clearvoice_pkg):
            raise FileNotFoundError(
                f"clearvoice 包目录不存在: {self._clearvoice_pkg}\n"
                "请确认 ClearerVoice-Studio 已克隆到正确位置，"
                "或通过 clearvoice_pkg 参数显式指定路径。"
            )

        if self._clearvoice_pkg not in sys.path:
            sys.path.insert(0, self._clearvoice_pkg)

        from clearvoice import ClearVoice  # noqa: PLC0415

        logger.info("正在加载 MossFormer2_SS_16K 人声分离模型（路径: %s）...", self._clearervoice_root)

        # 模型加载时内部用相对路径 checkpoints/MossFormer2_SS_16K，
        # 需切换到 ClearerVoice-Studio 根目录确保路径可解析。
        original_cwd = os.getcwd()
        try:
            os.chdir(self._clearervoice_root)
            self._cv = ClearVoice(
                task="speech_separation",
                model_names=["MossFormer2_SS_16K"],
            )
        finally:
            os.chdir(original_cwd)

        logger.info("MossFormer2_SS_16K 加载完成，设备: %s", self._device)

    # ------------------------------------------------------------------
    # 内部：同步分离（在线程池中执行）
    # ------------------------------------------------------------------

    def _separate_sync(self, audio_bytes: bytes) -> list[bytes]:
        """PCM s16le bytes → 各说话人 PCM s16le bytes 列表。

        Returns
        -------
        list[bytes]
            长度等于模型输出的说话人数（通常为 2）。
            output[0] 为第一路说话人，output[1] 为第二路。
        """
        # FunASR 推理后 PyTorch 保留了大量碎片化显存，归还后 MossFormer2 才能申请到连续块
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # PCM s16le → float32 [1, N]
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_f32 = audio_int16.astype(np.float32) / 32768.0
        audio_batch = audio_f32.reshape(1, -1)  # [1, N]

        # ClearVoice 推理 → [n_spks, 1, N] float32 numpy
        output = self._cv(audio_batch)  # call_t2t_mode

        results: list[bytes] = []
        n_spks = output.shape[0]
        for spk_idx in range(n_spks):
            spk_audio = output[spk_idx, 0, :]  # [N] float32
            spk_int16 = np.clip(spk_audio * 32768.0, -32768, 32767).astype(np.int16)
            results.append(spk_int16.tobytes())

        logger.debug(
            "分离完成：%d 路说话人，输入 %d 帧，各路 %d bytes",
            n_spks,
            len(audio_int16),
            len(results[0]) if results else 0,
        )
        return results

    # ------------------------------------------------------------------
    # 公共异步接口
    # ------------------------------------------------------------------

    async def separate(self, audio_bytes: bytes) -> list[bytes]:
        """异步说话人分离，在独立线程中运行，不阻塞事件循环。

        Parameters
        ----------
        audio_bytes : bytes
            PCM s16le 单声道 16kHz 字节流（完整一句话）。

        Returns
        -------
        list[bytes]
            各说话人 PCM s16le bytes，output[0] 为第一路。
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._separate_sync, audio_bytes)
