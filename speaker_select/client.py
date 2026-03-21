#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
说话人分离服务客户端：将 PCM 音频字节 POST 到本地 speaker 服务，取回第一路说话人 PCM。

与 speaker_select/server.py 配套使用，供 stt_server_novad 在 ENABLE_SPEAKER_SELECT=True 时调用。
"""

import urllib.error
import urllib.request


def separate_via_service(
    audio_bytes: bytes,
    service_url: str,
    timeout: float = 30.0,
) -> bytes:
    """
    将 PCM s16le 16kHz 单声道音频发送到 speaker 服务，返回第一路说话人 PCM 字节。

    Parameters
    ----------
    audio_bytes : bytes
        原始混合音频，PCM s16le 16kHz 单声道。
    service_url : str
        服务地址，例如 "http://127.0.0.1:8000/separate"。
    timeout : float
        请求超时秒数。

    Returns
    -------
    bytes
        第一路说话人 PCM s16le 16kHz 单声道。若请求失败则抛出异常。
    """
    req = urllib.request.Request(
        service_url,
        data=audio_bytes,
        method="POST",
        headers={"Content-Type": "application/octet-stream"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()
