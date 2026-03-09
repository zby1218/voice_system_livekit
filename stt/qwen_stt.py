# stt/qwen_stt.py
from __future__ import annotations

import asyncio
import json
import ssl
import time
from dataclasses import dataclass
from typing import Any, Optional, List
import logging
import numpy as np

import aiohttp
from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
)

# ===== Qwen3-ASR Constants =====
SAMPLE_RATE = 16000
NUM_CHANNELS = 1


@dataclass
class QwenSTTOptions:
    host: str = "localhost"
    port: int = 10096
    use_ssl: bool = False  # True => wss://
    
    # Qwen Specific
    context: str = ""  # Hotwords / Context
    language: str | None = None  # Force language if needed

    # Audio sending parameters
    # Sends chunks every `chunk_interval_ms` ms (approx)
    chunk_interval_ms: int = 60 # 60ms chunks

    
class QwenSTT(stt.STT):
    """
    LiveKit Agents STT plugin for Qwen3-ASR WebSocket server.
    """

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 10096,
        ssl: bool = False,
        context: str = "",
        language: str | None = None,
        chunk_interval_ms: int = 60,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                aligned_transcript=False,
            )
        )

        self._opts = QwenSTTOptions(
            host=host,
            port=port,
            use_ssl=ssl,
            context=context,
            language=language,
            chunk_interval_ms=chunk_interval_ms,
        )

        self._session: aiohttp.ClientSession | None = None
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            max_session_duration=10 * 60, 
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
        )

    @property
    def provider(self) -> str:
        scheme = "wss" if self._opts.use_ssl else "ws"
        return f"{scheme}://{self._opts.host}:{self._opts.port}"

    def stream(
        self,
        *,
        language: Any = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "QwenSpeechStream":
        return QwenSpeechStream(stt=self, pool=self._pool, conn_options=conn_options)

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        scheme = "wss" if self._opts.use_ssl else "ws"
        url = f"{scheme}://{self._opts.host}:{self._opts.port}"

        ssl_context: Optional[ssl.SSLContext] = None
        if self._opts.use_ssl:
            ssl_context = ssl.SSLContext()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

        session = self._ensure_session()

        try:
            print(f"[QwenSTT] connecting url={url}", flush=True)
            ws = await asyncio.wait_for(
                session.ws_connect(
                    url,
                    ssl=ssl_context,
                    heartbeat=None,
                    autoping=False,
                ),
                timeout=timeout,
            )
            print(f"[QwenSTT] ✅ WebSocket connected!", flush=True)
            return ws
        except asyncio.TimeoutError:
            print(f"[QwenSTT] ❌ Connection timeout", flush=True)
            raise APITimeoutError() from None
        except Exception as e:
            print(f"[QwenSTT] ❌ Connection error: {e}", flush=True)
            raise APIConnectionError() from e

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()
    
    # Optional: Implement _recognize_impl if one-shot recognition is needed
    # using valid streaming logic but closing immediately.
    # Leaving unimplemented if only streaming is used.
    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: Any = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
         raise NotImplementedError("Single-shot recognition not implemented for QwenSTT yet")


class QwenSpeechStream(stt.SpeechStream):
    def __init__(
        self,
        *,
        stt: QwenSTT,
        conn_options: APIConnectOptions,
        pool: utils.ConnectionPool[aiohttp.ClientWebSocketResponse],
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=SAMPLE_RATE)
        self._stt = stt
        self._pool = pool
        self._reconnect_event = asyncio.Event()
        self._speaking = False

    def push_frame(self, frame: rtc.AudioFrame | stt.RecognizeStream._FlushSentinel) -> None:
        # intercept FlushSentinel
        from livekit.agents.types import FlushSentinel
        if isinstance(frame, FlushSentinel):
            self.flush()
            return
        
        super().push_frame(frame)

    def _build_start_config(self) -> dict[str, Any]:
        o = self._stt._opts
        cfg = {"is_speaking": True}
        if o.context:
            cfg["context"] = o.context
        if o.language:
            cfg["language"] = o.language
        return cfg

    @utils.log_exceptions(logger=logging.getLogger("qwen_stt"))
    async def _run(self) -> None:
        closing_ws = False
        
        # Determine chunk size in samples
        # Qwen server works fine with small chunks, but batching slightly is efficient
        chunk_ms = max(20, self._stt._opts.chunk_interval_ms)
        samples_per_chunk = int(SAMPLE_RATE * chunk_ms / 1000.0)

        @utils.log_exceptions(logger=logging.getLogger("qwen_stt"))
        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws

            # AudioByteStream buffers input frames into fixed-size chunks
            # Note: Input is int16 from LiveKit
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                samples_per_channel=samples_per_chunk,
            )

            async def ensure_started() -> None:
                if self._speaking:
                    return
                # Send config to start/init state
                cfg = self._build_start_config()
                await ws.send_str(json.dumps(cfg, ensure_ascii=False))
                self._speaking = True

            async def send_audio_chunk(float32_bytes: bytes) -> None:
                await ws.send_bytes(float32_bytes)

            async for data in self._input_ch:
                frames_to_send: list[bytes] = []

                if isinstance(data, rtc.AudioFrame):
                    await ensure_started()
                    # Buffer audio
                    # get int16 bytes
                    int16_chunks = audio_bstream.write(data.data.tobytes())
                    
                    # Convert to float32
                    for chunk in int16_chunks:
                        # Convert int16 bytes -> np.int16 -> float32 -> bytes
                        data_int16 = np.frombuffer(chunk.data.tobytes(), dtype=np.int16)
                        data_float32 = data_int16.astype(np.float32) / 32768.0
                        frames_to_send.append(data_float32.tobytes())

                elif isinstance(data, self._FlushSentinel):
                    # Flush buffer
                    int16_chunks = audio_bstream.flush()
                    for chunk in int16_chunks:
                        data_int16 = np.frombuffer(chunk.data.tobytes(), dtype=np.int16)
                        data_float32 = data_int16.astype(np.float32) / 32768.0
                        frames_to_send.append(data_float32.tobytes())

                    # Send all pending audio
                    for b in frames_to_send:
                         await send_audio_chunk(b)
                    frames_to_send.clear()

                    # Send end of speech
                    if self._speaking:
                        print(f"[QwenSTT] 📤 Sending is_speaking=False", flush=True)
                        await ws.send_str(json.dumps({"is_speaking": False}))
                        self._speaking = False
                    continue

                # Normal send
                for b in frames_to_send:
                     await send_audio_chunk(b)

            # End of stream
            closing_ws = True
            if self._speaking:
                try:
                    await ws.send_str(json.dumps({"is_speaking": False}))
                except Exception:
                    pass
                self._speaking = False

        @utils.log_exceptions(logger=logging.getLogger("qwen_stt"))
        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws
            # 收到 is_final:True 后进入"封锁"状态，丢弃后续消息，直到下一轮 is_speaking:True
            # 防止服务端重置 state 后的残留 interim 结果被当成新一轮识别内容
            received_final = False

            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                    aiohttp.WSMsgType.ERROR,
                ):
                    if closing_ws:
                        return
                    raise APIStatusError(message="QwenSTT connection closed unexpectedly")

                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                try:
                    data = json.loads(msg.data)
                except Exception:
                    continue

                text = data.get("text", "")
                is_final = data.get("is_final", False)

                # 收到 final 后，后续任何消息（包括残留 interim）全部丢弃，
                # 等待 send_task 发出下一个 is_speaking:True 来重置
                if received_final:
                    continue

                if is_final:
                    received_final = True
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                            alternatives=[stt.SpeechData(text=text, language="")],
                        )
                    )
                else:
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                            alternatives=[stt.SpeechData(text=text, language="")],
                        )
                    )

        while True:
            closing_ws = False
            async with self._pool.connection(timeout=self._conn_options.timeout) as ws:
                tasks = [
                    asyncio.create_task(send_task(ws)),
                    asyncio.create_task(recv_task(ws)),
                ] 
                
                # Simplified reconnect logic compared to custom_stt, mostly relying on pool
                try:
                     done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                     for task in done:
                         task.result() 
                     # If one finishes (e.g. error), cancel other
                finally:
                    for t in tasks:
                        t.cancel()
                        try:
                            await t
                        except (asyncio.CancelledError, Exception):
                            pass
                    
                    if closing_ws:
                        break # Normal exit
