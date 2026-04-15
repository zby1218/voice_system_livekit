# my_stt.py
from __future__ import annotations

import asyncio
import json
import ssl
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, List
import logging

import aiohttp
from datetime import datetime
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
from livekit.agents.metrics import STTMetrics
from livekit.agents.metrics.base import Metadata


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

# ===== FunASR 常用输入参数 =====
SAMPLE_RATE = 16000
NUM_CHANNELS = 1


@dataclass
class FunASRSTTOptions:
    host: str = "localhost"
    port: int = 10095
    use_ssl: bool = False  # True => wss://
    mode: str = "2pass"  # "online" | "offline" | "2pass"

    chunk_size: List[int] = None  # e.g. [5,10,5]
    chunk_interval: int = 10
    encoder_chunk_look_back: int = 4
    decoder_chunk_look_back: int = 0

    hotwords: str = ""  # 可以是 json 字符串 or 直接文本（看你服务端支持啥）
    itn: bool = True

    # 发送给 FunASR 的音频 chunk 时长（ms）
    # 你原始 client 里等价于: 60 * chunk_size[1] / chunk_interval
    # 默认 chunk_size[1]=10, chunk_interval=10 -> 60ms
    chunk_ms: int = 60

    # ws 连接子协议（你原 client 用了 subprotocols=["binary"]）
    subprotocol: str = "binary"


class MySTT(stt.STT):
    """
    LiveKit Agents STT plugin for FunASR WebSocket server (the same protocol as your demo client).
    """

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 10095,
        ssl: bool = False,
        mode: str = "2pass",
        chunk_size: List[int] | None = None,
        chunk_interval: int = 10,
        encoder_chunk_look_back: int = 4,
        decoder_chunk_look_back: int = 0,
        hotwords: str = "",
        itn: bool = True,
        chunk_ms: int | None = None,
        on_segment_submitted: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                aligned_transcript=False,
            )
        )

        if chunk_size is None:
            chunk_size = [5, 10, 5]

        # 让行为尽量贴近你那份 client：chunk_ms 默认由 chunk_size/interval 推出来
        if chunk_ms is None:
            # 等价于你 client 的: 60 * chunk_size[1] / chunk_interval
            # 注意：这是“每次发送的 stride 对应时长”
            chunk_ms = int(60 * chunk_size[1] / chunk_interval)

        self._opts = FunASRSTTOptions(
            host=host,
            port=port,
            use_ssl=ssl,
            mode=mode,
            chunk_size=chunk_size,
            chunk_interval=chunk_interval,
            encoder_chunk_look_back=encoder_chunk_look_back,
            decoder_chunk_look_back=decoder_chunk_look_back,
            hotwords=hotwords,
            itn=itn,
            chunk_ms=chunk_ms,
        )
        self._on_segment_submitted = on_segment_submitted  # 音频段提交给 STT 时回调，用于 E2E 起点

        self._session: aiohttp.ClientSession | None = None

        # 用 LiveKit 的 ConnectionPool，断线/超时更好处理
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            max_session_duration=10 * 60,  # 可调
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
        language: Any = None,  # FunASR ws 协议里一般不需要 language，这里保持兼容签名
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "SpeechStream":
        return SpeechStream(
            stt=self,
            pool=self._pool,
            conn_options=conn_options,
            on_segment_submitted=self._on_segment_submitted,
        )

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

        # aiohttp 的 ws_connect 参数跟 websockets 略不同：
        # - protocols: 子协议
        # - ssl: SSLContext
        # - heartbeat: ping
        try:
            print(f"[MySTT] connecting url={url} ssl={bool(ssl_context)} subprotocol={self._opts.subprotocol} timeout={timeout}", flush=True)
            ws = await asyncio.wait_for(
                session.ws_connect(
                    url,
                    protocols=(self._opts.subprotocol,),
                    ssl=ssl_context,
                    heartbeat=None,
                    autoping=False,
                ),
                timeout=timeout,
            )
            print(f"[MySTT] ✅ WebSocket 连接成功!", flush=True)
            return ws
        except asyncio.TimeoutError:
            print(f"[MySTT] ❌ 连接超时", flush=True)
            raise APITimeoutError() from None
        except Exception as e:
            print(f"[MySTT] ❌ 连接错误: {e}", flush=True)
            raise APIConnectionError() from e

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language="",  # 保持签名兼容（实际上 FunASR ws 不用）
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """
        Non-streaming (one-shot) recognition required by abstract STT.
        Implemented by using FunASR websocket in offline mode for a single utterance.
        """
        try:
            # 1) 合并音频帧 -> wav bytes
            wav_bytes = rtc.combine_audio_frames(buffer).to_wav_bytes()

            # 2) 从 wav bytes 里取 PCM 数据（跳过 wav header）
            # 最稳的方法：用 Python wave 解码（Windows 也行）
            import io, wave

            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                # 期望：16k, mono, int16；如果不是，也能读出来，但服务端可能要求 16k
                pcm_bytes = wf.readframes(wf.getnframes())
                sr = wf.getframerate()

            # 如果你的 LiveKit 输入不是 16k，这里建议直接报错或你自己做重采样
            # 先给最小可用：强要求 16k
            if sr != SAMPLE_RATE:
                raise ValueError(f"Expected {SAMPLE_RATE}Hz audio, got {sr}Hz. Please resample before STT.")

            # 3) 建立 ws 连接
            timeout = conn_options.timeout
            async with self._pool.connection(timeout=timeout) as ws:
                # 4) 发配置包（强制 offline，保证一次性出 final）
                o = self._opts
                cfg = {
                    "mode": "offline",
                    "chunk_size": o.chunk_size,
                    "chunk_interval": o.chunk_interval,
                    "encoder_chunk_look_back": o.encoder_chunk_look_back,
                    "decoder_chunk_look_back": o.decoder_chunk_look_back,
                    "audio_fs": SAMPLE_RATE,
                    "wav_name": "livekit_recognize",
                    "wav_format": "pcm",
                    "is_speaking": True,
                    "hotwords": o.hotwords or "",
                    "itn": bool(o.itn),
                }
                await ws.send_str(json.dumps(cfg, ensure_ascii=False))

                # 5) 发送音频 bytes（按 stride 分块，行为贴近你 demo）
                stride = int(o.chunk_ms / 1000.0 * SAMPLE_RATE * 2)
                if stride <= 0:
                    stride = 1920  # fallback ~60ms@16k

                for i in range(0, len(pcm_bytes), stride):
                    await ws.send_bytes(pcm_bytes[i : i + stride])

                # 6) 句末 finalize
                await ws.send_str(json.dumps({"is_speaking": False}))

                # 7) 等 final
                final_text = ""
                t0 = time.time()
                while True:
                    # 手动超时兜底
                    if time.time() - t0 > max(5.0, timeout):
                        raise APITimeoutError()

                    msg = await ws.receive()
                    if msg.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSING,
                        aiohttp.WSMsgType.ERROR,
                    ):
                        raise APIStatusError(message="FunASR STT connection closed while recognizing")

                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue

                    try:
                        data = json.loads(msg.data)
                    except Exception:
                        continue

                    text = data.get("text", "")
                    mode = data.get("mode", "")
                    is_final = bool(data.get("is_final", False))
                    # demo 里 offline 会直接来一次；2pass-offline 也算最终
                    if text and (mode in ("offline", "2pass-offline") or is_final):
                        final_text = text
                        break

            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[stt.SpeechData(text=final_text, language="")],
            )

        except APITimeoutError:
            raise
        except APIStatusError:
            raise
        except Exception as e:
            raise APIConnectionError() from e

class SpeechStream(stt.SpeechStream):
    """
    One SpeechStream per AgentSession.
    Supports multiple utterances by:
      - start: send config JSON with is_speaking=True
      - end_input(): send {"is_speaking": false}
      - next audio after flush: send a new config JSON again
    """

    def __init__(
        self,
        *,
        stt: MySTT,
        conn_options: APIConnectOptions,
        pool: utils.ConnectionPool[aiohttp.ClientWebSocketResponse],
        on_segment_submitted: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=SAMPLE_RATE)
        self._stt: MySTT = stt
        self._pool = pool
        self._on_segment_submitted = on_segment_submitted

        self._reconnect_event = asyncio.Event()

        # utterance state
        self._utt_id = 0
        self._speaking = False  # whether we've started an utterance on server
        self._utt_audio_bytes = 0  # 当前句已发送的 PCM 字节数，用于 STTMetrics.audio_duration
        self._inference_start_at = None  # 发送 is_speaking=False 的时间，用于 STTMetrics.duration
        self._trace = {
            "utt_id": 0,
            "cfg_sent_ts": None,
            "first_audio_sent_ts": None,
            "last_audio_sent_ts": None,
            "finalize_sent_ts": None,
            "first_text_ts": None,
            "final_text_ts": None,
        }

    def _build_start_config(self, wav_name: str) -> dict[str, Any]:
        o = self._stt._opts
        return {
            "mode": o.mode,
            "chunk_size": o.chunk_size,
            "chunk_interval": o.chunk_interval,
            "encoder_chunk_look_back": o.encoder_chunk_look_back,
            "decoder_chunk_look_back": o.decoder_chunk_look_back,
            "audio_fs": SAMPLE_RATE,
            "wav_name": wav_name,
            "wav_format": "pcm",
            "is_speaking": True,
            "hotwords": o.hotwords or "",
            "itn": bool(o.itn),
        }

    def _ts(self) -> float:
        return time.perf_counter()


    def _reset_trace_for_new_utt(self) -> None:
        self._trace.update(
            cfg_sent_ts=None,
            first_audio_sent_ts=None,
            last_audio_sent_ts=None,
            finalize_sent_ts=None,
            first_text_ts=None,
            final_text_ts=None,
        )

    def push_frame(self, frame: rtc.AudioFrame | stt.RecognizeStream._FlushSentinel) -> None:
        # intercept FlushSentinel
        from livekit.agents.types import FlushSentinel
        if isinstance(frame, FlushSentinel):
            # print(f"[MySTT] 🛑 Intercepted FlushSentinel in push_frame, triggering flush()", flush=True)
            self.flush()
            return
        
        super().push_frame(frame)


    @utils.log_exceptions(logger=logging.getLogger("my_stt"))
    async def _run(self) -> None:
        closing_ws = False

        # 以 opts.chunk_ms 为单位拼音频块
        chunk_ms = max(10, int(self._stt._opts.chunk_ms))
        samples_per_chunk = int(SAMPLE_RATE * chunk_ms / 1000.0)

        @utils.log_exceptions(logger=logging.getLogger("my_stt"))
        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws

            # DEBUG: 计数器
            audio_frame_count = 0
            bytes_sent = 0

            # 将上游 AudioFrame 变成“固定 chunk 大小”的 PCM bytes
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                samples_per_channel=samples_per_chunk,
            )

            async def ensure_started() -> None:
                if self._speaking:
                    return
                self._utt_id += 1
                self._utt_audio_bytes = 0
                cfg = self._build_start_config(wav_name=f"livekit_{self._utt_id}")
                # print(f"[MySTT] 🎙️ 开始新话语 utt_id={self._utt_id}", flush=True)
                await ws.send_str(json.dumps(cfg, ensure_ascii=False))
                self._speaking = True

            async for data in self._input_ch:
                # DEBUG: 打印收到的音频帧信息
                # if isinstance(data, rtc.AudioFrame) and audio_frame_count % 100 == 0:
                    # print(f"[MySTT] 📥 收到音频帧 #{audio_frame_count}, 已发送 {bytes_sent} bytes", flush=True)
                frames: list[rtc.AudioFrame] = []

                if isinstance(data, rtc.AudioFrame):
                    audio_frame_count += 1
                    # 新一句话开始（上一句如果 end_input 过，会把 _speaking=False）
                    await ensure_started()

                    frames.extend(audio_bstream.write(data.data.tobytes()))

                elif isinstance(data, self._FlushSentinel):
                    # flush 可能是：end_input() 触发 或 stream 关闭时的收尾
                    frames.extend(audio_bstream.flush())

                    # 先把残余 chunk 发完，并计入当前句音频长度
                    for frame in frames:
                        frame_bytes = frame.data.tobytes()
                        self._utt_audio_bytes += len(frame_bytes)
                        await ws.send_bytes(frame_bytes)

                    frames.clear()

                    # 发送 is_speaking=false 作为句末 finalize（音频流已给到 STT server，用于 E2E 起点）
                    if self._speaking:
                        self._inference_start_at = time.time()
                        if self._on_segment_submitted:
                            self._on_segment_submitted()
                        print(f"[TIMING] {_ts()} ② STT 收到 FlushSentinel，发送 is_speaking=False 给 FunASR（2nd pass 开始）", flush=True)
                        await ws.send_str(json.dumps({"is_speaking": False}))
                        self._speaking = False

                    continue

                # 正常发送音频 chunk
                for frame in frames:
                    frame_bytes = frame.data.tobytes()
                    bytes_sent += len(frame_bytes)
                    self._utt_audio_bytes += len(frame_bytes)
                    await ws.send_bytes(frame_bytes)

            # 输入通道结束：关闭连接
            closing_ws = True

            # 如果还处于 speaking，补一个 false
            if self._speaking:
                try:
                    await ws.send_str(json.dumps({"is_speaking": False}))
                except Exception:
                    pass
                self._speaking = False

        @utils.log_exceptions(logger=None)
        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws

            # 2pass-online 会不断来 delta；2pass-offline 会来最终修正
            # 为了给 LiveKit 更好的 interim，我们维护一个 current_text
            current_text = ""

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
                    raise APIStatusError(message="FunASR STT connection closed unexpectedly")

                if msg.type != aiohttp.WSMsgType.TEXT:
                    # FunASR 返回通常是 TEXT(JSON)，binary 直接忽略
                    continue

                try:
                    data = json.loads(msg.data)
                except Exception:
                    continue

                # 你的 FunASR 回包里常见字段：text / mode / wav_name / is_final / timestamp ...
                mode = data.get("mode", "")
                text = data.get("text", "")
                if not text:
                    continue

                # 按你的 client 逻辑：
                # - online / 2pass-online => interim
                # - offline / 2pass-offline => final
                is_interim = mode in ("online", "2pass-online")
                is_final = mode in ("offline", "2pass-offline") or bool(data.get("is_final", False))

                if is_interim and not is_final:
                    # 累积显示更像你 demo 那样
                    current_text += text
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                            alternatives=[stt.SpeechData(text=current_text, language="")],
                        )
                    )
                else:
                    # final：用服务端给的 transcript（通常更干净）
                    current_text = ""
                    print(f"[TIMING] {_ts()} ★ FunASR 2nd pass 完成，FINAL_TRANSCRIPT='{text[:40]}'", flush=True)
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                            alternatives=[stt.SpeechData(text=text, language="")],
                        )
                    )
                    # 上报 STT 推理耗时，供 pipeline 打日志
                    if self._inference_start_at is not None and self._stt._label:
                        duration = time.time() - self._inference_start_at
                        audio_duration = self._utt_audio_bytes / (SAMPLE_RATE * 2) if self._utt_audio_bytes else 0.0
                        self._stt.emit(
                            "metrics_collected",
                            STTMetrics(
                                label=self._stt._label,
                                request_id=f"livekit_{self._utt_id}",
                                timestamp=time.time(),
                                duration=duration,
                                audio_duration=audio_duration,
                                streamed=True,
                                metadata=Metadata(model_name="FunASR", model_provider=self._stt.provider),
                            ),
                        )
                        self._inference_start_at = None

        # 主循环：从连接池拿 ws，起 send/recv 两个协程
        while True:
            closing_ws = False
            async with self._pool.connection(timeout=self._conn_options.timeout) as ws:
                tasks = [
                    asyncio.create_task(send_task(ws)),
                    asyncio.create_task(recv_task(ws)),
                ]
                tasks_group = asyncio.gather(*tasks)
                wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())

                try:
                    done, _ = await asyncio.wait(
                        (tasks_group, wait_reconnect_task),
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    for task in done:
                        if task != wait_reconnect_task:
                            task.result()

                    if wait_reconnect_task not in done:
                        break

                    self._reconnect_event.clear()

                finally:
                    await utils.aio.gracefully_cancel(*tasks, wait_reconnect_task)
                    tasks_group.cancel()
                    try:
                        tasks_group.exception()
                    except Exception:
                        pass
