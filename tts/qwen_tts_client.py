from __future__ import annotations

import asyncio
import json
import re
import base64
from dataclasses import dataclass, replace
from typing import Optional, Literal, Tuple, Dict, Any, List

import httpx
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

DEFAULT_SAMPLE_RATE = 24000
DEFAULT_NUM_CHANNELS = 1

EndpointType = Literal["custom_voice", "voice_design", "voice_clone"]


@dataclass
class _QwenTTSOptions:
    base_url: str
    endpoint: EndpointType

    # custom_voice 用
    speaker: str
    language: str
    instruct: Optional[str]

    # voice_clone 用
    ref_audio_path: Optional[str]
    ref_text: Optional[str]

    sample_rate: int
    num_channels: int

    # 文本切段参数
    max_chars: int = 140
    min_chars: int = 25
    hard_delims: str = "。！？!?；;\n"
    soft_delims: str = "，,、:："
    add_silence_ms: int = 80

    # HTTP 行为
    connect_timeout_s: float = 15.0
    total_timeout_s: float = 600.0
    first_audio_deadline_s: float = 60.0
    segment_deadline_s: float = 180.0


class QwenTTS(tts.TTS):
    """
    Qwen3-TTS 客户端，对接 qwen_tts_server.py。支持 HTTP 或 WebSocket（use_websocket=True）。
    用法与 CosyVoiceTTS 一致，直接替换 stt_llm_agent.py 中的 my_tts 即可。

    需先启动服务端：
        python tts/qwen_tts_server.py --mode custom_voice --port 50001
    本模块命名为 qwen_tts_client，避免与 pip 包 qwen_tts（Qwen3TTSModel）重名。

    示例:
        # HTTP（默认）
        my_tts = QwenTTS(base_url="http://localhost:50001", endpoint="custom_voice", speaker="Serena", language="Chinese")
        # WebSocket（长连接，多段复用）
        my_tts = QwenTTS(base_url="http://localhost:50001", endpoint="custom_voice", speaker="Serena", use_websocket=True)
    """

    def __init__(
        self,
        *,
        base_url: str,
        endpoint: EndpointType = "custom_voice",
        # custom_voice 参数
        speaker: str = "Serena",
        language: str = "Chinese",
        instruct: Optional[str] = None,
        # voice_clone 参数
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        # 音频参数
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        num_channels: int = DEFAULT_NUM_CHANNELS,
        # 文本切段参数
        max_chars: int = 140,
        min_chars: int = 25,
        add_silence_ms: int = 80,
        # 超时参数
        connect_timeout_s: float = 15.0,
        total_timeout_s: float = 600.0,
        first_audio_deadline_s: float = 60.0,
        segment_deadline_s: float = 180.0,
        # 传输方式：False=HTTP（每段一次 POST），True=WebSocket（单连接多段）
        use_websocket: bool = False,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )

        self._opts = _QwenTTSOptions(
            base_url=base_url.rstrip("/"),
            endpoint=endpoint,
            speaker=speaker,
            language=language,
            instruct=instruct,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            sample_rate=int(sample_rate),
            num_channels=int(num_channels),
            max_chars=int(max_chars),
            min_chars=int(min_chars),
            add_silence_ms=int(add_silence_ms),
            connect_timeout_s=float(connect_timeout_s),
            total_timeout_s=float(total_timeout_s),
            first_audio_deadline_s=float(first_audio_deadline_s),
            segment_deadline_s=float(segment_deadline_s),
        )

        self._use_websocket = bool(use_websocket)
        self._synth_lock = asyncio.Lock()
        self._ref_audio_cache: Optional[bytes] = None

        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=self._opts.connect_timeout_s,
                read=None,
                write=None,
                pool=None,
            ),
            follow_redirects=True,
            trust_env=False,
            limits=httpx.Limits(
                max_connections=50,
                max_keepalive_connections=50,
                keepalive_expiry=120,
            ),
        )

    @property
    def model(self) -> str:
        return f"qwen3-tts-{self._opts.endpoint}"

    @property
    def provider(self) -> str:
        return self._opts.base_url

    def update_options(
        self,
        *,
        endpoint: Optional[EndpointType] = None,
        speaker: Optional[str] = None,
        language: Optional[str] = None,
        instruct: Optional[str] = None,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        max_chars: Optional[int] = None,
        min_chars: Optional[int] = None,
        add_silence_ms: Optional[int] = None,
        first_audio_deadline_s: Optional[float] = None,
        segment_deadline_s: Optional[float] = None,
        total_timeout_s: Optional[float] = None,
    ) -> None:
        if endpoint is not None:
            self._opts.endpoint = endpoint
        if speaker is not None:
            self._opts.speaker = speaker
        if language is not None:
            self._opts.language = language
        if instruct is not None:
            self._opts.instruct = instruct
        if ref_text is not None:
            self._opts.ref_text = ref_text
        if ref_audio_path is not None:
            self._opts.ref_audio_path = ref_audio_path
            self._ref_audio_cache = None
        if max_chars is not None:
            self._opts.max_chars = int(max_chars)
        if min_chars is not None:
            self._opts.min_chars = int(min_chars)
        if add_silence_ms is not None:
            self._opts.add_silence_ms = int(add_silence_ms)
        if first_audio_deadline_s is not None:
            self._opts.first_audio_deadline_s = float(first_audio_deadline_s)
        if segment_deadline_s is not None:
            self._opts.segment_deadline_s = float(segment_deadline_s)
        if total_timeout_s is not None:
            self._opts.total_timeout_s = float(total_timeout_s)

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "QwenChunkedStream":
        return QwenChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    async def aclose(self) -> None:
        await self._client.aclose()


class QwenChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: QwenTTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: QwenTTS = tts
        self._opts = replace(tts._opts)

    # ------------------------------------------------------------------
    # 文本切段（与 CosyVoiceTTS 保持一致）
    # ------------------------------------------------------------------

    def _split_by_delims(self, s: str, delims: str) -> List[str]:
        if not s:
            return []
        out: List[str] = []
        start = 0
        for i, ch in enumerate(s):
            if ch in delims:
                seg = s[start : i + 1].strip()
                if seg:
                    out.append(seg)
                start = i + 1
        tail = s[start:].strip()
        if tail:
            out.append(tail)
        return out

    def _pack_to_target_len(self, pieces: List[str]) -> List[str]:
        out: List[str] = []
        buf = ""
        for p in pieces:
            if not buf:
                buf = p
                continue
            if len(buf) + len(p) <= self._opts.max_chars:
                buf += p
            else:
                out.append(buf)
                buf = p
        if buf:
            out.append(buf)
        return out

    def _merge_short(self, pieces: List[str]) -> List[str]:
        merged: List[str] = []
        for seg in pieces:
            if merged and len(seg) < self._opts.min_chars:
                merged[-1] += seg
            else:
                merged.append(seg)
        return merged

    def _ensure_ending_punct(self, seg: str) -> str:
        if not seg:
            return seg
        if seg[-1] in "。！？!?；;":
            return seg
        return seg + "。"

    def _split_dialogue(self, text: str) -> List[str]:
        s = (text or "").strip()
        if not s:
            return []
        s = re.sub(r"\s+", " ", s)

        hard = self._split_by_delims(s, self._opts.hard_delims)

        pieces: List[str] = []
        for seg in hard:
            if len(seg) <= self._opts.max_chars:
                pieces.append(seg)
            else:
                soft = self._split_by_delims(seg, self._opts.soft_delims)
                pieces.extend(soft if soft else [seg])

        packed = self._pack_to_target_len(pieces)
        merged = self._merge_short(packed)
        return [self._ensure_ending_punct(x.strip()) for x in merged if x.strip()]

    def _silence_bytes(self, ms: int) -> bytes:
        if ms <= 0:
            return b""
        samples = int(self._opts.sample_rate * ms / 1000.0)
        return b"\x00\x00" * samples * self._opts.num_channels

    # ------------------------------------------------------------------
    # 构建 HTTP 请求
    # ------------------------------------------------------------------

    def _read_ref_audio_bytes(self) -> bytes:
        if not self._opts.ref_audio_path:
            raise ValueError("voice_clone 模式需要 ref_audio_path")
        if self._tts._ref_audio_cache is None:
            with open(self._opts.ref_audio_path, "rb") as f:
                self._tts._ref_audio_cache = f.read()
        return self._tts._ref_audio_cache

    def _build_request(
        self, seg_text: str
    ) -> Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]:
        files = None

        if self._opts.endpoint == "custom_voice":
            url = f"{self._opts.base_url}/inference_custom_voice"
            data = {
                "tts_text": seg_text,
                "speaker": self._opts.speaker,
                "language": self._opts.language,
                "instruct": self._opts.instruct or "",
            }
            return url, data, files

        if self._opts.endpoint == "voice_design":
            url = f"{self._opts.base_url}/inference_voice_design"
            if not self._opts.instruct:
                raise ValueError("voice_design 模式需要 instruct")
            data = {
                "tts_text": seg_text,
                "language": self._opts.language,
                "instruct": self._opts.instruct,
            }
            return url, data, files

        if self._opts.endpoint == "voice_clone":
            url = f"{self._opts.base_url}/inference_voice_clone"
            data = {
                "tts_text": seg_text,
                "language": self._opts.language,
                "ref_text": self._opts.ref_text or "",
            }
            # ref_audio_path 有值则上传；服务端有预置时可不传
            if self._opts.ref_audio_path:
                wav_bytes = self._read_ref_audio_bytes()
                files = {"ref_audio": ("ref.wav", wav_bytes, "audio/wav")}
            return url, data, files

        raise ValueError(f"未知 endpoint: {self._opts.endpoint}")

    def _ws_url(self) -> str:
        base = self._opts.base_url
        if base.startswith("https://"):
            return base.replace("https://", "wss://", 1).rstrip("/") + "/ws/tts"
        return base.replace("http://", "ws://", 1).rstrip("/") + "/ws/tts"

    def _build_ws_message(self, seg_text: str) -> Dict[str, Any]:
        msg: Dict[str, Any] = {
            "action": "synthesize",
            "text": seg_text,
            "speaker": self._opts.speaker,
            "language": self._opts.language,
            "instruct": self._opts.instruct or "",
            "ref_text": self._opts.ref_text or "",
        }
        if self._opts.endpoint == "voice_clone" and self._opts.ref_audio_path:
            b = self._read_ref_audio_bytes()
            msg["ref_audio_b64"] = base64.b64encode(b).decode("ascii")
        return msg

    # ------------------------------------------------------------------
    # livekit-agents _run
    # ------------------------------------------------------------------

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        import logging
        _logger = logging.getLogger("qwen_tts")
        _logger.info(f"[QwenTTS] 开始合成: {self.input_text[:60]}")

        async with self._tts._synth_lock:
            output_emitter.initialize(
                request_id="qwen_tts",
                sample_rate=self._opts.sample_rate,
                num_channels=self._opts.num_channels,
                mime_type="audio/pcm",
            )

            segments = self._split_dialogue(self.input_text)
            loop = asyncio.get_event_loop()
            start_ts = loop.time()
            total_deadline = float(
                getattr(self._conn_options, "timeout", self._opts.total_timeout_s)
            )

            try:
                if self._tts._use_websocket:
                    await self._run_websocket(output_emitter, segments, start_ts, total_deadline, _logger)
                else:
                    await self._run_http(output_emitter, segments, start_ts, total_deadline, _logger)
            except httpx.TimeoutException:
                raise APITimeoutError() from None
            except APIStatusError:
                raise
            except APITimeoutError:
                raise
            except Exception as e:
                raise APIConnectionError() from e

    async def _run_websocket(
        self,
        output_emitter: tts.AudioEmitter,
        segments: List[str],
        start_ts: float,
        total_deadline: float,
        _logger,
    ) -> None:
        import websockets
        loop = asyncio.get_event_loop()
        ws_url = self._ws_url()
        seg_timeout = self._opts.segment_deadline_s

        async with websockets.connect(
            ws_url,
            open_timeout=float(self._opts.connect_timeout_s),
            close_timeout=5.0,
        ) as ws:
            for idx, seg in enumerate(segments):
                if total_deadline and (loop.time() - start_ts) > total_deadline:
                    raise APITimeoutError() from None

                await ws.send(json.dumps(self._build_ws_message(seg)))
                got_first = False
                seg_end_at = loop.time() + seg_timeout

                while True:
                    remain = seg_end_at - loop.time()
                    if remain <= 0:
                        raise APITimeoutError() from None
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=remain)
                    except asyncio.TimeoutError:
                        raise APITimeoutError() from None

                    if isinstance(raw, bytes):
                        if not got_first:
                            got_first = True
                        output_emitter.push(raw)
                        continue

                    obj = json.loads(raw)
                    if obj.get("done"):
                        break
                    if obj.get("error"):
                        raise APIConnectionError(obj["error"])

                if self._opts.add_silence_ms > 0 and idx != len(segments) - 1:
                    output_emitter.push(self._silence_bytes(self._opts.add_silence_ms))

        output_emitter.flush()
        _logger.info("[QwenTTS] 合成完成，已 flush (WebSocket)")

    async def _run_http(
        self,
        output_emitter: tts.AudioEmitter,
        segments: List[str],
        start_ts: float,
        total_deadline: float,
        _logger,
    ) -> None:
        loop = asyncio.get_event_loop()
        for idx, seg in enumerate(segments):
            if total_deadline and (loop.time() - start_ts) > total_deadline:
                raise APITimeoutError() from None

            url, data, files = self._build_request(seg)
            candidates = (url, url + "/")

            last_err: Optional[Exception] = None
            ok = False

            for candidate in candidates:
                try:
                    async with self._tts._client.stream(
                        "POST",
                        candidate,
                        data=data,
                        files=files,
                    ) as resp:
                        if resp.status_code == 404:
                            continue
                        if resp.status_code >= 400:
                            body = await resp.aread()
                            raise APIStatusError(
                                f"QwenTTS HTTP {resp.status_code}",
                                status_code=resp.status_code,
                                request_id="qwen_tts",
                                body=body.decode("utf-8", errors="ignore"),
                            )

                        it = resp.aiter_bytes()
                        try:
                            first = await asyncio.wait_for(
                                it.__anext__(),
                                timeout=float(self._opts.first_audio_deadline_s),
                            )
                        except StopAsyncIteration:
                            raise APIConnectionError() from Exception("empty audio stream")
                        except asyncio.TimeoutError:
                            raise APITimeoutError() from None

                        if first:
                            output_emitter.push(first)

                        seg_end_at = loop.time() + float(self._opts.segment_deadline_s)

                        while True:
                            remain = seg_end_at - loop.time()
                            if remain <= 0:
                                raise APITimeoutError() from None
                            try:
                                chunk = await asyncio.wait_for(
                                    it.__anext__(), timeout=remain
                                )
                            except StopAsyncIteration:
                                break
                            if chunk:
                                output_emitter.push(chunk)

                    ok = True
                    break

                except Exception as e:
                    last_err = e
                    continue

            if not ok:
                if isinstance(last_err, APIStatusError):
                    raise last_err
                if isinstance(
                    last_err,
                    (httpx.TimeoutException, asyncio.TimeoutError, APITimeoutError),
                ):
                    raise APITimeoutError() from None
                raise APIConnectionError() from (
                    last_err or Exception("Unknown error")
                )

            if self._opts.add_silence_ms > 0 and idx != len(segments) - 1:
                output_emitter.push(self._silence_bytes(self._opts.add_silence_ms))

        output_emitter.flush()
        _logger.info("[QwenTTS] 合成完成，已 flush")
