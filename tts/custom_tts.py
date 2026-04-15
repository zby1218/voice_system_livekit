from __future__ import annotations

import asyncio
import os
import re
import socket
from datetime import datetime
from dataclasses import dataclass, replace
from typing import Callable, Optional, Literal, Tuple, Dict, Any, List

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

# ✅ 新增 zero_shot
EndpointType = Literal["sft", "instruct", "cross_lingual", "zero_shot"]


@dataclass
class _CosyVoiceTTSOptions:
    base_url: str
    endpoint: EndpointType

    # sft / instruct 用
    voice: str
    instructions: Optional[str]

    # cross_lingual / zero_shot 用
    prompt_wav_path: Optional[str]

    # ✅ zero_shot 用
    prompt_text: Optional[str]

    sample_rate: int
    num_channels: int

    # ===== 通用对话切段参数 =====
    max_chars: int = 140
    min_chars: int = 25

    hard_delims: str = "。！？!?；;\n"
    # 对话更自然：逗号只在超长时辅助切
    soft_delims: str = "，,、:："

    add_silence_ms: int = 80

    # ===== HTTP 行为 =====
    connect_timeout_s: float = 15.0
    total_timeout_s: float = 600.0
    session_id: str = ""
    interrupt_mode: str = "latest_only"

    # 首包等待（zero_shot / cross_lingual 首包可能较慢）
    first_audio_deadline_s: float = 60.0

    # 单段最长允许时间（防卡死）
    segment_deadline_s: float = 180.0


class CosyVoiceTTS(tts.TTS):
    def __init__(
        self,
        *,
        base_url: str,
        endpoint: EndpointType = "cross_lingual",
        voice: str = "0",
        instructions: Optional[str] = None,

        # ✅ zero_shot / cross_lingual 都需要 prompt_wav
        prompt_wav_path: Optional[str] = None,

        # ✅ zero_shot 需要 prompt_text
        prompt_text: Optional[str] = None,

        sample_rate: int = DEFAULT_SAMPLE_RATE,
        num_channels: int = DEFAULT_NUM_CHANNELS,

        max_chars: int = 140,
        min_chars: int = 25,
        add_silence_ms: int = 80,

        connect_timeout_s: float = 15.0,
        total_timeout_s: float = 600.0,
        first_audio_deadline_s: float = 60.0,
        segment_deadline_s: float = 180.0,
        session_id: Optional[str] = None,
        interrupt_mode: str = "latest_only",

        client: Optional[httpx.AsyncClient] = None,
        on_first_frame_pushed: Optional[Callable[[], None]] = None,
        on_tts_request_sent: Optional[Callable[[], None]] = None,
        on_first_segment_ready: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )

        self._opts = _CosyVoiceTTSOptions(
            base_url=base_url.rstrip("/"),
            endpoint=endpoint,
            voice=voice,
            instructions=instructions,
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text,
            sample_rate=int(sample_rate),
            num_channels=int(num_channels),
            max_chars=int(max_chars),
            min_chars=int(min_chars),
            add_silence_ms=int(add_silence_ms),
            connect_timeout_s=float(connect_timeout_s),
            total_timeout_s=float(total_timeout_s),
            first_audio_deadline_s=float(first_audio_deadline_s),
            segment_deadline_s=float(segment_deadline_s),
            session_id=session_id or f"{socket.gethostname()}:{os.getpid()}",
            interrupt_mode=interrupt_mode or "latest_only",
        )

        # ✅ 全局串行锁：避免多 stream 交错导致“乱序/重复读体感”
        self._synth_lock = asyncio.Lock()

        # ✅ prompt wav bytes 缓存（zero_shot/cross_lingual 每段都要带）
        self._prompt_cache: Optional[bytes] = None
        self._on_first_frame_pushed = on_first_frame_pushed  # TTS 首帧 push 时回调，用于 E2E 终点
        self._on_tts_request_sent = on_tts_request_sent  # 向 TTS server 发送请求前回调，用于计时「请求→首帧」
        self._on_first_segment_ready = on_first_segment_ready  # 首句段就绪并发送到TTS前回调

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
        return f"cosyvoice-{self._opts.endpoint}"

    @property
    def provider(self) -> str:
        return self._opts.base_url

    def update_options(
        self,
        *,
        endpoint: Optional[EndpointType] = None,
        voice: Optional[str] = None,
        instructions: Optional[str] = None,
        prompt_wav_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
        max_chars: Optional[int] = None,
        min_chars: Optional[int] = None,
        add_silence_ms: Optional[int] = None,
        total_timeout_s: Optional[float] = None,
        first_audio_deadline_s: Optional[float] = None,
        segment_deadline_s: Optional[float] = None,
        session_id: Optional[str] = None,
        interrupt_mode: Optional[str] = None,
    ) -> None:
        if endpoint is not None:
            self._opts.endpoint = endpoint
        if voice is not None:
            self._opts.voice = voice
        if instructions is not None:
            self._opts.instructions = instructions
        if prompt_text is not None:
            self._opts.prompt_text = prompt_text
        if prompt_wav_path is not None:
            self._opts.prompt_wav_path = prompt_wav_path
            self._prompt_cache = None  # prompt 音频变了就清缓存
        if max_chars is not None:
            self._opts.max_chars = int(max_chars)
        if min_chars is not None:
            self._opts.min_chars = int(min_chars)
        if add_silence_ms is not None:
            self._opts.add_silence_ms = int(add_silence_ms)
        if total_timeout_s is not None:
            self._opts.total_timeout_s = float(total_timeout_s)
        if first_audio_deadline_s is not None:
            self._opts.first_audio_deadline_s = float(first_audio_deadline_s)
        if segment_deadline_s is not None:
            self._opts.segment_deadline_s = float(segment_deadline_s)
        if session_id is not None:
            self._opts.session_id = session_id
        if interrupt_mode is not None:
            self._opts.interrupt_mode = interrupt_mode

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "CosyVoiceChunkedStream":
        return CosyVoiceChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    async def aclose(self) -> None:
        await self._client.aclose()


class CosyVoiceChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: CosyVoiceTTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: CosyVoiceTTS = tts
        self._opts = replace(tts._opts)

    # ---------- 无 lookbehind 的切段 ----------
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

    def _hms_ms(self) -> str:
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def _silence_bytes(self, ms: int) -> bytes:
        if ms <= 0:
            return b""
        samples = int(self._opts.sample_rate * ms / 1000.0)
        return b"\x00\x00" * samples * self._opts.num_channels

    # ---------- 构建请求（新增 zero_shot） ----------
    def _read_prompt_bytes(self) -> bytes:
        if not self._opts.prompt_wav_path:
            raise ValueError(f"{self._opts.endpoint} requires prompt_wav_path")
        if self._tts._prompt_cache is None:
            with open(self._opts.prompt_wav_path, "rb") as f:
                self._tts._prompt_cache = f.read()
        return self._tts._prompt_cache

    def _build_request_for_segment(self, seg_text: str) -> Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]:
        files = None
        common = {
            "session_id": self._opts.session_id,
            "interrupt_mode": self._opts.interrupt_mode,
        }

        if self._opts.endpoint == "sft":
            url = f"{self._opts.base_url}/inference_sft"
            data = {"tts_text": seg_text, "spk_id": self._opts.voice, **common}
            return url, data, files

        if self._opts.endpoint == "instruct":
            url = f"{self._opts.base_url}/inference_instruct"
            data = {
                "tts_text": seg_text,
                "spk_id": self._opts.voice,
                "instruct_text": self._opts.instructions or "",
                **common,
            }
            return url, data, files

        if self._opts.endpoint == "cross_lingual":
            url = f"{self._opts.base_url}/inference_cross_lingual"
            data = {"tts_text": seg_text, **common}
            wav_bytes = self._read_prompt_bytes()
            files = {"prompt_wav": ("prompt.wav", wav_bytes, "audio/wav")}
            return url, data, files

        # ✅ 新增：zero_shot
        if self._opts.endpoint == "zero_shot":
            url = f"{self._opts.base_url}/inference_zero_shot"
            if not self._opts.prompt_text:
                raise ValueError("zero_shot requires prompt_text")
            data = {
                "tts_text": seg_text,
                "prompt_text": self._opts.prompt_text,
                **common,
            }
            wav_bytes = self._read_prompt_bytes()
            files = {"prompt_wav": ("prompt.wav", wav_bytes, "audio/wav")}
            return url, data, files

        raise ValueError(f"Unknown endpoint: {self._opts.endpoint}")

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        import logging
        _logger = logging.getLogger("custom_tts")
        _logger.info(f"🎵 [CosyVoiceTTS] 开始合成: {self.input_text}")
        
        async with self._tts._synth_lock:
            output_emitter.initialize(
                request_id="cosyvoice",
                sample_rate=self._opts.sample_rate,
                num_channels=self._opts.num_channels,
                mime_type="audio/pcm",
            )

            segments = self._split_dialogue(self.input_text)
            # _logger.info(f"🎵 [CosyVoiceTTS] 切分为 {len(segments)} 段")

            loop = asyncio.get_event_loop()
            start_ts = loop.time()
            e2e_end_recorded = False  # 整轮只记一次首帧（第一段的首帧）
            _push_count = 0
            _push_start = loop.time()

            total_deadline = float(getattr(self._conn_options, "timeout", self._opts.total_timeout_s))

            try:
                for idx, seg in enumerate(segments):
                    if total_deadline and (loop.time() - start_ts) > total_deadline:
                        raise APITimeoutError() from None

                    print(
                        f"[TIMING] {self._hms_ms()} 📝 句段就绪并发送到TTS "
                        f"seg={idx} text='{seg}'"
                    )
                    if idx == 0 and self._tts._on_first_segment_ready:
                        self._tts._on_first_segment_ready()
                    url, data, files = self._build_request_for_segment(seg)
                    candidates = (url, url + "/")  # 兼容尾部 /

                    if idx == 0 and self._tts._on_tts_request_sent:
                        self._tts._on_tts_request_sent()

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
                                        f"CosyVoice HTTP {resp.status_code}",
                                        status_code=resp.status_code,
                                        request_id="cosyvoice",
                                        body=body.decode("utf-8", errors="ignore"),
                                    )

                                # ========= 关键：首包 deadline =========
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
                                    _push_count += 1
                                    print(
                                        f"[TIMING] {self._hms_ms()} ⬆️  [SERVER] 首帧推入输出emitter "
                                        f"seg={idx} size={len(first)}B elapsed={loop.time()-_push_start:.3f}s",
                                        flush=True,
                                    )
                                    if self._tts._on_first_frame_pushed and not e2e_end_recorded:
                                        self._tts._on_first_frame_pushed()
                                        e2e_end_recorded = True

                                # ========= 单段整体 deadline（防卡死）=========
                                seg_end_at = loop.time() + float(self._opts.segment_deadline_s)

                                while True:
                                    remain = seg_end_at - loop.time()
                                    if remain <= 0:
                                        raise APITimeoutError() from None
                                    try:
                                        chunk = await asyncio.wait_for(it.__anext__(), timeout=remain)
                                    except StopAsyncIteration:
                                        break
                                    if chunk:
                                        output_emitter.push(chunk)
                                        _push_count += 1

                            ok = True
                            break

                        except Exception as e:
                            last_err = e
                            continue

                    if not ok:
                        if isinstance(last_err, APIStatusError):
                            raise last_err
                        if isinstance(last_err, (httpx.TimeoutException, asyncio.TimeoutError, APITimeoutError)):
                            raise APITimeoutError() from None
                        raise APIConnectionError() from (last_err or Exception("Unknown error"))

                    if self._opts.add_silence_ms > 0 and idx != len(segments) - 1:
                        output_emitter.push(self._silence_bytes(self._opts.add_silence_ms))

                output_emitter.flush()
                _logger.info("🎵 [CosyVoiceTTS] 合成完成，已 flush")

            except httpx.TimeoutException:
                raise APITimeoutError() from None
            except APIStatusError:
                raise
            except APITimeoutError:
                raise
            except Exception as e:
                raise APIConnectionError() from e
