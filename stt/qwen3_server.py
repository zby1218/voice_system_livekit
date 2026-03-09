# stt/qwen3_server.py
# coding=utf-8
"""
Qwen3-ASR WebSocket 服务端（单用户版）。
每条连接独立维护 streaming state，推理通过 asyncio.to_thread 在后台线程执行。

启动: python stt/qwen3_server.py [--host 0.0.0.0] [--port 10096] [--model /path/to/Qwen3-ASR-1.7B]
"""

import os
import sys

os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")

import json
import asyncio
import logging
import argparse
import time

import websockets
import numpy as np

try:
    from qwen_asr import Qwen3ASRModel
except ImportError:
    print("❌ 错误: 未安装 qwen-asr。请运行: pip install qwen-asr[vllm]")
    sys.exit(1)


DEFAULT_MODEL_PATH = "/home/zhangchi/project/Qwen3-ASR/Qwen3-ASR-1.7B"

SAMPLE_RATE = 16000
CHUNK_SIZE_SEC = 1.0
UNFIXED_CHUNK_NUM = 1
UNFIXED_TOKEN_NUM = 3

# 一次有效 session 至少需要的音频时长（秒）
# 低于此值视为背景噪音误触发，直接丢弃，不做识别
MIN_SESSION_DURATION_SEC = 1.0
MIN_SESSION_SAMPLES = int(SAMPLE_RATE * MIN_SESSION_DURATION_SEC)


def _setup_logging(log_subdir: str, log_filename: str) -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(project_root, "log", log_subdir)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


logger = logging.getLogger("Qwen3ASR-Server")


def load_model(model_path: str, gpu_memory_utilization: float, max_model_len: int) -> Qwen3ASRModel:
    if not os.path.exists(model_path):
        logger.error("❌ 模型路径不存在: %s", model_path)
        sys.exit(1)
    logger.info("📦 正在加载 ASR 模型 (vLLM) 从: %s ...", model_path)
    model = Qwen3ASRModel.LLM(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_new_tokens=32,
        max_model_len=max_model_len,
    )
    logger.info("✅ ASR 模型加载完成")
    return model

# 
def _make_state(asr, context: str = "", language: str | None = None):
    return asr.init_streaming_state(
        context=context or "",
        language=language or None,
        unfixed_chunk_num=UNFIXED_CHUNK_NUM,
        unfixed_token_num=UNFIXED_TOKEN_NUM,
        chunk_size_sec=CHUNK_SIZE_SEC,
    )


async def ws_handler(websocket, asr):
    remote = getattr(websocket, "remote_address", "unknown")
    logger.info("新连接: %s", remote)

    context = ""
    language = None
    state = _make_state(asr, context, language)
    last_sent_text = ""
    session_active = False
    session_samples = 0  # 本轮 session 收到的音频样本数，用于过滤噪音误触发

    try:
        async for message in websocket:
            # ── JSON 控制消息 ──────────────────────────────────────────
            if isinstance(message, str):
                try:
                    msg = json.loads(message)
                except json.JSONDecodeError:
                    continue

                if "context" in msg or "language" in msg:
                    context = msg.get("context", context) or ""
                    language = msg.get("language") or None
                    if not session_active:
                        state = _make_state(asr, context, language)
                    logger.info("更新 context/language [%s]", remote)

                if msg.get("is_speaking") is True:
                    state = _make_state(asr, context, language)
                    last_sent_text = ""
                    session_active = True
                    session_samples = 0
                    logger.debug("会话开始 [%s]", remote)

                elif msg.get("is_speaking") is False:
                    session_active = False
                    duration_sec = round(session_samples / SAMPLE_RATE, 2)

                    # 音频时长不足，视为噪音误触发，直接丢弃
                    if session_samples < MIN_SESSION_SAMPLES:
                        logger.info(
                            "会话丢弃 [%s] 音频时长 %.2fs < %.2fs（噪音误触发）",
                            remote, duration_sec, MIN_SESSION_DURATION_SEC,
                        )
                        state = _make_state(asr, context, language)
                        last_sent_text = ""
                        continue

                    # 流式推理从未产出任何文字：说明音频不足 chunk_size_sec（2s），
                    # 模型没有积累到足够内容就结束了，finish 的结果可信度极低，直接丢弃。
                    # 典型场景：短暂背景声/噪音触发 VAD，但内容不足以让模型正常识别。
                    if not last_sent_text:
                        logger.info(
                            "会话丢弃 [%s] 流式推理无输出（音频时长 %.2fs < %.1fs chunk窗口），跳过 finish",
                            remote, duration_sec, CHUNK_SIZE_SEC,
                        )
                        state = _make_state(asr, context, language)
                        last_sent_text = ""
                        continue

                    t0 = time.perf_counter()
                    await asyncio.to_thread(asr.finish_streaming_transcribe, state)
                    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

                    final_text = state.text or ""

                    # 识别结果为空，不发送
                    if not final_text:
                        logger.info("会话结束 [%s] 结果为空，已跳过发送（音频时长 %.2fs）", remote, duration_sec)
                        state = _make_state(asr, context, language)
                        last_sent_text = ""
                        continue

                    await websocket.send(json.dumps({
                        "text": final_text,
                        "language": state.language or "",
                        "is_final": True,
                        "inference_ms": elapsed_ms,
                    }))
                    logger.info("会话结束 [%s] %.2fs → %s", remote, duration_sec, final_text[:50])

                    state = _make_state(asr, context, language)
                    last_sent_text = ""

                continue

            # ── 二进制音频（Float32 bytes）─────────────────────────────
            if not session_active:
                continue

            chunk = np.frombuffer(message, dtype=np.float32)
            if chunk.size == 0:
                continue

            session_samples += chunk.size

            t0 = time.perf_counter()
            await asyncio.to_thread(asr.streaming_transcribe, chunk, state)
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

            if session_active and state.text != last_sent_text:
                last_sent_text = state.text
                await websocket.send(json.dumps({
                    "text": state.text or "",
                    "language": state.language or "",
                    "is_final": False,
                    "inference_ms": elapsed_ms,
                }))

    except websockets.ConnectionClosed:
        logger.info("连接关闭: %s", remote)
    except Exception:
        logger.exception("处理异常 [%s]", remote)


async def main_async(host: str, port: int, model_path: str, gpu_memory_utilization: float, max_model_len: int):
    asr = load_model(model_path, gpu_memory_utilization, max_model_len)

    async with websockets.serve(
        lambda ws: ws_handler(ws, asr),
        host, port,
        ping_interval=None,
    ):
        logger.info("🚀 Qwen3-ASR 服务已启动 ws://%s:%s", host, port)
        await asyncio.Future()


def main():
    _setup_logging("stt", "qwen3.log")
    parser = argparse.ArgumentParser(description="Qwen3-ASR WebSocket Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10096)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--gpu-mem", type=float, default=0.3)
    parser.add_argument("--max-model-len", type=int, default=8192)
    args = parser.parse_args()

    try:
        asyncio.run(main_async(args.host, args.port, args.model, args.gpu_mem, args.max_model_len))
    except KeyboardInterrupt:
        logger.info("已停止服务")


if __name__ == "__main__":
    main()
