#!/usr/bin/env python3
"""
Qwen3-TTS 流式服务：WebSocket 接收 text，整段一次推理后按块流式返回 PCM，保证语气一致。

启动: python qwen_tts_server.py --model-name /path/to/model --port 50001

协议: 客户端连 WS /ws/tts，发 JSON { "text": "要合成的内容" }（可选 speaker, language, instruct）
      服务端对全文做一次合成，再将 PCM 按块推送，最后发 { "done": true }
"""
import argparse
import asyncio
import json

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from qwen_tts import Qwen3TTSModel

CHUNK_BYTES = 8192


def _to_pcm_bytes(wav) -> bytes:
    if hasattr(wav, "cpu"):
        wav = wav.cpu().numpy()
    wav = np.asarray(wav, dtype=np.float32)
    wav = np.clip(wav, -1.0, 1.0)
    return (wav * 32767).astype(np.int16).tobytes()


# ---------------------------------------------------------------------------
# 模型（启动时加载一次）
# ---------------------------------------------------------------------------

model = None
executor = None


def _load_model(model_name: str, device: str = "cuda:0"):
    global model, executor
    from concurrent.futures import ThreadPoolExecutor
    model = Qwen3TTSModel.from_pretrained(
        model_name,
        device_map=device,
        dtype=torch.bfloat16,
    )
    executor = ThreadPoolExecutor(max_workers=1)
    print("模型加载完成")


def _generate_full(text: str, speaker: str, language: str, instruct: str) -> bytes:
    """整段一次推理，语气一致。"""
    wavs, sr = model.generate_custom_voice(
        text=text,
        speaker=speaker,
        language=language,
        instruct=instruct or "",
    )
    return _to_pcm_bytes(wavs[0])


# ---------------------------------------------------------------------------
# FastAPI + WebSocket
# ---------------------------------------------------------------------------

app = FastAPI(title="Qwen3-TTS Stream")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.websocket("/ws/tts")
async def ws_tts(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_event_loop()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "invalid json"})
                continue
            text = (msg.get("text") or "").strip()
            if not text:
                await websocket.send_json({"error": "text empty"})
                continue
            if model is None:
                await websocket.send_json({"error": "model not loaded"})
                continue

            speaker = msg.get("speaker", "Vivian")
            language = msg.get("language", "Chinese")
            instruct = msg.get("instruct", "")

            # 整段一次推理，避免按标点切段导致每段语气不一致
            pcm = await loop.run_in_executor(
                executor,
                _generate_full,
                text,
                speaker,
                language,
                instruct,
            )
            for i in range(0, len(pcm), CHUNK_BYTES):
                await websocket.send_bytes(pcm[i : i + CHUNK_BYTES])
            await websocket.send_json({"done": True})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS 流式 WebSocket 服务")
    parser.add_argument("--model-name", type=str, default="/home/zhangchi/project/Qwen3-TTS/Qwen3-TTS-12Hz-1.7B-CustomVoice", help="模型路径或 HuggingFace id")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50001)
    args = parser.parse_args()

    print(f"加载模型: {args.model_name}")
    _load_model(args.model_name, args.device)
    print(f"监听 {args.host}:{args.port}  WebSocket: /ws/tts")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
