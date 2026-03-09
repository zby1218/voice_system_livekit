"""
流式 TTS 测试：可选整段一次合成（语气一致）或按句切段边生成边播。

- USE_SEGMENT_SPLIT=False（默认）：整段一次推理，再按块写入 PyAudio，音色/语气必然统一。
- USE_SEGMENT_SPLIT=True：按标点切段逐段推理，用 do_sample=False 贪心解码尽量统一语气。
"""
import re
import threading
import queue
import time
from typing import List

import numpy as np
import torch
import pyaudio
from qwen_tts import Qwen3TTSModel

# False = 整段一次合成，语气一致；True = 按标点切段边生成边播（需 do_sample=False 尽量统一）
USE_SEGMENT_SPLIT = True

TTS_TEMPERATURE = 0.4
TTS_TOP_P = 0.9
TTS_SUBTALKER_TEMPERATURE = 0.4
CHUNK_FRAMES = 1024
SEGMENT_DELIMS = "。！？!?；;\n，,"  # 按这些符号切段

model = Qwen3TTSModel.from_pretrained(
    "/home/zhangchi/project/Qwen3-TTS/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)


def _split_text(text: str) -> List[str]:
    """按标点切分成多段，避免单段过长且便于流式播放。"""
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(f"([{re.escape(SEGMENT_DELIMS)}])", text)
    segments = []
    buf = ""
    for p in parts:
        buf += p
        if p.strip() and (not buf.strip() or buf.strip()[-1] in SEGMENT_DELIMS):
            seg = buf.strip()
            if seg:
                segments.append(seg)
            buf = ""
    if buf.strip():
        segments.append(buf.strip())
    return segments if segments else [text]


def _to_float32_bytes(samples) -> bytes:
    """模型输出转成 float32 字节（PyAudio paFloat32 用）。"""
    if hasattr(samples, "cpu"):
        samples = samples.cpu().numpy()
    samples = np.asarray(samples, dtype=np.float32)
    return np.clip(samples, -1.0, 1.0).tobytes()


def main():
    text = "此外，该模型具备强大的上下文理解能力，可根据指令和文本语义自适应地控制语调、语速和情感表达，并对含噪声的输入文本展现出显著增强的鲁棒性。"
    if USE_SEGMENT_SPLIT:
        segments = _split_text(text)
        if not segments:
            segments = [text]
    else:
        segments = [text]  # 整段一次合成，语气必然一致
    print("切段:", segments)

    audio_queue = queue.Queue(maxsize=4)
    sr = 24000  # Qwen3-TTS CustomVoice 固定采样率

    def generate_worker():
        total_inference_time = 0.0
        for i, seg in enumerate(segments):
            t0 = time.perf_counter()
            wavs, sample_rate = model.generate_custom_voice(
                text=seg,
                language="Chinese",
                speaker="Vivian",
                instruct="用亲切的语气说，语速为1.1",
                do_sample=False,  # 贪心解码，分段时各段语气最统一
                temperature=TTS_TEMPERATURE,
                top_p=TTS_TOP_P,
                subtalker_temperature=TTS_SUBTALKER_TEMPERATURE,
            )
            elapsed = time.perf_counter() - t0
            total_inference_time += elapsed
            num_samples = wavs[0].numel() if hasattr(wavs[0], "numel") else len(wavs[0])
            duration_s = num_samples / sample_rate if sample_rate else 0
            print(f"[推理] 段 {i + 1}/{len(segments)}: {elapsed:.3f} s (音频时长 {duration_s:.2f} s, RTF={elapsed / duration_s:.3f})")
            pcm = _to_float32_bytes(wavs[0])
            audio_queue.put((pcm, sample_rate))
        audio_queue.put(None)
        print(f"[推理] 合计: {total_inference_time:.3f} s ({len(segments)} 段)")

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=sr,
        output=True,
        frames_per_buffer=CHUNK_FRAMES,
    )
    print("PyAudio 播放流已打开，开始流式播放…")

    gen_thread = threading.Thread(target=generate_worker)
    gen_thread.start()

    while True:
        item = audio_queue.get()
        if item is None:
            break
        pcm, sample_rate = item
        if sample_rate != sr:
            # 若某段采样率不同可在此重采样，一般同模型同 sr
            pass
        stream.write(pcm)

    gen_thread.join()
    stream.stop_stream()
    stream.close()
    pa.terminate()
    print("播放结束。")


if __name__ == "__main__":
    main()
