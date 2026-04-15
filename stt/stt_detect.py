import os
import tempfile

# 推理 wav 目录（可用环境变量 STT_INFERENCE_WAV_DIR 覆盖）
DEFAULT_INFERENCE_WAV_DIR = os.environ.get(
    "STT_INFERENCE_WAV_DIR",
    "/home/zhangchi/project/kws_deploy/log/stt/inference_wav",
)
# 只改这里：上述目录下的 wav 文件名，然后运行 python stt_detect.py
INFERENCE_WAV_FILENAME = "20260321_182002_59858_0bad18fd.wav"

# FunASR 1.3.x 不会在 import funasr 时加载 fun_asr_nano，必须先导入该子模块，
# 否则 config.yaml 里的 model: FunASRNano 无法注册，会报「FunASRNano is not registered」
import funasr.models.fun_asr_nano.model  # noqa: F401

import torch
import torchaudio
from funasr import AutoModel

# 整段推理时，SenseVoice 编码器显存随时长近似平方增长；超过阈值则按时长切分多次推理再拼接。
MAX_DURATION_SEC_DIRECT = 45.0
CHUNK_SEC = 20.0
CHUNK_OVERLAP_SEC = 0.5
SAMPLE_RATE = 16000


def _load_16k_mono(path: str) -> torch.Tensor:
    """[1, T] float32, 16kHz mono。"""
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav


def audio_duration_sec(path: str) -> float:
    w = _load_16k_mono(path)
    return w.shape[1] / SAMPLE_RATE


def transcribe_long_audio(model, audio_path: str, **generate_kw) -> str:
    """
    短音频：一次 generate。
    长音频：切成多段 wav 逐段识别后拼接（避免 OOM）。
    """
    dur = audio_duration_sec(audio_path)
    if dur <= MAX_DURATION_SEC_DIRECT:
        res = model.generate(input=[audio_path], cache={}, batch_size=1, **generate_kw)
        return res[0].get("text", "").strip()

    w = _load_16k_mono(audio_path)
    total = w.shape[1]
    chunk_samples = int(CHUNK_SEC * SAMPLE_RATE)
    step = int((CHUNK_SEC - CHUNK_OVERLAP_SEC) * SAMPLE_RATE)
    texts: list[str] = []

    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        for start in range(0, total, step):
            end = min(start + chunk_samples, total)
            seg = w[:, start:end]
            if seg.shape[1] < int(0.3 * SAMPLE_RATE):
                break
            torchaudio.save(tmp, seg.cpu(), SAMPLE_RATE)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            res = model.generate(input=[tmp], cache={}, batch_size=1, **generate_kw)
            t = res[0].get("text", "").strip()
            if t:
                texts.append(t)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

    return "".join(texts)


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "model", "FunAudioLLM", "Fun-ASR-Nano-2512")
    inference_wav_dir = DEFAULT_INFERENCE_WAV_DIR
    wav_path = os.path.join(inference_wav_dir, INFERENCE_WAV_FILENAME)

    model = AutoModel(
        model=model_dir,
        device="cuda:0",
        disable_update=True,
        disable_log=True,
        disable_pbar=True,
    )

    if not os.path.isfile(wav_path):
        print(
            f"未找到 wav 文件。\n"
            f"  路径: {wav_path}\n"
            f"请确认 INFERENCE_WAV_FILENAME（当前为 {INFERENCE_WAV_FILENAME!r}）"
            f" 与目录内实际文件名一致。"
        )
        return

    print(f"使用音频: {wav_path}")
    gen_kw = dict(language="中文", itn=True)
    dur = audio_duration_sec(wav_path)
    print(f"音频时长约 {dur:.1f}s，阈值 {MAX_DURATION_SEC_DIRECT}s → ", end="")
    if dur > MAX_DURATION_SEC_DIRECT:
        print(f"分段推理（每段约 {CHUNK_SEC}s）")
    else:
        print("整段推理")

    text = transcribe_long_audio(model, wav_path, **gen_kw)
    print(f"text: {text}")


if __name__ == "__main__":
    main()
