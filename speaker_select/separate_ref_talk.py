#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

try:
    from speaker_select.separator import SpeakerSeparator
except ModuleNotFoundError:
    from separator import SpeakerSeparator


DEFAULT_INPUT = Path(__file__).with_name("ref_talk.wav")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("separated_ref_talk")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="分离 ref_talk.wav 中的说话人音轨，并导出为多个 wav 文件。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"输入 wav 路径，默认: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录，默认: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="推理设备，默认: cuda",
    )
    parser.add_argument(
        "--clearvoice-pkg",
        default=None,
        help="可选，显式指定 clearvoice Python 包目录。",
    )
    return parser


def load_audio_as_pcm16(input_path: Path, sample_rate: int = 16000) -> bytes:
    audio, _ = librosa.load(input_path, sr=sample_rate, mono=True)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16).tobytes()


def save_pcm16_wav(output_path: Path, pcm_bytes: bytes, sample_rate: int = 16000) -> None:
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    sf.write(output_path, samples, sample_rate, subtype="PCM_16")


def main() -> None:
    args = build_parser().parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"输入文件不存在: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pcm_bytes = load_audio_as_pcm16(args.input)
    separator_kwargs = {"device": args.device}
    if args.clearvoice_pkg:
        separator_kwargs["clearvoice_pkg"] = args.clearvoice_pkg

    separator = SpeakerSeparator(**separator_kwargs)
    separated_tracks = separator._separate_sync(pcm_bytes)

    stem = args.input.stem
    for idx, track_bytes in enumerate(separated_tracks, start=1):
        output_path = args.output_dir / f"{stem}_speaker{idx}.wav"
        save_pcm16_wav(output_path, track_bytes)
        duration = len(track_bytes) / 2 / 16000
        print(f"已保存: {output_path} ({duration:.2f}s)")

    print(f"完成，共导出 {len(separated_tracks)} 路说话人音轨到: {args.output_dir}")


if __name__ == "__main__":
    main()
