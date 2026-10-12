#!/usr/bin/env python3
"""Codec-ceiling / data-quality check: encode real audio with Mimi at several
codebook counts and decode it straight back (no language model involved).

Listen to  <orig>  vs  <q8>/<q16>/<q32>.  If the ORIGINAL already sounds
compressed/artifacty, the training data is the limit. If the original is clean
but the round-trip is not, the codec setup is the limit. If both are clean,
the language model is the limit.

Usage:
    # any audio file(s)
    python scripts/codec_roundtrip.py --audio some.wav other.mp3 --out_dir rt
    # first few clips (spread out) from an Emilia tar
    python scripts/codec_roundtrip.py --tar data/raw/emilia_yodas_en/EN-B000000.tar --n 4 --out_dir rt
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_emilia import AUDIO_EXTS, decode_audio, iter_tar_pairs  # noqa: E402


def pick_clips(tar_path: Path, n: int, skip: int = 300) -> list[tuple[str, bytes, str]]:
    """Every `skip`-th clip of the tar -> [(key, audio_bytes, ext)] (spread over many videos)."""
    out = []
    for i, (key, parts) in enumerate(iter_tar_pairs(tar_path)):
        if i % skip:
            continue
        ext = next(k for k in parts if k in AUDIO_EXTS)
        out.append((key, parts[ext], ext))
        if len(out) >= n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", nargs="*", type=Path, default=[])
    ap.add_argument("--tar", type=Path)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--qs", nargs="+", type=int, default=[8, 16, 32])
    ap.add_argument("--out_dir", type=Path, default=Path("roundtrip"))
    ap.add_argument("--mimi_repo", default="kyutai/mimi")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    items: list[tuple[str, bytes, str]] = [(p.stem, p.read_bytes(), p.suffix) for p in a.audio]
    if a.tar:
        items += pick_clips(a.tar, a.n)
    if not items:
        raise SystemExit("give --audio files and/or --tar")

    from gemma_mimi_tts.model.mimi_codec import MIMI_SAMPLE_RATE, MimiCodec

    codecs = {q: MimiCodec.load(a.mimi_repo, num_quantizers=q, device=a.device) for q in a.qs}
    for key, data, ext in items:
        name = key.replace("/", "_")
        (a.out_dir / f"{name}_orig{ext}").write_bytes(data)       # untouched original
        wav, sr = decode_audio(data)
        for q, codec in codecs.items():
            codes = codec.encode_waveform(wav, sample_rate=sr)
            rec = codec.decode_codes(codes).float().numpy()
            sf.write(str(a.out_dir / f"{name}_q{q}.wav"), rec, MIMI_SAMPLE_RATE)
        print(f"{name}: {len(wav)/sr:.1f}s  sr={sr}  -> orig + {['q%d' % q for q in a.qs]}")
    print("files in", a.out_dir)


if __name__ == "__main__":
    main()
