#!/usr/bin/env python3
"""Downloads LJSpeech via torchaudio and writes a `manifest.csv`
(`relative_wav_path|text`) that prepare_dataset.py can consume.

Usage:
    python scripts/download_ljspeech.py --root data/raw/ljspeech
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/ljspeech"))
    parser.add_argument(
        "--raw_text",
        action="store_true",
        help="use LJSpeech's raw transcript (digits/abbreviations) instead of the "
        "normalized one (numbers and abbreviations spelled out; default)",
    )
    args = parser.parse_args()

    import torchaudio

    args.root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading LJSpeech to {args.root} (skips if already present)...")
    dataset = torchaudio.datasets.LJSPEECH(root=str(args.root), download=True)

    # torchaudio's LJSPEECH doesn't expose a public per-item metadata API
    # across versions, so we read its internal file list directly: each row
    # is [fileid, transcript, normalized_transcript], and dataset._path is
    # the directory the wavs actually live in (avoids hardcoding the
    # "LJSpeech-1.1/wavs" layout, which has moved across releases).
    wav_dir = dataset._path
    manifest_path = args.root / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")
        for fileid, transcript, normalized in dataset._flist:
            text = transcript if args.raw_text else normalized
            wav_path = wav_dir / f"{fileid}.wav"
            wav_rel_path = wav_path.relative_to(args.root)
            writer.writerow([str(wav_rel_path), text])

    print(f"Wrote manifest with {len(dataset)} utterances to {manifest_path}")


if __name__ == "__main__":
    main()