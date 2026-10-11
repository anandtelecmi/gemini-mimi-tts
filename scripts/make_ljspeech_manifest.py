#!/usr/bin/env python3
"""Rebuild manifest.csv from an already-downloaded LJSpeech-1.1/metadata.csv,
using the NORMALIZED transcript (no torchaudio / no re-download needed).

Usage:
    python scripts/make_ljspeech_manifest.py --root data/raw/ljspeech
"""
import argparse
import csv
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, default=Path("data/raw/ljspeech"))
args = parser.parse_args()

meta = args.root / "LJSpeech-1.1" / "metadata.csv"
out = args.root / "manifest_normalized.csv"
n = 0
with open(meta, encoding="utf-8") as f, open(out, "w", newline="", encoding="utf-8") as g:
    w = csv.writer(g, delimiter="|")
    # metadata.csv rows: id|raw|normalized  (quotes inside text, so no csv quoting)
    for line in f:
        parts = line.rstrip("\n").split("|")
        if len(parts) < 3:
            continue
        w.writerow([f"LJSpeech-1.1/wavs/{parts[0]}.wav", parts[2]])
        n += 1
print(f"Wrote {n} rows to {out}")
