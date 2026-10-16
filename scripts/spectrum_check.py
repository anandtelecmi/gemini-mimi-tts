#!/usr/bin/env python3
"""Where does the energy of a wav sit?  Compare real recordings with generated samples.

For each file prints sample rate, the frequency below which 99% / 99.9% of the spectral energy lies
(rolloff), and the average level (dB, relative to the 0-1 kHz band) of 4-8 kHz, 8-12 kHz and >12 kHz.
A muffled model output shows a much earlier rolloff / lower high-band level than real recordings.

    python scripts/spectrum_check.py real/*.wav hi_dec_out/*.wav
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import soundfile as sf


def band_report(x: np.ndarray, sr: int, n_fft: int = 2048) -> dict:
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(1)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    win = np.hanning(n_fft)
    hop = n_fft // 2
    frames = np.stack([x[i : i + n_fft] * win for i in range(0, len(x) - n_fft + 1, hop)])
    # skip near-silent frames so pauses do not dominate the average
    en = (frames ** 2).sum(1)
    frames = frames[en > 0.05 * en.mean()] if (en > 0).any() else frames
    p = (np.abs(np.fft.rfft(frames, axis=1)) ** 2).mean(0)
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    cum = np.cumsum(p) / p.sum()

    def band(lo, hi):
        m = (f >= lo) & (f < hi)
        return p[m].sum() if m.any() else 0.0

    ref = band(0, 1000) + 1e-20

    def db(v):
        return 10 * np.log10(v / ref + 1e-20) if v > 0 else float("-inf")

    return {
        "sr": sr,
        "roll99": float(f[np.searchsorted(cum, 0.99)]),
        "roll999": float(f[np.searchsorted(cum, 0.999)]),
        "4-8k": db(band(4000, 8000)),
        "8-12k": db(band(8000, 12000)),
        ">12k": db(band(12000, sr / 2 + 1)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    paths = [p for pat in args.files for p in sorted(glob.glob(pat))]
    print(f"{'file':<48}{'sr':>7}{'99%':>7}{'99.9%':>8}{'4-8k':>8}{'8-12k':>8}{'>12k':>8}   (band levels dB re 0-1 kHz)")
    for p in paths:
        x, sr = sf.read(p, dtype="float32", always_2d=False)
        r = band_report(x, sr)
        print(f"{p[-47:]:<48}{r['sr']:>7}{r['roll99']:>7.0f}{r['roll999']:>8.0f}{r['4-8k']:>8.1f}{r['8-12k']:>8.1f}{r['>12k']:>8.1f}")


if __name__ == "__main__":
    main()
