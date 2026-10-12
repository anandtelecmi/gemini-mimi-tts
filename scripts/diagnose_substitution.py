#!/usr/bin/env python3
"""Model-free test: how sensitive is Mimi's decoder to WRONG fine codebooks?

Takes one real utterance A and replaces some of its codebooks with *natural but wrong* values
(from another utterance B, or shifted in time, or random). No language model involved.

  sub_deep8_other   A[0:8] + B[8:16]          (like fp32_k8: wrong deep codebooks, but real-speech statistics)
  sub_deep8_shift   A[0:8] + A[8:16] shifted by 1 frame
  sub_deep8_rand    A[0:8] + uniform-random entries
  sub_from4_other   A[0:4] + B[4:16]
  sub_from1_other   A[0]   + B[1:16]
  sub_from1_shift   A[0]   + A[1:16] shifted by 1 frame
  base_q8 / base_q16  A truncated / full (clean references)

Reading: if the sub_* files echo like your fp32_k8.wav, the codec simply cannot tolerate wrong fine
codebooks -> the LM must be (near) exact there, or we must generate fewer codebooks.
If they stay clean while the model's samples echo, the model's fine-codebook distribution is the problem.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch
import yaml

from gemma_mimi_tts.csm.dataset import CodesDataset
from gemma_mimi_tts.model.mimi_codec import MIMI_SAMPLE_RATE, MimiCodec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--val_pattern", default="data/processed/ljspeech_csm/val-*.pt")
    ap.add_argument("--out_dir", type=Path, default=Path("diag3"))
    ap.add_argument("--utt", type=int, default=0)
    ap.add_argument("--other", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    Q = cfg["model"]["num_quantizers"]
    V = 2048
    ds = CodesDataset(args.val_pattern, num_quantizers=Q)
    codec = MimiCodec.load(cfg["model"]["mimi_repo"], num_quantizers=Q, device=args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    A = ds[args.utt]["codes"].long()
    B = ds[args.other]["codes"].long()
    T = A.shape[1]
    reps = (T + B.shape[1] - 1) // B.shape[1]
    B = B.repeat(1, reps)[:, :T]                       # loop B to A's length
    g = torch.Generator().manual_seed(0)

    def save(name, c):
        wav = codec.decode_codes(c.tolist())
        sf.write(str(args.out_dir / f"{name}.wav"), wav.float().numpy(), MIMI_SAMPLE_RATE)
        print(f"wrote {name}.wav")

    def cat(*parts):
        return torch.cat(parts, dim=0)

    shifted = torch.roll(A, 1, dims=1)
    save("base_q16", A)
    save("base_q8", A[:8])
    save("sub_deep8_other", cat(A[:8], B[8:]))
    save("sub_deep8_shift", cat(A[:8], shifted[8:]))
    save("sub_deep8_rand", cat(A[:8], torch.randint(0, V, (Q - 8, T), generator=g)))
    save("sub_from4_other", cat(A[:4], B[4:]))
    save("sub_from1_other", cat(A[:1], B[1:]))
    save("sub_from1_shift", cat(A[:1], shifted[1:]))


if __name__ == "__main__":
    main()
