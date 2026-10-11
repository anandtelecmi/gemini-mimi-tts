#!/usr/bin/env python3
"""Convert existing flattened-RVQ jsonl shards (data/processed/ljspeech) to the
CSM `.pt` format WITHOUT re-running Mimi. Keeps the 8 codebooks and the text
tokens exactly as they were encoded.

Usage:
    python scripts/convert_legacy_shards.py \\
        --src data/processed/ljspeech --dst data/processed/ljspeech_csm8
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch


def convert_example(ids: list[int], m: dict):
    sos, eos = m["start_of_speech_id"], m["end_of_speech_id"]
    Q, V, off = m["num_quantizers"], m["codebook_size"], m["audio_offset"]
    s = ids.index(sos)
    e = ids.index(eos, s)
    audio = ids[s + 1 : e]
    T = len(audio) // Q
    if T < 2:
        return None
    codes = torch.empty(Q, T, dtype=torch.int16)
    for t in range(T):
        for q in range(Q):
            v = audio[t * Q + q] - off - q * V
            if not 0 <= v < V:
                raise ValueError(f"token out of codebook {q} range: {audio[t*Q+q]}")
            codes[q, t] = v
    return {"text_ids": torch.tensor(ids[:s], dtype=torch.int32), "codes": codes}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    args = ap.parse_args()

    m = json.load(open(args.src / "vocab_meta.json"))
    args.dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(glob.glob(str(args.src / "*.jsonl"))):
        out = []
        for line in open(f):
            ex = convert_example(json.loads(line)["input_ids"], m)
            if ex is not None:
                out.append(ex)
        dst = args.dst / (Path(f).stem + ".pt")
        torch.save(out, dst)
        print(f"{f} -> {dst} ({len(out)} examples)")
    (args.dst / "meta.json").write_text(json.dumps({
        "num_quantizers": m["num_quantizers"], "codebook_size": m["codebook_size"],
    }, indent=2))


if __name__ == "__main__":
    main()
