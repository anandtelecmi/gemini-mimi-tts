#!/usr/bin/env python3
"""manifest.csv (relative_wav_path|text) -> `.pt` shards for the CSM-style model.

Each example: text_ids = [bos] + gemma tokens, codes = Mimi codes [Q, T].
Unlike the flattened-RVQ pipeline, sequences don't grow with Q, so you can
raise `model.num_quantizers` (e.g. 16) at almost no sequence-length cost.

Usage:
    python scripts/prepare_csm_dataset.py \\
        --manifest data/raw/ljspeech/manifest_normalized.csv \\
        --audio_root data/raw/ljspeech \\
        --output_dir data/processed/ljspeech_csm \\
        --config configs/gemma_csm.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import soundfile as sf
import torch
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

from gemma_mimi_tts.model.mimi_codec import MimiCodec


def read_manifest(path: Path) -> list[tuple[str, str]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in csv.reader(f, delimiter="|"):
            if len(line) >= 2:
                rows.append((line[0], line[1]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--audio_root", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--val_fraction", type=float, default=0.02)
    ap.add_argument("--shard_size", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    Q = cfg["model"]["num_quantizers"]
    max_seq = cfg["data"]["max_seq_len"]

    codec = MimiCodec.load(cfg["model"]["mimi_repo"], num_quantizers=Q, device=args.device)
    tok = AutoTokenizer.from_pretrained(cfg["model"]["text_backbone"])

    rows = read_manifest(args.manifest)
    random.Random(cfg["training"]["seed"]).shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]
    n_val = max(1, int(len(rows) * args.val_fraction))
    splits = {"val": rows[:n_val], "train": rows[n_val:]}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, split_rows in splits.items():
        buf, shard_idx, skipped = [], 0, 0

        def flush():
            torch.save(buf, args.output_dir / f"{name}-{shard_idx:05d}.pt")

        for rel, text in tqdm(split_rows, desc=f"encoding {name}"):
            audio, sr = sf.read(str(args.audio_root / rel), dtype="float32", always_2d=True)
            wav = torch.from_numpy(audio.T).mean(0)
            codes = torch.tensor(codec.encode_waveform(wav, sample_rate=sr), dtype=torch.int16)  # [Q,T]
            text_ids = [tok.bos_token_id] + tok(text, add_special_tokens=False)["input_ids"]
            if len(text_ids) + 1 + codes.shape[1] > max_seq or codes.shape[1] < 2:
                skipped += 1
                continue
            buf.append({"text_ids": torch.tensor(text_ids, dtype=torch.int32), "codes": codes})
            if len(buf) >= args.shard_size:
                flush(); shard_idx += 1; buf = []
        if buf:
            flush(); shard_idx += 1
        print(f"{name}: {shard_idx} shard(s), skipped {skipped}")

    meta = {
        "num_quantizers": Q,
        "codebook_size": codec.codebook_size,
        "text_backbone": cfg["model"]["text_backbone"],
        "mimi_repo": cfg["model"]["mimi_repo"],
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print("wrote", args.output_dir / "meta.json")


if __name__ == "__main__":
    main()
