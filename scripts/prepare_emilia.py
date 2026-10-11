#!/usr/bin/env python3
"""Emilia WebDataset tars (mp3 + json) -> `.pt` shards for the CSM model.

Streams each tar once, decodes the mp3, runs frozen Mimi, tokenizes the text
with Gemma's tokenizer. Run several workers in parallel on the same GPU
(Mimi is small): each worker takes tars  idx % num_workers == worker.
Resumable at tar granularity (finished tars are marked in <out>/.done/).

Usage (4 parallel workers for train, then val):
    for w in 0 1 2 3; do
      python scripts/prepare_emilia.py --tar_dir data/raw/emilia_yodas_en \\
        --split train --worker $w --num_workers 4 \\
        --output_dir data/processed/emilia_en --config configs/gemma_csm_emilia.yaml &
    done; wait
    python scripts/prepare_emilia.py --tar_dir data/raw/emilia_yodas_en --split val \\
        --keep_prob 0.05 --max_utts 1500 \\
        --output_dir data/processed/emilia_en --config configs/gemma_csm_emilia.yaml
    python scripts/prepare_emilia.py --output_dir data/processed/emilia_en --summarize
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import tarfile
import time
from pathlib import Path

import soundfile as sf
import torch

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a"}


def iter_tar_pairs(path: Path):
    """Yield (key, {ext: bytes}) once both the audio and the json of a key
    have been seen. Streams the tar sequentially (no random access)."""
    pending: dict[str, dict[str, bytes]] = {}
    with tarfile.open(path, "r|") as tf:
        for m in tf:
            if not m.isfile():
                continue
            key, ext = os.path.splitext(os.path.basename(m.name))
            ext = ext.lower()
            if ext != ".json" and ext not in AUDIO_EXTS:
                continue
            slot = pending.setdefault(key, {})
            slot[ext] = tf.extractfile(m).read()
            if ".json" in slot and len(slot) == 2:
                yield key, pending.pop(key)


def decode_audio(data: bytes) -> tuple[torch.Tensor, int]:
    """bytes -> (mono float32 tensor, sample_rate). soundfile first (libsndfile
    >= 1.1 reads mp3), librosa/ffmpeg as fallback."""
    try:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception:
        import librosa  # noqa: WPS433

        y, sr = librosa.load(io.BytesIO(data), sr=None, mono=False)
        audio = y.reshape(-1, 1) if y.ndim == 1 else y.T
    return torch.from_numpy(audio.mean(axis=1).astype("float32")), int(sr)


def process_tar(tar_path, tar_idx, a, codec, tok, Q, max_seq, out_dir, rng):
    for old in out_dir.glob(f"{a.split}-t{tar_idx:03d}-*.pt"):
        old.unlink()
    buf, shard_i, kept, secs = [], 0, 0, 0.0
    skipped = {"meta": 0, "duration": 0, "dnsmos": 0, "rate": 0, "decode": 0, "too_long": 0}
    attempts = 0

    def flush():
        nonlocal buf, shard_i
        if buf:
            torch.save(buf, out_dir / f"{a.split}-t{tar_idx:03d}-{shard_i:04d}.pt")
            shard_i += 1
            buf = []

    t0 = time.time()
    for key, parts in iter_tar_pairs(tar_path):
        if a.max_utts and kept >= a.max_utts:
            break
        if a.keep_prob < 1.0 and rng.random() > a.keep_prob:
            continue
        try:
            meta = json.loads(parts[".json"])
            text = (meta.get("text") or "").strip()
            dur = float(meta.get("duration") or 0.0)
            dnsmos = meta.get("dnsmos")
        except Exception:
            skipped["meta"] += 1
            continue
        if not text or not (a.min_dur <= dur <= a.max_dur):
            skipped["duration"] += 1
            continue
        if a.min_dnsmos and dnsmos is not None and float(dnsmos) < a.min_dnsmos:
            skipped["dnsmos"] += 1
            continue
        cps = len(text) / dur
        if not (a.min_cps <= cps <= a.max_cps):  # bad transcript/audio alignment
            skipped["rate"] += 1
            continue

        audio_bytes = next(v for k, v in parts.items() if k != ".json")
        attempts += 1
        try:
            wav, sr = decode_audio(audio_bytes)
            codes = torch.tensor(codec.encode_waveform(wav, sample_rate=sr), dtype=torch.int16)  # [Q,T]
        except Exception as e:
            skipped["decode"] += 1
            if attempts >= 50 and kept == 0:
                raise RuntimeError(
                    "first 50 clips all failed to decode/encode - install ffmpeg + `pip install librosa` "
                    f"or upgrade soundfile (libsndfile>=1.1 reads mp3). Last error: {e}") from e
            continue
        text_ids = [tok.bos_token_id] + tok(text, add_special_tokens=False)["input_ids"]
        if len(text_ids) + 1 + codes.shape[1] > max_seq or codes.shape[1] < 2:
            skipped["too_long"] += 1
            continue
        buf.append({"text_ids": torch.tensor(text_ids, dtype=torch.int32), "codes": codes})
        kept += 1
        secs += codes.shape[1] / 12.5
        if len(buf) >= a.shard_size:
            flush()
            print(f"[{tar_path.name}] {kept} utts, {secs/3600:.1f} h, {kept/(time.time()-t0):.1f} utt/s", flush=True)
    flush()
    return {"split": a.split, "tar": tar_path.name, "utts": kept, "seconds": secs, "skipped": skipped}


def summarize(out_dir: Path) -> None:
    tot = {}
    for f in sorted((out_dir / ".done").glob("*.json")):
        d = json.loads(f.read_text())
        s = tot.setdefault(d["split"], {"utts": 0, "hours": 0.0})
        s["utts"] += d["utts"]
        s["hours"] += d["seconds"] / 3600
    for k, v in tot.items():
        print(f"{k}: {v['utts']} utterances, {v['hours']:.1f} hours")
    if not tot:
        print("nothing finished yet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar_dir", type=Path)
    ap.add_argument("--selected", type=Path, default=None, help="default: <tar_dir>/selected.json")
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard_size", type=int, default=2000)
    ap.add_argument("--min_dur", type=float, default=2.0)
    ap.add_argument("--max_dur", type=float, default=25.0)
    ap.add_argument("--min_dnsmos", type=float, default=0.0, help="0 = keep all (Emilia already filters >= 3.0)")
    ap.add_argument("--min_cps", type=float, default=4.0, help="min characters per second")
    ap.add_argument("--max_cps", type=float, default=30.0)
    ap.add_argument("--keep_prob", type=float, default=1.0, help="randomly keep this fraction (use for val)")
    ap.add_argument("--max_utts", type=int, default=0, help="stop a tar after this many kept utts (0 = no cap)")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()

    if a.summarize:
        summarize(a.output_dir)
        return
    import yaml
    from transformers import AutoTokenizer

    from gemma_mimi_tts.model.mimi_codec import MimiCodec

    cfg = yaml.safe_load(open(a.config))
    Q, max_seq = cfg["model"]["num_quantizers"], cfg["data"]["max_seq_len"]
    sel = json.loads((a.selected or a.tar_dir / "selected.json").read_text())[a.split]

    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / ".done").mkdir(exist_ok=True)
    codec = MimiCodec.load(cfg["model"]["mimi_repo"], num_quantizers=Q, device=a.device)
    tok = AutoTokenizer.from_pretrained(cfg["model"]["text_backbone"])
    meta_path = a.output_dir / "meta.json"
    if not meta_path.exists():
        meta_path.write_text(json.dumps({
            "num_quantizers": Q, "codebook_size": codec.codebook_size,
            "text_backbone": cfg["model"]["text_backbone"], "mimi_repo": cfg["model"]["mimi_repo"],
            "source": "Emilia-YODAS EN"}, indent=2))

    rng = random.Random(1234 + a.worker)
    for idx, name in enumerate(sel):
        if idx % a.num_workers != a.worker:
            continue
        done = a.output_dir / ".done" / f"{name}.json"
        if done.exists():
            print(f"skip {name} (done)")
            continue
        print(f"== worker {a.worker}: {name} ({a.split})", flush=True)
        stats = process_tar(a.tar_dir / name, idx, a, codec, tok, Q, max_seq, a.output_dir, rng)
        done.write_text(json.dumps(stats))
        print(f"finished {name}: {stats['utts']} utts, {stats['seconds']/3600:.1f} h, skipped {stats['skipped']}", flush=True)


if __name__ == "__main__":
    main()
