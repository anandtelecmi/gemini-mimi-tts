#!/usr/bin/env python3
"""{"audio": path, "text": ...} JSONL (from data_prep.py) -> `.pt` shards for the CSM-style model.

Use --dry_run FIRST: it reads only audio headers and prints hours, sample rates, duration / chars-per-second
percentiles, speakers, and a text-vs-duration correlation per source (a quick detector for text/audio
misalignment), without touching the GPU.

Source and speaker are derived from the collision-safe file name written by data_prep.py:
    <bucket>__<lang>__[<speaker_or_session>...]__<original_name>
(or taken from optional "source" / "speaker" keys in the JSONL if you add them).

Examples:
    python scripts/prepare_jsonl_dataset.py --jsonl training_1/hindi_metadata.jsonl --dry_run
    python scripts/prepare_jsonl_dataset.py --jsonl training_1/hindi_metadata.jsonl \\
        --output_dir data/processed/hindi_csm --config configs/gemma_csm_hindi.yaml --speaker_tags

Shards are named  train-<source>-00000.pt / val-<source>-00000.pt  so a config can use
`train-*.pt` (everything) or `train-studio_recordings-*.pt` (one source).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


# ------------------------------------------------------------------ pure helpers (unit tested)
def parse_name(audio_path: str) -> tuple[str, str]:
    """-> (source, speaker_key) from '<bucket>__<lang>__[ids...]__<orig>'. Falls back to ('unknown', 'unknown')."""
    parts = Path(audio_path).name.split("__")
    if len(parts) >= 4:
        return parts[0], "__".join(parts[2:-1])
    if len(parts) == 3:
        return parts[0], parts[1]
    return "unknown", "unknown"


def normalize_text(t: str) -> str:
    t = unicodedata.normalize("NFC", t)
    t = t.replace("​", "").replace("﻿", "")
    return re.sub(r"\s+", " ", t).strip()


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def pct(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))]


def keep(rec: dict, min_dur: float, max_dur: float, min_cps: float, max_cps: float) -> str | None:
    """None if kept, else the reason it is dropped."""
    if not rec["text"]:
        return "empty_text"
    if not (min_dur <= rec["dur"] <= max_dur):
        return "duration"
    cps = len(rec["text"]) / max(rec["dur"], 1e-6)
    if not (min_cps <= cps <= max_cps):
        return "cps"
    return None


def split_speakers(speakers_by_source: dict[str, list[str]], val_speakers: int, seed: int) -> set[tuple[str, str]]:
    """Hold out `val_speakers` whole speakers per source (never the only speaker)."""
    rng = random.Random(seed)
    held = set()
    for src, spks in speakers_by_source.items():
        spks = sorted(set(spks))
        if len(spks) > val_speakers:
            held |= {(src, s) for s in rng.sample(spks, val_speakers)}
    return held


def speaker_tag_map(counts: Counter, min_utts: int) -> dict[str, str]:
    """Most frequent speakers get s000, s001, ...; speakers below min_utts get no tag."""
    tags = {}
    for i, (spk, n) in enumerate(counts.most_common()):
        if n >= min_utts:
            tags[spk] = f"s{i:03d}"
    return tags


# ------------------------------------------------------------------ main
def load_records(jsonl_paths: list[Path]) -> list[dict]:
    import soundfile as sf

    recs, missing = [], 0
    for p in jsonl_paths:
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            ap = Path(j["audio"])
            if not ap.exists():
                missing += 1
                continue
            try:
                info = sf.info(str(ap))
            except Exception:
                missing += 1
                continue
            src, spk = parse_name(str(ap))
            recs.append({
                "audio": str(ap), "text": normalize_text(j.get("text", "")),
                "dur": info.frames / info.samplerate, "sr": info.samplerate,
                "source": j.get("source", src), "speaker": j.get("speaker", spk),
            })
    if missing:
        print(f"[load] {missing} record(s) skipped (audio missing/unreadable)")
    return recs


def report(recs: list[dict], title: str) -> None:
    print(f"\n=== {title}: {len(recs)} utterances, {sum(r['dur'] for r in recs)/3600:.1f} h ===")
    by_src = defaultdict(list)
    for r in recs:
        by_src[r["source"]].append(r)
    for src, rs in sorted(by_src.items()):
        durs = [r["dur"] for r in rs]
        cps = [len(r["text"]) / r["dur"] for r in rs if r["dur"] > 0]
        spk = Counter(r["speaker"] for r in rs)
        srs = Counter(r["sr"] for r in rs)
        corr = pearson([len(r["text"]) for r in rs], durs)
        print(f"\n[{src}] {len(rs)} utts, {sum(durs)/3600:.1f} h, {len(spk)} speaker key(s)")
        print(f"  sample rates : {dict(srs.most_common(5))}")
        print(f"  duration s   : p5 {pct(durs,.05):.1f}  p50 {pct(durs,.5):.1f}  p95 {pct(durs,.95):.1f}  max {max(durs):.1f}")
        print(f"  chars/sec    : p5 {pct(cps,.05):.1f}  p50 {pct(cps,.5):.1f}  p95 {pct(cps,.95):.1f}")
        flag = "" if corr != corr or corr >= 0.8 else "   <-- LOW: text and audio may be misaligned!"
        print(f"  corr(chars, duration) = {corr:.2f}{flag}")
        top = ", ".join(f"{k}:{v}" for k, v in spk.most_common(6))
        print(f"  speakers (utts): {top}{' ...' if len(spk) > 6 else ''}")
        if any(sr < 24000 for sr in srs):
            n_low = sum(v for sr, v in srs.items() if sr < 24000)
            print(f"  NOTE: {n_low} file(s) are below 24 kHz; Mimi is a 24 kHz codec, so these will be band-limited.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, nargs="+", required=True)
    ap.add_argument("--output_dir", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--only_source", nargs="+", default=None, help="keep only these sources (bucket names)")
    ap.add_argument("--min_dur", type=float, default=1.5)
    ap.add_argument("--max_dur", type=float, default=25.0)
    ap.add_argument("--min_cps", type=float, default=4.0, help="chars (unicode code points) per second")
    ap.add_argument("--max_cps", type=float, default=35.0)
    ap.add_argument("--val_speakers", type=int, default=1, help="whole speakers held out per source for validation")
    ap.add_argument("--val_fraction", type=float, default=0.01, help="extra random utterances held out (seen speakers)")
    ap.add_argument("--speaker_tags", action="store_true", help="prefix text with a speaker tag like '[s003] '")
    ap.add_argument("--min_spk_utts", type=int, default=50)
    ap.add_argument("--shard_size", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    recs = load_records(args.jsonl)
    if args.only_source:
        recs = [r for r in recs if r["source"] in set(args.only_source)]
    report(recs, "ALL (before filtering)")

    drops = Counter()
    kept = []
    for r in recs:
        why = keep(r, args.min_dur, args.max_dur, args.min_cps, args.max_cps)
        if why:
            drops[why] += 1
        else:
            kept.append(r)
    print(f"\n[filter] kept {len(kept)}/{len(recs)}; dropped {dict(drops)}")
    report(kept, "KEPT")
    if args.dry_run:
        return
    if not args.output_dir or not args.config:
        raise SystemExit("--output_dir and --config are required unless --dry_run")

    # heavy imports only when encoding
    import soundfile as sf
    import torch
    import yaml
    from tqdm import tqdm
    from transformers import AutoTokenizer

    from gemma_mimi_tts.model.mimi_codec import MimiCodec

    cfg = yaml.safe_load(open(args.config))
    Q, max_seq = cfg["model"]["num_quantizers"], cfg["data"]["max_seq_len"]
    codec = MimiCodec.load(cfg["model"]["mimi_repo"], num_quantizers=Q, device=args.device)
    tok = AutoTokenizer.from_pretrained(cfg["model"]["text_backbone"])

    rng = random.Random(args.seed)
    by_src_spk = defaultdict(list)
    for r in kept:
        by_src_spk[r["source"]].append(r["speaker"])
    held = split_speakers(by_src_spk, args.val_speakers, args.seed)
    tags = speaker_tag_map(Counter((r["source"], r["speaker"]) for r in kept), args.min_spk_utts) if args.speaker_tags else {}
    tag_of = {k: v for k, v in tags.items()}
    print(f"[split] held-out speakers: {sorted(held)}")
    if args.speaker_tags:
        print(f"[tags] {len(tags)} speaker tag(s)")

    splits: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in kept:
        key = (r["source"], r["speaker"])
        is_val = key in held or rng.random() < args.val_fraction
        splits[("val" if is_val else "train", r["source"])].append(r)
    if args.limit:
        for k in splits:
            rng.shuffle(splits[k]); splits[k] = splits[k][: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for (name, src), rs in sorted(splits.items()):
        buf, shard_idx, skipped, hours = [], 0, 0, 0.0
        safe_src = re.sub(r"[^A-Za-z0-9_.-]+", "_", src)

        def flush():
            torch.save(buf, args.output_dir / f"{name}-{safe_src}-{shard_idx:05d}.pt")

        for r in tqdm(rs, desc=f"{name}/{src}"):
            audio, sr = sf.read(r["audio"], dtype="float32", always_2d=True)
            wav = torch.from_numpy(audio.T).mean(0)
            codes = torch.tensor(codec.encode_waveform(wav, sample_rate=sr), dtype=torch.int16)
            text = r["text"]
            tag = tag_of.get((r["source"], r["speaker"]))
            if tag:
                text = f"[{tag}] {text}"
            ids = [tok.bos_token_id] + tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) + 1 + codes.shape[1] > max_seq or codes.shape[1] < 2:
                skipped += 1
                continue
            buf.append({"text_ids": torch.tensor(ids, dtype=torch.int32), "codes": codes})
            hours += r["dur"] / 3600
            if len(buf) >= args.shard_size:
                flush(); shard_idx += 1; buf = []
        if buf:
            flush(); shard_idx += 1
        stats[f"{name}/{src}"] = {"shards": shard_idx, "hours": round(hours, 2), "skipped": skipped}
        print(f"{name}/{src}: {shard_idx} shard(s), {hours:.1f} h, skipped {skipped}")

    meta = {"num_quantizers": Q, "codebook_size": codec.codebook_size,
            "text_backbone": cfg["model"]["text_backbone"], "mimi_repo": cfg["model"]["mimi_repo"]}
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (args.output_dir / "speakers.json").write_text(
        json.dumps({f"{s}|{k}": t for (s, k), t in tags.items()}, ensure_ascii=False, indent=2))
    (args.output_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print("wrote meta.json, speakers.json, stats.json in", args.output_dir)


if __name__ == "__main__":
    main()
