"""IndicVoices-R (HF parquet, e.g. SPRINGLab/IndicVoices-R_Hindi) -> 24 kHz wavs + data_prep-style JSONL.

Output names follow the prepare_jsonl_dataset.py convention so real speaker ids become tags:
    indicvoices_r__hindi__<speaker_id>__<idx>.wav
Then:  python scripts/prepare_jsonl_dataset.py --jsonl <out>/hindi_metadata.jsonl --only_source indicvoices_r \
         --output_dir data/processed/hindi_csm --config configs/gemma_csm_hindi_iv.yaml --speaker_tags --val_speakers 0 --val_fraction 0.01
"""
import argparse, io, json, math, os, re
from collections import Counter
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def clean_text(t):
    t = re.sub(r"\[[^\]]*\]|<[^>]*>", " ", t or "")   # drop [noise]-style tags
    return re.sub(r"\s+", " ", t).strip()


def keep(row, a):
    d = row.get("duration") or 0
    if not (a.min_dur <= d <= a.max_dur): return False
    if row.get("snr") is not None and row["snr"] < a.min_snr: return False
    if row.get("cer") is not None and row["cer"] > a.max_cer: return False
    return True


def to_24k(wav, sr):
    if wav.ndim > 1: wav = wav.mean(1)
    if sr != 24000:
        g = math.gcd(sr, 24000)
        wav = resample_poly(wav, 24000 // g, sr // g)
    pk = np.abs(wav).max()
    if pk > 0.99: wav = wav * (0.99 / pk)
    return wav.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="SPRINGLab/IndicVoices-R_Hindi")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--lang", default="hindi")
    ap.add_argument("--text_field", default="normalized", choices=["normalized", "verbatim", "text"])
    ap.add_argument("--min_dur", type=float, default=1.5)
    ap.add_argument("--max_dur", type=float, default=25.0)
    ap.add_argument("--min_snr", type=float, default=25.0)
    ap.add_argument("--max_cer", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report_only", action="store_true")
    a = ap.parse_args()
    from datasets import load_dataset, Audio
    ds = load_dataset(a.dataset, split="train").cast_column("audio", Audio(decode=False))
    wdir = os.path.join(a.out_dir, "wavs"); os.makedirs(wdir, exist_ok=True)
    meta = [c for c in ds.column_names if c != "audio"]
    spk, hrs, n, seen = Counter(), 0.0, 0, 0
    jl = open(os.path.join(a.out_dir, f"{a.lang}_metadata.jsonl"), "w", encoding="utf-8")
    for i, row in enumerate(ds.select_columns(meta + ["audio"])):
        seen += 1
        if a.limit and n >= a.limit: break
        text = clean_text(row.get(a.text_field) or row.get("text"))
        if not text or not keep(row, a): continue
        sid = re.sub(r"\W+", "", str(row["speaker_id"]))
        spk[sid] += 1; hrs += row["duration"] / 3600; n += 1
        if a.report_only: continue
        au = row["audio"]
        wav, sr = sf.read(io.BytesIO(au["bytes"]), dtype="float32") if au.get("bytes") else sf.read(au["path"], dtype="float32")
        name = f"indicvoices_r__{a.lang}__{sid}__{i:06d}.wav"
        sf.write(os.path.join(wdir, name), to_24k(wav, sr), 24000, subtype="PCM_16")
        jl.write(json.dumps({"audio": os.path.abspath(os.path.join(wdir, name)), "text": text}, ensure_ascii=False) + "\n")
    print(f"seen {seen}  kept {n}  {hrs:.1f} h  speakers {len(spk)}  top5 {spk.most_common(5)}")


if __name__ == "__main__":
    main()
