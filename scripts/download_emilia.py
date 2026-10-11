#!/usr/bin/env python3
"""Download a diverse slice of Emilia-YODAS English (CC BY 4.0 subset).

Each English tar is ~1.09 GB and holds roughly 55-60 hours of speech, so
~14 tars ~= 750-800 hours (~15 GB). Tars are picked EVENLY SPACED across the
whole English range (not the first N) so you get many different
videos/speakers. One extra tar from the end of the range is kept apart for
validation, so val speakers don't overlap train.

Needs (gated dataset): accept the terms at
https://huggingface.co/datasets/amphion/Emilia-Dataset and run
`huggingface-cli login` (or export HF_TOKEN).

Usage:
    python scripts/download_emilia.py --out data/raw/emilia_yodas_en --num_tars 14
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = "amphion/Emilia-Dataset"


def pick_tars(all_tars: list[str], num_train: int, num_val: int = 1) -> dict[str, list[str]]:
    """Evenly spaced over the sorted list; the LAST `num_val` picks are val."""
    all_tars = sorted(all_tars)
    total = num_train + num_val
    if total > len(all_tars):
        raise ValueError(f"asked for {total} tars but only {len(all_tars)} exist")
    if total == 1:
        picks = [all_tars[0]]
    else:
        picks = [all_tars[round(i * (len(all_tars) - 1) / (total - 1))] for i in range(total)]
    assert len(set(picks)) == total, "tar selection collided"
    return {"train": picks[:num_train], "val": picks[num_train:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/raw/emilia_yodas_en"))
    ap.add_argument("--num_tars", type=int, default=14, help="train tars (~55-60 h each)")
    ap.add_argument("--num_val_tars", type=int, default=1)
    ap.add_argument("--subset", default="Emilia-YODAS", help="Emilia-YODAS (CC BY 4.0) or Emilia (CC BY-NC 4.0)")
    ap.add_argument("--lang", default="EN")
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    prefix = f"{args.subset}/{args.lang}/"
    files = [f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.startswith(prefix) and f.endswith(".tar")]
    if not files:
        raise SystemExit(f"no tars under {prefix} - check you accepted the dataset terms / are logged in")
    print(f"{len(files)} tars available under {prefix}")

    sel = pick_tars(files, args.num_tars, args.num_val_tars)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "selected.json").write_text(json.dumps(
        {k: [Path(f).name for f in v] for k, v in sel.items()}, indent=2))
    print("train tars:", [Path(f).name for f in sel["train"]])
    print("val tars:  ", [Path(f).name for f in sel["val"]])

    for f in sel["train"] + sel["val"]:
        for attempt in range(1, args.retries + 1):
            try:
                p = hf_hub_download(REPO, f, repo_type="dataset", local_dir=args.out / "_hf")
                dst = args.out / Path(f).name
                if not dst.exists():
                    dst.symlink_to(Path(p).resolve())
                print("ok", dst)
                break
            except Exception as e:  # network blips on 1 GB files are normal
                print(f"[retry {attempt}/{args.retries}] {f}: {e}")
                time.sleep(5)
        else:
            raise SystemExit(f"failed to download {f}")
    print("done. next: scripts/prepare_emilia.py")


if __name__ == "__main__":
    main()
