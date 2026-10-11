"""Dataset for the CSM-style model.

Shards are `.pt` files (torch.save) holding a list of dicts:
    {"text_ids": int32 tensor [Lt]   (bos + text tokens),
     "codes":    int16 tensor [Q, T]  (raw Mimi codes, codebook-major)}
"""

from __future__ import annotations

import glob
from pathlib import Path

import torch
from torch.utils.data import Dataset


class CodesDataset(Dataset):
    def __init__(self, pattern: str, num_quantizers: int | None = None, max_frames: int | None = None):
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"no shards match {pattern!r}")
        self.items: list[dict] = []
        for f in files:
            for ex in torch.load(f, map_location="cpu"):
                codes = ex["codes"]
                if num_quantizers is not None:
                    if codes.shape[0] < num_quantizers:
                        raise ValueError(
                            f"{f}: shard has {codes.shape[0]} codebooks but config asks for {num_quantizers}"
                        )
                    codes = codes[:num_quantizers]
                if max_frames is not None and codes.shape[1] > max_frames:
                    continue
                self.items.append({"text_ids": ex["text_ids"], "codes": codes})
        print(f"[data] loaded {len(self.items)} utterances from {len(files)} shard(s) ({pattern})", flush=True)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]


class CodesCollator:
    def __init__(self, pad_token_id: int = 0):
        self.pad = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        B = len(batch)
        Q = batch[0]["codes"].shape[0]
        text_len = torch.tensor([len(b["text_ids"]) for b in batch])
        n_frames = torch.tensor([b["codes"].shape[1] for b in batch])
        text_ids = torch.full((B, int(text_len.max())), self.pad, dtype=torch.long)
        codes = torch.zeros((B, Q, int(n_frames.max())), dtype=torch.long)
        for i, b in enumerate(batch):
            text_ids[i, : text_len[i]] = b["text_ids"].long()
            codes[i, :, : n_frames[i]] = b["codes"].long()
        return {"text_ids": text_ids, "text_len": text_len, "codes": codes, "n_frames": n_frames}
