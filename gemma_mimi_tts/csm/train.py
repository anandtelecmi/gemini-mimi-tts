"""Training entrypoint for the CSM-style Gemma model (single GPU).

Usage:
    python -m gemma_mimi_tts.csm.train --config configs/gemma_csm.yaml
    python -m gemma_mimi_tts.csm.train --config configs/gemma_csm.yaml --init_from runs/gemma_csm/best
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from gemma_mimi_tts.csm.dataset import CodesCollator, CodesDataset
from gemma_mimi_tts.csm.model import CSMConfig, GemmaCSM


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device, use_bf16, max_batches) -> dict:
    model.eval()
    sums, n = {}, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            out = model(**to_device(batch, device))
        for k, v in out.items():
            sums[k] = sums.get(k, 0.0) + float(v)
        n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in sums.items()}


def save_ckpt(model, tokenizer, path: Path) -> None:
    model.save_pretrained(path)
    if tokenizer is not None:
        tokenizer.save_pretrained(str(path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--init_from", type=Path, default=None, help="checkpoint dir to start weights from")
    ap.add_argument("--max_steps", type=int, default=None, help="override total optimizer steps (debug)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    mc, dc, tc = cfg["model"], cfg["data"], cfg["training"]
    device = torch.device(args.device)

    random.seed(tc["seed"]); torch.manual_seed(tc["seed"])

    shard_dir = Path(dc["train_shards"]).parent
    meta = json.load(open(shard_dir / "meta.json"))
    Q = mc["num_quantizers"]
    if Q > meta["num_quantizers"]:
        raise SystemExit(f"config wants {Q} codebooks but shards only have {meta['num_quantizers']}; re-run prepare")

    csm_cfg = CSMConfig(
        text_backbone=mc["text_backbone"],
        num_quantizers=Q,
        codebook_size=meta["codebook_size"],
        decoder_hidden_size=mc["decoder"]["hidden_size"],
        decoder_layers=mc["decoder"]["num_layers"],
        decoder_heads=mc["decoder"]["num_heads"],
        decoder_intermediate=mc["decoder"]["intermediate_size"],
        decoder_frame_fraction=mc.get("decoder_frame_fraction", 1.0),
        c0_loss_weight=mc.get("c0_loss_weight", 1.0),
        acoustic_loss_weight=mc.get("acoustic_loss_weight", 1.0),
    )
    model = GemmaCSM(csm_cfg, dtype=torch.float32)
    if args.init_from:
        model.load_weights_from(args.init_from)
        print(f"[train] initialised weights from {args.init_from}")
    model.to(device).train()
    if tc.get("gradient_checkpointing"):
        model.backbone.gradient_checkpointing_enable()
        model.backbone.config.use_cache = False

    try:
        tokenizer = AutoTokenizer.from_pretrained(mc["text_backbone"])
    except Exception as e:  # tokenizer is only needed to ship it with checkpoints
        print(f"[train] WARNING: couldn't load tokenizer ({e}); checkpoints won't include it")
        tokenizer = None
    pad_id = (tokenizer.pad_token_id if tokenizer is not None and tokenizer.pad_token_id is not None else 0)

    train_ds = CodesDataset(dc["train_shards"], num_quantizers=Q, max_frames=dc["max_seq_len"])
    val_ds = CodesDataset(dc["val_shards"], num_quantizers=Q, max_frames=dc["max_seq_len"])
    collate = CodesCollator(pad_token_id=pad_id)
    bs, accum = tc["per_device_train_batch_size"], tc["gradient_accumulation_steps"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True, collate_fn=collate,
                              num_workers=tc.get("num_workers", 2), pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate)
    print(f"[train] {len(train_ds)} train / {len(val_ds)} val utterances, Q={Q}, "
          f"params: backbone {sum(p.numel() for p in model.backbone.parameters())/1e6:.0f}M, "
          f"new {sum(p.numel() for p in model.new_parameters())/1e6:.0f}M")

    # --- optimizer: lower LR for pretrained Gemma, higher for new modules ---
    def groups(params, lr):
        decay = [p for p in params if p.ndim >= 2]
        no_decay = [p for p in params if p.ndim < 2]
        return [{"params": decay, "lr": lr, "weight_decay": tc["weight_decay"]},
                {"params": no_decay, "lr": lr, "weight_decay": 0.0}]

    opt = torch.optim.AdamW(
        groups(list(model.backbone.parameters()), tc["backbone_lr"])
        + groups(model.new_parameters(), tc["new_lr"]),
        betas=(0.9, 0.95),
    )
    steps_per_epoch = max(1, len(train_loader) // accum)
    total_steps = args.max_steps or steps_per_epoch * int(tc["num_train_epochs"])
    warmup = max(1, int(total_steps * tc["warmup_ratio"]))
    min_ratio = tc.get("min_lr_ratio", 0.1)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_bf16 = bool(tc.get("bf16", True)) and device.type == "cuda"

    out_dir = Path(tc["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    best_val = float("inf")
    step, micro, t0 = 0, 0, time.time()
    run = {"loss": 0.0, "loss_c0": 0.0, "loss_ac": 0.0, "c0_acc": 0.0, "n": 0}
    print(f"[train] total optimizer steps: {total_steps} ({steps_per_epoch}/epoch), effective batch {bs*accum}")

    done = False
    while not done:
        for batch in train_loader:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                out = model(**to_device(batch, device))
            (out["loss"] / accum).backward()
            for k in ("loss", "loss_c0", "loss_ac", "c0_acc"):
                run[k] += float(out[k].detach())
            run["n"] += 1
            micro += 1
            if micro % accum:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["max_grad_norm"])
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1

            if step % tc["logging_steps"] == 0:
                n = run["n"]
                print(f"step {step}/{total_steps} loss {run['loss']/n:.4f} c0 {run['loss_c0']/n:.4f} "
                      f"ac {run['loss_ac']/n:.4f} c0_acc {run['c0_acc']/n:.3f} "
                      f"lr {sched.get_last_lr()[-1]:.2e} {(time.time()-t0)/step:.2f}s/step", flush=True)
                run = {k: 0.0 for k in run}; run["n"] = 0

            if step % tc["eval_steps"] == 0 or step == total_steps:
                v = evaluate(model, val_loader, device, use_bf16, tc.get("eval_batches", 50))
                print(f"[eval] step {step} val loss {v['loss']:.4f} c0 {v['loss_c0']:.4f} "
                      f"ac {v['loss_ac']:.4f} c0_acc {v['c0_acc']:.3f}", flush=True)
                if v["loss"] < best_val:
                    best_val = v["loss"]
                    save_ckpt(model, tokenizer, out_dir / "best")
                    print(f"[eval] new best -> {out_dir/'best'}")

            if step % tc["save_steps"] == 0 and step != total_steps:
                p = out_dir / f"checkpoint-{step}"
                save_ckpt(model, tokenizer, p)
                saved.append(p)
                while len(saved) > tc.get("save_total_limit", 3):
                    shutil.rmtree(saved.pop(0), ignore_errors=True)

            if step >= total_steps:
                done = True
                break

    save_ckpt(model, tokenizer, out_dir / "final")
    print(f"[train] done. final -> {out_dir/'final'}  (best val loss {best_val:.4f})")


if __name__ == "__main__":
    main()
