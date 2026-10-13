#!/usr/bin/env python3
"""text -> wav for the CSM-style model.

Usage:
    python -m gemma_mimi_tts.csm.generate \\
        --config configs/gemma_csm.yaml --checkpoint runs/gemma_csm/best \\
        --text "Hello, this is a test." --out out_csm.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch
import yaml
from transformers import AutoTokenizer

from gemma_mimi_tts.csm.model import GemmaCSM
from gemma_mimi_tts.model.mimi_codec import MIMI_SAMPLE_RATE, MimiCodec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--min_frames", type=int, default=0)
    ap.add_argument("--temperature_c0", type=float, default=None)
    ap.add_argument("--top_k_c0", type=int, default=None)
    ap.add_argument("--temperature_ac", type=float, default=None)
    ap.add_argument("--top_k_ac", type=int, default=None)
    ap.add_argument("--no_cache", action="store_true", help="disable KV cache (slow, but a safe fallback)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--deep_from", type=int, default=8, help="codebooks >= this use the *_deep sampling settings")
    ap.add_argument("--temperature_deep", type=float, default=None, help="e.g. 0 = greedy for the deep codebooks")
    ap.add_argument("--top_k_deep", type=int, default=None)
    ap.add_argument("--decode_q", type=int, nargs="+", default=None,
                    help="decode the SAME generated codes with only the first K codebooks (writes <out>_q<K>.wav each)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    inf = cfg.get("inference", {})
    dtype = torch.bfloat16 if args.dtype == "bf16" and args.device != "cpu" else torch.float32

    model = GemmaCSM.from_pretrained(args.checkpoint, dtype=dtype).to(args.device).eval()
    try:
        tok = AutoTokenizer.from_pretrained(str(args.checkpoint))
    except Exception:
        tok = AutoTokenizer.from_pretrained(model.cfg.text_backbone)
    codec = MimiCodec.load(cfg["model"]["mimi_repo"], num_quantizers=model.cfg.num_quantizers, device=args.device)

    ids = [tok.bos_token_id] + tok(args.text, add_special_tokens=False)["input_ids"]
    gen = torch.Generator(device=args.device)
    if args.seed is not None:
        gen.manual_seed(args.seed)
    codes = model.generate(
        torch.tensor(ids, device=args.device),
        max_frames=args.max_frames or inf.get("max_frames", 600),
        min_frames=args.min_frames,
        temperature_c0=args.temperature_c0 if args.temperature_c0 is not None else inf.get("temperature_c0", 0.9),
        top_k_c0=args.top_k_c0 if args.top_k_c0 is not None else inf.get("top_k_c0", 50),
        temperature_ac=args.temperature_ac if args.temperature_ac is not None else inf.get("temperature_ac", 0.7),
        top_k_ac=args.top_k_ac if args.top_k_ac is not None else inf.get("top_k_ac", 20),
        use_cache=not args.no_cache,
        generator=gen if args.seed is not None else None,
        deep_from=args.deep_from,
        temperature_deep=args.temperature_deep,
        top_k_deep=args.top_k_deep,
    )
    if not codes[0]:
        raise RuntimeError("model produced no frames (stopped immediately) - try --min_frames 20 or train longer")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.decode_q:
        for k in args.decode_q:
            wav = codec.decode_codes(codes[:k])
            p = args.out.with_name(f"{args.out.stem}_q{k}{args.out.suffix}")
            sf.write(str(p), wav.float().numpy(), MIMI_SAMPLE_RATE)
            print(f"Wrote {p} ({len(codes[0])} frames, first {k} codebooks)")
    else:
        wav = codec.decode_codes(codes)
        sf.write(str(args.out), wav.float().numpy(), MIMI_SAMPLE_RATE)
        print(f"Wrote {args.out} ({len(codes[0])} frames, {wav.shape[0]/MIMI_SAMPLE_RATE:.2f}s)")


if __name__ == "__main__":
    main()
