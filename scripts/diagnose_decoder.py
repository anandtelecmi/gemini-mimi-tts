#!/usr/bin/env python3
"""Second-round diagnostic: is the acoustic decoder weak, or is it a precision/sampling effect?

Prints, for the validation set, per-codebook cross-entropy of the decoder next to the
marginal (unigram) entropy of that codebook. If CE is close to the marginal entropy the decoder
barely uses its conditioning for that codebook.

Writes wavs (all use the REAL codebook 0..k-1 and let the decoder sample the rest, with the
backbone teacher-forced on the real frames):
  fp32_k1_*     fp32 model, real c0 only         (compare with the bf16 oracle you already heard)
  bf16_k1_*     bf16 model, same
  fp32_k2/k4/k8 real first k codebooks           (which codebook range makes it robotic?)
  fp32_k1_greedy greedy fine codebooks
  fp32_k1_t0.6 / t1.0   other temperatures
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import yaml

from gemma_mimi_tts.csm.dataset import CodesDataset
from gemma_mimi_tts.csm.model import GemmaCSM, _sample
from gemma_mimi_tts.model.mimi_codec import MIMI_SAMPLE_RATE, MimiCodec


@torch.no_grad()
def teacher_hidden(model, item, dev, dtype):
    text = item["text_ids"].long().unsqueeze(0).to(dev)
    codes = item["codes"].long().unsqueeze(0).to(dev)
    T = codes.shape[2]
    x, mask = model._build_inputs(text, torch.tensor([text.shape[1]]), codes, torch.tensor([T]))
    hidden = model.backbone(inputs_embeds=x.to(dtype), attention_mask=mask).last_hidden_state
    s = text.shape[1]
    return hidden[0, s : s + T], codes[0].T                      # h [T,D], frames [T,Q]


@torch.no_grad()
def sample_rest(model, h, frames, k, temperature, top_k, gen):
    """Keep real codebooks 0..k-1, sample k..Q-1 autoregressively with the decoder."""
    if model.cfg.codec_embeds:
        raise SystemExit('this oracle path does not implement codec_embeds; use model.generate-based outputs')
    Q, V = model.cfg.num_quantizers, model.cfg.codebook_size
    T = h.shape[0]
    offs = torch.arange(k, device=h.device) * V
    xe = model.dec_embed(frames[:, :k] + offs)                    # [T,k,dh]
    x0 = model.dec_in_proj(h).unsqueeze(1)
    x = torch.cat([x0, xe.to(x0.dtype)], dim=1)
    out = [frames[:, i].tolist() for i in range(k)]
    for i in range(k, Q):
        o = model.decoder(inputs_embeds=x).last_hidden_state[:, -1]
        lg = o.float() @ model.ac_heads[i - 1].float()
        ci = torch.tensor([_sample(lg[t], temperature, top_k, 1.0, gen) for t in range(T)], device=h.device)
        out.append(ci.tolist())
        if i < Q - 1:
            nxt = model.dec_embed((ci + i * V).view(T, 1)).to(x.dtype)
            x = torch.cat([x, nxt], dim=1)
    return out


@torch.no_grad()
def per_codebook_ce(model, ds, dev, dtype, n_utts):
    Q, V = model.cfg.num_quantizers, model.cfg.codebook_size
    ce = torch.zeros(Q - 1)
    n = 0
    counts = torch.zeros(Q, V)
    for j in range(min(n_utts, len(ds))):
        h, fr = teacher_hidden(model, ds[j], dev, dtype)
        lg = model.decoder_logits(h, fr).float()                  # [T,Q-1,V]
        for q in range(Q - 1):
            ce[q] += F.cross_entropy(lg[:, q], fr[:, q + 1], reduction="sum").cpu()
        n += fr.shape[0]
        for q in range(Q):
            counts[q] += torch.bincount(fr[:, q].cpu(), minlength=V).float()
    ce /= n
    p = counts / counts.sum(-1, keepdim=True)
    ent = -(p * torch.where(p > 0, p.log(), torch.zeros_like(p))).sum(-1)
    return ce, ent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--val_pattern", default="data/processed/ljspeech_csm/val-*.pt")
    ap.add_argument("--out_dir", type=Path, default=Path("diag2"))
    ap.add_argument("--n_ce", type=int, default=150, help="val utterances for the CE table")
    ap.add_argument("--utt", type=int, default=0)
    ap.add_argument("--temperature_ac", type=float, default=0.8)
    ap.add_argument("--top_k_ac", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dev = args.device
    Q = None
    ds = None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    codec = None

    def save(name, codes):
        wav = codec.decode_codes(codes)
        sf.write(str(args.out_dir / f"{name}.wav"), wav.float().numpy(), MIMI_SAMPLE_RATE)
        print(f"wrote {name}.wav")

    for tag, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        if dev == "cpu" and tag == "bf16":
            continue
        model = GemmaCSM.from_pretrained(args.checkpoint, dtype=dtype).to(dev).eval()
        if ds is None:
            Q = model.cfg.num_quantizers
            ds = CodesDataset(args.val_pattern, num_quantizers=Q)
            codec = MimiCodec.load(cfg["model"]["mimi_repo"], num_quantizers=Q, device=dev)

        ce, ent = per_codebook_ce(model, ds, dev, dtype, args.n_ce)
        print(f"\n[{tag}] per-codebook validation CE (nats) vs marginal entropy")
        print("  codebook   CE    marginal   gain")
        for q in range(Q - 1):
            print(f"  {q+1:>8}  {ce[q]:.2f}   {ent[q+1]:.2f}      {ent[q+1]-ce[q]:+.2f}")
        print(f"  mean CE {ce.mean():.3f}   mean marginal {ent[1:].mean():.3f}\n")

        item = ds[args.utt]
        h, fr = teacher_hidden(model, item, dev, dtype)
        g = torch.Generator(device=dev); g.manual_seed(args.seed)
        ks = (1, 2, 4, 8) if tag == "fp32" else (1,)
        for k in ks:
            save(f"{tag}_k{k}", sample_rest(model, h, fr, k, args.temperature_ac, args.top_k_ac, g))
        if tag == "fp32":
            save("fp32_k1_greedy", sample_rest(model, h, fr, 1, 0.0, 1, g))
            for t in (0.6, 1.0):
                save(f"fp32_k1_t{t}", sample_rest(model, h, fr, 1, t, args.top_k_ac, g))
            save("real", item["codes"].long().tolist())
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
