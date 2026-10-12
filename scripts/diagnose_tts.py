#!/usr/bin/env python3
"""Find out WHERE the robotic sound comes from. Writes several wavs to --out_dir.

  gen_q16 / gen_q8 / gen_q4   one normal generation, decoded with 16 / 8 / 4 codebooks
                              (same codes, just truncated -> are fine codebooks adding buzz?)
  gen_greedyac                generation with greedy fine codebooks (temperature_ac=0)
  oracle_c0_ac{T}             REAL codebook-0 from a val utterance (backbone teacher-forced),
                              fine codebooks sampled by the decoder  -> tests the decoder alone
  real_q16                    the val utterance's real codes decoded (codec ceiling)

How to read it:
  oracle_c0 clean, gen_* robotic   -> prosody/codebook-0 from the backbone is the weak part
  oracle_c0 robotic                -> the acoustic decoder is the weak part
  gen_q8 cleaner than gen_q16      -> sampled deep codebooks add noise: just decode fewer codebooks
"""
from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch
import yaml
from transformers import AutoTokenizer

from gemma_mimi_tts.csm.dataset import CodesDataset
from gemma_mimi_tts.csm.model import GemmaCSM, _sample
from gemma_mimi_tts.model.mimi_codec import MIMI_SAMPLE_RATE, MimiCodec


@torch.no_grad()
def oracle_c0(model, item, device, dtype, temperature, top_k, gen):
    """Teacher-force the backbone with the real frames; sample codebooks 1..Q-1 given the REAL c0."""
    Q, V = model.cfg.num_quantizers, model.cfg.codebook_size
    text = item["text_ids"].long().unsqueeze(0).to(device)
    codes = item["codes"].long().unsqueeze(0).to(device)          # [1,Q,T]
    T = codes.shape[2]
    x, mask = model._build_inputs(text, torch.tensor([text.shape[1]]), codes, torch.tensor([T]))
    hidden = model.backbone(inputs_embeds=x.to(dtype), attention_mask=mask).last_hidden_state
    s = text.shape[1]
    h = hidden[0, s : s + T]                                      # [T,D]
    c0 = codes[0, 0]                                              # [T]
    x_dec = torch.cat([model.dec_in_proj(h).unsqueeze(1), model.dec_embed(c0.view(T, 1))], dim=1)
    out_codes = [c0.tolist()]
    for i in range(1, Q):
        o = model.decoder(inputs_embeds=x_dec).last_hidden_state[:, -1]           # [T,dh]
        lg = (o.float() @ model.ac_heads[i - 1].float())                          # [T,V]
        ci = torch.tensor([_sample(lg[t], temperature, top_k, 1.0, gen) for t in range(T)], device=device)
        out_codes.append(ci.tolist())
        if i < Q - 1:
            nxt = model.dec_embed((ci + i * V).view(T, 1)).to(x_dec.dtype)
            x_dec = torch.cat([x_dec, nxt], dim=1)
    return out_codes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--val_pattern", default="data/processed/ljspeech_csm/val-*.pt")
    ap.add_argument("--text", default="The weather today is quite pleasant, and I think we should go for a walk.")
    ap.add_argument("--out_dir", type=Path, default=Path("diag"))
    ap.add_argument("--n_oracle", type=int, default=2)
    ap.add_argument("--temperature_c0", type=float, default=0.9)
    ap.add_argument("--top_k_c0", type=int, default=50)
    ap.add_argument("--temperature_ac", type=float, default=0.8)
    ap.add_argument("--top_k_ac", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dev = args.device
    dtype = torch.bfloat16 if dev != "cpu" else torch.float32
    model = GemmaCSM.from_pretrained(args.checkpoint, dtype=dtype).to(dev).eval()
    try:
        tok = AutoTokenizer.from_pretrained(str(args.checkpoint))
    except Exception:
        tok = AutoTokenizer.from_pretrained(model.cfg.text_backbone)
    Q = model.cfg.num_quantizers
    codec = MimiCodec.load(cfg["model"]["mimi_repo"], num_quantizers=Q, device=dev)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def save(name, codes):
        wav = codec.decode_codes(codes)
        sf.write(str(args.out_dir / f"{name}.wav"), wav.float().numpy(), MIMI_SAMPLE_RATE)
        print(f"wrote {name}.wav  ({len(codes)} codebooks, {len(codes[0])} frames)")

    ids = torch.tensor([tok.bos_token_id] + tok(args.text, add_special_tokens=False)["input_ids"], device=dev)

    def gen(temp_ac, top_k_ac):
        g = torch.Generator(device="cpu"); g.manual_seed(args.seed)
        return model.generate(ids, max_frames=600, min_frames=10,
                              temperature_c0=args.temperature_c0, top_k_c0=args.top_k_c0,
                              temperature_ac=temp_ac, top_k_ac=top_k_ac, generator=g)

    codes = gen(args.temperature_ac, args.top_k_ac)
    for k in (Q, 8, 4):
        save(f"gen_q{k}", codes[:k])
    save("gen_greedyac", gen(0.0, 1))

    ds = CodesDataset(args.val_pattern, num_quantizers=Q)
    g = torch.Generator(device="cpu"); g.manual_seed(args.seed)
    for j in range(args.n_oracle):
        item = ds[j * 37 % len(ds)]
        print(f"val utt {j}: {tok.decode(item['text_ids'][1:].tolist())[:90]!r}")
        save(f"real_q16_{j}", item["codes"].long().tolist())
        save(f"oracle_c0_ac_{j}", oracle_c0(model, item, dev, dtype, args.temperature_ac, args.top_k_ac, g))


if __name__ == "__main__":
    main()
