#!/usr/bin/env python3
"""text -> wav CLI.

Usage:
    python -m gemma_mimi_tts.infer.generate \\
        --config configs/gemma_mimi_small.yaml \\
        --checkpoint runs/gemma_mimi_small/final \\
        --text "Hello, this is a test." \\
        --out out.wav
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from gemma_mimi_tts.infer.constraints import AudioFrameConstrainedLogitsProcessor
from gemma_mimi_tts.model.audio_vocab import AudioVocab
from gemma_mimi_tts.model.mimi_codec import MIMI_SAMPLE_RATE, MimiCodec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--min_new_tokens",
        type=int,
        default=0,
        help=(
            "force at least this many audio tokens before end_of_speech is "
            "allowed to end generation. Useful for a plumbing sanity check "
            "on a barely-trained checkpoint, which otherwise tends to emit "
            "end_of_speech almost immediately -- has nothing to do with "
            "real speech quality, just lets you confirm multi-frame decode "
            "doesn't crash. e.g. --min_new_tokens 80 forces 10 frames "
            "(80 / num_quantizers=8)."
        ),
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    vocab_meta_path = args.checkpoint / "vocab_meta.json"
    if not vocab_meta_path.exists():
        # fall back to wherever the training shards' metadata lives
        vocab_meta_path = Path(cfg["data"]["train_shards"]).parent / "vocab_meta.json"
    with open(vocab_meta_path) as f:
        vocab_meta = json.load(f)

    audio_vocab = AudioVocab(
        text_vocab_size=vocab_meta["text_vocab_size"],
        num_quantizers=vocab_meta["num_quantizers"],
        codebook_size=vocab_meta["codebook_size"],
        start_of_speech_id=vocab_meta["start_of_speech_id"],
        end_of_speech_id=vocab_meta["end_of_speech_id"],
    )

    dtype = torch.bfloat16 if cfg.get("training", {}).get("bf16", True) else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint))
    model = AutoModelForCausalLM.from_pretrained(str(args.checkpoint), dtype=dtype)
    model.to(args.device).eval()

    codec = MimiCodec.load(
        repo_id=cfg["model"]["mimi_repo"],
        num_quantizers=audio_vocab.num_quantizers,
        device=args.device,
    )

    text_ids = tokenizer(args.text, add_special_tokens=False)["input_ids"]
    prompt_ids = [tokenizer.bos_token_id] + text_ids + [audio_vocab.start_of_speech_id]
    prompt = torch.tensor([prompt_ids], device=args.device)

    inf_cfg = cfg.get("inference", {})
    logits_processor = LogitsProcessorList(
        [AudioFrameConstrainedLogitsProcessor(audio_vocab, prompt_len=prompt.shape[1])]
    )

    with torch.no_grad():
        generated = model.generate(
            prompt,
            max_new_tokens=inf_cfg.get("max_new_audio_tokens", 2048),
            do_sample=True,
            temperature=inf_cfg.get("temperature", 0.7),
            top_p=inf_cfg.get("top_p", 0.9),
            min_new_tokens=args.min_new_tokens,
            logits_processor=logits_processor,
            eos_token_id=audio_vocab.end_of_speech_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    new_tokens = generated[0, prompt.shape[1] :].tolist()
    if new_tokens and new_tokens[-1] == audio_vocab.end_of_speech_id:
        new_tokens = new_tokens[:-1]
    audio_tokens = [t for t in new_tokens if audio_vocab.is_audio_token(t)]

    if not audio_tokens:
        raise RuntimeError(
            "model generated no audio tokens -- it likely hasn't been trained "
            "yet, or emitted end_of_speech immediately"
        )

    codes = audio_vocab.token_ids_to_codes(audio_tokens)
    waveform = codec.decode_codes(codes)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.out), waveform.numpy(), MIMI_SAMPLE_RATE)
    print(f"Wrote {args.out} ({waveform.shape[0] / MIMI_SAMPLE_RATE:.2f}s at {MIMI_SAMPLE_RATE} Hz)")


if __name__ == "__main__":
    main()