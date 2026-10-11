#!/usr/bin/env python3
"""Turns a manifest.csv (relative_wav_path|text) into pre-tokenized training
shards: JSONL files of {"input_ids": [...], "label_ids": [...]}.

Sequence layout per utterance:

    <bos> <text tokens...> <|start_of_speech|> <audio tokens...> <|end_of_speech|> <eos>

`label_ids` mirrors `input_ids` except positions outside the audio region
(and the end-of-speech/eos tokens) are set to -100 when
`data.loss_on_text: false` in the config (the default) -- the model still
sees the text as context, it just isn't penalized for "predicting" it.

Usage:
    python scripts/prepare_dataset.py \\
        --manifest data/raw/ljspeech/manifest.csv \\
        --audio_root data/raw/ljspeech \\
        --output_dir data/processed/ljspeech \\
        --config configs/gemma_mimi_small.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import soundfile as sf
import torch
import yaml
from tqdm import tqdm

from gemma_mimi_tts.model.backbone import load_tokenizer_and_vocab
from gemma_mimi_tts.model.mimi_codec import MimiCodec


def read_manifest(manifest_path: Path) -> list[tuple[str, str]]:
    rows = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in csv.reader(f, delimiter="|"):
            if len(line) < 2:
                continue
            rows.append((line[0], line[1]))
    return rows


def build_example(
    text: str,
    wav_path: Path,
    tokenizer,
    audio_vocab,
    codec: MimiCodec,
    max_seq_len: int,
    loss_on_text: bool,
) -> dict | None:
    # soundfile rather than torchaudio.load: recent torchaudio (2.9+) needs
    # the separate torchcodec package for load/save, soundfile doesn't.
    audio_np, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio_np.T)  # [channels, samples]
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    codes = codec.encode_waveform(waveform[0], sample_rate=sr)
    audio_tokens = audio_vocab.codes_to_token_ids(codes)

    text_ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    input_ids = (
        [bos_id]
        + text_ids
        + [audio_vocab.start_of_speech_id]
        + audio_tokens
        + [audio_vocab.end_of_speech_id, eos_id]
    )
    if len(input_ids) > max_seq_len:
        return None  # skip utterances that don't fit; log/report at call site

    if loss_on_text:
        label_ids = list(input_ids)
    else:
        text_region_len = 1 + len(text_ids)  # bos + text
        label_ids = [-100] * text_region_len + input_ids[text_region_len:]

    return {"input_ids": input_ids, "label_ids": label_ids}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--val_fraction", type=float, default=0.02)
    parser.add_argument("--shard_size", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="cap utterances, for a fast dry run")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # codebook_size isn't known until Mimi loads, so load the codec first
    # and build the (tokenizer, audio_vocab) pair from its actual config.
    codec = MimiCodec.load(
        repo_id=cfg["model"]["mimi_repo"],
        num_quantizers=cfg["model"]["num_quantizers"],
        device=args.device,
    )

    tokenizer, audio_vocab = load_tokenizer_and_vocab(
        model_name=cfg["model"]["text_backbone"],
        num_quantizers=cfg["model"]["num_quantizers"],
        codebook_size=codec.codebook_size,
        start_of_speech=cfg["special_tokens"]["start_of_speech"],
        end_of_speech=cfg["special_tokens"]["end_of_speech"],
    )
    print(
        f"Extended vocab: text={audio_vocab.text_vocab_size} "
        f"audio_offset={audio_vocab.audio_offset} "
        f"total={audio_vocab.total_vocab_size}"
    )

    rows = read_manifest(args.manifest)
    random.Random(cfg["training"]["seed"]).shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]

    n_val = max(1, int(len(rows) * args.val_fraction))
    val_rows, train_rows = rows[:n_val], rows[n_val:]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_rows in [("train", train_rows), ("val", val_rows)]:
        n_skipped = 0
        shard_idx = 0
        buffer: list[dict] = []

        def flush(idx: int) -> None:
            path = args.output_dir / f"{split_name}-{idx:05d}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for ex in buffer:
                    f.write(json.dumps(ex) + "\n")

        for wav_rel, text in tqdm(split_rows, desc=f"encoding {split_name}"):
            example = build_example(
                text=text,
                wav_path=args.audio_root / wav_rel,
                tokenizer=tokenizer,
                audio_vocab=audio_vocab,
                codec=codec,
                max_seq_len=cfg["data"]["max_seq_len"],
                loss_on_text=cfg["data"]["loss_on_text"],
            )
            if example is None:
                n_skipped += 1
                continue
            buffer.append(example)
            if len(buffer) >= args.shard_size:
                flush(shard_idx)
                shard_idx += 1
                buffer = []
        if buffer:
            flush(shard_idx)
            shard_idx += 1

        print(f"{split_name}: wrote {shard_idx} shard(s), skipped {n_skipped} (too long)")

    vocab_meta = {
        "text_vocab_size": audio_vocab.text_vocab_size,
        "num_quantizers": audio_vocab.num_quantizers,
        "codebook_size": audio_vocab.codebook_size,
        "start_of_speech_id": audio_vocab.start_of_speech_id,
        "end_of_speech_id": audio_vocab.end_of_speech_id,
        "audio_offset": audio_vocab.audio_offset,
        "total_vocab_size": audio_vocab.total_vocab_size,
    }
    with open(args.output_dir / "vocab_meta.json", "w") as f:
        json.dump(vocab_meta, f, indent=2)
    print(f"Wrote vocab metadata to {args.output_dir / 'vocab_meta.json'}")


if __name__ == "__main__":
    main()