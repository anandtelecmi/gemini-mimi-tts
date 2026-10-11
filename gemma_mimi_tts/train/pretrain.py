"""Training entrypoint, shared by the pretrain and finetune stages.

Usage:
    python -m gemma_mimi_tts.train.pretrain --config configs/gemma_mimi_small.yaml
    python -m gemma_mimi_tts.train.pretrain --config configs/gemma_mimi_small.yaml --stage finetune
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import shutil
from pathlib import Path

import torch
import yaml
from transformers import Trainer, TrainingArguments

from gemma_mimi_tts.data.dataset import JsonlTokenDataset, PadCollator
from gemma_mimi_tts.model.backbone import load_backbone, load_tokenizer_and_vocab


def build_training_arguments(stage_cfg: dict, num_train_examples: int) -> TrainingArguments:
    """Builds TrainingArguments defensively against API drift across
    transformers releases (confirmed on a real install: 'warmup_ratio' is
    not a field at all in some versions -- only 'warmup_steps' is -- and
    'eval_strategy' vs the older 'evaluation_strategy' also varies).
    Rather than pin an exact version, this only passes kwargs the
    installed version actually accepts, converting warmup_ratio into an
    equivalent warmup_steps count when that's the only knob available, and
    prints whatever it had to drop so a real behavior change (e.g. no
    warmup at all) is visible instead of silently swallowed.
    """
    accepted = set(inspect.signature(TrainingArguments.__init__).parameters)

    desired = {
        "output_dir": stage_cfg["output_dir"],
        "per_device_train_batch_size": stage_cfg["per_device_train_batch_size"],
        "gradient_accumulation_steps": stage_cfg["gradient_accumulation_steps"],
        "learning_rate": stage_cfg["learning_rate"],
        "num_train_epochs": stage_cfg["num_train_epochs"],
        "bf16": stage_cfg.get("bf16", True),
        "logging_steps": stage_cfg["logging_steps"],
        "save_steps": stage_cfg["save_steps"],
        "eval_steps": stage_cfg["eval_steps"],
        "save_total_limit": stage_cfg.get("save_total_limit", 3),
        "seed": stage_cfg.get("seed", 42),
        "report_to": [],
    }

    warmup_ratio = stage_cfg.get("warmup_ratio", 0.03)
    if "warmup_ratio" in accepted:
        desired["warmup_ratio"] = warmup_ratio
    elif "warmup_steps" in accepted:
        steps_per_epoch = max(
            1,
            math.ceil(
                num_train_examples
                / (stage_cfg["per_device_train_batch_size"] * stage_cfg["gradient_accumulation_steps"])
            ),
        )
        total_steps = steps_per_epoch * stage_cfg["num_train_epochs"]
        warmup_steps = max(1, round(total_steps * warmup_ratio))
        desired["warmup_steps"] = warmup_steps
        print(
            f"[pretrain] this transformers version has no warmup_ratio field; "
            f"converted to warmup_steps={warmup_steps} (~{warmup_ratio:.0%} of "
            f"an estimated {total_steps} total steps, single-process estimate -- "
            f"actual step count may differ under multi-GPU/distributed training)"
        )
    # else: neither field exists -- no warmup control available, fall through
    # to the "dropped" report below rather than guessing further.

    if "eval_strategy" in accepted:
        desired["eval_strategy"] = "steps"
    elif "evaluation_strategy" in accepted:
        desired["evaluation_strategy"] = "steps"

    kwargs = {k: v for k, v in desired.items() if k in accepted}
    dropped = sorted(set(desired) - set(kwargs))
    if dropped:
        print(
            f"[pretrain] TrainingArguments in this transformers version doesn't "
            f"accept: {dropped} -- continuing without them. If this drops "
            f"something you actually need (warmup control, periodic eval), check "
            f"this version's real field names with: `python -c \"from transformers "
            f"import TrainingArguments; import inspect; print(sorted("
            f"inspect.signature(TrainingArguments.__init__).parameters))\"` and "
            f"tell me the output so I can fix this properly."
        )
    if "eval_strategy" not in kwargs and "evaluation_strategy" not in kwargs:
        print(
            "[pretrain] WARNING: neither eval_strategy nor evaluation_strategy "
            "was accepted -- periodic eval during training is off."
        )

    return TrainingArguments(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["pretrain", "finetune"], default="pretrain")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    stage_cfg = cfg["training"] if args.stage == "pretrain" else {**cfg["training"], **cfg["finetune"]}
    data_cfg = cfg["data"] if args.stage == "pretrain" else {**cfg["data"], **cfg["finetune"]}

    # vocab_meta.json is written by prepare_dataset.py next to the shards;
    # it pins down num_quantizers/codebook_size exactly as they were when
    # the data was encoded, so training can't silently drift from them.
    shard_dir = Path(data_cfg["train_shards"]).parent
    with open(shard_dir / "vocab_meta.json") as f:
        vocab_meta = json.load(f)

    tokenizer, audio_vocab = load_tokenizer_and_vocab(
        model_name=cfg["model"]["text_backbone"],
        num_quantizers=vocab_meta["num_quantizers"],
        codebook_size=vocab_meta["codebook_size"],
        start_of_speech=cfg["special_tokens"]["start_of_speech"],
        end_of_speech=cfg["special_tokens"]["end_of_speech"],
    )
    assert audio_vocab.total_vocab_size == vocab_meta["total_vocab_size"], (
        "tokenizer/vocab built here doesn't match the data's vocab_meta.json -- "
        "did the config change (num_quantizers, backbone) since prepare_dataset.py ran?"
    )

    init_from = stage_cfg.get("init_from", cfg["model"]["text_backbone"])
    model = load_backbone(
        model_name=init_from,
        audio_vocab=audio_vocab,
        dtype=torch.bfloat16 if stage_cfg.get("bf16", True) else torch.float32,
        freeze_text_embeddings=(args.stage == "finetune") and cfg["model"].get("freeze_text_embeddings", False),
    )

    train_ds = JsonlTokenDataset(data_cfg["train_shards"])
    val_ds = JsonlTokenDataset(data_cfg["val_shards"])
    collator = PadCollator(pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)

    training_args = build_training_arguments(stage_cfg, num_train_examples=len(train_ds))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )
    trainer.train()
    final_dir = Path(stage_cfg["output_dir"]) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    shutil.copy(shard_dir / "vocab_meta.json", final_dir / "vocab_meta.json")


if __name__ == "__main__":
    main()