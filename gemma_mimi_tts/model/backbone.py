"""Loads a Gemma causal-LM and resizes its embedding matrix / LM head so it
has rows for the extended (text + audio-control + audio-code) vocabulary.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from gemma_mimi_tts.model.audio_vocab import AudioVocab, extend_tokenizer_and_build_vocab


def load_tokenizer_and_vocab(
    model_name: str,
    num_quantizers: int,
    codebook_size: int,
    start_of_speech: str,
    end_of_speech: str,
) -> tuple[PreTrainedTokenizerBase, AudioVocab]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    audio_vocab = extend_tokenizer_and_build_vocab(
        tokenizer,
        num_quantizers=num_quantizers,
        codebook_size=codebook_size,
        start_of_speech=start_of_speech,
        end_of_speech=end_of_speech,
    )
    return tokenizer, audio_vocab


def load_backbone(
    model_name: str,
    audio_vocab: AudioVocab,
    dtype: torch.dtype = torch.bfloat16,
    freeze_text_embeddings: bool = False,
) -> PreTrainedModel:
    # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)

    old_size = model.get_input_embeddings().weight.shape[0]
    new_size = audio_vocab.total_vocab_size
    if new_size < old_size:
        raise ValueError(
            f"computed extended vocab ({new_size}) is smaller than the "
            f"model's existing embedding table ({old_size}) -- check "
            "num_quantizers/codebook_size in the config"
        )
    model.resize_token_embeddings(new_size)

    if freeze_text_embeddings:
        # Freeze the original text rows, leave the newly added audio rows
        # (and the control tokens just before them) trainable. Useful once
        # you're fine-tuning on a small single-speaker set and don't want
        # to disturb Gemma's general text understanding.
        embed = model.get_input_embeddings()

        def _mask_text_grad(grad: torch.Tensor) -> torch.Tensor:
            grad = grad.clone()
            grad[:old_size] = 0
            return grad

        embed.weight.register_hook(_mask_text_grad)

    return model