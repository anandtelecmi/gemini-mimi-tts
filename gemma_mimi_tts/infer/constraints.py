"""Constrained decoding: once generation enters the speech region (right
after <|start_of_speech|>), only allow tokens that are valid audio-codebook
tokens for the *current* position within the frame, plus <|end_of_speech|>
so the model can still choose to stop. This keeps the model from emitting
a stray text token (or a codebook-0 token where codebook-3 was expected)
partway through generation, which would desync the de-interleaving step.
"""

from __future__ import annotations

import torch
from transformers import LogitsProcessor

from gemma_mimi_tts.model.audio_vocab import AudioVocab


class AudioFrameConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, audio_vocab: AudioVocab, prompt_len: int):
        self.audio_vocab = audio_vocab
        self.prompt_len = prompt_len  # index where generation starts (after start_of_speech)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        step = input_ids.shape[1] - self.prompt_len
        if step < 0:
            return scores  # still inside the prompt, don't touch anything

        codebook = step % self.audio_vocab.num_quantizers
        lo, hi = self.audio_vocab.codebook_token_range(codebook)

        mask = torch.full_like(scores, float("-inf"))
        mask[:, lo:hi] = 0
        mask[:, self.audio_vocab.end_of_speech_id] = 0
        return scores + mask