"""Maps Mimi's per-codebook discrete codes onto new token ids appended to
Gemma's text vocabulary, and back.

Layout of the extended vocabulary (ids >= original tokenizer length):

    [ control tokens ]  [ codebook 0 codes ] [ codebook 1 codes ] ... [ codebook Q-1 codes ]

Each codebook gets its own contiguous block of `codebook_size` ids so a
given raw Mimi code (0..codebook_size-1) maps to a different token
depending on which codebook it came from -- this mirrors how Llama-Mimi
and VyvoTTS extend their base LLM's vocabulary.

A single audio "frame" (one time step out of Mimi's 12.5 Hz stream) is
flattened into `num_quantizers` consecutive tokens, codebook 0 first, so
the frame's semantic (most important) codebook stays adjacent to the text
that came before it -- see the `frame_to_token_ids` / `token_ids_to_frames`
pair below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class AudioVocab:
    text_vocab_size: int
    num_quantizers: int
    codebook_size: int
    start_of_speech_id: int
    end_of_speech_id: int

    @property
    def audio_offset(self) -> int:
        """First token id used for audio codes (right after control tokens)."""
        return self.text_vocab_size + 2  # +2 for start/end-of-speech

    @property
    def audio_vocab_size(self) -> int:
        return self.num_quantizers * self.codebook_size

    @property
    def total_vocab_size(self) -> int:
        return self.audio_offset + self.audio_vocab_size

    def codebook_token_range(self, q: int) -> tuple[int, int]:
        """Inclusive-exclusive [start, end) token-id range for codebook q."""
        start = self.audio_offset + q * self.codebook_size
        return start, start + self.codebook_size

    def is_audio_token(self, token_id: int) -> bool:
        return self.audio_offset <= token_id < self.total_vocab_size

    def code_to_token(self, codebook: int, code: int) -> int:
        if not (0 <= codebook < self.num_quantizers):
            raise ValueError(f"codebook {codebook} out of range [0, {self.num_quantizers})")
        if not (0 <= code < self.codebook_size):
            raise ValueError(f"code {code} out of range [0, {self.codebook_size})")
        return self.audio_offset + codebook * self.codebook_size + code

    def token_to_code(self, token_id: int) -> tuple[int, int]:
        """Inverse of `code_to_token`. Raises if token_id isn't an audio token."""
        if not self.is_audio_token(token_id):
            raise ValueError(f"token {token_id} is not an audio token")
        offset = token_id - self.audio_offset
        codebook, code = divmod(offset, self.codebook_size)
        return codebook, code

    def frame_to_token_ids(self, frame_codes: Sequence[int]) -> list[int]:
        """frame_codes[q] = raw Mimi code for codebook q at this time step."""
        if len(frame_codes) != self.num_quantizers:
            raise ValueError(
                f"expected {self.num_quantizers} codes per frame, got {len(frame_codes)}"
            )
        return [self.code_to_token(q, c) for q, c in enumerate(frame_codes)]

    def codes_to_token_ids(self, codes: Sequence[Sequence[int]]) -> list[int]:
        """codes: [num_quantizers][num_frames] (Mimi's native layout) ->
        a single flattened, interleaved token sequence, frame-major."""
        num_frames = len(codes[0])
        assert all(len(c) == num_frames for c in codes), "ragged codebooks"
        tokens: list[int] = []
        for t in range(num_frames):
            tokens.extend(self.frame_to_token_ids([codes[q][t] for q in range(self.num_quantizers)]))
        return tokens

    def token_ids_to_codes(self, token_ids: Sequence[int]) -> list[list[int]]:
        """Inverse of `codes_to_token_ids`. Silently drops a trailing
        incomplete frame (e.g. generation stopped mid-frame)."""
        usable = (len(token_ids) // self.num_quantizers) * self.num_quantizers
        codes: list[list[int]] = [[] for _ in range(self.num_quantizers)]
        for i in range(0, usable, self.num_quantizers):
            for q in range(self.num_quantizers):
                _, code = self.token_to_code(token_ids[i + q])
                codes[q].append(code)
        return codes


def extend_tokenizer_and_build_vocab(
    tokenizer: PreTrainedTokenizerBase,
    num_quantizers: int,
    codebook_size: int,
    start_of_speech: str = "<|start_of_speech|>",
    end_of_speech: str = "<|end_of_speech|>",
) -> AudioVocab:
    """Adds the control tokens as real tokenizer special tokens (so they
    round-trip through encode/decode cleanly), then reserves a contiguous
    block of ids after them for audio codes. The audio-code ids are NOT
    added to the tokenizer's vocab (there's no textual form for them) --
    callers build/parse them directly via the returned AudioVocab, and the
    model's embedding matrix must be resized to `AudioVocab.total_vocab_size`
    to have rows for them.
    """
    text_vocab_size = len(tokenizer)
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": [start_of_speech, end_of_speech]}
    )
    if num_added != 2:
        raise RuntimeError(
            "expected to add exactly 2 new special tokens; got "
            f"{num_added} -- tokenizer may already contain one of them"
        )
    start_id = tokenizer.convert_tokens_to_ids(start_of_speech)
    end_id = tokenizer.convert_tokens_to_ids(end_of_speech)
    assert start_id == text_vocab_size, (start_id, text_vocab_size)
    assert end_id == text_vocab_size + 1, (end_id, text_vocab_size)

    return AudioVocab(
        text_vocab_size=text_vocab_size,
        num_quantizers=num_quantizers,
        codebook_size=codebook_size,
        start_of_speech_id=start_id,
        end_of_speech_id=end_id,
    )