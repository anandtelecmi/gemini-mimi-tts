"""Pure-logic tests for the codebook <-> token-id mapping. No GPU, no
network, no transformers model download -- just checks the math that every
other stage in the pipeline depends on being right.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gemma_mimi_tts.model.audio_vocab import AudioVocab


def make_vocab(text_vocab_size=100, num_quantizers=8, codebook_size=2048) -> AudioVocab:
    return AudioVocab(
        text_vocab_size=text_vocab_size,
        num_quantizers=num_quantizers,
        codebook_size=codebook_size,
        start_of_speech_id=text_vocab_size,
        end_of_speech_id=text_vocab_size + 1,
    )


def test_offsets_and_total_size():
    v = make_vocab()
    assert v.audio_offset == 102
    assert v.audio_vocab_size == 8 * 2048
    assert v.total_vocab_size == 102 + 8 * 2048


def test_codebook_ranges_dont_overlap():
    v = make_vocab()
    ranges = [v.codebook_token_range(q) for q in range(v.num_quantizers)]
    for i in range(len(ranges) - 1):
        assert ranges[i][1] == ranges[i + 1][0], "ranges must be contiguous, no gaps/overlaps"
    assert ranges[0][0] == v.audio_offset
    assert ranges[-1][1] == v.total_vocab_size


def test_code_token_roundtrip():
    v = make_vocab()
    rng = random.Random(0)
    for _ in range(1000):
        q = rng.randrange(v.num_quantizers)
        c = rng.randrange(v.codebook_size)
        token = v.code_to_token(q, c)
        assert v.is_audio_token(token)
        q2, c2 = v.token_to_code(token)
        assert (q, c) == (q2, c2)


def test_frame_and_full_sequence_roundtrip():
    v = make_vocab(num_quantizers=4, codebook_size=16)
    rng = random.Random(1)
    num_frames = 5
    codes = [[rng.randrange(v.codebook_size) for _ in range(num_frames)] for _ in range(v.num_quantizers)]

    tokens = v.codes_to_token_ids(codes)
    assert len(tokens) == v.num_quantizers * num_frames

    # frame-major ordering: first num_quantizers tokens are frame 0
    frame0_tokens = tokens[: v.num_quantizers]
    expected_frame0 = v.frame_to_token_ids([codes[q][0] for q in range(v.num_quantizers)])
    assert frame0_tokens == expected_frame0

    recovered = v.token_ids_to_codes(tokens)
    assert recovered == codes


def test_partial_trailing_frame_is_dropped():
    v = make_vocab(num_quantizers=4, codebook_size=16)
    codes = [[1, 2], [3, 4], [5, 6], [7, 8]]
    tokens = v.codes_to_token_ids(codes)
    # simulate generation stopping mid-frame (2 extra tokens, not a full frame)
    truncated = tokens[:-2]
    recovered = v.token_ids_to_codes(truncated)
    assert len(recovered[0]) == 1  # only the complete first frame survives


def test_is_audio_token_rejects_text_and_control_tokens():
    v = make_vocab()
    assert not v.is_audio_token(0)
    assert not v.is_audio_token(v.text_vocab_size - 1)
    assert not v.is_audio_token(v.start_of_speech_id)
    assert not v.is_audio_token(v.end_of_speech_id)
    assert not v.is_audio_token(v.total_vocab_size)  # one past the end
    assert v.is_audio_token(v.audio_offset)
    assert v.is_audio_token(v.total_vocab_size - 1)


def test_out_of_range_raises():
    v = make_vocab(num_quantizers=4, codebook_size=16)
    try:
        v.code_to_token(4, 0)  # codebook out of range
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        v.code_to_token(0, 16)  # code out of range
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        v.token_to_code(v.text_vocab_size - 1)  # a text token, not audio
        assert False, "expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)} tests passed")