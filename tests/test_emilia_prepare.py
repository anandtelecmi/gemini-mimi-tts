"""No-download tests for the Emilia scripts: tar streaming, mp3 decode,
filters, shard writing and tar selection, using a stub codec/tokenizer."""

import argparse
import io
import json
import random
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import prepare_emilia as pe
from download_emilia import pick_tars


class StubCodec:
    """12.5 frames/s, Q codebooks, deterministic codes."""
    def __init__(self, Q=4):
        self.Q = Q
    def encode_waveform(self, wav, sample_rate):
        T = int(len(wav) / sample_rate * 12.5)
        return [[(q * 7 + t) % 2048 for t in range(T)] for q in range(self.Q)]


class StubTok:
    bos_token_id = 2
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [3 + (ord(c) % 50) for c in text]}


def mp3_bytes(seconds, sr=24000):
    t = np.arange(int(seconds * sr)) / sr
    x = (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32")
    b = io.BytesIO()
    sf.write(b, x, sr, format="MP3")
    return b.getvalue()


def add(tf, name, data):
    ti = tarfile.TarInfo(name); ti.size = len(data); tf.addfile(ti, io.BytesIO(data))


def make_tar(path):
    with tarfile.open(path, "w") as tf:
        # (key, seconds, text, dnsmos)  - json BEFORE mp3 for some, after for others
        rows = [
            ("EN_B0_S0_W0", 5.0, "hello there how are you doing today", 3.4),   # ok
            ("EN_B0_S0_W1", 1.0, "too short clip", 3.4),                           # duration
            ("EN_B0_S0_W2", 5.0, "x", 3.4),                                        # chars/sec too low
            ("EN_B0_S0_W3", 6.0, "this one has lower quality audio really", 2.1),  # dnsmos (if filtered)
            ("EN_B0_S0_W4", 4.0, "another good sentence for the test", 3.3),       # ok
        ]
        for i, (k, s, text, d) in enumerate(rows):
            js = json.dumps({"id": k, "text": text, "duration": s, "dnsmos": d, "language": "en"}).encode()
            mp = mp3_bytes(s)
            if i % 2 == 0:
                add(tf, f"{k}.json", js); add(tf, f"{k}.mp3", mp)
            else:
                add(tf, f"{k}.mp3", mp); add(tf, f"{k}.json", js)


def args(**kw):
    d = dict(split="train", shard_size=2, min_dur=2.0, max_dur=25.0, min_dnsmos=0.0,
             min_cps=4.0, max_cps=30.0, keep_prob=1.0, max_utts=0)
    d.update(kw)
    return argparse.Namespace(**d)


def test_tar_pairs_and_decode():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.tar"; make_tar(p)
        pairs = list(pe.iter_tar_pairs(p))
        assert len(pairs) == 5
        wav, sr = pe.decode_audio(pairs[0][1][".mp3"])
        assert sr == 24000 and abs(len(wav) / sr - 5.0) < 0.2


def test_process_tar_filters_and_shards():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d); p = d / "a.tar"; make_tar(p)
        out = d / "out"; out.mkdir()
        st = pe.process_tar(p, 0, args(), StubCodec(4), StubTok(), 4, 1024, out, random.Random(0))
        assert st["utts"] == 3, st            # W0, W3 (dnsmos not filtered by default), W4
        assert st["skipped"]["duration"] == 1 and st["skipped"]["rate"] == 1, st
        shards = sorted(out.glob("train-t000-*.pt"))
        assert len(shards) == 2               # shard_size=2 -> 2 + 1
        ex = torch.load(shards[0])[0]
        assert ex["codes"].dtype == torch.int16 and ex["codes"].shape[0] == 4
        assert ex["text_ids"][0] == 2         # bos first
        assert abs(ex["codes"].shape[1] - 5.0 * 12.5) <= 3

        st2 = pe.process_tar(p, 0, args(min_dnsmos=3.0), StubCodec(4), StubTok(), 4, 1024, out, random.Random(0))
        assert st2["utts"] == 2 and st2["skipped"]["dnsmos"] == 1
        assert len(list(out.glob("train-t000-*.pt"))) == 1   # old shards of this tar were cleaned


def test_seq_len_filter_and_max_utts():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d); p = d / "a.tar"; make_tar(p); out = d / "o"; out.mkdir()
        st = pe.process_tar(p, 0, args(), StubCodec(4), StubTok(), 4, 40, out, random.Random(0))
        assert st["utts"] == 0 and st["skipped"]["too_long"] >= 1
        st = pe.process_tar(p, 0, args(max_utts=1), StubCodec(4), StubTok(), 4, 1024, out, random.Random(0))
        assert st["utts"] == 1


def test_pick_tars_even_and_disjoint():
    tars = [f"Emilia-YODAS/EN/EN-B{i:06d}.tar" for i in range(1600)]
    sel = pick_tars(tars, 14, 1)
    assert len(sel["train"]) == 14 and len(sel["val"]) == 1
    assert sel["val"][0] == tars[-1] and sel["train"][0] == tars[0]
    assert not set(sel["train"]) & set(sel["val"])
    idx = [tars.index(t) for t in sel["train"] + sel["val"]]
    gaps = np.diff(idx)
    assert gaps.max() - gaps.min() <= 1       # evenly spaced


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print("PASS:", t.__name__)
    print(f"\n{len(tests)} tests passed")
