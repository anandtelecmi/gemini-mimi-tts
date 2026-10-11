"""CPU smoke tests for the CSM-style model using a tiny random Gemma3 backbone
(no downloads). Checks shapes, loss at init, padding invariance, KV-cache vs
no-cache greedy decoding, and save/load round-trip.
"""

import math
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import Gemma3TextConfig, Gemma3TextModel

from gemma_mimi_tts.csm.dataset import CodesCollator
from gemma_mimi_tts.csm.model import CSMConfig, GemmaCSM

V, Q = 32, 4


def make_model(fraction=1.0) -> GemmaCSM:
    torch.manual_seed(0)
    bcfg = Gemma3TextConfig(
        vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, sliding_window=8,
        max_position_embeddings=512,
    )
    cfg = CSMConfig(num_quantizers=Q, codebook_size=V, decoder_hidden_size=32, decoder_layers=2,
                    decoder_heads=4, decoder_intermediate=64, decoder_frame_fraction=fraction)
    return GemmaCSM(cfg, backbone=Gemma3TextModel(bcfg)).eval()


def make_examples():
    g = torch.Generator().manual_seed(1)
    return [
        {"text_ids": torch.randint(1, 100, (5,), generator=g, dtype=torch.int32),
         "codes": torch.randint(0, V, (Q, 7), generator=g, dtype=torch.int16)},
        {"text_ids": torch.randint(1, 100, (9,), generator=g, dtype=torch.int32),
         "codes": torch.randint(0, V, (Q, 12), generator=g, dtype=torch.int16)},
    ]


def test_initial_loss_is_near_uniform_and_backward_works():
    m = make_model().train()
    out = m(**CodesCollator()(make_examples()))
    assert torch.isfinite(out["loss"])
    assert abs(out["loss_ac"].item() - math.log(V)) < 1.0
    out["loss"].backward()
    assert m.c0_head.weight.grad is not None and m.ac_heads.grad is not None
    assert m.audio_embed.weight.grad is not None
    assert next(m.backbone.parameters()).grad is not None


def test_padding_invariance():
    m = make_model()
    ex = make_examples()
    with torch.no_grad():
        alone = m(**CodesCollator()([ex[0]]))
        batched = m(**CodesCollator()(ex))
        other = m(**CodesCollator()([ex[1]]))
    # batch loss is a mean over all positions, so check per-example via weighted combo
    n0 = ex[0]["codes"].shape[1]; n1 = ex[1]["codes"].shape[1]
    c0 = (alone["loss_c0"] * (n0 + 1) + other["loss_c0"] * (n1 + 1)) / (n0 + n1 + 2)
    ac = (alone["loss_ac"] * n0 + other["loss_ac"] * n1) / (n0 + n1)
    assert torch.allclose(batched["loss_c0"], c0, atol=1e-4), (batched["loss_c0"], c0)
    assert torch.allclose(batched["loss_ac"], ac, atol=1e-4), (batched["loss_ac"], ac)


def test_cache_matches_no_cache_greedy():
    m = make_model()
    ids = torch.tensor([1, 5, 9, 12, 3])
    kw = dict(max_frames=14, min_frames=14, temperature_c0=0, temperature_ac=0)
    a = m.generate(ids, use_cache=True, **kw)
    b = m.generate(ids, use_cache=False, **kw)
    assert len(a) == Q and len(a[0]) == 14
    assert a == b


def test_generation_matches_teacher_forced_logits():
    """The decoder used step-by-step in generate() must agree with teacher forcing."""
    m = make_model()
    ids = torch.tensor([1, 5, 9, 12, 3])
    codes = m.generate(ids, max_frames=6, min_frames=6, temperature_c0=0, temperature_ac=0, use_cache=False)
    t = len(codes[0])
    c = torch.tensor(codes)  # [Q,T]
    coll = CodesCollator()([{"text_ids": ids.int(), "codes": c.to(torch.int16)}])
    with torch.no_grad():
        x, mask = m._build_inputs(coll["text_ids"], coll["text_len"], coll["codes"], coll["n_frames"])
        h = m.backbone(inputs_embeds=x, attention_mask=mask).last_hidden_state[0]
        s = len(ids)
        logits0 = m.c0_head(h[s : s + t])
        assert logits0[:, :V].argmax(-1).tolist() == codes[0]
        lg = m.decoder_logits(h[s : s + t], c.T)  # [T,Q-1,V]
        assert lg.argmax(-1).T.tolist() == codes[1:]


def test_save_load_roundtrip():
    m = make_model()
    batch = CodesCollator()(make_examples())
    with tempfile.TemporaryDirectory() as d:
        m.save_pretrained(d)
        m2 = GemmaCSM.from_pretrained(d).eval()
    with torch.no_grad():
        a, b = m(**batch), m2(**batch)
    assert torch.allclose(a["loss"], b["loss"], atol=1e-5)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)} tests passed")
