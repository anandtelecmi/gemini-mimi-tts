"""Tests for the optional codec-aware path (codec_embeds=True), tiny random model, CPU."""
import sys, tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import Gemma3TextConfig, Gemma3TextModel

from gemma_mimi_tts.csm.dataset import CodesCollator
from gemma_mimi_tts.csm.model import CSMConfig, GemmaCSM

V, Q, DC = 32, 4, 8


def make(codec: bool, seed=0) -> GemmaCSM:
    torch.manual_seed(seed)
    bcfg = Gemma3TextConfig(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=16, sliding_window=8,
                            max_position_embeddings=512)
    cfg = CSMConfig(num_quantizers=Q, codebook_size=V, decoder_hidden_size=32, decoder_layers=2,
                    decoder_heads=4, decoder_intermediate=64, codec_embeds=codec, codec_dim=DC)
    m = GemmaCSM(cfg, backbone=Gemma3TextModel(bcfg))
    if codec:
        m.set_codec_table(torch.randn(Q, V, DC))
    return m.eval()


def batch():
    g = torch.Generator().manual_seed(1)
    ex = [{"text_ids": torch.randint(1, 100, (5,), generator=g, dtype=torch.int32),
           "codes": torch.randint(0, V, (Q, 7), generator=g, dtype=torch.int16)}]
    return CodesCollator()(ex)


def test_zero_init_matches_plain_model():
    plain = make(False)
    geo = make(True)
    res = geo.load_state_dict(plain.state_dict(), strict=False)
    assert all(k.startswith(("codec_table", "cg_", "ac_geo")) for k in res.missing_keys), res.missing_keys
    b = batch()
    with torch.no_grad():
        a = plain(**b)["loss"]; c = geo(**b)["loss"]
    assert torch.allclose(a, c, atol=1e-5), (a, c)


def test_gradients_reach_new_weights():
    m = make(True).train()
    m(**batch())["loss"].backward()
    for n in ("cg_back_sem", "cg_back_ac", "cg_dec_sem", "cg_dec_ac"):
        assert getattr(m, n).weight.grad.abs().sum() > 0, n
    assert m.ac_geo.grad.abs().sum() > 0


def test_incremental_sampling_matches_teacher_forced():
    """sample_fine (inference) must agree with decoder_logits (training) once the geo weights are non-zero."""
    m = make(True)
    with torch.no_grad():
        for p in (m.cg_back_sem.weight, m.cg_back_ac.weight, m.cg_dec_sem.weight, m.cg_dec_ac.weight, m.ac_geo):
            p.normal_(0, 0.5)
        h = torch.randn(1, 64)
        codes = m.sample_fine(h, c0=3, temperature=0.0, top_k=1)
        lg = m.decoder_logits(h, torch.tensor([codes]))[0]            # [Q-1,V]
    assert [int(lg[i].argmax()) for i in range(Q - 1)] == codes[1:], (codes, lg.argmax(-1))


def test_cache_equals_nocache_and_roundtrip():
    m = make(True)
    with torch.no_grad():
        for p in (m.cg_back_sem.weight, m.cg_back_ac.weight, m.cg_dec_sem.weight, m.cg_dec_ac.weight, m.ac_geo):
            p.normal_(0, 0.3)
    ids = torch.tensor([1, 5, 7, 9])
    a = m.generate(ids, max_frames=6, min_frames=6, temperature_c0=0, top_k_c0=1, temperature_ac=0, top_k_ac=1)
    b = m.generate(ids, max_frames=6, min_frames=6, temperature_c0=0, top_k_c0=1, temperature_ac=0, top_k_ac=1,
                   use_cache=False)
    assert a == b
    with tempfile.TemporaryDirectory() as d:
        m.save_pretrained(d)
        m2 = GemmaCSM.from_pretrained(d).eval()
    assert m2.cfg.codec_embeds and torch.equal(m2.codec_table, m.codec_table)
    c = m2.generate(ids, max_frames=6, min_frames=6, temperature_c0=0, top_k_c0=1, temperature_ac=0, top_k_ac=1)
    assert a == c


def test_freeze_for_decoder_training():
    m = make(True).train()
    ref = m(**batch())["loss"].detach()
    m.freeze_for_decoder_training()
    out = m(**batch())
    assert torch.allclose(out["loss"].detach(), ref, atol=1e-5)
    out["loss"].backward()
    for mod in (m.backbone, m.audio_embed, m.c0_head, m.cg_back_sem, m.cg_back_ac):
        assert all(p.grad is None for p in mod.parameters())
    assert m.sos_embed.grad is None
    assert m.dec_embed.weight.grad.abs().sum() > 0
    assert m.ac_heads.grad.abs().sum() > 0
    assert m.dec_in_proj.weight.grad.abs().sum() > 0


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
