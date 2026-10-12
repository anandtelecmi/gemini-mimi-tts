"""CSM-style TTS model with a Gemma backbone.

Architecture (same idea as Sesame's CSM, with Gemma as the big model):

    text tokens ──┐
    <SOS> vector ─┼─> Gemma backbone (causal) ──> hidden state per position
    audio frames ─┘        |
       (one position       ├─> c0_head   -> codebook-0 code of the NEXT frame
        per Mimi frame,    |               (+1 extra class = end-of-speech)
        embedding = sum    └─> h  ──> small Llama "audio decoder" ──> codebooks 1..Q-1
        of the Q code                    of that same frame, autoregressive
        embeddings)                      over the codebook axis

So the backbone sequence is only ~12.5 steps per second of audio (instead of
12.5 * Q for the flattened-RVQ approach), and the Q-1 fine codebooks are
predicted by a tiny dedicated decoder conditioned on the backbone state.

Sequence layout fed to the backbone, per utterance:

    [bos, text...]  [SOS]  [frame_0] [frame_1] ... [frame_{T-1}]

    hidden at SOS        -> predicts c0 of frame_0
    hidden at frame_t    -> predicts c0 of frame_{t+1}   (t < T-1)
    hidden at frame_{T-1}-> predicts END_OF_SPEECH
    The decoder for frame_t is conditioned on the hidden state that predicted
    its c0 (i.e. the hidden at SOS for t=0, at frame_{t-1} otherwise).

No new text tokens are added to the Gemma tokenizer.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, LlamaConfig, LlamaModel


@dataclass
class CSMConfig:
    text_backbone: str = "google/gemma-3-1b-pt"
    num_quantizers: int = 16
    codebook_size: int = 2048
    decoder_hidden_size: int = 1024
    decoder_layers: int = 6
    decoder_heads: int = 8
    decoder_intermediate: int = 3072
    # fraction of frames the audio decoder is trained on each step (CSM uses
    # 1/16 to save memory; with <=16 codebooks you can usually afford 1.0)
    decoder_frame_fraction: float = 1.0
    c0_loss_weight: float = 1.0
    acoustic_loss_weight: float = 1.0
    # Codec-aware inputs/outputs: feed Mimi's own codeword vectors (and the running sum of the
    # acoustic residual chain) to the backbone and the decoder, and add a codeword-similarity term
    # to the fine-codebook logits. All new weights start at zero, so an existing checkpoint
    # behaves identically at the start.
    codec_embeds: bool = False
    codec_dim: int = 256

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "CSMConfig":
        return cls(**json.loads(path.read_text()))


def _sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float = 1.0,
            generator: torch.Generator | None = None) -> int:
    """logits: 1-D. temperature <= 0 means greedy."""
    logits = logits.float()
    if temperature <= 0:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k and 0 < top_k < logits.numel():
        kth = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        before = probs.cumsum(-1) - probs
        sorted_logits = sorted_logits.masked_fill(before > top_p, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(0, idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))


class GemmaCSM(nn.Module):
    def __init__(self, cfg: CSMConfig, backbone: nn.Module | None = None, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.cfg = cfg
        Q, V = cfg.num_quantizers, cfg.codebook_size
        if Q < 2:
            raise ValueError("need at least 2 codebooks (1 for the backbone, >=1 for the decoder)")

        if backbone is None:
            backbone = AutoModel.from_pretrained(cfg.text_backbone)
        self.backbone = backbone
        D = backbone.config.hidden_size
        dh = cfg.decoder_hidden_size
        self.hidden_size = D

        # --- audio-side embeddings for the backbone input ---------------------
        # Match the scale of Gemma's (already sqrt(D)-scaled) text embeddings so
        # audio and text inputs live in a comparable range.
        with torch.no_grad():
            emb = backbone.get_input_embeddings()
            n = min(4096, emb.num_embeddings)
            text_std = emb(torch.arange(n)).float().std().item()
        text_std = max(text_std, 1e-3)
        self.audio_embed = nn.Embedding(Q * V, D)
        nn.init.normal_(self.audio_embed.weight, std=text_std / math.sqrt(Q))
        self.sos_embed = nn.Parameter(torch.randn(D) * text_std)

        # codebook-0 head, with one extra class = end-of-speech
        self.c0_head = nn.Linear(D, V + 1, bias=False)
        nn.init.normal_(self.c0_head.weight, std=D ** -0.5)

        # --- audio decoder (codebooks 1..Q-1) ---------------------------------
        dcfg = LlamaConfig(
            vocab_size=1,  # unused: we feed inputs_embeds
            hidden_size=dh,
            intermediate_size=cfg.decoder_intermediate,
            num_hidden_layers=cfg.decoder_layers,
            num_attention_heads=cfg.decoder_heads,
            num_key_value_heads=cfg.decoder_heads,
            max_position_embeddings=max(64, Q + 2),
            bos_token_id=None, eos_token_id=None, pad_token_id=None,
        )
        self.decoder = LlamaModel(dcfg)
        self.dec_in_proj = nn.Linear(D, dh, bias=False)
        self.dec_embed = nn.Embedding((Q - 1) * V, dh)  # codebooks 0..Q-2 are decoder inputs
        nn.init.normal_(self.dec_embed.weight, std=0.02)
        self.ac_heads = nn.Parameter(torch.randn(Q - 1, dh, V) * dh ** -0.5)

        if cfg.codec_embeds:
            Dc = cfg.codec_dim
            # frozen Mimi codeword vectors: [Q, V, Dc] (row 0 = semantic codebook, rows 1.. = acoustic chain)
            self.register_buffer("codec_table", torch.zeros(Q, V, Dc))
            self.cg_back_sem = nn.Linear(Dc, D, bias=False)
            self.cg_back_ac = nn.Linear(Dc, D, bias=False)
            self.cg_dec_sem = nn.Linear(Dc, dh, bias=False)
            self.cg_dec_ac = nn.Linear(Dc, dh, bias=False)
            self.ac_geo = nn.Parameter(torch.zeros(Q - 1, dh, Dc))
            for m in (self.cg_back_sem, self.cg_back_ac, self.cg_dec_sem, self.cg_dec_ac):
                nn.init.zeros_(m.weight)

        self.to(dtype)

    # ------------------------------------------------------------------ utils
    def new_parameters(self):
        """Everything except the pretrained Gemma backbone."""
        bb = {id(p) for p in self.backbone.parameters()}
        return [p for p in self.parameters() if id(p) not in bb]

    # ------------------------------------------------------- codec geometry
    @staticmethod
    def build_codec_table(mimi_model, num_quantizers: int) -> torch.Tensor:
        """Mimi codeword vectors as a [Q, V, Dc] tensor. Mimi is a *split* RVQ: codebook 0 is its own
        (semantic) quantizer; codebooks 1.. form one residual chain whose codeword sum is the
        reconstruction. A single global scale keeps their relative magnitudes."""
        q = mimi_model.quantizer
        rows = [q.semantic_residual_vector_quantizer.layers[0].codebook.embed]
        for k in range(num_quantizers - 1):
            rows.append(q.acoustic_residual_vector_quantizer.layers[k].codebook.embed)
        table = torch.stack([r.detach().float().cpu() for r in rows])
        return table / table[1:].std().clamp(min=1e-6)

    def set_codec_table(self, table: torch.Tensor) -> None:
        if not self.cfg.codec_embeds:
            raise ValueError("model was built with codec_embeds=False")
        if tuple(table.shape) != tuple(self.codec_table.shape):
            raise ValueError(f"codec table {tuple(table.shape)} != expected {tuple(self.codec_table.shape)}")
        self.codec_table.copy_(table.to(self.codec_table.dtype))

    def _geo_vectors(self, frames: torch.Tensor):
        """frames [N, Q'] -> (sem [N,Dc], ac_cum [N,Q'-1,Dc]) where ac_cum[:, m] is the running sum of
        acoustic codewords of codebooks 1..m+1 (the residual-chain state)."""
        Qp = frames.size(1)
        idx = torch.arange(Qp, device=frames.device)
        vec = self.codec_table[idx, frames]                            # [N,Q',Dc]
        return vec[:, 0], vec[:, 1:].cumsum(dim=1)

    def _logit_table(self) -> torch.Tensor:
        """Per-codebook-normalised acoustic codeword table used for the similarity logit term."""
        t = self.codec_table[1:].float()
        t = t / (t.std(dim=(1, 2), keepdim=True) + 1e-6)
        return t / math.sqrt(t.size(-1))                                # [Q-1,V,Dc]

    def frame_embeds(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: [T, Q] long -> [T, D] (sum of per-codebook embeddings)."""
        Q, V = self.cfg.num_quantizers, self.cfg.codebook_size
        offs = torch.arange(Q, device=frames.device) * V
        e = self.audio_embed(frames + offs).sum(dim=1)
        if self.cfg.codec_embeds:
            sem, ac_cum = self._geo_vectors(frames)
            e = e + self.cg_back_sem(sem).to(e.dtype) + self.cg_back_ac(ac_cum[:, -1]).to(e.dtype)
        return e

    def decoder_logits(self, h: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        """Teacher-forced. h: [N, D] backbone states, frames: [N, Q] long.
        Returns logits [N, Q-1, V] for codebooks 1..Q-1."""
        Q, V = self.cfg.num_quantizers, self.cfg.codebook_size
        x0 = self.dec_in_proj(h).unsqueeze(1)                       # [N,1,dh]
        offs = torch.arange(Q - 1, device=frames.device) * V
        xe = self.dec_embed(frames[:, : Q - 1] + offs)               # [N,Q-1,dh]
        if self.cfg.codec_embeds:
            sem, ac_cum = self._geo_vectors(frames[:, : Q - 1])      # ac_cum: [N,Q-2,Dc]
            add = torch.zeros_like(xe)
            add[:, 0] = self.cg_dec_sem(sem).to(xe.dtype)
            if Q > 2:
                add[:, 1:] = self.cg_dec_ac(ac_cum).to(xe.dtype)
            xe = xe + add
        x = torch.cat([x0, xe.to(x0.dtype)], dim=1)                  # [N,Q,dh]
        out = self.decoder(inputs_embeds=x).last_hidden_state        # [N,Q,dh]
        o = out[:, 1:]
        logits = torch.einsum("nqd,qdv->nqv", o, self.ac_heads.to(o.dtype))
        if self.cfg.codec_embeds:
            g = torch.einsum("nqd,qdc->nqc", o, self.ac_geo.to(o.dtype))
            logits = logits + torch.einsum("nqc,qvc->nqv", g, self._logit_table().to(g.dtype))
        return logits

    @torch.no_grad()
    def sample_fine(self, h: torch.Tensor, c0: int, temperature: float, top_k: int,
                    generator: torch.Generator | None = None, tbln: torch.Tensor | None = None) -> list[int]:
        """Autoregressively sample codebooks 1..Q-1 of ONE frame given backbone state h [1,D] and c0."""
        Q, V = self.cfg.num_quantizers, self.cfg.codebook_size
        dev = h.device
        e0 = self.dec_embed(torch.tensor([[c0]], device=dev))
        geo = self.cfg.codec_embeds
        if geo:
            if tbln is None:
                tbln = self._logit_table()
            e0 = e0 + self.cg_dec_sem(self.codec_table[0, c0].view(1, 1, -1)).to(e0.dtype)
            acc = torch.zeros(1, self.cfg.codec_dim, device=dev, dtype=self.codec_table.dtype)
        x_dec = torch.cat([self.dec_in_proj(h).unsqueeze(1), e0.to(h.dtype)], dim=1)   # [1,2,dh]
        codes = [c0]
        for i in range(1, Q):
            o = self.decoder(inputs_embeds=x_dec).last_hidden_state[:, -1]              # [1,dh]
            lg = (o.float() @ self.ac_heads[i - 1].float())[0]
            if geo:
                lg = lg + ((o.float() @ self.ac_geo[i - 1].float()) @ tbln[i - 1].T)[0]
            ci = _sample(lg, temperature, top_k, 1.0, generator)
            codes.append(ci)
            if i < Q - 1:
                nxt = self.dec_embed(torch.tensor([[ci + i * V]], device=dev))
                if geo:
                    acc = acc + self.codec_table[i, ci].view(1, -1)
                    nxt = nxt + self.cg_dec_ac(acc).unsqueeze(1).to(nxt.dtype)
                x_dec = torch.cat([x_dec, nxt.to(x_dec.dtype)], dim=1)
        return codes

    # --------------------------------------------------------------- training
    def _build_inputs(self, text_ids, text_len, codes, n_frames):
        embed = self.backbone.get_input_embeddings()
        seqs = []
        for b in range(text_ids.size(0)):
            lt, t = int(text_len[b]), int(n_frames[b])
            parts = [embed(text_ids[b, :lt])]
            dt = parts[0].dtype
            parts.append(self.sos_embed.to(dt).unsqueeze(0))
            if t > 0:
                parts.append(self.frame_embeds(codes[b, :, :t].T).to(dt))
            seqs.append(torch.cat(parts, dim=0))
        L = max(s.size(0) for s in seqs)
        x = seqs[0].new_zeros(len(seqs), L, seqs[0].size(1))
        mask = torch.zeros(len(seqs), L, dtype=torch.long, device=x.device)
        for b, s in enumerate(seqs):
            x[b, : s.size(0)] = s
            mask[b, : s.size(0)] = 1
        return x, mask

    def forward(self, text_ids, text_len, codes, n_frames):
        """text_ids [B,Lt] long (right padded), text_len [B], codes [B,Q,Tmax] long
        (right padded), n_frames [B]."""
        cfg = self.cfg
        V = cfg.codebook_size
        codes = codes[:, : cfg.num_quantizers]
        x, mask = self._build_inputs(text_ids, text_len, codes, n_frames)
        hidden = self.backbone(inputs_embeds=x, attention_mask=mask).last_hidden_state

        h_c0, tgt_c0, h_dec, fr = [], [], [], []
        eos = torch.tensor([V], device=codes.device, dtype=torch.long)
        for b in range(text_ids.size(0)):
            s, t = int(text_len[b]), int(n_frames[b])
            h_c0.append(hidden[b, s : s + t + 1])
            tgt_c0.append(torch.cat([codes[b, 0, :t], eos]))
            h_dec.append(hidden[b, s : s + t])
            fr.append(codes[b, :, :t].T)
        h_c0, tgt_c0 = torch.cat(h_c0), torch.cat(tgt_c0)
        h_dec, fr = torch.cat(h_dec), torch.cat(fr)

        logits0 = self.c0_head(h_c0).float()
        loss_c0 = F.cross_entropy(logits0, tgt_c0)
        c0_acc = (logits0.argmax(-1) == tgt_c0).float().mean()

        if self.training and cfg.decoder_frame_fraction < 1.0 and h_dec.size(0) > 1:
            k = max(1, int(h_dec.size(0) * cfg.decoder_frame_fraction))
            idx = torch.randperm(h_dec.size(0), device=h_dec.device)[:k]
            h_dec, fr = h_dec[idx], fr[idx]
        logits_ac = self.decoder_logits(h_dec, fr).float()                # [N,Q-1,V]
        loss_ac = F.cross_entropy(logits_ac.reshape(-1, V), fr[:, 1:].reshape(-1))

        loss = cfg.c0_loss_weight * loss_c0 + cfg.acoustic_loss_weight * loss_ac
        return {"loss": loss, "loss_c0": loss_c0.detach(), "loss_ac": loss_ac.detach(), "c0_acc": c0_acc.detach()}

    # -------------------------------------------------------------- inference
    @torch.no_grad()
    def generate(
        self,
        text_ids: torch.Tensor,
        max_frames: int = 400,
        min_frames: int = 0,
        temperature_c0: float = 0.9,
        top_k_c0: int = 50,
        top_p_c0: float = 1.0,
        temperature_ac: float = 0.7,
        top_k_ac: int = 20,
        use_cache: bool = True,
        generator: torch.Generator | None = None,
    ) -> list[list[int]]:
        """text_ids: 1-D long tensor (bos + text). Returns codes[Q][T]."""
        Q, V = self.cfg.num_quantizers, self.cfg.codebook_size
        dev = text_ids.device
        embed = self.backbone.get_input_embeddings()
        x = torch.cat([embed(text_ids), self.sos_embed.to(embed.weight.dtype).unsqueeze(0)], dim=0).unsqueeze(0)
        full_x, past = x, None
        frames: list[list[int]] = []
        tbln = self._logit_table() if self.cfg.codec_embeds else None

        while True:
            if use_cache:
                out = self.backbone(inputs_embeds=x, past_key_values=past, use_cache=True)
                past = out.past_key_values
            else:
                out = self.backbone(inputs_embeds=full_x)
            h = out.last_hidden_state[:, -1]                                # [1,D]

            logits0 = self.c0_head(h)[0].float()
            if len(frames) < min_frames:
                logits0[V] = float("-inf")
            if len(frames) >= max_frames:
                break
            c0 = _sample(logits0, temperature_c0, top_k_c0, top_p_c0, generator)
            if c0 == V:
                break

            codes = self.sample_fine(h, c0, temperature_ac, top_k_ac, generator, tbln)
            frames.append(codes)

            nxt_x = self.frame_embeds(torch.tensor([codes], device=dev)).unsqueeze(0).to(full_x.dtype)
            x = nxt_x
            full_x = torch.cat([full_x, nxt_x], dim=1)

        if not frames:
            return [[] for _ in range(Q)]
        return [[f[q] for f in frames] for q in range(Q)]

    # -------------------------------------------------------------------- io
    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.cfg.save(path / "csm_config.json")
        self.backbone.config.save_pretrained(path / "backbone_config")
        torch.save(self.state_dict(), path / "model.pt")

    @classmethod
    def from_pretrained(cls, path: str | Path, dtype: torch.dtype = torch.float32, map_location="cpu") -> "GemmaCSM":
        path = Path(path)
        cfg = CSMConfig.load(path / "csm_config.json")
        backbone = AutoModel.from_config(AutoConfig.from_pretrained(path / "backbone_config"))
        model = cls(cfg, backbone=backbone, dtype=torch.float32)
        state = torch.load(path / "model.pt", map_location=map_location)
        model.load_state_dict(state)
        return model.to(dtype)

    def load_weights_from(self, path: str | Path) -> None:
        """Load weights from a checkpoint dir into this (already built) model."""
        state = torch.load(Path(path) / "model.pt", map_location="cpu")
        res = self.load_state_dict(state, strict=False)
        new_ok = ("codec_table", "cg_back_", "cg_dec_", "ac_geo")
        bad_missing = [k for k in res.missing_keys if not k.startswith(new_ok)]
        if bad_missing or res.unexpected_keys:
            raise RuntimeError(f"checkpoint mismatch: missing {bad_missing}, unexpected {res.unexpected_keys}")
        if res.missing_keys:
            print(f"[load] new (zero-initialised) weights not in checkpoint: {len(res.missing_keys)} tensors", flush=True)
