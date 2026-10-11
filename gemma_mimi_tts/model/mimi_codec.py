"""Thin wrapper around transformers' MimiModel (kyutai/mimi).

Mimi runs at 12.5 Hz and produces one code per active quantizer per frame.
We always keep it frozen -- it's a fixed, pretrained neural codec, not
something this pipeline trains.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from transformers import MimiModel, AutoFeatureExtractor


MIMI_SAMPLE_RATE = 24_000  # Mimi's native encode/decode sample rate


def _resample_numpy(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Polyphase resampling via scipy, deliberately NOT torchaudio's
    conv1d-based `functional.resample`. That implementation is known to
    trigger a native "Floating point exception (core dumped)" crash (a
    SIGFPE, not a catchable Python exception) on some cloud VM CPUs, due to
    an AVX-512 dispatch bug in the oneDNN/MKL backend PyTorch's conv1d
    falls back to. scipy's FFT/polyphase path avoids that codepath
    entirely and needs no exotic CPU feature detection.
    """
    from scipy.signal import resample_poly

    g = math.gcd(orig_sr, target_sr)
    up, down = target_sr // g, orig_sr // g
    return resample_poly(audio, up, down).astype(np.float32)


@dataclass
class MimiCodec:
    model: MimiModel
    feature_extractor: AutoFeatureExtractor
    num_quantizers: int
    device: torch.device

    @property
    def codebook_size(self) -> int:
        return self.model.config.codebook_size

    @classmethod
    def load(
        cls,
        repo_id: str = "kyutai/mimi",
        num_quantizers: int = 8,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> "MimiCodec":
        device = torch.device(device)
        # model = MimiModel.from_pretrained(repo_id, torch_dtype=dtype).to(device)
        model = MimiModel.from_pretrained(repo_id, dtype=dtype).to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        feature_extractor = AutoFeatureExtractor.from_pretrained(repo_id)
        return cls(
            model=model,
            feature_extractor=feature_extractor,
            num_quantizers=num_quantizers,
            device=device,
        )

    @torch.no_grad()
    def encode_waveform(self, waveform: torch.Tensor, sample_rate: int) -> list[list[int]]:
        """waveform: mono float tensor, any sample rate (resampled to
        Mimi's native rate here). Returns codes[quantizer][frame] as plain
        python ints, using only the first `self.num_quantizers` codebooks.
        """
        audio_np = waveform.squeeze().cpu().numpy()
        if sample_rate != MIMI_SAMPLE_RATE:
            audio_np = _resample_numpy(audio_np, sample_rate, MIMI_SAMPLE_RATE)

        inputs = self.feature_extractor(
            raw_audio=audio_np,
            sampling_rate=MIMI_SAMPLE_RATE,
            return_tensors="pt",
        ).to(self.device)

        encoder_outputs = self.model.encode(
            inputs["input_values"], num_quantizers=self.num_quantizers
        )
        # encoder_outputs.audio_codes: [batch, num_quantizers, num_frames]
        codes = encoder_outputs.audio_codes[0].to(torch.int64).cpu()
        return [codes[q].tolist() for q in range(self.num_quantizers)]

    @torch.no_grad()
    def decode_codes(self, codes: list[list[int]]) -> torch.Tensor:
        """codes[quantizer][frame] -> mono waveform tensor at MIMI_SAMPLE_RATE."""
        num_frames = len(codes[0])
        assert all(len(c) == num_frames for c in codes)
        codes_t = torch.tensor(codes, dtype=torch.int64, device=self.device).unsqueeze(0)
        decoded = self.model.decode(codes_t)
        return decoded.audio_values[0].squeeze().cpu()