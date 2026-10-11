# gemma-mimi-tts

A training pipeline that turns a **Gemma** language model into a text-to-speech
model by teaching it to autoregressively predict **Mimi** neural-audio-codec
tokens, the same recipe used by Llama-Mimi, Breeze TTS 2 and VyvoTTS — just
with Gemma as the backbone instead of Llama/Qwen/LFM2.

```
text (any language Gemma's tokenizer covers)
  -> Gemma backbone (vocab extended with Mimi codec tokens)
  -> autoregressively generates interleaved codec tokens
  -> Mimi decoder
  -> waveform
```

This repo is meant to be cloned onto a machine with a GPU. Nothing here
trains on CPU at any real speed — this environment was only used to write
and syntax-check the code.

## Status / intended path

1. **Smoke test on LJSpeech** (English, 24h, tiny, fast to download) to
   prove the plumbing works end-to-end: Mimi tokenization, vocab extension,
   training loop converges, generated tokens decode to audible (if bad)
   speech. This is what the scripts below default to.
2. **Swap in real data** for your actual target language(s) once the
   pipeline is verified — e.g. IndicTTS (IIT Madras), IndicVoices,
   CommonVoice, Shrutilipi. Nothing in the model/training code is
   English-specific; only `scripts/prepare_dataset.py`'s dataset loader
   changes.

Emilia (101k hours) was considered for step 1 but has **no Indian
languages** (en/zh/de/fr/ja/ko only), so it doesn't help you get to the
real target any faster than LJSpeech does — LJSpeech is smaller and faster
to iterate with for a pure plumbing test.

## Install

```bash
pip install -r requirements.txt
```

Requires a CUDA GPU with enough VRAM for whichever Gemma size you pick
(gemma-3-270m or gemma-3-1b recommended for single/few-GPU fine-tuning;
see `configs/gemma_mimi_small.yaml`).

## Pipeline

### 1. Get audio

```bash
python scripts/download_ljspeech.py --root data/raw/ljspeech
```

Downloads via `torchaudio.datasets.LJSPEECH` (wav + transcript pairs).

For your real target language, replace this step with your own
loader that produces the same `manifest.csv` format: `path|text` per line,
UTF-8, one utterance per line. See `scripts/prepare_dataset.py --help` for
the exact schema an "audiofolder"-style HF dataset needs if you'd rather
pull from the Hub directly (e.g. an IndicTTS or CommonVoice split).

### 2. Tokenize audio with Mimi + build training sequences

```bash
python scripts/prepare_dataset.py \
  --manifest data/raw/ljspeech/manifest.csv \
  --audio_root data/raw/ljspeech \
  --output_dir data/processed/ljspeech \
  --config configs/gemma_mimi_small.yaml
```

This runs the frozen Mimi encoder over every utterance, flattens the
per-frame codebook outputs into a single token stream (same "flattened
RVQ" approach as Llama-Mimi), wraps it with control tokens, tokenizes the
text with Gemma's tokenizer, and writes `.jsonl` shards of
`input_ids` / `label_ids` ready for training.

### 3. Train

```bash
python -m gemma_mimi_tts.train.pretrain --config configs/gemma_mimi_small.yaml
```

Extends Gemma's embedding matrix and LM head for the new audio tokens,
then does causal-LM training over the interleaved sequences (loss masked
to the audio region by default — the text portion is context only; flip
`loss_on_text` in the config to change that).

### 4. Fine-tune for a single voice (optional, once pretraining "works")

Same entrypoint, pointed at a small single-speaker manifest and a lower
learning rate — see the `finetune` section of the config.

### 5. Generate

```bash
python -m gemma_mimi_tts.infer.generate \
  --config configs/gemma_mimi_small.yaml \
  --checkpoint runs/gemma_mimi_small/checkpoint-XXXX \
  --text "your text here" \
  --out out.wav
```

Constrained decoding keeps the model emitting only valid audio-codebook
tokens once it enters the speech region (see
`gemma_mimi_tts/infer/constraints.py`), then the generated tokens are
de-interleaved back into per-codebook code sequences and decoded to a
waveform by Mimi.

## Licensing, read before training on real data

- **Gemma** weights: Google's Gemma license — check current terms before
  commercial use; generally more permissive than Sarvam-1's
  non-commercial license, but it is not a blanket "do anything" license.
- **Mimi** (`kyutai/mimi`): CC-BY 4.0, commercial-friendly.
- Whatever speech dataset you swap in for step 1/2 carries its own
  license — IndicTTS, CommonVoice, IndicVoices etc. all differ; check
  before shipping a product on top of them.

## Repo layout

```
gemma_mimi_tts/
  model/
    audio_vocab.py     special tokens + codebook <-> token-id mapping
    mimi_codec.py       thin wrapper around transformers' MimiModel
    backbone.py          loads Gemma, extends embeddings/LM head
  data/
    dataset.py            torch Dataset over the pre-tokenized jsonl shards
  train/
    pretrain.py            HF Trainer-based training entrypoint
  infer/
    constraints.py     logits processor for constrained audio decoding
    generate.py             text -> wav CLI
configs/
  gemma_mimi_small.yaml
scripts/
  download_ljspeech.py
  prepare_dataset.py
```
# gemini-mimi-tts
