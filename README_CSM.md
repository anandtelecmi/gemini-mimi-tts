# CSM-style variant (Gemma backbone + audio decoder)

Same Mimi codec and Gemma-3-1b-pt as the main pipeline, but the audio is
modelled the way Sesame's CSM does it instead of flattening all codebooks:

- **Backbone (Gemma)** sees one position per Mimi frame (12.5/s). Each frame's
  input embedding is the *sum* of its Q codebook embeddings. A head on the
  backbone predicts **codebook 0** of the next frame (plus an end-of-speech class).
- **Audio decoder** (small Llama, ~80M) predicts codebooks **1..Q-1** of that
  frame autoregressively, conditioned on the backbone's hidden state.
- Sequences no longer grow with Q, so you can use 16 codebooks (vs 8) at the
  same backbone cost. No new tokens are added to Gemma's tokenizer.

## Run

```bash
pip install -r requirements.txt            # needs transformers >= 4.50 (Gemma3)

# A) re-encode with 16 codebooks (recommended), using normalized transcripts
python scripts/make_ljspeech_manifest.py --root data/raw/ljspeech
python scripts/prepare_csm_dataset.py \
  --manifest data/raw/ljspeech/manifest_normalized.csv \
  --audio_root data/raw/ljspeech \
  --output_dir data/processed/ljspeech_csm \
  --config configs/gemma_csm.yaml

# B) or quick start from your existing 8-codebook shards (no Mimi pass)
python scripts/convert_legacy_shards.py \
  --src data/processed/ljspeech --dst data/processed/ljspeech_csm
#    ...and set model.num_quantizers: 8 in configs/gemma_csm.yaml

python -m gemma_mimi_tts.csm.train --config configs/gemma_csm.yaml

python -m gemma_mimi_tts.csm.generate --config configs/gemma_csm.yaml \
  --checkpoint runs/gemma_csm/best --text "Hello, this is a test." --out out_csm.wav
```

Single GPU. Weights/optimizer are fp32 with bf16 autocast (~16 GB for
params+Adam before activations); set `gradient_checkpointing: true` or lower
`per_device_train_batch_size` if you run out of memory.

## Sampling knobs (generate.py)
`--temperature_c0/--top_k_c0` control prosody and timing (codebook 0);
`--temperature_ac/--top_k_ac` control fine acoustic detail. Lower the `_ac`
temperature if the voice sounds buzzy, raise `_c0` if it sounds monotone.

## Tests
`python tests/test_csm_model.py` (CPU, tiny random Gemma3, no downloads).
