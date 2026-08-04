# vocal

Rust implementation scaffold of **RVC (Retrieval-based Voice Conversion)** in `~/repos/maolan/vocal`.

## Status

This project implements a functional **RVC-style pipeline** in pure Rust:

1. WAV I/O
2. Content feature extraction (HuBERT-like proxy features)
3. F0 extraction (frame-wise proxy)
4. Retrieval blending (nearest-neighbor feature mixing)
5. Voice conversion synthesis (baseline resynthesis/pitch-shift proxy)
6. Output post-processing (RMS mix)
7. Burnpack model artifact loading (`.bpk`) in the same style used by `maolan/generate`

It is designed to mirror RVC's architecture and CLI flow while remaining dependency-light and easy to extend.

## CLI

### Create config

```bash
cargo run --release -- new-config --output config.json
```

### Inference

```bash
cargo run --release -- infer \
  --input-wav input.wav \
  --output-wav output.wav \
  --burnpack rvc_model.bpk \
  --config config.json \
  --pitch-shift-semitones 3 \
  --index-mix 0.2 \
  --rms-mix-rate 0.3 \
  --target-sr 48000
```

### Inspect a burnpack

```bash
cargo run --release -- inspect-burnpack --model rvc_model.bpk
```

### Convert PyTorch checkpoint to burnpack

```bash
cargo run --release -- convert-pth \
  --input model.pth \
  --output rvc_model.bpk \
  --top-level-key state_dict
```

## Project layout

- `src/main.rs`: CLI (`infer`, `new-config`)
- `src/config.rs`: RVC config schema
- `src/audio.rs`: WAV I/O + linear resampler + RMS
- `src/rvc.rs`: model components (`HubertEncoder`, `F0Extractor`, `VoiceConverter`)
- `src/retrieval.rs`: retrieval index + nearest-neighbor blending
- `src/pipeline.rs`: end-to-end inference pipeline

## Notes on parity with upstream RVC

This is a Rust-native baseline and **not yet bit-exact to upstream Python/PyTorch RVC**.
To reach full parity, next steps are:

1. Full RVC synth graph in Burn modules loaded from burnpack tensors.
2. RMVPE/Harvest/DIO-grade F0 extraction.
3. FAISS-compatible retrieval index loading and querying.
4. Exact frame slicing, protection, and infer-time blending behavior.

The current code is a clean foundation to implement those pieces incrementally in Rust.
