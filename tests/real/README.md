# Real-checkpoint tests (need a download, so not part of the quick suite)

Checkpoint used on 2026-10-22: `rwkv7-g1d-0.1b-20260129-ctx8192.pth` from https://huggingface.co/BlinkDL/rwkv7-g1 (382,223,725 bytes, BF16).

```sh
# 1. talk to it (greedy), prints config, load report, ms/token
node --max-old-space-size=6000 tests/real/generate.js rwkv7-g1d-0.1b.pth "User: What is a goose?

Assistant:"
# 2. parity vs the numpy transcription of the official math, on the real weights
python3 ../brymar-college/scripts/convert_pth.py rwkv7-g1d-0.1b.pth g1d.safetensors   # torch-free, independent reader
node tests/real/parity.js x tok toks.json
python3 tests/real/parity_ref.py g1d.safetensors toks.json
node --max-old-space-size=6000 tests/real/parity.js rwkv7-g1d-0.1b.pth run toks.json
```

Result on 2026-10-22: 14 prompt tokens, max |Δlogit| **3.3e-5** (engine float32 vs reference float64), greedy pick agrees 14/14.

## Repetition control (v1.5)

```sh
# Multi-Pip Chat's loop, anti-repeat off (0) vs on (0.4), same seed; prints the chat and a score
node --max-old-space-size=3000 tests/real/repetition.js rwkv7-g1d-0.1b.pth 0   7 8
node --max-old-space-size=3000 tests/real/repetition.js rwkv7-g1d-0.1b.pth 0.4 7 8
# Pip's Room in headless Chromium with the real checkpoint (needs Playwright)
node tests/real/pips-room-e2e.js rwkv7-g1d-0.1b.pth
```

Score = repeated 4-word phrases across the chat (an exact-line count missed the real failure: near-identical lines that differ by a word). Results on 2026-10-22, 8 turns: seed 7 32% → 0%, seed 23 3% → 0%, seed 11 0% → 0%.
