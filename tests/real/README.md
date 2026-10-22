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
