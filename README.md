# HTMLNLM — Browser Neural Runtime

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Platform: Browser](https://img.shields.io/badge/platform-browser-blue.svg)
![Dependencies: Zero](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)
![Architecture: RWKV--v6](https://img.shields.io/badge/architecture-RWKV--v6-teal.svg)
![Quantization: b1.58](https://img.shields.io/badge/quantization-b1.58%20ternary-orange.svg)
![Size: Single File](https://img.shields.io/badge/size-single%20file-purple.svg)

A complete, self-contained neural language model runtime that runs 
entirely in your browser. No server. No Python. No CUDA. No 
dependencies. One HTML file.

Train a language model from scratch, align it with reinforcement 
learning, and run inference — all locally, all offline, all in 
the tab you already have open.

---

## What This Is

HTMLNLM implements a full LLM training and inference pipeline using 
only browser-native APIs. It is not a wrapper around an existing 
model. It is not a demo. It trains real weights from real data using 
real gradient descent.

**Architecture:** RWKV-v6 (Finch) — a linear-time recurrent 
architecture with O(1) inference memory. No KV cache. No quadratic 
attention. Runs efficiently on CPU.

**Quantization:** BitNet b1.58 — weights constrained to ternary 
values {-1, 0, +1} via T-MAC lookup table microkernel. Matrix 
multiplication replaced with cache-efficient table lookups.

**Training:** OOMB (Out of Memory Barrier) chunk-recurrent 
backpropagation. Activations recomputed on-the-fly during the 
backward pass — constant memory footprint regardless of sequence 
length.

**Optimizer:** AdamW (default, mobile-friendly) or Muon with 
Newton-Schulz quintic orthogonalization (desktop, higher quality).

**Alignment:** GRPO (Group Relative Policy Optimization) — 
critic-free reinforcement learning with Z-score normalized 
advantages and approximate KL divergence constraint.

**Tokenizer:** BPE (Byte Pair Encoding) compiled in a WebWorker 
so it doesn't block the UI thread.

---

## Features

- **Zero dependencies** — no npm, no pip, no CDN calls
- **Zero server** — runs fully offline after the first load
- **Mobile compatible** — auto-detects mobile and selects 
  AdamW optimizer for battery efficiency
- **Persistent storage** — save/load model weights via 
  IndexedDB or portable JSON export
- **Live training telemetry** — EMA-smoothed loss curve, 
  ternary weight distribution histogram, tok/sec throughput
- **Live inference sampling** — generates from corpus-seeded 
  prompts during training so you can watch it learn
- **Full GRPO alignment tab** — post-training RL with cohort 
  visualization
- **CSS state machine routing** — zero-JS navigation via 
  `:has()` selectors and radio inputs

---

## How To Use

1. Download `HTMLNLM.html`
2. Open it in any modern browser
3. Go to **1. ARCHITECTURE** — configure your model size
4. Drop or paste a `.txt` corpus
5. Click **COMPILE BPE** → **ALLOCATE VM**
6. Go to **2. PRE-TRAIN** → **START LOOP**
7. Watch it learn

That's it. No install step. No terminal. No accounts.

---

## Architecture Details

```
Corpus (.txt)
    │
    ▼
BPE Tokenizer (WebWorker)
    │
    ▼
RWKV-v6 Blocks × L
  ├─ Time Mix (WKV recurrent state)
  ├─ Channel Mix (gated FFN)
  └─ BitLinear (ternary weights via T-MAC)
    │
    ▼
Language Model Head
    │
    ▼
OOMB Backward Pass
(chunk-recurrent, O(1) activation memory)
    │
    ▼
AdamW / Muon Optimizer
    │
    ▼
Embedding Update (SGD, sparse)
```

**Recommended starting config:**
- Vocab: 2048
- Hidden dim: 256
- Layers: 4
- Context chunk: 128
- Optimizer: AdamW (mobile) / Muon (desktop)

---

## Technical Notes

**Why RWKV?**
Standard attention scales quadratically with sequence length and 
requires a growing KV cache at inference time. RWKV is a recurrent 
architecture with linear scaling and constant inference memory — 
making it viable for in-browser training without hitting memory 
walls.

**Why ternary weights?**
BitNet b1.58 replaces float multiplications with integer additions. 
This maps well to CPU execution (no FPU bottleneck) and reduces 
weight storage dramatically — a 2M parameter model fits comfortably 
in browser heap memory.

**Why OOMB?**
Standard backpropagation stores all intermediate activations for 
the backward pass. For long sequences this is catastrophic. OOMB 
checkpoints the recurrent state at chunk boundaries and recomputes 
activations on-the-fly during the backward pass, keeping memory 
footprint constant.

**Why GRPO instead of SFT?**
At small parameter counts, RL-based alignment significantly 
outperforms supervised fine-tuning. GRPO eliminates the need for 
a critic model (unlike PPO), halving the memory requirement during 
alignment.

---

## Limitations

This is a research and educational tool. Models trained in-browser 
will be small (sub-10M parameters practical) and require substantial 
training time to develop coherent outputs. Throughput is 
hardware-dependent — expect 0.5–5 tok/sec depending on device.

This is not a replacement for Python-based training pipelines for 
production models. It is a demonstration that the full pipeline 
can exist natively in a browser, and a genuinely useful tool for 
small-scale local model development.

---

## License

MIT. Do what you want with it.

---

## Built By

[ConsciousNode SoftWorks](https://github.com/ConsciousNode)
```
