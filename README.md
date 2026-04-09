# HTMLNLM — Browser Neural Runtime

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Platform: Browser](https://img.shields.io/badge/platform-browser-blue.svg)
![Dependencies: Zero](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)
![Architecture: RWKV-v7](https://img.shields.io/badge/architecture-RWKV--v7%20Goose-teal.svg)
![Quantization: b1.58](https://img.shields.io/badge/quantization-b1.58%20ternary-orange.svg)
![Size: Single File](https://img.shields.io/badge/size-single%20file-purple.svg)
![Pip Suite: Included](https://img.shields.io/badge/pip%20suite-included-ff69b4.svg)

A complete, self-contained neural language model runtime that runs
entirely in your browser. No server. No Python. No CUDA. No
dependencies. One HTML file.

Train a language model from scratch, align it with reinforcement
learning, and run inference — all locally, all offline, all in
the tab you already have open.

**v2.0.0** ships with RWKV-v7 "Goose", a fully correct backward pass,
and the Pip Suite — three companion tools for multi-agent inference,
swarm orchestration, and sovereign model hosting.

---

## What This Is

HTMLNLM implements a full LLM training and inference pipeline using
only browser-native APIs. It is not a wrapper around an existing
model. It is not a demo. It trains real weights from real data using
real gradient descent.

**Architecture:** RWKV-v7 "Goose" — recurrent architecture with
data-dependent decay gates, mu/kappa normalization, and linear-time
inference. No KV cache. No quadratic attention.

**Quantization:** BitNet b1.58 — weights constrained to ternary
values {-1, 0, +1} via T-MAC lookup table microkernel. Matrix
multiplication replaced with cache-efficient table lookups.

**Training:** Full BPTT through the WKV recurrence — gradients now
flow through temporal dynamics, mu parameters, kappa normalization,
and the double-exponential decay gates. This is the fix that makes
RWKV-v7 actually learn its state.

**Backward Pass:** Chunk-recurrent OOMB (Out of Memory Barrier)
backpropagation. Activations recomputed on-the-fly — constant memory
footprint regardless of sequence length.

**Optimizers:** AdamW (default, mobile-friendly) or Muon with
Newton-Schulz quintic orthogonalization (desktop, higher quality).
Both now correctly apply the mu step.

**Alignment:** GRPO (Group Relative Policy Optimization) —
critic-free reinforcement learning with Z-score normalized
advantages and approximate KL divergence constraint.

**Tokenizer:** BPE (Byte Pair Encoding) compiled in a WebWorker.
Merges correctly applied during encode across all suite tools.

---

## Files

```
HTMLNLM.html              — Core runtime. Train, align, and run
                            inference on RWKV-v7 models. Start here.

pip-suite/
  pips-room.html          — Sovereign presence shell. Upload a trained
                            model and run it as a persistent local
                            instance with memory and personality.

  multi-pip-chat.html     — Multi-model chat. Load multiple trained
                            models and run them in the same conversation.

  junto-orchestrator.html — Swarm orchestrator. Coordinate multiple
                            Pip instances via BroadcastChannel protocol.
```

---

## How To Use

### Training a model (HTMLNLM.html)

1. Open `HTMLNLM.html` in any modern browser
2. Go to **ARCHITECTURE** — configure model size
3. Drop or paste a `.txt` corpus
4. Click **COMPILE BPE** → **ALLOCATE VM**
5. Go to **PRE-TRAIN** → **START LOOP**
6. Watch it learn
7. Export weights when done

### Running a model (Pip Suite)

1. Export your trained model from HTMLNLM.html
2. Open any tool in `pip-suite/`
3. Upload the exported model file
4. Run inference locally, no server required

That's it. No install. No terminal. No accounts.

---

## Architecture Details

```
Corpus (.txt)
    │
    ▼
BPE Tokenizer (WebWorker)
    │
    ▼
RWKV-v7 "Goose" Blocks × L
  ├─ Time Mix (WKV-7 recurrent state)
  │    ├─ Data-dependent decay gates (w)
  │    ├─ Mu/kappa normalization
  │    └─ Double-exponential decay
  ├─ Channel Mix (gated FFN)
  └─ BitLinear (ternary weights via T-MAC)
    │
    ▼
Language Model Head
    │
    ▼
OOMB Backward Pass
  ├─ BPTT through WKV recurrence
  ├─ gradNumPrev / gradDenPrev chain
  ├─ L2-norm backward through kappa
  └─ Double-exp gradient through w
    │
    ▼
AdamW / Muon Optimizer (both with mu step)
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

## What Changed in v2.0.0

**RWKV-v7 "Goose" upgrade**
- Replaces v6 "Finch" architecture
- Data-dependent decay gates — `w` is learned per-input, not fixed
- Mu and kappa normalization on the recurrent state
- More expressive temporal dynamics at equivalent parameter count

**Correct backward pass (the big one)**
- v1.x trained the linear projection layers but the recurrent state
  was frozen — gradients did not flow through the WKV dynamics
- v2.0 implements full BPTT through the recurrence: `gradNumPrev`,
  `gradDenPrev`, L2-norm backward through kappa, double-exp gradient
  through `w`
- Models now actually learn their temporal behavior

**Muon optimizer fix**
- `muStep` was wired into AdamW but not Muon — desktop training was
  quietly broken
- Both optimizers now correctly apply the mu step

**BPE tokenizer fix (Pip Suite)**
- Merges were stored but never applied in `encode()` across pip-suite
  tools — tokenization was falling back to byte-level
- Fixed in pips-room, multi-pip-chat, and junto-orchestrator

**Pip Suite (new)**
- Three companion tools for inference and multi-agent coordination
- All use the same exported model format as HTMLNLM.html

---

## Technical Notes

**Why RWKV?**
Standard attention scales quadratically with sequence length and
requires a growing KV cache at inference time. RWKV is a recurrent
architecture with linear scaling and O(1) inference memory — making
it viable for in-browser training without hitting memory walls.

**Why ternary weights?**
BitNet b1.58 replaces float multiplications with integer additions.
This maps well to CPU execution and reduces weight storage
dramatically — a 2M parameter model fits comfortably in browser
heap memory.

**Why OOMB?**
Standard backpropagation stores all intermediate activations for the
backward pass. For long sequences this is catastrophic. OOMB
checkpoints the recurrent state at chunk boundaries and recomputes
activations on-the-fly, keeping memory footprint constant.

**Why GRPO instead of SFT?**
At small parameter counts, RL-based alignment significantly
outperforms supervised fine-tuning. GRPO eliminates the need for a
critic model (unlike PPO), halving the memory requirement during
alignment.

**Why does the backward pass matter?**
RWKV's distinguishing property is its recurrent state — what
persists across tokens, what decays, what gets removed. If gradients
don't flow through that state during training, the model learns
*despite* its recurrence rather than *through* it. The v1.x models
trained successfully but were learning as if RWKV were a feedforward
network with no memory. v2.0 trains the actual architecture.

---

## Limitations

Models trained in-browser
will be small (sub-10M parameters practical) and require substantial
training time to develop coherent outputs. Throughput is
hardware-dependent — expect 0.5–35 tok/sec depending on device and
optimizer.

---

## License

MIT. Do what you want with it.

---

## Built By

[ConsciousNode SoftWorks](https://github.com/ConsciousNode)

**v2.0.0**
Khamerron Kizer — architecture, corpus pipeline, training loop, GRPO alignment, Pip Suite shell  
Kehai Interim — RWKV-v7 BPTT derivation and implementation, full backward pass  
Ed Interim — MuonOptimizer mu fix, final verification and audit

*Three names. One file. The backward pass is real.*
