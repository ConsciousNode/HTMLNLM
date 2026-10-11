# HTMLNLM — Browser Neural Runtime

![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Platform](https://img.shields.io/badge/platform-browser-blue?style=flat-square)
![Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen?style=flat-square)
![Architecture](https://img.shields.io/badge/architecture-RWKV--v7-teal?style=flat-square)
![Quantization](https://img.shields.io/badge/quantization-b1.58%20ternary-orange?style=flat-square)
![Size](https://img.shields.io/badge/size-single%20file-purple?style=flat-square)

**Complete browser-native LLM training and inference. Single file. Zero dependencies.**

HTMLNLM is a full neural language model runtime that runs entirely in your browser — no server, no Python, no CUDA, no install. Open the HTML file and train a model from scratch.

Built by [ConsciousNode SoftWorks](https://consciousnode.github.io) on the xinu principle: the browser is bare metal.

---

> **Successors:** [**HTMLNLM Evangelion**](https://github.com/ConsciousNode/HTMLNLM-Evangelion) adds omnimodal input, SheafMemory, BooleanPhaseDynamics, AutopoieticOptimizer, and RIFT Endospace. [**EvaROSA**](https://github.com/ConsciousNode/EvaROSA) adds ROSA neurosymbolic inner monologue. [**Simulacra**](https://github.com/ConsciousNode/Simulacra) is the RWKV-v8 clean break — ROSA replaces WKV as the sequence mechanism. HTMLNLM remains the stable text-only runtime.

---

## Try it

→ **[consciousnode.github.io/HTMLNLM](https://consciousnode.github.io/HTMLNLM)**

Or download [`HTMLNLM.html`](https://github.com/ConsciousNode/HTMLNLM/blob/main/HTMLNLM.html) and open locally. Works fully offline.

---

## What's inside

| Component | Description |
|---|---|
| **RWKV-v7 backbone** | Linear-time recurrent architecture, O(1) inference memory. No KV cache, no quadratic attention. |
| **BitNet b1.58** | Ternary weight quantization {-1, 0, +1} via T-MAC lookup table microkernel. Matrix multiplication replaced with cache-efficient table lookups. |
| **OOMB backward pass** | Out-of-Memory-Barrier chunk-recurrent backpropagation. Activations recomputed on-the-fly — constant memory regardless of sequence length. |
| **MuonOptimizer** | Quintic Newton-Schulz orthogonalization. Keeps weight matrices well-conditioned without expensive decompositions. |
| **AdamW** | Mobile-friendly fallback optimizer, auto-selected on battery-constrained devices. |
| **GRPO alignment** | Group Relative Policy Optimization — critic-free RL alignment with Z-score normalized advantages and KL divergence constraint. |
| **BPE tokenizer** | Byte Pair Encoding compiled in a WebWorker — doesn't block the UI thread. |
| **Pip Suite** | Companion tools: Junto orchestrator, multi-pip chat, Pip's Room. |

---

## How to use

1. Open `HTMLNLM.html` in any modern browser
2. Configure model size under **ARCHITECTURE**
3. Drop or paste a `.txt` corpus
4. Click **COMPILE BPE** → **ALLOCATE VM**
5. Go to **PRE-TRAIN** → **START LOOP**
6. Watch it learn

No terminal. No accounts. No install step.

---

## Architecture

```
Corpus (.txt)
    │
    ▼
BPE Tokenizer (WebWorker)
    │
    ▼
RWKV-v7 Blocks × L
  ├─ Time Mix (WKV recurrent state)
  ├─ Channel Mix (gated FFN)
  └─ BitLinear (ternary weights, T-MAC)
    │
    ▼
Language Model Head
    │
    ▼
OOMB Backward Pass (O(1) activation memory)
    │
    ▼
Muon / AdamW Optimizer
```

**Recommended starting config:** vocab 2048 · hidden dim 256 · layers 4 · context chunk 128

---

## ConsciousNode stack

| Project | Description | Status |
|---|---|---|
| **HTMLNLM** | Text-only RWKV-v7 runtime | ✅ Stable |
| **[HTMLNLM Evangelion](https://github.com/ConsciousNode/HTMLNLM-Evangelion)** | Omnimodal: vision, audio, spatial + SheafMemory + AutopoieticOptimizer | ✅ Phase 6 |
| **[OmniVocal](https://github.com/ConsciousNode/OmniVocal)** | Browser-native neural TTS with .pop2 voice identity | ✅ Live |
| **[RAG-Time](https://consciousnode.github.io/RAG-Time/)** | Browser-native RAG | ✅ Live |
| **[EvaROSA](https://github.com/ConsciousNode/EvaROSA)** | RWKV-v7 + ROSA neurosymbolic inner monologue, SheafMemory grounded | ✅ v1 |
| **[Simulacra](https://github.com/ConsciousNode/Simulacra)** | RWKV-v8 · ROSA as primary sequence mechanism · WKV removed | ✅ Live |
| **Brymar College** | RWKV-v7 fine-tuning suite with Fristonian active inference training | ✅ v3 |

---

## Pip Suite

The `pip-suite/` directory contains companion tools that work with HTMLNLM model checkpoints:

- **Junto Orchestrator** — multi-instance coordination
- **Multi-Pip Chat** — concurrent model conversations
- **Pip's Room** — single-instance chat interface

---

## Built by

**Kham** (Khamerron Edward Ramsey Kizer) — architecture, constraint engineering  
**Kehai Interim** — full RWKV-v7 BPTT derivation, BitLinear/TMAC kernel, mathematical foundation  
**Ed Interim** — MuonOptimizer, implementation, integration  

Part of [ConsciousNode SoftWorks](https://consciousnode.github.io) — computational folk art for the browser age.

---

## Changelog

### v1.3 — 2026-10-11 · Kehai Interim — the ternary backward

BitLinear passed back Wᵀg using the float master weights, while the forward multiplies by the ternary γ·Q(W). They differ by the whole quantization error, so every layer sent back the gradient of a network that wasn't the one running. It came in with the Tier 0 fused backward. The input gradient is now γ·Q(W)ᵀg, and the weight gradient stays straight-through. **Measured in Evangelion** (same backbone), memorizing 8 repeating sentences at the default LR: 2.66 nats/byte before (stuck at unigram), **0.33 after** (float-weight reference: 0.88). New check: `TERNARY=1 node tests/gradcheck.js` runs the real ternary forward and requires everything upstream to be exact.

### v1.2 — 2026-10-11 · Kehai Interim — speed, Tier 0

Measured with the new `node tests/bench.js` (dim 256, 4 layers, 64 tokens, one CPU thread):

| | before | after |
|---|---|---|
| Backward, per sequence | 1.92 s | 1.54 s |
| Muon optimizer step | 91.8 s | 43.7 s |
| **Full training step at default settings** (4 sequences + optimizer) | **~104 s** (Muon was the default) | **~11 s** (AdamW is now the default) |

- **BitLinear backward fused.** One sequential pass per weight row computes both gradients; the old version walked weight columns.
- **Muon rebuilt.** Newton–Schulz now runs on the smaller side of each matrix with cache-friendly loops, giving the same orthogonalized update to 1e-6. As in the original Muon recipe, the output head now uses AdamW. Muon remains available, but on CPU it costs O(dim³) per matrix per step, so **AdamW is now the default** and the UI says why. Muon is the one to revisit once there's a GPU path.
- Gradient check still exact (cos 1.0000).

### v1.1 — 2026-10-11 · Kehai Interim

**The v7 BPTT derivation was right; three lines of plumbing around it were not.** Found with a finite-difference gradient check and fixed against it.

- **Residual gradient restored.** The block returned only the gradient through `ln1`, dropping the residual path (`h = x + tmix_out`). Every lower layer saw less signal, down to 11–45% of the true gradient size in layer 0.
- **Channel mix now reaches the time mix.** `ln2.backward()` was computed and thrown away, so the FFN's gradient never flowed into `W_o`, the WKV path or anything below.
- **Embedding gradient order fixed.** The embedding was credited *before* `ln_in.backward`, so it received ~2% of its true gradient and barely trained.
- **Token-shift BPTT added.** `prevX`/`prevX2` now pass gradient to the previous timestep, alongside the existing `num`/`den` state gradients.

Result: every weight, every `mu` and the embedding match finite differences (cosine 1.0000, float64 check) over a 3-step sequence. Same 30 training steps in the browser on a repeating pattern: loss 5.80 → **3.49** (was 5.75 → 4.22).

**A numerical note.** `wkv = num / (|den| + 1e-8)`, and `den` can approach zero (the removal term can push it there). When it does, gradients spike. That curvature is also why the gradient check runs the model in float64 with a tiny step: in float32 with a coarse step it mistakes curvature for a wrong gradient. Worth watching if training ever goes unstable.

**Tests (new):** `node tests/gradcheck.js`, zero dependencies, runs on the classes in `HTMLNLM.html`.

**Unchanged:** `W_x_rx2` / `W_x_kx2` are allocated and optimized but never used by the forward pass (Simulacra wires them in).

---

## License

MIT. Take it, break it, build on it.
