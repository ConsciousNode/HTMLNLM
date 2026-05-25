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

> **Successor available:** [**HTMLNLM Evangelion**](https://github.com/ConsciousNode/HTMLNLM-Evangelion) extends this work with omnimodal input (vision, audio, spatial), SheafMemory topological contradiction detection, BooleanPhaseDynamics, AutopoieticOptimizer self-correction, and RIFT Endospace visualization. HTMLNLM remains the stable text-only runtime.

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
| **EvaROSA** | RWKV-8 + ROSA neurosymbolic inner monologue, SheafMemory grounded | 🔧 In development |
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

## License

MIT. Take it, break it, build on it.
