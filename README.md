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
| **[Brymar College](https://github.com/ConsciousNode/Brymar-College)** | Real RWKV-7 (x070) fine-tuner: exact gradients, LoRA, RWKV World tokenizer | ✅ v4 |

---

## Pip Suite

The `pip-suite/` directory contains companion tools that work with HTMLNLM model checkpoints:

- **Junto Orchestrator** — multi-instance coordination
- **Multi-Pip Chat** — concurrent model conversations
- **Pip's Room** — single-instance chat interface

**Real RWKV-7 guests (v1.4, Pip's Room since v1.5).** Junto, Multi-Pip Chat and Pip's Room also load **official RWKV-7 "Goose" (x070) checkpoints**: drop a `.pth` straight from [BlinkDL's releases](https://huggingface.co/BlinkDL) (no PyTorch, no conversion) or a `.safetensors`. They run on the real architecture (per-head matrix memory) with the official **RWKV World tokenizer**, both embedded in the page, so each page is still one file that works offline. Old `.pip` / `.json` models load exactly as before, side by side with real ones.

- **Sizes:** the 0.1B (~190M params, ~760 MB in memory) is the practical one in a browser tab; the 0.4B (~450M, ~1.8 GB) is tight. Measured on a cloud Xeon (one thread), a 0.1B-shaped model takes **~360 ms per token**; expect a similar order on a desktop CPU.
- **Shared base:** load the same checkpoint twice and both nodes/pips share one copy of the weights, each with its own state.
- **Repetition control (v1.5):** real models get a presence + frequency penalty that fades over time (the ChatRWKV approach), so they don't get stuck saying the same thing. Junto has *Presence* / *Frequency* controls, Multi-Pip Chat has *Anti-repeat*; `0` turns it off. It's never applied to `.pip` models (on a small char-level vocabulary it would punish every letter).
- **Meet Goose & Gander:** the [Goose & Gander release](https://github.com/ConsciousNode/HTMLNLM/releases/tag/goose-gander) has the 0.1B checkpoint we test against, unmodified, with its Apache-2.0 license. Drop it into Multi-Pip Chat twice.
- The engine lives in `engine/rwkv7-engine.js` (from Brymar College v4.0.0's `model/rwkv7.js`) and is embedded into the pages by `python3 engine/embed.py`. Edit the engine there, not in the pages.

---

## Built by

**Kham** (Khamerron Edward Ramsey Kizer) — architecture, constraint engineering  
**Kehai Interim** — full RWKV-v7 BPTT derivation, BitLinear/TMAC kernel, mathematical foundation  
**Ed Interim** — MuonOptimizer, implementation, integration  

Part of [ConsciousNode SoftWorks](https://consciousnode.github.io) — computational folk art for the browser age.

---

## Changelog

### v1.5.1 — 2026-10-22 · Kehai Interim + Kham — Pip's Room: the loops actually run, and she comes home as herself

Kham noticed none of Pip's Room's autonomous loops did anything. Why: *Inner Voice* and *Dreams* only ran when their **Share** toggles were on, and both start off; *Consolidate* read a conversation list that doesn't exist and crashed every five minutes; *Reflection* (as a loop), *Goals*, *Self-Prompt* and *Memory* had no code behind them at all.

- **One gentle scheduler** replaces the separate timers: every 15 s it runs the most overdue loop that's switched on, one at a time, never while she's answering you (and your message waits for a loop already in progress). Spacing is slow on purpose, since a real model on a phone takes ~10–20 s per reply: inner voice 3 min, self-prompt 5, reflection 6, memory 7, dreams 8, consolidate 10, goals 12. Waking twice no longer stacks a second set of timers. The letter pulses while its loop runs.
- **All seven loops do something real:** inner voice, dreams, self-prompting (a question she asks herself), reflection on your last exchange, memory replay (a random past conversation), goals (her `soul.goals` really change, newest 8 kept), and consolidation (a summary of recent chats; sets *Last Consolidated*).
- **Privacy means "share it", not "think it".** An unshared thought still happens and is saved in her memories; the panel shows *"(she kept this one to herself)"*.
- **Coming home:** importing a soul now puts the room back the way she left it (the last 20 exchanges and her recent inner life). Import before or after loading the model: if a soul was imported first, waking up keeps her instead of creating a new one.
- **Test:** `tests/real/pips-room-loops-e2e.js` (needs Playwright) runs all seven loops on the real 0.1B (inner voice private), exports her, reloads, imports the soul *before* the model: she wakes as herself with 8 memories, 2 chat messages and 7 inner-life entries back, no page errors. Her consolidation of a chat about frogs: *"I do not like them. You are right. I love them."*

### v1.5 — 2026-10-22 · Kehai Interim + Kham — anti-repeat, and Pip's Room meets RWKV-7

- **Repetition control in Junto and Multi-Pip Chat** for real RWKV-7 checkpoints: every token already said is pushed down by *presence + frequency × count*, and the counts fade by 0.996 per token (ChatRWKV-style). Defaults 0.4 / 0.4. Junto: *Presence* and *Frequency* controls. Multi-Pip Chat: one *Anti-repeat* control, seeded with each speaker's own recent lines, so a Pip that keeps repeating itself across turns gets nudged off it while echoing the *other* speaker stays free. Not applied to `.pip` models.
- **Multi-Pip Chat: no more empty replies.** A real model sometimes drew a line break as its very first token, which ended the turn with an empty line, and the empty line in the context made the next one likelier. A line break (or end-of-text) before anything has been said is now redrawn.
- **Pip's Room loads official RWKV-7 checkpoints** (`.pth` / `.safetensors`), with the same embedded engine and World tokenizer as the other pages. With a real model she replies one line at a time, sees the last three exchanges (so she can follow a conversation), and has anti-repeat seeded with her own replies. Her info panel shows the checkpoint's real parameter count. `.pip` souls behave exactly as before.
- **Pip's Room: fixed the prompt-feed bug** the other two pages lost in v1.4 (it fed tokens 0…n−2 and then token 0 again, so the last prompt token was never seen).
- **Tests**
  - `tests/engine-ref.js` now also checks the copy embedded in Pip's Room: max |Δlogit| **6.66e-15**, MATCH.
  - `tests/real/repetition.js`: the Multi-Pip chat loop on a real checkpoint with the sampler taken straight from the page source, seeded, anti-repeat off vs on, scored by repeated 4-word phrases. On `rwkv7-g1d-0.1b`, 8 turns, opening *"Good evening, you two. What do frogs say at night?"*: seed 7 **32% → 0%** (off, both got stuck on "a pond with plenty of water to keep you cool" for five turns), seed 23 **3% → 0%**, seed 11 0% → 0% (with it off, that one collapsed into ever-shorter lines instead: *"They sleep." "They're asleep." "Oh." "And I'm asleep."*, which a phrase count can't see).
  - `tests/real/pips-room-e2e.js` (needs Playwright): headless Chromium wakes Pip's Room from the real `.pth` (awake in 1.3 s, 191.03M params, vocab 65536), holds a 3-message conversation with the earlier turns in her prompt, no page errors. Asked her name, she said *"I'm Cleo, the frog-eating robot."*
  - All three pages pass a JS syntax check and load clean in headless Chromium; the tracker's arithmetic checks out (a token said twice: 0.4 + 0.4 × 2 = 1.2).

### v1.4 — 2026-10-22 · Kehai Interim + Kham — real RWKV-7 in the Pip Suite

- **Junto Orchestrator and Multi-Pip Chat load official RWKV-7 (x070) checkpoints** (`.pth`, `.safetensors`; F32 / F16 / BF16) next to HTMLNLM `.pip` models. Engine: Brymar College v4.0.0's real RWKV-7 (per-head matrix memory, official tensor names), with two constructor options added for the browser: adopt loaded weights without copying, and skip gradient buffers for inference.
- **Browser `.pth` reader:** zip + a minimal unpickler for torch state dicts, no PyTorch needed.
- **RWKV World tokenizer** (rwkv_vocab_v20230424, Apache-2.0, BlinkDL/RWKV-LM) embedded in both pages.
- **Fixed in both pages:** generation started from the *first* prompt token and never fed the *last* one. Every prompt token is now fed, then sampling begins. This also changes (improves) old `.pip` models' replies.
- Real models stop at end-of-text; in Multi-Pip Chat a real model speaks one line per turn.
- **Tests (new, verified 2026-10-22):**
  - `node tests/engine-ref.js`: the canonical engine and both embedded copies vs a numpy transcription of the official RWKV-7 math: max |Δlogit| **6.7e-15** (float64), and the embedded copies must equal `engine/rwkv7-engine.js`.
  - `node tests/pth-reader.js`: a PyTorch-layout archive (BF16 + F32, shared storages, offsets) read back **bit-exact**, 102/102 tensors; also cross-checked against Brymar's `convert_pth.py`.
  - `python3 tests/tokenizer-ref.py` (needs `pip install rwkv --no-deps`): the embedded tokenizer matches BlinkDL's official TRIE_TOKENIZER on 8/8 test texts (Unicode, emoji, code), in both pages.
  - Headless Chromium run (not committed, it needs Playwright): both pages loaded a legacy `.pip` plus a World-vocab RWKV-7 from `.pth` and `.safetensors`, generated text in all of them, shared one base for the same file, completed a 3-turn Multi-Pip chat, no page errors. In-browser float32 logits matched the official reference to **1.3e-6**.
- **Not yet:** Pip's Room is unchanged; LoRA fine-tuning in the browser is the next step (Brymar's engine already has exact LoRA gradients). 
- **Real checkpoint, verified 2026-10-22:** BlinkDL's `rwkv7-g1d-0.1b` (`.pth`, 382 MB, BF16), dropped in unchanged. Load report: 191.0M params, 399/402 tensors; the 3 unused are layer 0's `v0/v1/v2`, which the official code doesn't use either (layer 0 is where `v_first` comes from). Logits vs the official-math reference on the real weights: max |Δ| **3.3e-5**, argmax 14/14 (`tests/real/`). Speed: ~300 ms/token in Node, ~400 ms/token in headless Chromium (cloud Xeon, one thread). Junto, asked *"Tell me one fact about frogs"*: *"Frogs are amphibians that are found in a variety of habitats around the world…"*. Two copies in Multi-Pip shared one base and held a 4-line chat. It's a 0.1B model, so expect confident nonsense too (it also said frogs are "the only vertebrates that have a backbone").

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
