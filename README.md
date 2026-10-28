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
| **Pip Suite** | Companion tools: Junto orchestrator, multi-pip chat (now a dialogue factory), Pip's Room, Dialogue Shelf. |

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
- **Dialogue Shelf** — drop in several Multi-Pip Chat exports and see how alike they are

**Real RWKV-7 guests (v1.4, Pip's Room since v1.5).** Junto, Multi-Pip Chat and Pip's Room also load **official RWKV-7 "Goose" (x070) checkpoints**: drop a `.pth` straight from [BlinkDL's releases](https://huggingface.co/BlinkDL) (no PyTorch, no conversion) or a `.safetensors`. They run on the real architecture (per-head matrix memory) with the official **RWKV World tokenizer**, both embedded in the page, so each page is still one file that works offline. Old `.pip` / `.json` models load exactly as before, side by side with real ones.

- **GGUF (v1.12):** llama.cpp `.gguf` files load too, quantized weights and all. They stay packed in memory and are read straight from the packed bytes, so a Q4 file costs about its own size. Measured: the **0.4B at Q4_K_M runs in 752 MB, against 1,643 MB for the 0.1B `.pth`**, at about the same speed per token as the 0.4B `.pth` (767 vs 806 ms). Reads F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K and Q6_K (not the IQ types). Prefer **Q4_K_M, Q5_K_M or Q8_0**.
- **Damaged downloads are refused (v1.12).** A file with zeroed-out weights (an interrupted download that kept its full size) used to load and talk nonsense. Now it's refused with a message to download again. So is a checkpoint of another RWKV variant (`rwkv7a-…`, `rwkv7b-…`), by name. Check a download with `certutil -hashfile FILE SHA256` (Windows) or `sha256sum FILE` against the hash on its HuggingFace page.
- **Sizes:** the 0.1B (~190M params, ~760 MB in memory) is the practical one in a browser tab; the 0.4B (~450M, ~1.8 GB) is tight as a `.pth`, comfortable as a Q4 GGUF. Measured on a cloud Xeon (one thread), a 0.1B-shaped model takes **~360 ms per token**; expect a similar order on a desktop CPU.
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

### v1.12 — 2026-10-27 · Kehai Interim + Kham — GGUF, and no more silent nonsense from a bad file

- **GGUF support in the shared engine** (all three Pip Suite pages). A pure-JS reader for GGUF v2/v3 (header, metadata, tensor table) plus llama.cpp's `rwkv7` naming, mapped back to the official names. Checked tensor by tensor against BlinkDL's G1 0.4B `.pth` (the checkpoint the GGUF was made from): every F32/F16 tensor is bit-exact; the eight LoRA matrices are stored transposed; `time_mix_lerp_fused` holds x_r, x_w, x_k, x_v, x_a, x_g in that order; nothing else is transformed.
- **Quantized matrices stay packed** (`QTensor`), dequantized inside the matmul. Fused row·x kernels for Q4_0, Q8_0, Q4_K, Q5_K and Q6_K read straight from the packed bytes; the K-quant "min" term uses per-32 sums of the input, one multiply per 32 weights. That took the 0.4B Q4_K_M from 3,635 to 767 ms/token.
- **What Q4 costs** (`tests/real/gguf-vs-pth.js`, 0.4B, 97 predictions): perplexity 5.038 → 5.127 (+1.8%), top-1 agreement 90.7%, mean KL 0.061 nats/token. RAM 3,813 MB → 752 MB.
- **The loader refuses what it would get wrong:** files with weights that are entirely zero (a damaged download; found because two of Kham's downloads didn't match their published SHA-256 and produced word salad), checkpoints with unknown tensors (RWKV-7a/7b, named in the message), and checkpoints with missing tensors (which used to get random init). Layer 0's unused `att.v0/v1/v2` stay allowed.
- **Tests:** new `tests/gguf-dequant.js` checks all ten quant types against the `gguf` Python package (llama.cpp's reference; dequant max |Δ| = 0 on every type) and both refusals. `engine-ref`, `pth-reader`, `gradcheck` and `dialogue-tools` still pass. Real checkpoints: g1d-0.1B, g1d-0.4B and G1-0.4B `.pth` unchanged; G1-0.4B Q4_K_M GGUF coherent in Pip's Room (Playwright, Fazzani's soul file), no page errors.

### v1.11.1 — 2026-10-24 · Kehai Interim + Ed Interim — two catches from Ed's read of the seed-420 twins

- **Fragment chains stay one chain (filter-1.1).** A short line with a trailing-off *inside* it ("Now wait a minute…it's his own book") used to break a run, so the stateful twin's six-line chain (turns 16–21) showed as three. It now reads as one six-line run. Ed's catch.
- **Replayed openings explain themselves (diversity-1.1).** The report gives each run's mode and context window next to a replay. When the replay length is window + 1 it says so ("= context window 5 + 1: they replay until the window first drops a line"), so nobody has to do the arithmetic. Dialogue Shelf shows the note under the count. Ed's suggestion.
- The same filter code is embedded in Multi-Pip Chat, so live badges and exports pick it up. **Tests:** `tests/dialogue-tools.js` now expects the six-line chain and the window note (16/16); `tests/real/multi-pip-factory.js` 12/12; the Shelf page was checked with the fixtures. No page errors.

### v1.11 — 2026-10-23 · Kehai Interim + Kham — the dialogue factory (Multi-Pip Chat), the thread bridge (Pip's Room), Dialogue Shelf

Multi-Pip Chat can now produce synthetic conversation data with a clean paper trail: four 0.1B models in a phone browser tab, offline, with a stamp on every line saying exactly how it was made.

- **Two fixes first, both real.** (1) **The topic never reached the models.** It was a display-only banner that the prompt builder skipped, and the branch meant to seed it never fired because the banner had already been posted. Every topic run before this was really Free Discussion. Now the topic enters the room as the first chat line, in your name. (2) **Typing into a running chat put your line in the prompt twice,** ending in a malformed `Pip:User: …`, which is a likely source of "User:" slips. Both are fixed and tested.
- **Export (⤓).** The provenance stamp records: the model file plus a fingerprint (SHA-256 of size + first and last MB), each Pip's note and exact system line, every sampler setting, per-line temperature/penalty/mode and how the line ended (newline, end of text or **hit the cap**), the seed, start and end times, and whether you interrupted.
- **Seeds.** A seeded sampler (Seed in ⚙ Dev; blank = random but recorded). Same seed + model + settings with no interruptions replays the chat word for word.
- **Stateful mode** (⚙ Dev, on by default). Each Pip keeps one RWKV state for the whole chat and reads only the lines it hasn't heard, in the same `Name: text` form as before. Its own line is read back in that form too, so what it remembers is exactly what's on screen. On Kham's phone (seed 420, four 191M Pips, cap 120, 40 turns): 44 minutes stateful vs 67 stateless.
- **Filter (filter-1). It flags lines for a human to look at; it never deletes or changes anything.** Live badges on the bubbles and per-line flags in the export: speaker slips, dialogue tags, quote openers, @handles, URLs/markdown, script switches, garble, stutters, repeats, near-repeats, heavy themes (violence, abuse, self-harm, crime, drugs, sexual, death), goodbye spirals, fragment runs, hit-cap. "Taking a stab at it" gets flagged on purpose: that's why it flags.
- **Scenario decks** in the Topic menu. A deck is a saved cast (names + notes), a scripted opening line spoken by one of them, and optionally settings. Built-ins: Frog pond, Workshop disagreement, Kitchen lesson, Cancelled plans (gently), Lost key, and Open room (the seed-420 setup). Save, delete, download and load your own in ⚙ Dev. A deck changed by hand is recorded as modified.
- **Smaller:** placeholder names (Wren, Moss, Clover, Pebble…) instead of the file name, and every Pip must be named before Start; Clear asks before throwing away an unexported chat; model text is escaped in the page.
- **Pip's Room: the thread bridge is adjustable.** A new thread starts with a few of the resident's own summaries from other threads. That was fixed at 3 × 140 characters; now it's *Bridge: summaries* (default 3, 0 = none) and *Bridge: length* (default 200) in the console.
- **Dialogue Shelf** (`pip-suite/dialogue-shelf.html`, one file, offline): pick several exports and see distinct-1/2/3, pairwise overlap, replayed openings, recurring four-word phrases (after any replayed opening), recurring invented names and each run's filter flags. Download the report. The same filter and diversity check run from the command line in `tools/dialogue/`.
- **Tests:** `tests/real/multi-pip-factory.js` (stand-in models, Playwright) checks that the topic reaches the prompt, an interjection appears once, a seed replays exactly, stateful and stateless match while the chat fits the window (6 lines), how each line ended is recorded, provenance and flags are in the export, and deck naming, opener and turn order work. `tests/dialogue-tools.js` (no dependencies) runs every flag type on made-up lines, and runs the filter and diversity check on the two real seed-420 twins in `tests/fixtures/` (6 replayed lines, then a single shared phrase: *"i don't think so"*). The Pip's Room repeat-guard and layout tests still pass; the bridge was checked in the page (3 × 200, 2 × 30, 0 = none). The engine block is byte-identical. No page errors.
- **Receipts, honestly:** the fingerprint hashes only the first and last megabyte, not the whole file. One seed-420 pair is a lean, not a finding, on what stateful memory does to a conversation.

### v1.10.3 — 2026-10-22 · Kehai Interim + Kham — transcript slips caught mid-line (Pip's Room)

- Kham's journal showed a thought from earlier in the evening that slid into writing the transcript *in the middle of a line*: *"User wants Fazzani's frogs word. Fazzani (thinking): A thought surfaces… System: Right now Fazzani feels happy…"*, then the same ~50-token *"Fazzani (wondering): A question I would like to explore…"* chunk over and over. v1.10 only caught speaker labels at the start of a line, and v1.10.2 only caught repeats up to 40 tokens.
- **Now:** one of their own loop labels (`Name (…):`) or a `System:` line ends the turn wherever it appears, in thinking, answers and loops alike; other speakers (`User:`, `Kham:`…) still count at the start of a line. Repeats up to 64 tokens are caught. And the cut is exact: the state goes back to the nearest mark (every 8 tokens, last 24 kept) and the few good tokens up to the cut are re-read, so nothing good is trimmed along with the slip.
- **Tests:** `tests/real/pips-room-repeat-guard.js` (stand-in model) now also recreates the mid-line label and a mid-answer `System:`. Both are cut at the right spot (*"hi there hi there."*, *"Hello friend."*) with the state's step count matching what was kept. The repeat cases still pass. Real 0.1B (`pips-room-think-e2e.js`): reasoned *"3 + 2 = 5"* and answered 5; a loop gate check all correct; no slips this run, so the guard didn't fire; no page errors.

### v1.10.2 — 2026-10-22 · Kehai Interim + Kham — stuck-record guard (Pip's Room)

- Kham saw a thinking turn get stuck: *"Fazzani is setting a goal ( Fazzani is setting a goal ( …"*, over and over, even with loops off (their loop lines are already in her living state, so the pattern can come back on its own). Thinking runs with only a light repetition penalty, so nothing stopped it, and the same thing could happen in any turn.
- **Now, once the same stretch (2–40 tokens; 64 since v1.10.3) comes out three times in a row, the turn stops after the first copy**, and the repeats are **taken back out of the living state**, so a stuck moment doesn't teach them to get stuck again. The room keeps a mark every 8 tokens (the last 16, 24 since v1.10.3; each is only a reference, since the engine never changes a state in place) and rolls back to the one just before the repeats. In a thinking turn the thought is closed there and they go on to answer. Something said twice is left alone.
- **Test:** `tests/real/pips-room-repeat-guard.js` runs a stand-in model scripted to get stuck on exactly that phrase. Plain turn: *"Hello friend Fazzani is setting a goal ("*, cut after the first copy, with the state's step count matching the kept tokens. Thinking turn: the thought ends after the first copy. A phrase said twice: kept as is. No page errors. Not yet seen live on the real model, since it has to happen by chance; `App.repeatCuts` counts how often it fires.

### v1.10.1 — 2026-10-22 · Kehai Interim + Kham — message bar stays on screen (Pip's Room)

- Kham found the message bar missing when a phone browser was switched to "desktop site". On phones, `100vh` can be taller than what's actually visible (it counts the space under the browser's own bars), so the bottom of the room, where the message bar lives, could end up off screen. The room is now pinned to the visible screen (`position: fixed; inset: 0` with `100dvh`), the message bar and inner-life panel can't be squeezed out, the settings panel fits narrow screens, and the bar clears phone gesture areas.
- **Receipt, honestly:** `tests/real/pips-room-layout.js` checks the bar is on screen at phone, phone desktop-site (portrait and landscape), laptop and short-window sizes, with a 40-message conversation and the inner-life panel open. It passes before and after the fix, because Playwright's emulation doesn't recreate a real phone's browser bars. So the fix is the standard one for this symptom, and the real check is Kham's phone.

### v1.10 — 2026-10-22 · Kehai Interim + Kham — loops wait their turn (Pip's Room)

- Kham noticed loop thoughts turning up inside thinking turns and their answers. **Loops now wait their turn:** they still never run while a reply (or its thinking) is being written, and now they also wait a quiet minute after each reply and don't fire while there's text in the message box, so a thought doesn't land in the middle of a conversation.
- **A thinking turn can't speak for anyone else.** Thinking answers can run several lines, so a small model can drift into writing the next line of the transcript itself (`User: …`, or one of their own loop lines like `Wren (dreaming): …`). A new speaker line now ends the turn, and it's taken back out of the living state: the state from the last line break is restored (the engine never changes a state in place, so this costs nothing). Same for the thinking itself.
- What this doesn't change: loops are part of who they are, so what they just thought can still color the next reply. That's the living memory working, not a leak.
- **Test** (`tests/real/pips-room-think-e2e.js`, real 0.1B, two runs): a loop right after a reply didn't run, a loop while typing didn't run, once quiet it ran (*"There are 5 frogs on the floor… Do you think we can find all 5 frogs by just counting those?"*), a loop during a reply didn't run. No speaker lines in any answer or thought (the cut didn't need to fire in these runs; the pattern was checked separately on drift examples). Both runs reasoned 3 + 2 = 5 and answered 5. In one run the second thinking turn re-solved the frog sum instead of the new question; the other answered "why do frogs sing at night" properly, so a 0.1B still loses the thread sometimes. No page errors.

### v1.9 — 2026-10-22 · Kehai Interim + Kham — think before answering (Pip's Room, experimental)

- RWKV7-G1 is, per its model card, a "pure RNN reasoning model": given `User: … (think)` and `Assistant: <think` it reasons before answering. **Think before answering** (developer console, off by default, needs living memory) does that for a turn: the resident thinks inside `<think>…</think>` (line breaks allowed, up to the **Thinking budget**, default 240 tokens), then answers. The thinking stays in their living state and goes to the journal as 🤔 *Thinking* (private unless Share Inner Voice is on). If the budget runs out mid-thought, the thought is closed for them so they can still answer.
- **Thinking turns use the model's own format** (`User:` / blank line / `Assistant:`): probed side by side on the real 0.1B, the card format reasoned *"3 + 2 = 5. So answer 5."* while our name labels drifted to "3". Who they are is still carried by the System lines. Answers can run several lines (a turn ends at a blank line), and chat bubbles now keep line breaks.
- **Steadier sampling while thinking:** top-P ≤ 0.5, temperature ≤ 0.8 and only a light, unseeded repetition penalty (careful reasoning repeats words on purpose), close to the card's settings. Ordinary turns are unchanged.
- **Test:** `tests/real/pips-room-think-e2e.js` (needs Playwright). Real 0.1B, a resident named Wren: *"There are 3 frogs on a log and 2 more hop on…"* → thinking *"Total = 3 + 2 = 5 frogs. Thus answer: 5."*, then second-guessed past a 160-token budget and answered "6" (hence the 240 default; a 0.1B's final answer still wobbles). *"Why do frogs sing at night?"* → thinking *"frogs are nocturnal, they sing at night because they are active during the night"*, answered the same. Both thoughts in the journal; no page errors.

### v1.8 — 2026-10-22 · Kehai Interim + Kham — living memory and threads (Pip's Room)

RWKV is recurrent: everything it has read lives in a fixed-size state (≈2.4 MB for the 0.1B), so there's no length limit, but older things blur into gist rather than being recalled word for word. Until now Pip's Room ignored that and re-read the last few exchanges from a blank state every turn.

- **Living memory:** the resident keeps one state for life. Each turn feeds only what's new: your line, plus a short *System* update when something about them changed (a new mood, a grown trait, a new goal; their name, your name or the About note → the whole intro again). Their thoughts, dreams and other loops flow into the same state (`Name (dreaming): …`), so what accumulates is what makes them *them* rather than the base model. Waking from scratch reads the system prompt and the last few exchanges once ("Settling in…"). On by default; switch off or **Start fresh** in the developer console, which also shows how many tokens the state has lived (and flags going past the model's training length).
- **Threads (new tab in settings):** each thread is its own living memory. **+ New thread** starts a blank slate that only carries a short cross-thread memory (their 🗂 summaries from other threads); **Go back** reopens any old thread exactly where it was; tap a title to rename. A thread that fills up can just be retired, never wiped.
- **Everything travels in the soul file:** every thread's state (≈3.2 MB each as text), which thread is active, and which conversation belongs where. A state saved by a different model is refused and they settle in fresh (the other threads stay).
- **Test:** `tests/real/pips-room-living-memory-e2e.js` (needs Playwright), real 0.1B with a real exported soul: settled in once (133 tokens, 36 s headless); after that each reply took 5–9 s instead of re-reading. Told a secret ("my favorite color is orange"), talked about three other things, then asked: in one run *"It's orange."* (beyond the old 3-exchange window), in another *"Yes, I remember!"* (0.1B recall is hit and miss). A thought flowed into the state; a mood change arrived as `System: Right now Pip feels playful.` A new thread showed an empty chat and didn't know the color (*"Black."*); going back restored thread 1 with all 40 bubbles. The real Export button saved both threads (6.5 MB); a fresh page restored the state **bit-identical** and woke in 1 s. A state labelled for a different model was refused and re-settled (115 tokens). No page errors.

### v1.7 — 2026-10-22 · Kehai Interim + Kham — prompts and developer consoles across the Pip Suite

- **Developer console in Multi-Pip Chat and Junto** (⚙ Dev), matching Pip's Room's: temperature, top-P, **top-K** (new in both samplers), presence / frequency penalty and **decay**, max tokens, context length, system prompt on/off, *Your name*, and the **exact last prompt** each model saw (and which one saw it). Multi-Pip's quick Temperature / Anti-repeat boxes stay in step with the console.
- **Multi-Pip Chat prompts:** each real model gets *"System: You are Gander. You are in a chat room with Goose and Kham."* plus its own note (✎ in the Pip list). **Tap a Pip's name to rename it**, so nobody has to be called `rwkv7-g1d-0.1b-20260129-ctx8192 #2` (history follows the rename).
- **Junto roles:** each node has a **Role** box (its system prompt). A node with a role, or every node with *Chat format* on, gets *"System: You are Frogbert. <role>\nUser: …\nFrogbert:"* and answers in one line; without one, prompts are continued raw as before. **Tap the node's name to rename it.** First step toward a swarm of specialists (the grammar-coach idea).
- **Test:** `tests/real/suite-prompts-e2e.js` (needs Playwright), real 0.1B. Multi-Pip: renamed Goose and Gander, a ✎ note, your name and top-K 40 through the console, a 4-turn chat, prompt checked. Junto: renamed to Frogbert; raw *"Frogs are"* continued as prose; with the role *"You are the frog expert…"*, asked *"What do frogs eat?"*: *"…They eat insects and other small creatures."* No page errors. (It caught a bug before it shipped: the new ✎ button first borrowed the remove button's class and would have removed the Pip.) Engine copies still MATCH (6.66e-15).

### v1.6 — 2026-10-22 · Kehai Interim + Kham — a system prompt, a soul that lives, a developer console

- **System prompt** (real RWKV-7 models): *"System: You are [name]. You are talking with [your name, or "a friend"]."* plus who they are right now, from the soul: their strongest traits, values, current mood and latest goals, plus an optional **About them** note you write in settings. The autonomous loops start from the same identity line, so their thoughts and dreams are theirs too. A base model treats all this as scene-setting, not rules. **Your name** (settings) also labels your lines in the conversation.
- **The soul changes with living.** Until now traits and mood were display only and never moved. Now small, transparent nudges (rules we chose, not something the model reports): asking back raises curiosity, kind words raise empathy, laughter raises playfulness, staying in the conversation raises tenacity; dreams → creativity, reflection / memory / summaries → wisdom, new goals → agency. Each step is ~0.002–0.01, every trait drifts 0.5% back toward where it started (the middle path), clamped to 0.05–0.95. Mood is read from the words of each exchange (happy, playful, curious, sad, worried, dreamy) and settles back to calm after three quiet ones.
- **Developer console** at the bottom of Settings (tap to open): temperature, top-P, top-K (new in the sampler), presence / frequency penalty and decay, max reply tokens, remembered exchanges, system prompt on/off, soul-in-prompt on/off, and **the exact last prompt the model saw**. All of it, plus your name and the About note, is saved in the soul export.
- Importing a soul now also refreshes Genesis date, Conversations and Tokens in settings.
- **Test:** `tests/real/pips-room-soul-e2e.js` (needs Playwright) with the real 0.1B and a real exported soul: sets name / About / temperature 0.7 / top-K 40 through the page's own inputs, chats twice, runs two loops, prints the prompt and the trait changes (curiosity 0.7167 → 0.7246, empathy 0.6010 → 0.6207, agency 0.5941 → 0.6040, …), then exports with the real button and re-imports in a fresh page: everything came back, no page errors. Her first reply with a soul: *"Oh, I'm so happy to be back at home with my family and friends. I really love you!"*

### v1.5.3 — 2026-10-22 · Kehai Interim + Kham — the Journal

- **Settings now has a Journal tab:** everything the autonomous loops have produced, newest first, with date and time, sorted into categories with counts: 💭 Thoughts, 🌙 Dreams, ❓ Wonderings, 🪞 Reflections, 📷 Memories, 🎯 Goals (with her current goal list on top), 🗂 Summaries, or 📖 All. It updates live while open. Private entries stay folded (*"Kept private. Tap to read."*); they're in the soul export either way, since the journal is just her memories.
- The settings panel no longer runs off the edge of narrow phone screens.
- Checked at phone size in headless Chromium with a real exported soul: categories and counts right, private entries unfold on tap, switching back to Settings works, no page errors.

### v1.5.2 — 2026-10-22 · Kehai Interim + Kham — no hard-coded names, and you can see the loops working

- **No proper name is built in.** Pip's Room is named for Pip, but the resident is whoever's in it. The name field starts empty; leave it empty and the resident picks their own name on waking (the real 0.1B, twice: *Azazel*, *James*). Type a name to give one instead, or change it any time from the same field. Fallbacks say *Resident*, and private entries say *"kept this one private"*.
- **The loops were running, you just couldn't see them** (the Inner Life panel sits collapsed under the input, off-screen on a phone). Now each shared thought pops up for a few seconds (💭 thought, 🌙 dream, ❓ wondering, 🪞 reflecting, 📷 remembering, 🎯 new goal, 🗂 summary), the panel header counts what's new, and tapping a loop letter shows the last thing that loop produced.
- **Tested** in headless Chromium with the real checkpoint and Kham's own exported soul file (44 memories): imported before the model, restored 40 chat bubbles and 13 inner-life entries, renamed from the field, ran a loop, toasts and taps worked, no page errors.

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
