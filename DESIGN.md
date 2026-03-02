# ERM Diffusion Overhaul — Design Document

## Overview

This document describes the tokens-in → tokens-out pipeline and
Mercury-style diffusion integration layered on top of the Emergent Route
Model (ERM) ant colony.

The key idea: replace the single-step denoising loss with a T-step
coarse-to-fine refinement loop where the ant colony acts as the parallel
edit proposer at each noise level.

---

## 1. Tokens-In → Tokens-Out Pipeline

### 1.1 Tokenizer API

All training and inference code is now generic over the `TokenizerApi` trait:

```rust
pub trait TokenizerApi: Send + Sync {
    fn encode_text(&self, text: &str) -> Vec<u32>;
    fn decode_text(&self, ids: &[u32]) -> String;
    fn vocab_size(&self) -> usize;
    fn mask_id(&self) -> u32;
    fn pad_id(&self) -> u32;
    // Stubs:
    fn encode_code(&self, _src: &str) -> ErmResult<Vec<u32>>;
    fn encode_image(&self, _bytes: &[u8]) -> ErmResult<Vec<u32>>;
}
```

`BpeTokenizer` implements this trait with BPE (byte-pair encoding).
`CharTokenizer` also implements it for backward compatibility.

### 1.2 BPE Tokenizer

- Vocabulary layout: `[PAD=0, MASK=1, UNK=2, subwords 3..]`
- Built by iterative merge of most-frequent adjacent byte pairs.
- Serialized to JSON for checkpoint portability.
- CLI auto-trains BPE from corpus sample (1M chars) or loads from file.

### 1.3 Training Example Format

Training examples are `(prompt_tokens, target_tokens)` pairs of shape `[B, L]`.

For books data:
- `use_paragraph_spans = true`: span boundaries at paragraph/sentence breaks
- `use_paragraph_spans = false`: sliding window with stride `L/2`

### 1.4 CPU Streaming Dataloader

```text
Disk files → [producer thread] → bounded_channel(2) → [trainer thread]
                tokenize+chunk on CPU                    GPU forward/backward
```

- Producer iterates `.txt` files in a directory, tokenizes on CPU, emits `TokenBatch`.
- Bounded channel (capacity 2) = double buffer: trainer always has a ready batch.
- Back-pressure: producer blocks when trainer is slow.
- No full corpus in GPU memory at any time.

---

## 2. Diffusion-Style Coarse-to-Fine Refinement

### 2.1 T Denoising Steps Per Training Iteration

At each training step, `T` diffusion levels are processed from heaviest
noise (t=T) down to lightest (t=1):

```
For t = T..1:
  z_t = corrupt(x, t)          # mask/replace tokens
  (logits, unc, h) = forward_with_hidden(z_t)
  colony_edit_step(z_t, logits, edit_scale=t/T)
  loss_t = γ(t) * CE(x | z_t)
  accumulated_loss += loss_t

accumulated_loss.backward()
optimizer.step()
```

### 2.2 Coarse vs. Fine Colony Edits

- At t=T (max noise): colony proposes `max_edits * (t/T) ≈ max_edits` edits (coarse).
- At t=1 (min noise): colony proposes `max_edits * 0.1` edits (fine).

This mirrors Mercury's parallel masked diffusion: early steps fill in large
spans, later steps refine single tokens.

### 2.3 Loss Formula

```
L = E_t[ γ(t) * CE(x | z_t) ]
```

where:
- `z_t` = noised latents (masked/replaced tokens) at level t
- `γ(t)` = noise level weight following configured schedule
- CE = cross-entropy between logits and clean tokens x

γ(t) schedule options (configured in `noise_schedule`):
- `"linear"`: γ(1) = γ_min → γ(T) = γ_max
- `"cosine"`: cosine ramp, smooth
- `"sqrt"`: square-root ramp

Pheromone deposit remains: `tanh(Δ/σ)` normalized (unchanged from original ERM).

### 2.4 RouteGraph Over Token Spans

`RouteGraph.route_aggregate()` consumes hidden states `[B, L, d]` from
`BurnScorer.forward_with_hidden()`. The graph aggregates evidence over token
positions (spans), not single characters. Colony proposals are now at the
token level.

### 2.5 Inference: Iterative Parallel Denoising

No autoregressive loop. Inference uses K refinement steps:

```
y_0 = [MASK, MASK, ..., prompt_prefix...]
For k = 0..K-1:
  t = (K-k)/K * T      # coarse→fine
  logits = scorer(y_k)
  y_{k+1} = fill top-(edit_scale * max_edits) masked positions greedily
Return y_K
```

CLI: `erm infer --checkpoint <dir> --length 256 --steps 8 --prompt "Once upon"`

---

## 3. Engineering Details

### 3.1 BurnScorer.forward_with_hidden()

Returns `(logits [B,L,V], uncertainty [B,L], hidden [B,L,d])`.
Used by both the diffusion loss (logits) and route aggregation (hidden).

### 3.2 Checkpointing

All checkpoints contain:
```
<dir>/
  scorer.bin     — burn binary weights
  scorer/        — burn directory format (alternate)
  graph.json     — RouteGraph state
  ant_state.json — AntState (streak counters, ant types)
  config.json    — ErmConfig
  step.json      — {"step": N}
```

Checkpoint structure applies to: `colony-train`, `diffusion-train`, warmstart/.

Mid-run save: `erm save-scorer --checkpoint <dir> [--output <out>]`

### 3.3 Metrics JSONL

Written every `log_every` steps (min 50) to `metrics.jsonl`:

```json
{"exp_id":"exp-a","step":50,"loss":3.42,"edits":47,"mean_phi":0.12,
 "deaths":18,"seq_len":512,"batch":2,"hidden_dim":192}
```

Fields: `exp_id, step, loss, edits, mean_phi, deaths, seq_len, batch, hidden_dim`

### 3.4 DiffusionTrainer

`DiffusionTrainer<B: AutodiffBackend>` owns:
- `BurnScorer<B>` — GPU scorer with autodiff
- Adam optimizer
- `RouteGraph` — pheromone routing structure
- `AntState` — ant lifecycle state
- `RunningDeltaStats` — normalized pheromone deposit
- `ErmConfig`, `PheromoneConfig`

---

## 4. K8s Experiments

6 variants with different seq_len/hidden_dim/ants/batch/T configurations
time-sliced across 4 GPUs on theshire.

Experiment IDs: exp-a through exp-f.
Each writes to `/home/kadajett/dev/erm-rust/data/experiments/<exp-id>/`.

See `k8s/experiments/` for manifests.

---

## 5. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| BPE over char-level | Subword tokens reduce sequence length 5–10×, reducing memory and improving semantics |
| T-step diffusion loop | Enables parallel refinement (Mercury-style) without AR bottleneck |
| Colony as parallel editor | Ants propose edits simultaneously; no sequential token sampling |
| Hidden states to RouteGraph | Graph aggregation has richer signal than just logits |
| CPU streaming dataset | No corpus in GPU memory; back-pressure prevents OOM |
| tanh(Δ/σ) pheromone | Prevents runaway feedback while preserving gradient signal |
