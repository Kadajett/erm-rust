# ERM Tensor Shape Reference

All shapes assume default config: `B=8, L=128, V=16384, d=256, Emax=16, A=256, k=8, pmax=8`.

## Notation

| Symbol | Value | Meaning |
|---|---|---|
| B | 8 | Batch size |
| L | 128 | Sequence length |
| V | 16384 | Vocabulary size (+ 1 for MASK sentinel = 16385) |
| d | 256 | Hidden dimension |
| Emax | 16 | Max edges per destination node |
| A | 256 | Total ants per sequence |
| A_L | 26 | Leader ants (10%) |
| A_F | 230 | Follower ants (90%) |
| k | 8 | Top-k candidates per position |
| pmax | 8 | Max positions proposed per ant |
| T | 6 | Refinement steps |

## Input / State

| Tensor | Shape | Dtype | Size (f16) |
|---|---|---|---|
| `y_t` | `[B, L]` | i32 | 131 KB |
| `x` (ground truth) | `[B, L]` | i32 | 131 KB |
| `editable` | `[B, L]` | bool | 4 KB |
| `t` (noise level) | `[B]` | i32 | <1 KB |

## Scorer Forward Pass

| Tensor | Shape | Dtype | Size (f16) |
|---|---|---|---|
| `emb` | `[B, L, d]` | f16 | 8.4 MB |
| `route_msg r` | `[B, L, d]` | f16 | 8.4 MB |
| `hidden` | `[B, L, d]` | f16 | 8.4 MB |
| `logits` | `[B, L, V]` | f16 | **33.6 MB** ⚠️ |
| `uncertainty u` | `[B, L]` | f16 | 262 KB |

## Top-k

| Tensor | Shape | Dtype | Size |
|---|---|---|---|
| `topk_ids` | `[B, L, k]` | i32 | 2.1 MB |
| `topk_scores` | `[B, L, k]` | f16 | 1.05 MB |

## Route Graph (dense-neighbor format)

| Tensor | Shape | Dtype | Size |
|---|---|---|---|
| `nbr_idx` | `[B, L, Emax]` | i32 | 655 KB |
| `phi` | `[B, L, Emax]` | f16 | 328 KB |
| `taint` | `[B, L, Emax]` | f16 | 328 KB |
| `age` | `[B, L, Emax]` | i16 | 328 KB |
| `etype` | `[B, L, Emax]` | u8 | 164 KB |
| **Total graph** | | | **~1.8 MB** ✅ |

## RouteAggregate Intermediate

| Tensor | Shape | Dtype | Size |
|---|---|---|---|
| `h_nbr` (DO NOT MATERIALIZE) | `[B, L, Emax, d]` | f16 | **84 MB** ⚠️ |
| `w` (edge weights) | `[B, L, Emax]` | f16 | 328 KB |

**→ Must fuse gather + softmax + weighted sum. Never allocate h_nbr.**

## Ant Proposals

| Tensor | Shape | Dtype | Size |
|---|---|---|---|
| `ant_pos` | `[B, A, pmax]` | i32 | 655 KB |
| `ant_tok` | `[B, A, pmax]` | i32 | 655 KB |
| `ant_gain` | `[B, A, pmax]` | f16 | 328 KB |
| **Total ant tensors** | | | **~1.6 MB** ✅ |

## Merge Output

| Tensor | Shape | Dtype | Size |
|---|---|---|---|
| `best_tok` | `[B, L]` | i32 | 131 KB |
| `best_gain` | `[B, L]` | f16 | 65 KB |
| `y_{t-1}` | `[B, L]` | i32 | 131 KB |

## Delta Computation

| Tensor | Shape | Dtype | Size |
|---|---|---|---|
| `logits_before` | `[B, L, V]` | f16 | 33.6 MB |
| `logits_after` | `[B, L, V]` | f16 | 33.6 MB |
| **Total for Δ** | | | **67.2 MB** ⚠️ |

## Edge Usage Trace

| Tensor | Shape | Dtype | Size |
|---|---|---|---|
| `edge_trace` | `[B, A, pmax, Emax]` | u8 | 2.6 MB |

## Model Parameters (approx)

| Component | Parameters | Size (f32) |
|---|---|---|
| Token embedding `[V+1, d]` | 4.2M | 16.8 MB |
| Position embedding `[L, d]` | 32K | 128 KB |
| 6 attn blocks (QKV+MLP) | ~6 * 1.3M = 7.8M | 31.3 MB |
| Output head `[d, V]` (tied) | 0 (tied to embed) | 0 |
| **Total scorer** | **~12M** | **~48 MB** |

## VRAM Budget Summary

| Item | Size | Notes |
|---|---|---|
| Scorer weights | ~50 MB | |
| AdamW optimizer state | ~100 MB | 2x params in f32 |
| Activations (1 step, no grad ckpt) | ~75 MB | logits + hidden |
| Logits for Δ (2 copies) | ~67 MB | |
| Route graph | ~2 MB | |
| Ant tensors + misc | ~5 MB | |
| **Model subtotal** | **~300 MB** | |
| CUDA/burn framework overhead | ~500–800 MB | Estimated |
| **Practical peak** | **~1.1 GB** | Well within 4 GB ✅ |
| Safety margin (headroom) | ~900 MB | Before 4 GB limit |

*Note: actual VRAM will differ. Measure with `nvidia-smi` during training. If OOM: reduce B, V, or Emax first.*
