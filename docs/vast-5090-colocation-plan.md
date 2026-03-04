# Vast.ai RTX 5090 (1x) Colocation Plan

## Scope
This is a planning doc for a **future** move to `RTX 5090 (1x)` on Vast.ai while keeping visibility in your current local Kubernetes/AIM workflow.

## Input Prices (from your quote)
- RTX 5090 1x on-demand: **$0.37/hr**
- H200 1x on-demand: **$2.32/hr**
- H200 2x on-demand: **$4.65/hr**
- H200 8x on-demand: **$18.58/hr**

## Current Baseline (local 3050 run)
Measured from recent live run window:
- Throughput: **~3.24 steps/sec**
- Config baseline: `batch=2, seq_len=128, T=2, hidden=512`
- Tokens/sec: `3.24 * 2 * 128 = ~829 tok/sec`
- 1M steps wall time: `1,000,000 / 3.24 = ~85.8h`
- Tokens at 1M baseline config: `1,000,000 * 2 * 128 = 256M tokens`

---

## Cost Math on 5090
Formula:
- `cost = hours * 0.37`

Examples:
- 24h: **$8.88**
- 72h: **$26.64**
- 1 week (168h): **$62.16**
- 1 month (720h): **$266.40**

---

## Scenario A: Use extra VRAM primarily to speed up training
Goal: keep model mostly similar, push data throughput hard.

Proposed config band:
- `batch=8`, `seq_len=128`, `T=2`, `hidden=512`
- Same objective/data pipeline, fewer architectural changes.

Expected range on 5090 (practical estimate):
- `10–16 steps/sec` (after warmup)
- Tokens/sec: `10–16 * 8 * 128 = 10,240–16,384 tok/sec`

Relative to current ~829 tok/sec:
- **~12x to ~20x token throughput**

Time/cost examples:
- 256M tokens target (same token budget as current 1M local baseline):
  - Time: `256M / (10,240–16,384) = ~6.9h to ~4.3h`
  - Cost: **~$2.55 to ~$1.59**
- 1B tokens target:
  - Time: `~27.1h to ~17.0h`
  - Cost: **~$10.03 to ~$6.29**

---

## Scenario B: Use extra VRAM to improve model capacity/quality
Goal: spend VRAM on harder settings not feasible on 6GB.

Proposed quality config band:
- `batch=4`, `seq_len=256`, `T=4`, `hidden=768`, `num_blocks=6`
- (Still single-GPU, but significantly larger capacity/context/refinement)

Expected range on 5090 (practical estimate):
- `2–4 steps/sec`
- Tokens/sec: `2–4 * 4 * 256 = 2,048–4,096 tok/sec`

Relative to current ~829 tok/sec:
- **~2.5x to ~5x token throughput**, with much stronger per-step model capacity.

Time/cost examples:
- 256M tokens:
  - Time: `~34.7h to ~17.4h`
  - Cost: **~$12.84 to ~$6.44**
- 1B tokens:
  - Time: `~135.6h to ~67.8h`
  - Cost: **~$50.17 to ~$25.09**

---

## Which path helps “intelligence” more?
Short answer:
- **Scenario A** gets you many more experiments and faster iteration cycles.
- **Scenario B** gives better chance at improved language quality/coherence per token trained.

Practical recommendation for your workflow:
1. Start with Scenario A for 1-2 calibration runs (cheap, fast).
2. Move to Scenario B once loss behavior and IO quality are stable and measurable.
3. Keep measuring with the same IO sample probes so changes are comparable.

---

## Vast.ai -> Local K8s visibility plan (no immediate migration required)
You wanted remote compute visible in your existing local stack.

### Recommended architecture
- Keep current local control/observability:
  - local Kubernetes control plane
  - AIM server/dashboard
- Add Vast 5090 worker as **remote GPU node** connected over a private mesh network.

### Implementation outline
1. Bring up Vast instance.
2. Connect Vast instance + local cluster network using Tailscale/WireGuard.
3. Join Vast host to cluster as a worker (k3s/kubelet-compatible path).
4. Install NVIDIA runtime/device plugin on remote node.
5. Add node labels/taints (e.g., `gpu-provider=vast`, `gpu-class=5090`).
6. Schedule selected training jobs to that node via `nodeSelector`/`affinity`.
7. Keep AIM endpoint pointing at your existing local AIM server.

### Why this approach
- You preserve your existing dashboards/log habits.
- You can gradually move only selected runs.
- No forked monitoring stack.

---

## Important current code constraint
Your trainer is currently single-GPU bound in code paths (device `0`), so:
- 2x/8x GPU boxes do **not** automatically speed one job.
- 1x 5090 is the highest value immediate upgrade.

---

## Conversation-Capable Target Matrix (6GB now vs 5090 later)
Goal: move from pure denoise replication toward usable prompt->completion chat behavior.

### What is hard-limited today
- Current run is fixed-window (`seq_len=128`) per generation call.
- Prompt + output must fit in that window.
- This limits real thread memory unless you implement chunked continuation/summarized memory.

### Practical profiles
| Profile | Hardware | Train config target | Inference `length` | Practical per-turn history budget* | Practical reply budget* | Risk |
|---|---|---|---:|---:|---:|---|
| C0 (current short-chat) | 6GB | `B=2, L=128, T=2, H=512, blocks=3` | 128 | ~80 tok (~60 words) | ~48 tok (~35 words) | Low |
| C1 (entry chat) | 5090 | `B=4, L=256, T=2, H=512, blocks=3` | 256 | ~160 tok (~120 words) | ~96 tok (~70 words) | Low |
| C2 (better chat quality) | 5090 | `B=4, L=384, T=3, H=768, blocks=6` | 384 | ~256 tok (~190 words) | ~128 tok (~95 words) | Medium |
| C3 (longer turn budget) | 5090 | `B=2, L=512, T=4, H=768, blocks=6` | 512 | ~352 tok (~260 words) | ~160 tok (~120 words) | Medium/High |

\\*Budgets assume you reserve output space each turn; 1 English word is typically ~1.3 tokens.

### Why this matters
- 6GB can do very short-turn Q/A, but not robust multi-turn memory.
- 5090 enables larger `L` + stronger model capacity at the same time.
- For true “normal chat thread” behavior, you still eventually need:
  1. Thread packing policy (history truncation/summarization).
  2. Completion-style training mix (not denoise-only).
  3. Optional chunked continuation for outputs longer than one window.

### Suggested rollout
1. **Now on 6GB**: keep current run and add completion-style evals only.
2. **First 5090 pass**: use C1 to validate stability and prompt-following quickly.
3. **Second 5090 pass**: move to C2 for higher quality responses.
4. **Only if needed**: C3 for larger single-turn context/output.

---

## Quick Decision Summary
- If objective is **fast iteration now**: 5090 Scenario A first.
- If objective is **better model behavior per run**: 5090 Scenario B after one calibration cycle.
- Monthly ceiling at on-demand 5090 is low enough to run long experiments without H200-class cost.
