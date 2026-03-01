# ERM-Rust TODO

Emergent Route Model — full implementation plan, organized by milestone.

Source: architecture specification document (2026-03-01).  
Hardware target: RTX 3050 (4GB VRAM), burn + CUDA backend.  
Parallel project: [rust-pcn](../rust-pcn) — trained together for comparison.

---

## Legend

- `[ ]` — not started
- `[x]` — complete
- `⚠️ VRAM` — memory/VRAM concern for RTX 3050
- `🔗 PCN` — requires coordination with rust-pcn

---

## Phase 0: Project Setup

> Get the repo, toolchain, and CI ready before writing any model code.

### Repository

- [ ] Initialize git repo with `main` branch
- [ ] Add `.gitignore` for Rust, build artifacts, model checkpoints
- [ ] Add `LICENSE` (MIT or Apache-2.0)
- [ ] Create `Cargo.toml` workspace manifest
- [ ] Create member crates:
  - [ ] `erm-core` — model, graph, ant logic
  - [ ] `erm-train` — training loop, data pipeline
  - [ ] `erm-cli` — CLI entrypoint
- [ ] Add `Cargo.lock` to git (binary project)

### Dependencies (add to Cargo.toml)

- [ ] `burn` with features: `cuda`, `autodiff`
- [ ] `burn-cuda` backend
- [ ] `tokenizers` (HuggingFace) for BPE
- [ ] `serde` + `serde_json` for config serialization
- [ ] `clap` for CLI
- [ ] `anyhow` / `thiserror` for error handling
- [ ] `rand` + `rand_chacha` for deterministic seeding
- [ ] `criterion` (dev) for benchmarks
- [ ] `proptest` (dev) for property-based tests

### CI (.github/workflows/)

- [ ] `ci.yml`: `cargo check`, `cargo test`, `cargo clippy -- -D warnings`, `cargo fmt --check`
- [ ] Run on push to `main` and all PRs
- [ ] Cache `~/.cargo/registry` and `target/`
- [ ] Add badge to README

### Config & Hyperparameters

- [ ] Create `config/default.toml` with all hyperparameters (see inline notes throughout)
- [ ] `ErmConfig` struct (serde) with fields:
  ```
  vocab_size: usize       // V = 16_384 (start); up to 32_768
  seq_len: usize          // L = 128 (start); grow to 256, 512
  hidden_dim: usize       // d = 256
  num_blocks: usize       // B = 6
  num_heads: usize        // h = 4 (if attention)
  mlp_expansion: usize    // 4 (i.e., 4*d = 1024)
  emax: usize             // Emax = 16 (max edges per destination node)
  num_ants: usize         // A = 256 per sequence
  topk: usize             // k = 8 candidates per position
  refinement_steps: usize // T = 6
  batch_size: usize       // B = 8 (tune with VRAM)
  mask_rate_max: f32      // α_T = 0.8
  mask_rate_min: f32      // α_1 = 0.15
  replace_rate_max: f32   // β_T = 0.1
  replace_rate_min: f32   // β_2 = 0.02
  pheromone_evap: f32     // ρ = (TBD, ~0.1)
  pheromone_eta: f32      // η = deposit rate
  taint_zeta: f32         // ζ = taint deposit rate
  taint_max: f32          // τ_max = cap
  death_streak: usize     // K = streak before ant dies
  max_edits_per_step: f32 // M = 0.15 * L
  leader_fraction: f32    // 0.10
  ```

### Documentation

- [ ] Archive architecture spec to `docs/architecture-plan.md`
- [ ] Add `docs/math.md` summarizing objective and update rules
- [ ] Add `docs/shapes.md` with all tensor shape reference table

---

## Phase 1 (Milestone 1): Baseline Discrete Denoiser

> Train a small scorer network as a plain masked-token denoiser. No route graph yet — just establish that the backbone learns.

### Tokenizer & Dataset

- [ ] Integrate HuggingFace `tokenizers` crate
- [ ] Support loading a pre-trained BPE tokenizer (e.g., GPT-2 or custom 16k vocab)
- [ ] Implement dataset loader: read plain text, tokenize, chunk into sequences of length `L=128` with stride
- [ ] Implement `DataBatch` struct:
  ```
  x:        [B, L] i32   // ground truth tokens
  y_t:      [B, L] i32   // corrupted tokens (V = MASK sentinel)
  editable: [B, L] bool  // which positions can be edited
  t:        [B]    i32   // noise level per sample
  ```
- [ ] Implement `DataLoader` with shuffling and deterministic seeding
- [ ] 🔗 PCN: use same tokenizer and dataset splits as rust-pcn for fair comparison

### Corruption Schedule

- [ ] Implement `corrupt(x, t, config) -> y_t` with:
  - Linear schedule: `α_t` from `α_T=0.8` (heavy) to `α_1=0.15` (light) over `T=6` steps
  - Linear schedule: `β_t` from `β_T=0.1` to `β_1=0.02`
  - Per-token decision: mask (prob `α_t`) → replace with confuser (prob `β_t`) → keep
  - Confuser token: sample from unigram distribution
- [ ] Store MASK sentinel as `V` (one past last real token id)
- [ ] Sample random `t ∈ 1..T` per batch item

### Neural Scorer Sθ

- [ ] Implement token embedding: `E ∈ [V+1, d]` f32 (V+1 to include MASK)
- [ ] Implement position embedding: `P ∈ [L, d]` f32 (learned or rotary — start learned)
- [ ] Implement `B=6` transformer encoder blocks, each:
  - Multi-head self-attention (h=4 heads, d=256) — causal mask OFF (encoder, not AR)
  - MLP: Linear(d → 4d) → GELU → Linear(4d → d)
  - LayerNorm pre-norm style
- [ ] Implement output head: Linear(d → V), no bias (tie weights with embedding table optionally)
- [ ] Implement uncertainty head: Linear(d → 1) → sigmoid, output `u ∈ [B, L]`
- [ ] Forward signature:
  ```
  fn forward(y_t: [B,L], r: [B,L,d]) -> (logits: [B,L,V], uncertainty: [B,L])
  ```
  (Phase 1: r = zeros, route graph not wired yet)

#### ⚠️ VRAM: Logits tensor
```
logits: [B=8, L=128, V=16384] f16 = 8 * 128 * 16384 * 2 bytes = 33.6 MB
logits: [B=8, L=128, V=32768] f16 = 67.1 MB   ← likely too large
```
**Mitigation:** Start with `V=16_384`. If still tight, reduce `B` to 4 or use blockwise vocab projection (compute logits in chunks, reduce before materializing full tensor).

### Denoising Loss

- [ ] Implement `denoise_loss(logits: [B,L,V], x: [B,L], y_t: [B,L]) -> scalar`:
  - Cross-entropy only on corrupted positions: `i where y_t[i] != x[i]`
  - Sum (not mean) over corrupted positions, mean over batch
  - `L_denoise = Σ_{i: corrupted} CE(softmax(logits[i]), x[i])`
- [ ] Gradient flows through scorer only (graph updates are non-differentiable)

### Training Loop (Phase 1)

- [ ] Implement basic training loop:
  1. Sample batch `(x, t)` → corrupt → `y_t`
  2. `r = zeros([B, L, d])` (no route graph yet)
  3. Forward scorer → `logits, u`
  4. Compute `L_denoise`
  5. Backward + optimizer step (AdamW, lr=1e-3, weight_decay=0.01)
  6. Log loss every N steps
- [ ] Implement LR warmup (1000 steps) + cosine decay

### Acceptance Criteria — Phase 1

- [ ] Training loss decreases monotonically over first 10k steps on any text dataset
- [ ] Masked-token accuracy on held-out set improves above random baseline (`1/V`)
- [ ] Forward pass produces correct shapes: `logits=[B,L,V]`, `u=[B,L]`

### Tests — Phase 1

- [ ] `test_corruption_shape`: output shapes match input, MASK sentinel in range `[0, V]`
- [ ] `test_corruption_rate`: empirically verify mask rate matches schedule (±2%)
- [ ] `test_corruption_deterministic`: same seed → same output
- [ ] `test_scorer_forward_shapes`: assert `logits.shape() == [B, L, V]`
- [ ] `test_scorer_no_nan`: no NaN/Inf in logits after random init
- [ ] `test_denoise_loss_only_corrupted`: loss is zero if no positions are corrupted
- [ ] `test_denoise_loss_decreases`: 100 gradient steps reduce loss on toy dataset

---

## Phase 2 (Milestone 2): Route Graph Storage + RouteAggregate

> Build the dynamic sparse memory G and the differentiable RouteAggregate kernel.

### Route Graph Representation

Use **dense-neighbor format** (recommended for v1 on GPU over CSR):

```
nbr_idx:  [B, L, Emax]  i32   // neighbor position indices (-1 = empty slot)
phi:      [B, L, Emax]  f16   // pheromone strength (≥ 0)
taint:    [B, L, Emax]  f16   // taint level (≥ 0, ≤ τ_max)
age:      [B, L, Emax]  i16   // edge age (steps since creation)
etype:    [B, L, Emax]  u8    // edge type: 0=local, 1=long-range, 2=concept
```

With `B=8, L=128, Emax=16`:
```
nbr_idx:  8 * 128 * 16 * 4 bytes =  655 KB
phi:      8 * 128 * 16 * 2 bytes =  328 KB
taint:    8 * 128 * 16 * 2 bytes =  328 KB
age:      8 * 128 * 16 * 2 bytes =  328 KB
etype:    8 * 128 * 16 * 1 byte  =  164 KB
Total route graph: ~1.8 MB per batch  ✅ safe
```

- [ ] Implement `RouteGraph` struct holding all 5 arrays on device
- [ ] Implement `RouteGraph::new_empty(B, L, Emax) -> Self`: initialize `nbr_idx = -1`, `phi = 0.1` (small warmstart), `taint = 0`, `age = 0`
- [ ] Implement `RouteGraph::add_edge(b, dst, src, etype)`: insert into first empty slot at `[b, dst, *]`; if full, reject (caller handles pruning first)
- [ ] Implement `RouteGraph::remove_edge(b, dst, slot)`: swap-remove (move last valid slot into this slot, mark last as -1)
- [ ] Implement `RouteGraph::prune(b, dst)`: remove edges with lowest `phi - λ*taint` score until `count ≤ Emax`
- [ ] Serialize/deserialize `RouteGraph` via serde for checkpoint saving

### RouteAggregate Kernel

- [ ] Implement `route_aggregate(graph: &RouteGraph, h: [B,L,d]) -> r: [B,L,d]`:
  1. Gather: `h_nbr[b,i,e,:] = h[b, nbr_idx[b,i,e], :]` (zero-fill for empty slots)
  2. Compute weights: `w_raw[b,i,e] = log(phi[b,i,e] + ε) - λ * taint[b,i,e] - μ * age[b,i,e]`
     - Mask empty slots (`nbr_idx == -1`) to `-inf` before softmax
  3. Softmax over `e` dimension: `w[b,i,:] = softmax(w_raw[b,i,:])`
  4. Weighted sum: `r[b,i,:] = Σ_e w[b,i,e] * h_nbr[b,i,e,:]`
  - Output: `r ∈ [B, L, d]`
  - Hyperparams inline: `ε=1e-6`, `λ=1.0` (taint penalty), `μ=0.01` (age penalty)

#### ⚠️ VRAM: h_nbr intermediate
```
h_nbr: [B=8, L=128, Emax=16, d=256] f16 = 8 * 128 * 16 * 256 * 2 bytes = 84 MB
```
**Mitigation:** Fuse gather + weighted sum in a single kernel pass — never materialize the full `h_nbr` tensor. Process one destination `i` at a time, or chunk over `L`. Alternatively, keep `Emax=8` initially (halves to 42 MB).

- [ ] Wire `RouteAggregate` into scorer forward: `h = emb(y_t) + pos; r = route_aggregate(G, h); out = scorer_blocks(h + r)`
- [ ] Ensure `RouteAggregate` is **not** in the autodiff graph for pheromone fields (graph updates are non-differentiable); only `h` carries gradients

### CPU↔GPU Efficiency

- [ ] Keep all graph tensors resident on GPU for the full training run
- [ ] CPU builds only: minibatch token ids + corruption masks (compact, O(B·L) per step)
- [ ] Use pinned host memory for token upload buffers
- [ ] Implement double-buffering: CPU prepares batch N+1 while GPU processes batch N

### Acceptance Criteria — Phase 2

- [ ] `RouteAggregate` output matches a CPU reference implementation within `1e-3` tolerance on random inputs
- [ ] Training loss with route conditioning ≤ training loss without (or equal at init)
- [ ] VRAM usage stays within 3.5 GB budget with `B=8, L=128, d=256, Emax=16`
- [ ] No graph corruption: `nbr_idx` slots always contain valid positions or -1

### Tests — Phase 2

- [ ] `test_route_graph_init`: all `nbr_idx == -1`, `phi == 0.1`, shapes correct
- [ ] `test_route_aggregate_empty_graph`: with all slots empty, `r` should be `zeros([B,L,d])`
- [ ] `test_route_aggregate_single_edge`: manually set one edge, verify `r` matches hand-computed expected value
- [ ] `test_route_aggregate_cpu_gpu_match`: random graph, compare CPU vs GPU outputs `< 1e-3`
- [ ] `test_graph_add_remove_edge`: add edges to capacity, remove one, verify count and slot integrity
- [ ] `test_graph_prune_keeps_best`: insert `Emax+1` edges with known phi, prune → weakest removed
- [ ] `test_vram_budget`: forward pass with max config does not OOM (integration test, GPU required)

---

## Phase 3 (Milestone 3): Followers-Only Colony Refinement

> Implement ant sampling and edit application. Leaders and graph updates come later. Prove that one refinement step improves over no refinement.

### Top-k Extraction

- [ ] Implement `topk(logits: [B,L,V], k: usize) -> (ids: [B,L,k] i32, scores: [B,L,k] f16)`:
  - Use burn's built-in topk or implement via sort
  - `k = 8` default

### Route Strength

- [ ] Implement `route_strength(graph: &RouteGraph) -> [B,L] f16`:
  - `route_strength[b,i] = Σ_e phi[b,i,e]` (sum of incoming pheromone for position i)
  - Used by followers to prefer strongly-supported positions

### Follower Ant Sampler

Each ant selects a subset of positions `P_k` (size `p`, e.g., 1..8), then proposes a token per position.

- [ ] Implement `follower_position_sample(conf: [B,L], route_str: [B,L], editable: [B,L], rng, p) -> [B, A_F, pmax] i32`:
  - Score per position: `score_i = editable[i] * conf_i * route_strength_i`
  - Sample `p` positions without replacement, weighted by score (Gumbel-top-k trick)
  - `A_F = ceil(0.9 * A) = 231` followers (with `A=256`)
  - `pmax = 8` positions per ant
- [ ] Implement `follower_token_propose(topk_ids: [B,L,k], topk_scores: [B,L,k], positions: [B,A_F,pmax], temp: f32) -> (ant_tok: [B,A_F,pmax] i32, ant_gain: [B,A_F,pmax] f16)`:
  - Temperature `< 1.0` (e.g., `0.7`) for conservative sampling from top-k
  - `predicted_gain` = top-k score (proxy for improvement)

### Ant Proposals Tensor Shapes

```
ant_pos:  [B=8, A=256, pmax=8]  i32  =  8 * 256 * 8 * 4 = 65.5 KB
ant_tok:  [B=8, A=256, pmax=8]  i32  =  65.5 KB
ant_gain: [B=8, A=256, pmax=8]  f16  =  32.8 KB
Total: ~164 KB  ✅ trivial
```

### Conflict-Free Edit Merge

- [ ] Implement `merge_edits(ant_pos, ant_tok, ant_gain, editable, M) -> (best_tok: [B,L] i32, best_gain: [B,L] f16)`:
  - For each position `i`: keep candidate with highest `predicted_gain`
  - Apply constraint: at most `M = ceil(0.15 * L) = 20` edits per step (select top-M by gain)
  - Clamped positions (`!editable`) are never edited
  - Positions with no proposals keep current token (sentinel = no-edit)
- [ ] Implement `apply_edits(y_t: [B,L], best_tok: [B,L]) -> y_{t-1}: [B,L]`:
  - In-place update on GPU

### Improvement Metric Δ

- [ ] Implement `compute_delta(logits_before: [B,L,V], logits_after: [B,L,V], x: [B,L], edited_mask: [B,L]) -> [B,L] f32`:
  - `Δ_i = CE_before(i) - CE_after(i)` for edited positions
  - Requires one additional forward pass after applying edits
  - ⚠️ VRAM: two copies of logits simultaneously = `2 * 33.6 MB = 67.2 MB` — manageable but track total budget

### Acceptance Criteria — Phase 3

- [ ] One refinement step (followers only) reduces masked-token CE vs. zero-step baseline on average over 1000 eval batches
- [ ] Merge is deterministic: same inputs + same seed → identical `y_{t-1}`
- [ ] No edits applied to clamped prefix positions
- [ ] Max-edits constraint respected: count of changed tokens ≤ `ceil(0.15 * L)`

### Tests — Phase 3

- [ ] `test_topk_shape`: output shapes `[B,L,k]`, all ids in `[0, V)`
- [ ] `test_merge_deterministic`: same ant proposals → same merged output (no randomness in merge)
- [ ] `test_merge_conflict_resolution`: two ants propose different tokens for same position → higher gain wins
- [ ] `test_merge_max_edits`: with 100 proposals all at different positions, only `M` are applied
- [ ] `test_merge_respects_editable`: proposals to clamped positions are ignored
- [ ] `test_apply_edits_no_side_effects`: original `y_t` tensor unchanged if in-place logic clones
- [ ] `test_one_step_improves_ce`: integration test — CE after one follower step < CE before on average

---

## Phase 4 (Milestone 4): Pheromone Update Loop (Stigmergy)

> Wire up the graph update rules. Edges concentrate on useful routes over training. Enforce Emax via pruning.

### Edge Usage Tracing

Each ant must record which edges it relied on during position sampling (for pheromone deposit).

- [ ] Implement `EdgeTrace` struct per ant:
  ```
  traces: [B, A, pmax, Emax]  u8  // boolean: was edge [b, pos, e] used by this ant?
  ```
  Or equivalently, record `(dst_pos, edge_slot)` pairs per ant proposal.
  - During `route_aggregate`, record which edges had nonzero weight for each chosen position
  - ⚠️ VRAM: full trace tensor = `8 * 256 * 8 * 16 = 262 KB` ✅ fine

### Pheromone Update Rules

For each edge `e = (j → i)` at slot `[b, i, s]`:

- [ ] Implement `evaporate(graph, ρ)`: `phi[b,i,s] *= (1 - ρ)`; `ρ ≈ 0.05..0.1`
- [ ] Implement `deposit(graph, traces, delta_per_ant, η)`:
  - For each ant k that used edge e with improvement `Δ_k > 0`:
  - `phi[b,i,s] += η * relu(Δ_k)`
  - `η ≈ 0.01..0.1` (start small for stability)
- [ ] Implement `taint_update(graph, traces, delta_per_ant, ζ, ρ_τ)`:
  - For each ant k that used edge e with `Δ_k ≤ 0`:
  - `taint[b,i,s] = clamp(taint[b,i,s] + ζ * relu(-Δ_k), 0, τ_max)`
  - Taint decay: `taint[b,i,s] *= (1 - ρ_τ)`
  - `ζ ≈ 0.05`, `τ_max = 5.0`, `ρ_τ ≈ 0.01`
- [ ] Apply all updates in order: evaporate → deposit → taint_update
- [ ] Increment `age` for all active edges each step

### Pruning

- [ ] Implement `prune_edges(graph, λ=1.0)` called every `N_prune=100` steps:
  - For each `(b, i)`: compute score `phi[b,i,e] - λ * taint[b,i,e]` for all edges
  - If any slot is -1 (empty), skip
  - If all slots used: remove edge with minimum score (swap-remove)
  - This runs on CPU (compact pass over neighbor slots) — upload only modified slices

### Stability Tricks (Phase 4)

- [ ] Warm-start: run Phase 1+2+3 training for `W=1000` steps before enabling pheromone deposits
- [ ] Start with `η=0.001`, linearly increase to target over 5000 steps
- [ ] Clip `phi` to `[0, phi_max=100.0]` to prevent runaway reinforcement

### Acceptance Criteria — Phase 4

- [ ] After 10k training steps, edges used by successful ants accumulate higher phi than unused edges (measurable)
- [ ] Graph growth is bounded: no `(b, i)` ever exceeds `Emax` active edges
- [ ] `phi` values remain in `[0, phi_max]`, `taint` in `[0, τ_max]` at all times
- [ ] CSR/dense graph integrity: no invalid `nbr_idx` values (all in `[-1, L)`)

### Tests — Phase 4

- [ ] `test_evaporation_rate`: phi decreases by factor `(1-ρ)` per step from known initial value
- [ ] `test_deposit_increases_phi`: ant with `Δ > 0` using edge → phi increases
- [ ] `test_taint_for_negative_delta`: ant with `Δ ≤ 0` → taint increases, phi unchanged by deposit
- [ ] `test_phi_non_negative`: after 1000 random updates, no phi < 0
- [ ] `test_taint_bounded`: after 1000 updates with negative delta, taint ≤ τ_max
- [ ] `test_pruning_removes_weakest`: insert Emax edges, force one to accumulate taint, prune → that edge removed
- [ ] `test_graph_integrity_after_prune`: all nbr_idx still valid after prune
- [ ] `proptest_phi_invariants`: property-based test — phi always non-negative after any sequence of updates

---

## Phase 5 (Milestone 5): Leaders + Exploration + Death/Respawn

> Add the 10% leader population, edge proposal mechanics, and ant death/taint-streak logic.

### Leader Ant Sampler

- [ ] Implement `leader_position_sample(u: [B,L], route_str: [B,L], editable: [B,L], rng, p) -> [B, A_L, pmax] i32`:
  - Score: `score_i = editable[i] * u_i * (1 / (route_strength_i + ε))`
  - Leaders seek high-uncertainty, low-support positions
  - `A_L = ceil(0.1 * A) = 26` leaders (with `A=256`)
- [ ] Implement `leader_token_propose(topk_ids, topk_scores, positions, temp_high=1.5) -> (ant_tok, ant_gain)`:
  - Higher temperature (> 1.0) for exploration
  - Include novelty bonus: prefer tokens not currently in `y_t` at that position

### Leader Edge Proposals

Leaders can propose **new edges** to the route graph:

- [ ] Implement `leader_propose_edge(b: usize, src: usize, dst: usize, etype: u8) -> Option<EdgeProposal>`:
  - Leader that edits position `dst` using evidence from position `src` → propose edge `src → dst`
  - Reject if slot full (Emax reached at dst); return None
  - Assign provisional pheromone: `phi_new = phi_init = 0.05`
- [ ] Implement `apply_edge_proposals(graph, proposals: &[EdgeProposal])`: batch insert into graph
- [ ] Track proposed edges separately for utility tracking (below)

### Leader Utility Tracking

- [ ] Implement `LeaderUtility` struct:
  ```
  utility: HashMap<EdgeId, f32>  // EMA of positive contributions
  gamma: f32                      // EMA decay, e.g., 0.9
  ```
- [ ] Update utility after each step: for each newly-introduced edge that was used:
  - `U(e) = (1 - γ) * U(e) + γ * relu(Δ_when_used)`
- [ ] Log average utility of leader-introduced edges vs. follower edges as a training metric

### Death and Respawn

Each ant maintains a "streak" counter (consecutive steps with no improvement):

- [ ] Implement `AntState` array: `streak: [B, A] i32`, `ant_type: [B, A] u8` (0=follower, 1=leader)
- [ ] Per step: `if Δ_k ≤ ε: streak[k] += 1 else: streak[k] = 0`
- [ ] If `streak[k] >= K` (e.g., `K=5`):
  - Apply taint to recently used edges (same as negative delta, fixed penalty)
  - Reset: `streak[k] = 0`, reassign type (maintain 10/90 ratio), reseed
- [ ] ⚠️ VRAM: `AntState` at `[B=8, A=256]` = trivial (~16 KB)

### Stability Tricks (Phase 5)

- [ ] Introduce leaders only after `W=5000` steps (let followers stabilize first)
- [ ] For the first 2000 leader steps: leaders can propose edits but NOT new edges (edge proposals start later)
- [ ] Cap leader edge proposals at `max_proposals_per_step = 4` per sequence

### Acceptance Criteria — Phase 5

- [ ] Leader-introduced edges accumulate nonzero utility U(e) over training (measurable)
- [ ] Follower ants preferentially use edges introduced by successful leaders (usage count metric)
- [ ] Training remains stable (loss does not diverge) after leader introduction
- [ ] Ant death/respawn fires correctly: streak resets on improvement, death at K, type distribution maintained at 10/90

### Tests — Phase 5

- [ ] `test_leader_targets_uncertainty`: leader position scores are inversely correlated with route_strength
- [ ] `test_follower_targets_confidence`: follower position scores positively correlated with conf and route_strength
- [ ] `test_death_fires_at_k`: ant with streak=K is marked for respawn
- [ ] `test_streak_resets_on_improvement`: ant with streak=K-1 that improves → streak=0
- [ ] `test_type_ratio_maintained`: after 100 deaths/respawns, 10/90 ratio within ±2%
- [ ] `test_edge_proposal_respects_emax`: leader cannot insert edge if destination slot is full
- [ ] `test_utility_ema`: utility EMA updates correctly from known delta sequence
- [ ] `test_reproducible_under_seed`: same seed → identical ant behavior (leaders, followers, deaths)

---

## Phase 6 (Milestone 6): Multi-Step Unrolled Refinement Training

> Unroll multiple refinement steps during training. Train from all-mask initialization. Achieve coherent generation.

### Multi-Step Unrolling

- [ ] Implement `refine_loop(y_init: [B,L], graph: &mut RouteGraph, scorer: &Sθ, steps: usize) -> y_final: [B,L]`:
  - For `t = T..1` (or truncated to 2..4 steps during early training):
    1. Compute `h = embed(y_t) + pos`
    2. `r = route_aggregate(graph, h)`
    3. `logits, u = scorer.forward(y_t, r)`
    4. Build top-k candidates
    5. Sample ants (leaders + followers)
    6. Merge → apply edits → `y_{t-1}`
    7. Compute `Δ` (one extra forward pass)
    8. Update pheromones
  - Gradient flows through scorer forward passes only (graph updates non-differentiable)
  - Truncate gradient at step boundaries to limit memory: only keep last 2 steps in autodiff graph

#### ⚠️ VRAM: Unrolled forward passes
```
Each step materializes: hidden [B,L,d] + logits [B,L,V] + logits_after [B,L,V]
Per step VRAM ≈ 33.6 MB (logits) * 2 + 2.1 MB (hidden) ≈ 70 MB per step
2 steps = ~140 MB, 4 steps = ~280 MB for activations
```
**Mitigation:** Use gradient checkpointing (recompute activations on backward pass). Truncate unrolling to 2 steps initially. Increase to 4 when VRAM budget confirmed safe.

### Stop Criteria (Inference)

- [ ] Stop when: `no_edits_applied OR mean_confidence > θ_stop OR steps >= T_max`
  - `θ_stop = 0.9` (mean max-softmax across editable positions)
  - `T_max = T = 6`
- [ ] Implement `should_stop(y_t, y_prev, logits, editable) -> bool`

### Training Curriculum

- [ ] Phase 6a (first N steps): 1 refinement step per batch, heavy corruption (`α=0.8`)
- [ ] Phase 6b: 2 steps, medium corruption
- [ ] Phase 6c: full T=6 steps, light corruption + full pipeline

### Evaluation

- [ ] Implement eval loop:
  - Start from `y_T = [MASK]*L`
  - Run T refinement steps
  - Compute perplexity of output under a reference language model (or CE against ground truth)
- [ ] Log: denoising accuracy, generation perplexity, route entropy, prune rate, tainted edge fraction
- [ ] 🔗 PCN: run equivalent eval on rust-pcn at same training step for comparison plots

### Acceptance Criteria — Phase 6

- [ ] Generation from all-mask produces coherent text (subjective) with perplexity < all-mask baseline
- [ ] VRAM stays stable across 1000 steps of multi-step unrolling (no memory leak)
- [ ] Loss decreases across all refinement steps (not just the first)
- [ ] Checkpoint save/load round-trips correctly (scorer weights + route graph state)

### Tests — Phase 6

- [ ] `test_refine_loop_shape`: output `y_final` shape = `[B, L]`, all tokens in `[0, V)`
- [ ] `test_stop_criteria_no_edits`: if merge produces no edits, loop terminates early
- [ ] `test_stop_criteria_confidence`: high-confidence output triggers stop
- [ ] `test_checkpoint_roundtrip`: save scorer + graph, reload, assert parameter equality
- [ ] `test_vram_stable`: run 1000 steps of unrolled training, measure peak VRAM at step 1 vs step 1000 (within 5% variation)
- [ ] `test_generation_from_mask`: integration — start from mask, run full loop, output has no MASK tokens remaining

---

## Phase 7 (Milestone 7): End-to-End CLI and Benchmarks

> Ship a usable tool with train/eval/generate commands and performance benchmarks.

### CLI (`erm-cli`)

- [ ] `erm train --config config/default.toml --data <path> --output <checkpoint_dir>`:
  - Load config, build model, run training loop
  - Save checkpoints every N steps
  - Log to stdout (and optionally a log file)
- [ ] `erm eval --checkpoint <path> --data <path>`:
  - Load checkpoint, run eval loop
  - Report: masked-token accuracy, generation perplexity, route stats
- [ ] `erm generate --checkpoint <path> --prompt "<text>" --steps T --seed 42`:
  - Tokenize prompt, set prefix, run refinement loop
  - Output generated text
- [ ] `erm generate --checkpoint <path> --unconditional --length 128`:
  - Start from all-mask, generate freely
- [ ] Implement `--dry-run` flag: validate config and shapes without GPU

### Benchmarks (`benches/`)

- [ ] `bench_scorer_forward`: throughput (tokens/sec) for scorer forward pass at various B, L
- [ ] `bench_route_aggregate`: throughput for RouteAggregate at various Emax
- [ ] `bench_full_step`: end-to-end single refinement step (tokens/sec)
- [ ] `bench_generation`: tokens/sec for full T=6 step generation at B=1, L=128
- [ ] Report VRAM peak for each benchmark

### Snapshot Tests

- [ ] `test_generate_snapshot`: with fixed seed and checkpoint, output matches expected string (regression)
- [ ] Store expected outputs in `tests/snapshots/`

### Acceptance Criteria — Phase 7

- [ ] `erm train` runs for 100 steps without crash on RTX 3050 with `B=8, L=128, V=16384`
- [ ] `erm generate` produces non-empty, non-mask output under 10 seconds for L=128
- [ ] Throughput benchmark: at least 1000 tokens/sec for single-step scorer forward
- [ ] VRAM peak ≤ 3.5 GB for full training config (leaves 500 MB headroom on 4 GB card)

### Tests — Phase 7

- [ ] `test_cli_train_dry_run`: config loads, shapes validate, no GPU required
- [ ] `test_cli_generate_prompt`: with known checkpoint, prompted generation includes prefix tokens
- [ ] `test_cli_generate_unconditional`: unconditional generate produces L tokens, no MASKs in output
- [ ] `test_snapshot_regression`: generated output matches stored snapshot (deterministic seed)

---

## Phase 8: Comparison with rust-pcn

> This phase is ongoing alongside all others. Document comparison methodology and collect results.

### 🔗 PCN Coordination

- [ ] Agree on shared dataset and tokenizer with rust-pcn (same splits, same vocab)
- [ ] Agree on eval protocol: same test set, same perplexity calculation method
- [ ] Synchronize training compute budget for fair comparison (e.g., same number of gradient steps or same wall-clock time)
- [ ] Implement a shared `eval-harness` script or Makefile target that runs both and outputs a comparison table

### Metrics to Compare

| Metric | ERM | PCN | Notes |
|---|---|---|---|
| Masked-token accuracy | `erm eval` | `pcn eval` | Same corruption schedule |
| Generation perplexity | Under reference LM | Under reference LM | Same test prompts |
| Training VRAM | Peak MB | Peak MB | Same B, L |
| Training throughput | Steps/sec | Steps/sec | |
| Sample quality (subjective) | Human eval | Human eval | Small sample, 10 examples |
| Convergence speed | Loss vs steps | Loss vs steps | Plot together |
| Parameter count | Σ θ | Σ θ | ERM uses smaller NN + graph |
| Graph/memory overhead | Route graph MB | N/A | ERM-specific |

- [ ] Create `docs/comparison.md` to record results at each milestone
- [ ] Generate side-by-side loss curves at Milestones 1, 3, 6
- [ ] Note any qualitative differences: does ERM make different kinds of errors than PCN?

### Hypotheses to Test

- [ ] **H1:** ERM achieves similar masked-token accuracy to PCN with fewer learned parameters (graph compensates)
- [ ] **H2:** ERM shows worse early convergence but better late convergence as the graph matures
- [ ] **H3:** ERM's generated text has more local coherence (route graph enforces local dependencies)
- [ ] **H4:** PCN converges faster per gradient step but ERM is more VRAM-efficient at equivalent quality

---

## Ongoing / Cross-Cutting

### Memory & VRAM Budget Summary (RTX 3050, 4 GB)

| Component | Size | Notes |
|---|---|---|
| Scorer weights (d=256, B=6, V=16k) | ~50 MB | Embedding table dominates |
| Optimizer state (AdamW, 2x params) | ~100 MB | fp32 optimizer states |
| Activations (hidden [B,L,d]) | ~2.1 MB | Per step |
| Logits [B,L,V] f16 | ~33.6 MB | Per step, biggest tensor |
| Logits (delta, 2 copies) | ~67.2 MB | When computing Δ |
| Route graph [B,L,Emax] | ~1.8 MB | Negligible |
| h_nbr intermediate | ~84 MB | **Must fuse/chunk** |
| Ant tensors | ~0.2 MB | Negligible |
| **Estimated peak** | **~350 MB** | Rough, excludes framework overhead |
| CUDA framework overhead | ~500–800 MB | Burn/CUDA baseline |
| **Practical budget** | **~3.5 GB** | 500 MB headroom on 4 GB |

⚠️ If OOM: reduce `B` (4 instead of 8), reduce `V` (8k), reduce `Emax` (8), use gradient checkpointing, use `f16` throughout with `f32` accumulation only at loss.

### Code Quality

- [ ] All public types have `///` doc comments
- [ ] No `unwrap()` or `expect()` in library code — use `Result<_, ErmError>`
- [ ] All code passes `cargo clippy -- -D warnings`
- [ ] `cargo fmt` on every file before commit
- [ ] Group imports: std → external crates → internal modules

### Checkpointing

- [ ] Checkpoint format: `scorer_weights.safetensors` + `graph_state.bin` + `config.toml` + `train_state.json`
- [ ] `train_state.json` contains: step, loss history, pheromone stats, ant death count
- [ ] Resume training from checkpoint without loss of state

### Logging

- [ ] Log every 100 steps: loss, masked-token accuracy, mean phi, mean taint, prune events, ant deaths
- [ ] Log every 1000 steps: eval metrics, generate one sample (unconditional)
- [ ] Structured log format (JSON lines) for easy parsing

---

*Last updated: 2026-03-01*  
*Architecture source: docs/architecture-plan.md*
