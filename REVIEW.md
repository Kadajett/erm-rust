# ERM TODO Plan — Audit Review

Reviewed: 2026-03-01  
Reviewer: Architecture review sub-agent  
Files audited: `TODO.md`, original architecture spec, `docs/math.md`, `docs/shapes.md`

---

## 1. Coverage Gaps

### 🔴 Critical

**1a. Concept nodes and global nodes omitted entirely**  
The spec describes optional "concept nodes" (token cluster IDs, n-gram nodes) and "global nodes" (sentence intent, topic) as part of the route graph's node set. The TODO only implements position nodes `i = 0..L-1`. This is acceptable for v1 but should be called out explicitly as a future extension — the graph storage format (`nbr_idx` range, `etype` semantics) must be designed now to not preclude concept nodes later.  
**Fix:** Add a note in Phase 2's `RouteGraph` section: "Design `nbr_idx` value range to accommodate future concept/global nodes (e.g., indices `L..L+C` for concept nodes). Phase 2 only uses position nodes."

**1b. No length handling for generation**  
The spec mentions an "optional length head: `len_logits ∈ R^{Bsz × Lmax}`" for picking output length during unconditional generation. The TODO's Phase 7 `erm generate --unconditional --length 128` hardcodes length. The spec says "Pick L (or sample length from a small length model)."  
**Fix:** Add a task to Phase 6 or 7: "Implement optional length sampling head (Linear(d → Lmax) on [CLS] or pooled representation), or document that length is user-specified for v1."

**1c. Follower confidence signal (`conf_i`) definition mismatch**  
`docs/math.md` defines `conf_i = max(softmax(logits[i]))`. The TODO's Phase 3 `follower_position_sample` takes a parameter `conf: [B,L]` but never specifies how it's computed. The scorer forward pass in Phase 1 doesn't produce a `conf` tensor — it produces `logits` and `uncertainty`.  
**Fix:** Add to Phase 3 before `follower_position_sample`: "Compute `conf[b,i] = max(softmax(logits[b,i,:]))` from scorer logits." This is a trivial computation but must be explicit in the data flow.

**1d. Leader uncertainty signal — learned head vs entropy ambiguity**  
The spec says `u_i` could be "entropy of logits or a learned head." `docs/math.md` says "entropy(softmax(logits[i])) # or learned uncertainty head." Phase 1 implements a learned uncertainty head (`Linear(d → 1) → sigmoid`). Phase 5 leaders use `u: [B,L]` from the scorer. These are two different signals. The learned head produces a `[0,1]` scalar; entropy of a 16k-softmax is on a different scale.  
**Fix:** Clarify in Phase 1 that the uncertainty head IS the signal leaders will use in Phase 5. Add: "The uncertainty head replaces logit-entropy as the leader targeting signal. Ensure its output correlates with actual prediction uncertainty (add auxiliary loss or rely on denoising gradient to shape it)."

**1e. Scorer forward signature includes `r` but Phase 1 says `r = zeros`**  
This is handled in the TODO — good. But the forward signature `fn forward(y_t: [B,L], r: [B,L,d])` doesn't show how `r` is integrated. The spec says `scorer consumes emb(y_t) + pos + r` (additive). The TODO's Phase 2 says `h = emb(y_t) + pos; r = route_aggregate(G, h); out = scorer_blocks(h + r)`. This means the route message is added to the hidden state *after* embedding but *before* transformer blocks. That's consistent with the spec.  
**Status:** ✅ Covered, just confirming.

### 🟡 Important

**1f. No sampled/adaptive softmax mentioned as a concrete implementation option**  
The spec explicitly suggests "adaptive/sampled softmax" or "project to a smaller candidate set per step using top-k from a smaller head" as VRAM mitigations. The TODO only mentions "blockwise vocab projection (compute logits in chunks)" and batch size reduction.  
**Fix:** Add to Phase 1 VRAM mitigation: "If blockwise is insufficient, implement sampled softmax or two-stage projection (small head → candidate set → full logits on candidates only)."

**1g. No `route consolidation` mechanism**  
The spec describes: "if many edges originate from tokens with same embedding cluster, replace with a concept node edge." This is an optional growth/scaling mechanism. Not needed for v1, but worth noting.  
**Fix:** Add to "Ongoing / Cross-Cutting": "Future: route consolidation (merge cluster-origin edges into concept nodes)."

**1h. Edge `cost c` field omitted**  
The spec's edge structure lists `cost c (optional, for pruning)`. The TODO's dense-neighbor format doesn't include it. This is fine if pruning uses `phi - λ*taint` as scored, but should be noted.  
**Status:** Acceptable omission — `phi - λ*taint` subsumes it.

**1i. No explicit "online inference learning" toggle**  
The spec's inference recipe says: "update graph optionally (online inference learning can be disabled)." The TODO's Phase 7 `erm generate` doesn't mention whether graph updates happen during inference.  
**Fix:** Add to Phase 7 `erm generate`: flag `--update-graph` (default off for deterministic generation, on for adaptive inference).

**1j. Edge `usage count n` from spec not tracked**  
The spec lists per-edge `usage count n`. The TODO tracks `age` but not `n`. Usage count could inform pruning (drop rarely-used old edges).  
**Fix (nice-to-have):** Add `n_used: [B, L, Emax] u16` to the graph struct, or document why it's omitted.

### 🟢 Nice-to-have

**1k. No mention of "same token class buckets" for confuser sampling**  
The spec offers three confuser options: unigram, top frequent, or same token class. TODO only implements unigram. Fine for v1.

**1l. Triple-buffering mentioned in spec but TODO only says double-buffering**  
Spec: "double or triple buffering." TODO: "double-buffering." Fine — triple is an optimization.

**1m. "Dirty-bit range list" for upload optimization**  
Spec describes per-batch-item dirty tracking to avoid copying entire graph on prune. TODO's Phase 4 says "upload only modified slices" but doesn't spec the dirty-tracking mechanism. Worth adding as a performance task.

---

## 2. Ordering / Dependency Issues

### 🔴 Critical

**2a. Phase 3 `compute_delta` requires two forward passes but Phase 3 depends on Phase 1 scorer only**  
`compute_delta` needs `logits_before` and `logits_after`. But `logits_after` requires running the scorer on `y_{t-1}` (the edited sequence). This means Phase 3 must have a working scorer forward pass AND the ability to run it twice per step. This is fine — Phase 1 delivers the scorer. But the TODO doesn't mention that `compute_delta` requires Phase 2's `route_aggregate` to be wired in for the second forward pass, or that it should use `r = zeros` if Phase 2 isn't done.  
**Fix:** Add note to Phase 3's `compute_delta`: "Uses `r = zeros` for both forward passes until Phase 2 is integrated."

**2b. Phase 3 follower uses `route_strength` but Phase 2 might not be fully integrated**  
`follower_position_sample` takes `route_str: [B,L]` which is defined as sum of incoming pheromone. This requires the route graph from Phase 2. Phase 3 says "Followers-Only Colony Refinement" and is listed AFTER Phase 2, so the ordering is correct. But the acceptance criteria says "One refinement step (followers only) reduces masked-token CE vs. zero-step baseline" — with an empty graph (all `phi = 0.1`), `route_strength` will be uniform and followers degenerate to confidence-only sampling. This is fine but should be noted.  
**Status:** Technically correct ordering. Add note: "With empty graph, follower sampling reduces to confidence-weighted. Route strength becomes meaningful only after Phase 4 deposits."

### 🟡 Important

**2c. Phase 4 pheromone deposits depend on `delta_per_ant` which is per-ant Δ, but Phase 3's Δ is per-position**  
`docs/math.md`: `Δ_k = Σ_{i ∈ edits_by_ant_k} Δ_i` — the per-ant delta is the sum of per-position deltas for positions that ant edited. Phase 3 computes per-position `Δ_i` but doesn't aggregate to per-ant. Phase 4's deposit function takes `delta_per_ant`.  
**Fix:** Add to Phase 3 or Phase 4: "Compute per-ant delta: `Δ_k = Σ_{i ∈ ant_k's edits} Δ_i`. Requires tracking which ant proposed which winning edit through the merge step."

**2d. Phase 4 edge usage tracing references RouteAggregate but trace recording isn't in Phase 2**  
The `EdgeTrace` says "During `route_aggregate`, record which edges had nonzero weight for each chosen position." But Phase 2's `route_aggregate` doesn't mention recording traces. Trace recording changes the RouteAggregate implementation.  
**Fix:** Add to Phase 2's `route_aggregate`: "Return or store edge weight mask for trace recording (Phase 4). Prepare the API: `route_aggregate(...) -> (r, Option<EdgeWeights>)` where `EdgeWeights: [B, L, Emax] f16`."

**2e. Phase 5 `LeaderUtility` uses `HashMap<EdgeId, f32>` — not GPU-friendly**  
This is a CPU-side data structure, which is fine since leader utility tracking is lightweight. But the TODO doesn't specify what `EdgeId` is. With dense-neighbor format, an edge is identified by `(batch, dst_pos, slot_idx)`. Slots can change via swap-remove, so `EdgeId` needs to be `(batch, dst_pos, src_pos)` or the utility map breaks when slots shift.  
**Fix:** Define `EdgeId = (batch: usize, dst: usize, src: usize)` and note that swap-remove in `remove_edge` must update the utility map if an edge moves slots.

---

## 3. Math / Shape Errors

### 🔴 Critical

**3a. Denoising loss: TODO says "Sum over corrupted, mean over batch" but `docs/math.md` says mean over corrupted**  
TODO Phase 1: "Sum (not mean) over corrupted positions, mean over batch."  
`docs/math.md`: `L_denoise = (1 / |corrupted|) * Σ_{i: y_t[i] != x[i]} CE(...)` — this is mean over corrupted positions.  
The spec says: `L_denoise = Σ_{i: y_t[i] != x[i]} CE(pθ(i), x[i])` — sum, no normalization by |corrupted|.  
**Three different things.** This matters: sum-over-corrupted makes the loss scale with corruption rate α_t, creating an implicit curriculum (high-t steps have higher loss magnitude). Mean-over-corrupted normalizes this away.  
**Fix:** Pick one and make all three documents consistent. Recommendation: mean over corrupted positions (as in `docs/math.md`) for training stability, since different samples in a batch may have different corruption rates. Update TODO Phase 1 and the spec to match.

**3b. `shapes.md` has `A_F = 230` but TODO has `A_F = 231`**  
`shapes.md`: `A_F = 230` (= 256 - 26).  
TODO Phase 3: `A_F = ceil(0.9 * A) = 231`.  
`ceil(0.9 * 256) = ceil(230.4) = 231`, but `A - A_L = 256 - 26 = 230`.  
The issue: `A_L = ceil(0.1 * 256) = ceil(25.6) = 26`, and `A_F = A - A_L = 230`. OR `A_F = ceil(0.9 * 256) = 231`, which means `A_L + A_F = 257 > A`.  
**Fix:** Use `A_L = ceil(0.1 * A)`, `A_F = A - A_L`. So `A_L = 26, A_F = 230`. Update TODO Phase 3.

**3c. Embedding table size: TODO says `[V+1, d]` but shapes.md says `[V+1, d]` with V=16384 → 16385 rows**  
This is actually consistent — V+1 to include the MASK sentinel. But the config struct has `vocab_size: usize // V = 16_384` and the logits shape is `[B, L, V]` (not V+1). The output head predicts V classes (real tokens), but the embedding must embed V+1 inputs (including MASK).  
**Status:** ✅ Correct — embedding is `[V+1, d]`, output logits are `[B, L, V]`. The scorer sees MASK as input but never predicts it as output.

**3d. `shapes.md` logits size says 33.6 MB but uses V=16384**  
`8 * 128 * 16384 * 2 = 33,554,432 bytes = 32 MB`, not 33.6 MB. Minor rounding error.  
**Status:** Cosmetic. The estimate is close enough for VRAM planning.

### 🟡 Important

**3e. Route aggregation weight formula — age type mismatch**  
`age` is `i16` in the graph struct but used in `w_raw = log(φ + ε) - λ * τ - μ * age`. The `μ * age` term mixes f16/f32 with i16. Need explicit cast.  
**Fix (nice-to-have):** Note in Phase 2: "Cast `age` to f16/f32 before computing `w_raw`."

**3f. Corruption schedule direction**  
`docs/math.md`: `α_t = α_T + (α_1 - α_T) * (T - t) / (T - 1)`. At `t=T`: `α_T + 0 = α_T = 0.8` (heavy). At `t=1`: `α_T + (α_1 - α_T) * (T-1)/(T-1) = α_1 = 0.15` (light). This is correct — refinement goes from `t=T` (heavy corruption) down to `t=1` (light). ✅

**3g. Taint update order**  
`docs/math.md` applies taint deposit then decay:
```
τ_e ← clamp(τ_e + ζ * relu(-Δ_k), 0, τ_max)
τ_e ← (1 - ρ_τ) * τ_e
```
TODO Phase 4 says same. But note: decay after deposit means newly-deposited taint is immediately partially decayed. This is probably intended (prevents taint spikes) but differs from how pheromone works (evaporate BEFORE deposit). Consistency would suggest decay-then-deposit for taint too.  
**Status:** Design choice, not a bug. Flag for Jeremy to confirm intent.

---

## 4. VRAM Realism

### Overall Assessment: ✅ Credible, with caveats

The `shapes.md` VRAM budget totals ~1.1 GB practical peak, well within the 4 GB RTX 3050. The TODO's inline estimates are more conservative (~350 MB model + 500-800 MB CUDA overhead = ~850 MB-1.15 GB). Both are plausible.

### Specific Concerns

**4a. `h_nbr` materialization is the real risk — correctly flagged** ✅  
The 84 MB intermediate at `[B=8, L=128, Emax=16, d=256]` is correctly identified as must-fuse. If burn doesn't support fused gather-softmax-sum, this becomes a hard blocker. The TODO mentions fusing but doesn't have a fallback plan if burn's API can't do it.  
**Fix:** Add fallback: "If fused kernel not feasible in burn, iterate over `L` dimension in a loop (128 iterations, each gathering `[B, Emax, d]` = 65 KB — trivially fits). Slower but correct."

**4b. Multi-step unrolling VRAM (Phase 6) is underestimated**  
TODO says ~140 MB for 2 unrolled steps (activations only). But autodiff graphs in burn store more than just activations — they store the full computation graph for backward. With 6 transformer blocks per step, the actual autodiff memory could be 3-5x the activation size. For 2 steps: potentially 350-700 MB of autodiff state.  
**Fix:** Revise Phase 6 VRAM estimate: "Autodiff state for 2 unrolled steps ≈ 300-500 MB (depends on burn's autodiff implementation). With gradient checkpointing: ~150-250 MB. Measure empirically at Phase 6 start. Budget: max 1 GB for autodiff."

**4c. Framework overhead estimate (500-800 MB) is reasonable but wide**  
CUDA context + burn runtime + cuDNN/cuBLAS typically costs 300-600 MB. The estimate is conservative, which is correct.  
**Status:** ✅ Fine.

**4d. Optimizer state at 100 MB is correct**  
AdamW stores 2 extra fp32 copies per parameter (~12M params × 4 bytes × 2 = 96 MB ≈ 100 MB). ✅

**4e. V=32768 is correctly flagged as risky**  
67 MB for logits alone. With optimizer, activations, and framework overhead, total would push to ~1.5 GB+. The TODO correctly starts at V=16384.  
**Status:** ✅ Correct decision.

### Verdict
The VRAM estimates are credible for `V=16384, B=8, L=128`. The main risks are:
1. Fused `h_nbr` kernel (must be solved, not optional)
2. Multi-step autodiff (likely underestimated by 2-3x)
3. burn's CUDA backend overhead (unknown until measured)

Recommendation: add "VRAM measurement gate" tasks at Phases 1, 2, and 6 — measure actual VRAM with `nvidia-smi` and record in `docs/vram-measurements.md`.

---

## 5. PCN Comparison Gaps

### 🔴 Critical

**5a. No shared evaluation harness exists or is specified**  
Phase 8 says "Implement a shared `eval-harness` script or Makefile target" but doesn't specify what it looks like. For a fair comparison, the harness must:
- Use identical tokenizer, vocabulary, and dataset splits
- Compute perplexity the same way (teacher-forced CE under same reference model, or intrinsic CE)
- Run both models on the same hardware under the same conditions
- Report identical metrics (accuracy, perplexity, VRAM, throughput)  
**Fix:** Create a concrete task in Phase 0: "Create `eval-harness/` workspace member crate (or script) with shared types: `EvalResult { masked_accuracy: f32, perplexity: f32, vram_peak_mb: u32, throughput_tok_sec: f32 }`. Both ERM and PCN implement a trait `Evaluable` or output this struct."

**5b. "Same number of gradient steps" is not a fair comparison**  
The TODO says "Synchronize training compute budget for fair comparison (e.g., same number of gradient steps or same wall-clock time)." But ERM does more computation per gradient step (route aggregation, ant sampling, pheromone updates). Fair comparison options:
1. Same wall-clock time (favors whichever is more efficient per second)
2. Same number of scorer forward passes (closer to compute-equivalent)
3. Same FLOPs (hardest to measure but most rigorous)  
**Fix:** Define comparison axes explicitly: "Compare at (a) same gradient steps, (b) same wall-clock time, (c) same number of scorer forward passes. Report all three."

**5c. PCN architecture not specified — can't verify equivalence**  
The TODO references `rust-pcn` but doesn't specify what PCN looks like. For a fair comparison, PCN should have the same scorer backbone (d=256, 6 blocks, same tokenizer/vocab). If PCN uses a different architecture, the comparison measures architecture + training paradigm differences, not just ERM vs PCN training.  
**Fix:** Add to Phase 8: "PCN baseline uses identical scorer architecture (d=256, B=6 blocks, h=4 heads, same embedding) trained as a standard masked language model with the same corruption schedule. The only difference: PCN has no route graph, no ants, no pheromone updates."

**5d. No statistical significance testing**  
Phase 8 says "Human eval: Small sample, 10 examples." For quantitative metrics, there's no mention of significance testing (confidence intervals, multiple runs with different seeds).  
**Fix:** Add: "Run both models with 3 different seeds. Report mean ± std for all quantitative metrics. Use paired t-test or bootstrap CI for perplexity comparison."

### 🟡 Important

**5e. Hypothesis H4 ("ERM is more VRAM-efficient at equivalent quality") may be false**  
ERM has the route graph + ant tensors as overhead. If the scorer is identical, ERM uses MORE VRAM than PCN by ~5-90 MB (graph + fused kernel overhead). ERM's hypothesis should be: "ERM achieves better quality with the same scorer size" (graph acts as external memory), not "less VRAM."  
**Fix:** Reframe H4: "ERM achieves better quality than PCN with the same parameter count, at the cost of modest additional VRAM for the route graph (~2 MB)."

**5f. No comparison at Milestone 1**  
Phase 8 says "Generate side-by-side loss curves at Milestones 1, 3, 6." At Milestone 1, ERM IS a plain denoiser (no graph). The comparison at M1 should show that both models are identical baselines.  
**Status:** This is actually good — it validates the experimental setup. Confirm: "M1 comparison serves as a sanity check: both models should have identical loss curves since ERM hasn't activated route graph yet."

---

## 6. Specific Fixes

### High Priority

| Location | Issue | Fix |
|---|---|---|
| TODO Phase 1, Denoising Loss | Loss normalization inconsistent across docs | Change to: "`L_denoise = (1/\|corrupted\|) * Σ CE(...)` (mean over corrupted, mean over batch)" and update spec |
| TODO Phase 3, line `A_F = ceil(0.9 * A) = 231` | Off-by-one: A_L + A_F > A | Change to: "`A_F = A - A_L = 256 - 26 = 230`" |
| TODO Phase 3, `follower_position_sample` | Missing `conf` computation | Add task before sampler: "Compute `conf[b,i] = max(softmax(logits[b,i,:]))` from scorer logits" |
| TODO Phase 2, `route_aggregate` return type | No trace recording for Phase 4 | Change signature to return `(r: [B,L,d], weights: [B,L,Emax] f16)` for Phase 4 edge tracing |
| TODO Phase 3 | No per-ant Δ aggregation | Add: "Implement `per_ant_delta(Δ_pos: [B,L], ant_pos: [B,A,pmax], winning_ant: [B,L]) -> Δ_ant: [B,A]`" |
| TODO Phase 6, VRAM estimate | Autodiff memory underestimated | Revise: "Autodiff state for 2 steps ≈ 300-500 MB. Budget max 1 GB. Measure at Phase 6 start." |
| `shapes.md`, A_F value | 230 vs 231 | Change to: "`A_F = 230` with note: `A_F = A - A_L`" (already 230 — update TODO to match) |
| TODO Phase 8 | No concrete PCN architecture spec | Add: "PCN uses identical scorer backbone. Only difference: no route graph, no ants." |

### Medium Priority

| Location | Issue | Fix |
|---|---|---|
| TODO Phase 2 | No fallback for h_nbr fusion | Add: "Fallback: loop over L, gather [B, Emax, d] per position" |
| TODO Phase 4 | `delta_per_ant` not defined | Cross-ref Phase 3 per-ant delta aggregation |
| TODO Phase 5 | `LeaderUtility` EdgeId undefined | Define: "`EdgeId = (batch, dst, src)`. Update on swap-remove." |
| TODO Phase 7 | No `--update-graph` flag | Add inference-time graph update toggle |
| TODO Phase 0 | No eval harness crate | Add `eval-harness` to workspace members |
| All phases | No VRAM measurement gates | Add: "Measure actual VRAM with nvidia-smi, record in docs/vram-measurements.md" after Phases 1, 2, 6 |
| TODO Phase 1 | Uncertainty head training signal unclear | Add: "Uncertainty head receives gradient from denoising loss only (no auxiliary loss in Phase 1)" |

### Low Priority

| Location | Issue | Fix |
|---|---|---|
| TODO Phase 2 | Concept node extensibility not mentioned | Add design note for future node types |
| TODO Phase 4 | Dirty-bit upload optimization not specified | Add as performance optimization task |
| `shapes.md`, logits size | 33.6 MB should be ~32 MB | Cosmetic — leave as-is |
| TODO Phase 8 | No statistical testing | Add: "3 seeds per model, report mean ± std" |

---

## Summary

**Overall quality: Strong.** The TODO is well-structured, phased correctly, and covers ~90% of the spec. The VRAM analysis is conservative and credible. The main gaps are:

1. **Loss normalization inconsistency** (3 documents disagree) — must fix before implementation
2. **A_L + A_F arithmetic** (off-by-one) — trivial fix
3. **PCN comparison methodology** is underspecified — needs concrete protocol before Phase 8
4. **Autodiff VRAM** for multi-step unrolling is likely 2-3x underestimated — add measurement gates
5. **Per-ant delta aggregation** is needed by Phase 4 but not built in Phase 3

None of these are architectural blockers. They're all fixable with the edits listed above.
