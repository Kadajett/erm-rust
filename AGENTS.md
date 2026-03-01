# AGENTS.md — ERM Coding Rules for AI Agents

All agents working on this codebase **must** follow these rules. Violations will be caught in review and added here as new rules.

---

## Rust Style

- **Strict Clippy:** All code must pass `cargo clippy -- -D warnings`
- **Formatting:** Run `cargo fmt` on every file before committing
- **No `unwrap()` or `expect()`** in library code (`erm-core/src/`). Use `Result<T, ErmError>` everywhere.
- `unwrap()` is allowed ONLY in tests and `erm-cli/src/main.rs`
- **Imports:** Group by: std, external crates, internal modules. One blank line between groups.
- **Doc comments:** All public types, functions, and modules must have `///` doc comments
- **Edition:** 2021

## Tensor Shape Conventions

- All tensor dimensions follow `[B, L, ...]` ordering (batch first, sequence second)
- **Document shapes in comments** on every function that creates or transforms tensors:
  ```rust
  /// Computes route message from neighbor embeddings.
  /// Input: h [B, L, d], nbr_idx [B, L, Emax]
  /// Output: r [B, L, d]
  ```
- Use type aliases from `types.rs` (e.g., `TokenId`, `SeqLen`)

## ERM-Specific Invariants

These invariants must hold at ALL times. Add assertions in debug builds.

- `phi >= 0.0` — pheromone values are never negative
- `taint` in `[0.0, config.taint_max]` — taint is bounded
- `nbr_idx` values are in `[0, L)` or `EMPTY_SLOT` sentinel — no out-of-bounds neighbors
- Edge count per destination node ≤ `Emax` — enforced by dense-neighbor format
- `A_L + A_F == A` — ant counts must sum exactly. Use `A_F = A - A_L`, never compute both independently
- Corruption rates: `mask_rate` + `replace_rate` ≤ 1.0

## Graph Format

- **v1 uses dense-neighbor format**, NOT CSR:
  - `nbr_idx: [B, L, Emax]` — neighbor indices (i32, EMPTY_SLOT = -1)
  - `phi: [B, L, Emax]` — pheromone (f32)
  - `taint: [B, L, Emax]` — taint (f32)
  - `age: [B, L, Emax]` — edge age (u32)
- Pruning: swap-remove within the `[Emax]` slots, keep compacted (no holes)
- RouteAggregate must return edge weights alongside the message vector (needed for pheromone tracing in Phase 4+)

## Loss Function

- `L_denoise` = **mean** over corrupted positions (not sum). This is the reconciled decision per REVIEW.md.
- Colony improvement Δ = sum of per-position CE deltas for edited positions only

## Testing Requirements

- Every public function needs at least one unit test
- Math kernels (corruption, pheromone updates, route aggregation) need **property-based tests** with `proptest`
- All RNG-dependent tests must use **deterministic seeding** (`rand_chacha::ChaCha8Rng`)
- Shape assertion tests for every forward-pass function
- Golden tests: fixed input → expected output, checked exactly

## Performance Rules

- **No allocations in hot loops** (refinement steps, ant sampling, pheromone updates)
- Prefer in-place tensor operations (`+=`, `.assign()`) for large tensors
- Keep all heavy state GPU-resident; CPU only emits small control buffers
- Profile VRAM at milestone boundaries (Phases 1, 2, 6 minimum)

## Error Handling

- Use `ErmError` from `error.rs` for all fallible operations
- Propagate with `?` — no manual matching unless you need to add context
- Add context with `.map_err(|e| ErmError::Training(format!("...: {e}")))`

## Commit Messages

- No AI attribution. No "Co-Authored-By: Claude" or similar.
- Format: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`
- Keep messages concise and descriptive

## Common Mistakes (grows over time)

1. **Computing A_L and A_F independently** — always derive: `A_F = A - A_L`
2. **Forgetting EMPTY_SLOT in graph operations** — filter out -1 entries before any index operation
3. **Using sum instead of mean for denoising loss** — it's mean over corrupted positions
4. **Not returning edge weights from RouteAggregate** — Phase 4 needs them for tracing
5. **Unbounded graph growth** — every edge insertion must check Emax and prune if needed

---

_This file is a living document. Every code review should check if new rules need to be added._
