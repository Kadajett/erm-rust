//! One-step refinement pipeline for the Emergent Route Model.
//!
//! Wires together:
//! 1. Scorer forward pass → logits
//! 2. Top-k extraction from logits
//! 3. Follower ant proposals (position scoring + token sampling)
//! 4. Conflict-free merge → `y_{t-1}`
//! 5. Per-ant Δ computation (bridges Phase 3 → Phase 4)
//!
//! This module operates on a **single batch element** at a time.
//! Batch-level orchestration is handled by the caller.

use rand::Rng;

use crate::ants::{AntColony, FollowerConfig};
use crate::config::ErmConfig;
use crate::error::{ErmError, ErmResult};
use crate::graph::RouteGraph;
use crate::merge::{compute_ant_deltas, merge_proposals};
use crate::scorer::Scorer;

/// Result of a single refinement step.
#[derive(Debug, Clone)]
pub struct RefinementResult {
    /// Refined tokens after merge. Shape: `[L]` as `Vec<u32>`.
    pub y_new: Vec<u32>,
    /// Per-ant improvement deltas. Shape: `[num_followers]`.
    pub ant_deltas: Vec<f32>,
    /// Number of edits applied.
    pub num_edits: usize,
}

/// Execute one refinement step on a single sequence.
///
/// Pipeline:
/// 1. Run scorer forward on `y_t` → logits `[L, V]`
/// 2. Follower ants sample proposals using logits + route graph
/// 3. Merge proposals (conflict-free, capped at `max_edits`)
/// 4. Run scorer forward on `y_new` → logits_after
/// 5. Compute per-ant Δ from logit differences
///
/// # Arguments
///
/// - `y_t`: current tokens for one sequence, `[L]` as `Vec<u32>`.
/// - `scorer`: the neural scorer network.
/// - `graph`: route graph providing pheromone signals.
/// - `batch_idx`: which batch element in the graph to use.
/// - `config`: ERM hyperparameters.
/// - `editable`: boolean mask `[L]`, `true` = position can be edited.
/// - `rng`: random number generator for deterministic sampling.
///
/// # Returns
///
/// [`RefinementResult`] containing `y_{t-1}`, per-ant deltas, and edit count.
///
/// # Errors
///
/// Returns [`ErmError`] on shape mismatches or scorer failures.
///
/// # Shape reference
///
/// | Tensor | Shape |
/// |---|---|
/// | `y_t` (input) | `[L]` |
/// | `editable` (input) | `[L]` |
/// | `y_new` (output) | `[L]` |
/// | `ant_deltas` (output) | `[num_followers]` |
pub fn refine_step<R: Rng>(
    y_t: &[u32],
    scorer: &Scorer,
    graph: &RouteGraph,
    batch_idx: usize,
    config: &ErmConfig,
    editable: &[bool],
    rng: &mut R,
) -> ErmResult<RefinementResult> {
    let seq_len = config.seq_len;
    let vocab_size = config.vocab_size;

    if y_t.len() != seq_len {
        return Err(ErmError::ShapeMismatch {
            expected: format!("y_t length = {seq_len}"),
            got: format!("{}", y_t.len()),
        });
    }
    if editable.len() != seq_len {
        return Err(ErmError::ShapeMismatch {
            expected: format!("editable length = {seq_len}"),
            got: format!("{}", editable.len()),
        });
    }

    // Step 1: Scorer forward pass → logits [1, L, V].
    let scorer_out = scorer.forward(y_t, 1)?;
    // logits is flat [1 * L * V] = [L * V]; extract just the [L, V] portion.
    let logits = &scorer_out.logits;

    // Step 2: Follower ant proposals.
    let num_followers = config.num_followers();
    let first_follower_id = config.num_leaders(); // A_F = A - A_L
    let follower_cfg = FollowerConfig::from_config(config);

    let proposals = AntColony::sample_follower_proposals(
        logits,
        graph,
        batch_idx,
        &follower_cfg,
        editable,
        num_followers,
        first_follower_id,
        seq_len,
        vocab_size,
        rng,
    )?;

    // Step 3: Merge proposals.
    let max_edits = config.max_edits();
    let y_new = merge_proposals(&proposals, y_t, editable, seq_len, max_edits)?;

    // Count actual edits.
    let num_edits = y_t.iter().zip(y_new.iter()).filter(|(a, b)| a != b).count();

    // Step 4: Scorer forward on y_new for delta computation.
    let scorer_out_after = scorer.forward(&y_new, 1)?;
    let logits_after = &scorer_out_after.logits;

    // Step 5: Per-ant Δ.
    let ant_deltas = compute_ant_deltas(
        &proposals,
        y_t,
        &y_new,
        logits,
        logits_after,
        vocab_size,
        num_followers,
    )?;

    Ok(RefinementResult {
        y_new,
        ant_deltas,
        num_edits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn test_config() -> ErmConfig {
        ErmConfig {
            batch_size: 1,
            seq_len: 8,
            vocab_size: 16,
            hidden_dim: 8,
            num_blocks: 2,
            num_heads: 2,
            mlp_expansion: 4,
            dropout: 0.0,
            num_ants: 10,
            leader_fraction: 0.10,
            pmax: 4,
            topk: 4,
            emax: 4,
            max_edits_per_step: 0.15,
            ..ErmConfig::default()
        }
    }

    #[test]
    fn test_refine_step_produces_valid_output() {
        let cfg = test_config();
        let scorer = Scorer::new(&cfg, cfg.vocab_size, 42);
        let graph = RouteGraph::new(&cfg);
        let editable = vec![true; cfg.seq_len];
        let y_t: Vec<u32> = (0..cfg.seq_len as u32).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(99);

        let result = refine_step(&y_t, &scorer, &graph, 0, &cfg, &editable, &mut rng).unwrap();

        // Output must have correct length.
        assert_eq!(result.y_new.len(), cfg.seq_len);

        // Tokens must be in valid range.
        for &t in &result.y_new {
            assert!(
                (t as usize) < cfg.vocab_size,
                "token {t} >= vocab_size {}",
                cfg.vocab_size
            );
        }

        // ant_deltas must have correct length (num_followers).
        assert_eq!(result.ant_deltas.len(), cfg.num_followers());

        // All deltas must be finite.
        for &d in &result.ant_deltas {
            assert!(d.is_finite(), "delta must be finite, got {d}");
        }
    }

    #[test]
    fn test_refine_step_deterministic() {
        let cfg = test_config();
        let scorer = Scorer::new(&cfg, cfg.vocab_size, 42);
        let graph = RouteGraph::new(&cfg);
        let editable = vec![true; cfg.seq_len];
        let y_t: Vec<u32> = (0..cfg.seq_len as u32).collect();

        let mut rng1 = ChaCha8Rng::seed_from_u64(42);
        let r1 = refine_step(&y_t, &scorer, &graph, 0, &cfg, &editable, &mut rng1).unwrap();

        let mut rng2 = ChaCha8Rng::seed_from_u64(42);
        let r2 = refine_step(&y_t, &scorer, &graph, 0, &cfg, &editable, &mut rng2).unwrap();

        assert_eq!(r1.y_new, r2.y_new);
        assert_eq!(r1.num_edits, r2.num_edits);
        assert_eq!(r1.ant_deltas.len(), r2.ant_deltas.len());
        for (a, b) in r1.ant_deltas.iter().zip(r2.ant_deltas.iter()) {
            assert!((a - b).abs() < 1e-9, "deltas differ: {a} vs {b}");
        }
    }

    #[test]
    fn test_refine_step_edit_cap() {
        let cfg = test_config();
        // max_edits = ceil(0.15 * 8) = 2
        let scorer = Scorer::new(&cfg, cfg.vocab_size, 42);
        let graph = RouteGraph::new(&cfg);
        let editable = vec![true; cfg.seq_len];
        let y_t: Vec<u32> = (0..cfg.seq_len as u32).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(55);

        let result = refine_step(&y_t, &scorer, &graph, 0, &cfg, &editable, &mut rng).unwrap();

        // Should not exceed max_edits.
        assert!(
            result.num_edits <= cfg.max_edits(),
            "num_edits {} > max_edits {}",
            result.num_edits,
            cfg.max_edits()
        );
    }

    #[test]
    fn test_refine_step_non_editable_preserved() {
        let cfg = test_config();
        let scorer = Scorer::new(&cfg, cfg.vocab_size, 42);
        let graph = RouteGraph::new(&cfg);
        // Only first 2 positions are editable.
        let mut editable = vec![false; cfg.seq_len];
        editable[0] = true;
        editable[1] = true;
        let y_t: Vec<u32> = (0..cfg.seq_len as u32).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(77);

        let result = refine_step(&y_t, &scorer, &graph, 0, &cfg, &editable, &mut rng).unwrap();

        // Non-editable positions must be unchanged.
        for i in 2..cfg.seq_len {
            assert_eq!(
                result.y_new[i], y_t[i],
                "position {i} should be unchanged (non-editable)"
            );
        }
    }

    #[test]
    fn test_refine_step_shape_mismatch() {
        let cfg = test_config();
        let scorer = Scorer::new(&cfg, cfg.vocab_size, 42);
        let graph = RouteGraph::new(&cfg);
        let editable = vec![true; cfg.seq_len];

        // Wrong y_t length.
        let bad_y_t = vec![0u32; 3];
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        assert!(refine_step(&bad_y_t, &scorer, &graph, 0, &cfg, &editable, &mut rng).is_err());
    }
}
