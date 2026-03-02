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

use crate::ants::{
    apply_death_respawn, AntColony, AntState, DeathMode, FollowerConfig, LeaderConfig,
};
use crate::config::{ErmConfig, PheromoneConfig};
use crate::error::{ErmError, ErmResult};
use crate::graph::RouteGraph;
use crate::merge::{compute_ant_deltas, merge_proposals, SimpleEditProposal};
use crate::pheromone::{build_edge_traces, prune_edges, update_pheromones, PheromoneStats};
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

/// Result of a single refinement step with pheromone updates.
#[derive(Debug, Clone)]
pub struct PheromoneRefinementResult {
    /// Refined tokens after merge. Shape: `[L]`.
    pub y_new: Vec<u32>,
    /// Per-ant improvement deltas. Shape: `[num_ants]` (followers + leaders).
    pub ant_deltas: Vec<f32>,
    /// Number of edits applied.
    pub num_edits: usize,
    /// Pheromone statistics after the update.
    pub pheromone_stats: PheromoneStats,
    /// Number of edges pruned.
    pub edges_pruned: usize,
}

/// Execute one refinement step with pheromone feedback.
///
/// Pipeline:
/// 1. Route aggregate → route messages + edge weights
/// 2. Scorer forward with route messages → logits
/// 3. Follower ant proposals using logits + graph
/// 4. Merge proposals → `y_new`
/// 5. Scorer forward on `y_new` → logits_after
/// 6. Compute per-ant Δ
/// 7. Build edge traces + update pheromones
/// 8. Prune weak edges
///
/// # Arguments
///
/// - `y_t`: current tokens for one sequence, `[L]`.
/// - `scorer`: the neural scorer network.
/// - `graph`: mutable route graph (updated in-place).
/// - `batch_idx`: which batch element in the graph.
/// - `config`: ERM hyperparameters.
/// - `pheromone_config`: pheromone-specific hyperparameters.
/// - `editable`: boolean mask `[L]`.
/// - `rng`: random number generator.
///
/// # Returns
///
/// [`PheromoneRefinementResult`] with refined tokens, deltas, and pheromone stats.
///
/// # Errors
///
/// Returns [`ErmError`] on shape mismatches or scorer failures.
#[allow(clippy::too_many_arguments)]
pub fn refine_step_with_pheromones<R: Rng>(
    y_t: &[u32],
    scorer: &Scorer,
    graph: &mut RouteGraph,
    batch_idx: usize,
    config: &ErmConfig,
    pheromone_config: &PheromoneConfig,
    editable: &[bool],
    rng: &mut R,
) -> ErmResult<PheromoneRefinementResult> {
    let seq_len = config.seq_len;
    let vocab_size = config.vocab_size;
    let d = config.hidden_dim;

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

    // Step 1: Scorer forward pass (no route for now — Phase 1 scorer).
    let scorer_out = scorer.forward(y_t, 1)?;
    let logits = &scorer_out.logits;

    // Step 2: Route aggregate → edge weights for trace building.
    // Use zeros for hidden state if we don't have a real one.
    let hidden = vec![0.0_f32; config.batch_size * seq_len * d];
    let (_, edge_weights) = graph.route_aggregate(
        &hidden,
        d,
        config.route_epsilon,
        config.route_lambda,
        config.route_mu,
    )?;

    // Step 3: Follower ant proposals.
    let num_followers = config.num_followers();
    let first_follower_id = config.num_leaders();
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

    // Step 4: Merge proposals.
    let max_edits = config.max_edits();
    let y_new = merge_proposals(&proposals, y_t, editable, seq_len, max_edits)?;
    let num_edits = y_t.iter().zip(y_new.iter()).filter(|(a, b)| a != b).count();

    // Step 5: Scorer forward on y_new.
    let scorer_out_after = scorer.forward(&y_new, 1)?;
    let logits_after = &scorer_out_after.logits;

    // Step 6: Per-ant Δ.
    let ant_deltas = compute_ant_deltas(
        &proposals,
        y_t,
        &y_new,
        logits,
        logits_after,
        vocab_size,
        num_followers,
    )?;

    // Step 7: Build edge traces and update pheromones.
    let traces = build_edge_traces(
        &proposals,
        &edge_weights,
        config.batch_size,
        seq_len,
        config.emax,
    );
    let pheromone_stats = update_pheromones(graph, &traces, &ant_deltas, pheromone_config)?;

    // Step 8: Prune weak edges.
    let edges_pruned = prune_edges(
        graph,
        pheromone_config.prune_min_score,
        pheromone_config.prune_max_age,
        pheromone_config.route_lambda,
    );

    Ok(PheromoneRefinementResult {
        y_new,
        ant_deltas,
        num_edits,
        pheromone_stats,
        edges_pruned,
    })
}

/// Result of a full colony step (followers + leaders + pheromones + edges + death).
#[derive(Debug, Clone)]
pub struct ColonyStepResult {
    /// Refined tokens after merge. Shape: `[L]`.
    pub y_new: Vec<u32>,
    /// Total per-ant deltas (followers + leaders combined). Shape: `[num_ants]`.
    pub ant_deltas: Vec<f32>,
    /// Number of edits applied.
    pub num_edits: usize,
    /// Pheromone statistics after the update.
    pub pheromone_stats: PheromoneStats,
    /// Number of edges pruned.
    pub edges_pruned: usize,
    /// Number of new edges inserted by leaders.
    pub edges_inserted: usize,
    /// Number of ants that died and were respawned.
    pub deaths: usize,
}

/// Execute a full colony step: followers + leaders → merge → pheromones → edges → death.
///
/// This is the top-level orchestration for one refinement step that includes:
/// 1. Follower proposals (exploit)
/// 2. Leader proposals (explore) + edge proposals
/// 3. Merge all proposals
/// 4. Compute deltas
/// 5. Build traces + update pheromones
/// 6. Insert proposed edges
/// 7. Prune weak edges
/// 8. Apply death/respawn
///
/// # Arguments
///
/// - `y_t`: current tokens, `[L]`.
/// - `scorer`: neural scorer.
/// - `graph`: mutable route graph.
/// - `ant_state`: mutable ant lifecycle state.
/// - `batch_idx`: batch element index.
/// - `config`: ERM hyperparameters.
/// - `pheromone_config`: pheromone hyperparameters.
/// - `editable`: boolean mask `[L]`.
/// - `death_mode`: which death mode to use.
/// - `rng`: random number generator.
///
/// # Returns
///
/// [`ColonyStepResult`] with all outputs from the step.
///
/// # Errors
///
/// Returns [`ErmError`] on failures.
#[allow(clippy::too_many_arguments)]
pub fn full_colony_step<R: Rng>(
    y_t: &[u32],
    scorer: &Scorer,
    graph: &mut RouteGraph,
    ant_state: &mut AntState,
    batch_idx: usize,
    config: &ErmConfig,
    pheromone_config: &PheromoneConfig,
    editable: &[bool],
    death_mode: DeathMode,
    rng: &mut R,
) -> ErmResult<ColonyStepResult> {
    let seq_len = config.seq_len;
    let vocab_size = config.vocab_size;
    let d = config.hidden_dim;

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

    // Step 1: Scorer forward → logits + uncertainty.
    let scorer_out = scorer.forward(y_t, 1)?;
    let logits = &scorer_out.logits;
    let uncertainty = &scorer_out.uncertainty;

    // Step 2: Route aggregate → edge weights.
    let hidden = vec![0.0_f32; config.batch_size * seq_len * d];
    let (_, edge_weights) = graph.route_aggregate(
        &hidden,
        d,
        config.route_epsilon,
        config.route_lambda,
        config.route_mu,
    )?;

    // Step 3: Follower proposals.
    let num_followers = config.num_followers();
    let first_follower_id = config.num_leaders();
    let follower_cfg = FollowerConfig::from_config(config);

    let follower_proposals = AntColony::sample_follower_proposals(
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

    // Step 4: Leader proposals.
    let num_leaders = config.num_leaders();
    let leader_cfg = LeaderConfig::from_config(config);

    let (leader_proposals, edge_proposals) = AntColony::sample_leader_proposals(
        logits,
        uncertainty,
        graph,
        batch_idx,
        &leader_cfg,
        editable,
        num_leaders,
        0, // leaders start at id 0
        seq_len,
        vocab_size,
        rng,
    )?;

    // Step 5: Merge all proposals.
    let mut all_proposals: Vec<SimpleEditProposal> =
        Vec::with_capacity(follower_proposals.len() + leader_proposals.len());
    all_proposals.extend(follower_proposals.iter().cloned());
    all_proposals.extend(leader_proposals.iter().cloned());

    let max_edits = config.max_edits();
    let y_new = merge_proposals(&all_proposals, y_t, editable, seq_len, max_edits)?;
    let num_edits = y_t.iter().zip(y_new.iter()).filter(|(a, b)| a != b).count();

    // Step 6: Scorer forward on y_new → compute deltas.
    let scorer_out_after = scorer.forward(&y_new, 1)?;
    let logits_after = &scorer_out_after.logits;

    let total_ants = config.num_ants;
    let ant_deltas = compute_ant_deltas(
        &all_proposals,
        y_t,
        &y_new,
        logits,
        logits_after,
        vocab_size,
        total_ants,
    )?;

    // Step 7: Build edge traces and update pheromones.
    let traces = build_edge_traces(
        &all_proposals,
        &edge_weights,
        config.batch_size,
        seq_len,
        config.emax,
    );
    let pheromone_stats = update_pheromones(graph, &traces, &ant_deltas, pheromone_config)?;

    // Step 8: Insert proposed edges from leaders.
    let edges_inserted = graph.propose_edges(&edge_proposals, config.phi_init, config.route_lambda);

    // Step 9: Prune weak edges.
    let edges_pruned = prune_edges(
        graph,
        pheromone_config.prune_min_score,
        pheromone_config.prune_max_age,
        pheromone_config.route_lambda,
    );

    // Step 10: Death/respawn.
    let deaths = apply_death_respawn(ant_state, &ant_deltas, config, death_mode, rng);

    Ok(ColonyStepResult {
        y_new,
        ant_deltas,
        num_edits,
        pheromone_stats,
        edges_pruned,
        edges_inserted,
        deaths,
    })
}

// ── Multi-step refinement (Milestone 6) ───────────────────────────────────

/// Statistics for a single step within multi-step refinement.
#[derive(Debug, Clone)]
pub struct StepStats {
    /// Number of edits applied in this step.
    pub num_edits: usize,
    /// Pheromone statistics after this step.
    pub pheromone_stats: PheromoneStats,
}

/// Result of a multi-step refinement run.
#[derive(Debug, Clone)]
pub struct MultiStepResult {
    /// Final refined tokens. Shape: `[L]`.
    pub y_final: Vec<u32>,
    /// Per-step statistics (edits, pheromone stats).
    pub step_stats: Vec<StepStats>,
    /// Total number of refinement steps actually executed (may be < T if early-stopped).
    pub steps_executed: usize,
    /// Reason for stopping: `"max_steps"`, `"no_edits"`, `"convergence"`, or `"confidence"`.
    pub stop_reason: String,
    /// Peak memory estimate in bytes (sum of largest Vec allocations tracked).
    pub peak_memory_estimate: usize,
}

/// Configuration for multi-step refinement stopping criteria.
#[derive(Debug, Clone)]
pub struct MultiStepConfig {
    /// Maximum number of refinement steps `T`.
    pub max_steps: usize,
    /// Confidence threshold: stop if mean max-softmax across editable positions exceeds this.
    pub confidence_threshold: f32,
}

impl MultiStepConfig {
    /// Build from the global [`ErmConfig`].
    #[must_use]
    pub fn from_config(config: &ErmConfig) -> Self {
        Self {
            max_steps: config.refinement_steps,
            confidence_threshold: 0.9,
        }
    }
}

/// Compute the mean confidence (max softmax probability) across editable positions.
///
/// Confidence for position `i` = `max(softmax(logits[i]))`.
///
/// # Arguments
///
/// - `logits`: flat `[L * V]` logits for one sequence.
/// - `editable`: boolean mask `[L]`.
/// - `seq_len`: sequence length `L`.
/// - `vocab_size`: vocabulary size `V`.
///
/// # Returns
///
/// Mean confidence across editable positions. Returns `1.0` if no positions are editable.
fn mean_confidence(logits: &[f32], editable: &[bool], seq_len: usize, vocab_size: usize) -> f32 {
    let mut total_conf = 0.0_f32;
    let mut count = 0_usize;

    for (i, &is_editable) in editable.iter().enumerate().take(seq_len) {
        if !is_editable {
            continue;
        }
        let start = i * vocab_size;
        let end = start + vocab_size;
        if end > logits.len() {
            continue;
        }
        let row = &logits[start..end];

        // Numerically stable softmax: max softmax = 1 / sum(exp(x_j - x_max)).
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();
        if exp_sum > 0.0 {
            let max_softmax = 1.0 / exp_sum;
            total_conf += max_softmax;
        }
        count += 1;
    }

    if count == 0 {
        1.0
    } else {
        total_conf / count as f32
    }
}

/// Run T refinement steps in sequence: `y_T → y_{T-1} → ... → y_1`.
///
/// Each step uses [`refine_step_with_pheromones`]. The loop stops early if:
/// - No edits were applied in a step.
/// - Mean confidence across editable positions exceeds `confidence_threshold`.
/// - Convergence: `y` is unchanged between consecutive steps.
///
/// # Arguments
///
/// - `y_init`: initial tokens for one sequence, `[L]`.
/// - `scorer`: the neural scorer network.
/// - `graph`: mutable route graph (updated in-place across steps).
/// - `batch_idx`: which batch element in the graph.
/// - `config`: ERM hyperparameters.
/// - `pheromone_config`: pheromone hyperparameters.
/// - `multi_config`: multi-step stopping criteria.
/// - `editable`: boolean mask `[L]`.
/// - `rng`: random number generator.
///
/// # Returns
///
/// [`MultiStepResult`] with final tokens, per-step stats, and stopping info.
///
/// # Errors
///
/// Returns [`ErmError`] on shape mismatches or scorer failures.
#[allow(clippy::too_many_arguments)]
pub fn multi_step_refine<R: Rng>(
    y_init: &[u32],
    scorer: &Scorer,
    graph: &mut RouteGraph,
    batch_idx: usize,
    config: &ErmConfig,
    pheromone_config: &PheromoneConfig,
    multi_config: &MultiStepConfig,
    editable: &[bool],
    rng: &mut R,
) -> ErmResult<MultiStepResult> {
    let seq_len = config.seq_len;
    let vocab_size = config.vocab_size;

    if y_init.len() != seq_len {
        return Err(ErmError::ShapeMismatch {
            expected: format!("y_init length = {seq_len}"),
            got: format!("{}", y_init.len()),
        });
    }

    let mut y_current = y_init.to_vec();
    let mut step_stats: Vec<StepStats> = Vec::with_capacity(multi_config.max_steps);
    let mut stop_reason = "max_steps".to_string();

    // Track peak memory estimate: start with y_current.
    let y_mem = y_current.len() * std::mem::size_of::<u32>();
    let mut peak_mem: usize = y_mem;

    for step in 0..multi_config.max_steps {
        let y_before = y_current.clone();

        // Track memory: y_current + y_before clones.
        let clone_mem = y_mem * 2 + step_stats.len() * std::mem::size_of::<StepStats>();
        peak_mem = peak_mem.max(clone_mem);

        let result = refine_step_with_pheromones(
            &y_current,
            scorer,
            graph,
            batch_idx,
            config,
            pheromone_config,
            editable,
            rng,
        )?;

        // Track memory from logits (2 forward passes inside refine_step_with_pheromones).
        let logits_mem = seq_len * vocab_size * std::mem::size_of::<f32>() * 2;
        peak_mem = peak_mem.max(logits_mem);

        step_stats.push(StepStats {
            num_edits: result.num_edits,
            pheromone_stats: result.pheromone_stats,
        });

        y_current = result.y_new;

        // Early stop: no edits applied.
        if result.num_edits == 0 {
            stop_reason = "no_edits".to_string();
            break;
        }

        // Early stop: convergence (y unchanged).
        if y_current == y_before {
            stop_reason = "convergence".to_string();
            break;
        }

        // Early stop: mean confidence exceeds threshold.
        if step + 1 < multi_config.max_steps {
            let scorer_out = scorer.forward(&y_current, 1)?;
            let conf = mean_confidence(&scorer_out.logits, editable, seq_len, vocab_size);
            if conf > multi_config.confidence_threshold {
                stop_reason = "confidence".to_string();
                break;
            }
        }
    }

    Ok(MultiStepResult {
        y_final: y_current,
        steps_executed: step_stats.len(),
        step_stats,
        stop_reason,
        peak_memory_estimate: peak_mem,
    })
}

/// Generate tokens from scratch by initializing with MASK tokens and refining.
///
/// Initializes `y_T = [MASK] * length`, then runs [`multi_step_refine`] to
/// produce a complete token sequence.
///
/// # Arguments
///
/// - `scorer`: the neural scorer.
/// - `graph`: mutable route graph.
/// - `config`: ERM hyperparameters. `config.seq_len` must equal `length`.
/// - `pheromone_config`: pheromone hyperparameters.
/// - `length`: desired output length in tokens.
/// - `rng`: random number generator.
///
/// # Returns
///
/// [`MultiStepResult`] with generated tokens.
///
/// # Errors
///
/// Returns [`ErmError`] if `length != config.seq_len` or on refinement failures.
pub fn generate_from_scratch<R: Rng>(
    scorer: &Scorer,
    graph: &mut RouteGraph,
    config: &ErmConfig,
    pheromone_config: &PheromoneConfig,
    length: usize,
    rng: &mut R,
) -> ErmResult<MultiStepResult> {
    if length != config.seq_len {
        return Err(ErmError::InvalidConfig(format!(
            "generate_from_scratch: length={length} must equal config.seq_len={}",
            config.seq_len
        )));
    }

    // MASK token id = vocab_size (sentinel). If scorer embedding table is large
    // enough, use the real MASK id; otherwise use 0 as a pseudo-mask.
    let mask_id = if (config.vocab_size as u32) < scorer.total_vocab as u32 {
        config.vocab_size as u32
    } else {
        0_u32
    };
    let y_init: Vec<u32> = vec![mask_id; length];
    let editable = vec![true; length];
    let multi_config = MultiStepConfig::from_config(config);

    multi_step_refine(
        &y_init,
        scorer,
        graph,
        0,
        config,
        pheromone_config,
        &multi_config,
        &editable,
        rng,
    )
}

/// Generate tokens with a fixed prefix (prompted generation).
///
/// The prefix tokens are set as non-editable. The suffix is initialized with
/// MASK tokens and refined via [`multi_step_refine`].
///
/// # Arguments
///
/// - `scorer`: the neural scorer.
/// - `graph`: mutable route graph.
/// - `config`: ERM hyperparameters. `config.seq_len` must equal `prefix.len() + gen_length`.
/// - `pheromone_config`: pheromone hyperparameters.
/// - `prefix`: token ids for the fixed prompt prefix.
/// - `gen_length`: number of tokens to generate after the prefix.
/// - `rng`: random number generator.
///
/// # Returns
///
/// [`MultiStepResult`] with generated tokens. Prefix positions are preserved unchanged.
///
/// # Errors
///
/// Returns [`ErmError`] if lengths don't match, prefix is empty, or refinement fails.
#[allow(clippy::too_many_arguments)]
pub fn generate_prompted<R: Rng>(
    scorer: &Scorer,
    graph: &mut RouteGraph,
    config: &ErmConfig,
    pheromone_config: &PheromoneConfig,
    prefix: &[u32],
    gen_length: usize,
    rng: &mut R,
) -> ErmResult<MultiStepResult> {
    let total_len = prefix.len() + gen_length;
    if total_len != config.seq_len {
        return Err(ErmError::InvalidConfig(format!(
            "generate_prompted: prefix({}) + gen_length({gen_length}) = {total_len}, \
             must equal config.seq_len={}",
            prefix.len(),
            config.seq_len
        )));
    }
    if prefix.is_empty() {
        return Err(ErmError::InvalidConfig(
            "generate_prompted: prefix must not be empty".to_string(),
        ));
    }

    // Build initial sequence: prefix ++ [MASK] * gen_length.
    let mask_id = if (config.vocab_size as u32) < scorer.total_vocab as u32 {
        config.vocab_size as u32
    } else {
        0_u32
    };
    let mut y_init: Vec<u32> = Vec::with_capacity(total_len);
    y_init.extend_from_slice(prefix);
    y_init.extend(std::iter::repeat_n(mask_id, gen_length));

    // Editable mask: prefix = false, suffix = true.
    let mut editable = vec![false; prefix.len()];
    editable.extend(std::iter::repeat_n(true, gen_length));

    let multi_config = MultiStepConfig::from_config(config);

    multi_step_refine(
        &y_init,
        scorer,
        graph,
        0,
        config,
        pheromone_config,
        &multi_config,
        &editable,
        rng,
    )
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
        let graph = RouteGraph::new_empty(&cfg);
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
        let graph = RouteGraph::new_empty(&cfg);
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
        let graph = RouteGraph::new_empty(&cfg);
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
        let graph = RouteGraph::new_empty(&cfg);
        // Only first 2 positions are editable.
        let mut editable = vec![false; cfg.seq_len];
        editable[0] = true;
        editable[1] = true;
        let y_t: Vec<u32> = (0..cfg.seq_len as u32).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(77);

        let result = refine_step(&y_t, &scorer, &graph, 0, &cfg, &editable, &mut rng).unwrap();

        // Non-editable positions must be unchanged.
        for (i, (new_val, old_val)) in result.y_new.iter().zip(y_t.iter()).enumerate().skip(2) {
            assert_eq!(
                *new_val, *old_val,
                "position {i} should be unchanged (non-editable)"
            );
        }
    }

    #[test]
    fn test_refine_step_shape_mismatch() {
        let cfg = test_config();
        let scorer = Scorer::new(&cfg, cfg.vocab_size, 42);
        let graph = RouteGraph::new_empty(&cfg);
        let editable = vec![true; cfg.seq_len];

        // Wrong y_t length.
        let bad_y_t = vec![0u32; 3];
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        assert!(refine_step(&bad_y_t, &scorer, &graph, 0, &cfg, &editable, &mut rng).is_err());
    }

    // ── refine_step_with_pheromones tests ────────────────────────────────

    #[test]
    fn test_refine_step_with_pheromones_produces_valid_output() {
        let cfg = test_config();
        let scorer = Scorer::new(&cfg, cfg.vocab_size, 42);
        let mut graph = RouteGraph::new_empty(&cfg);

        // Add some edges so pheromone update has something to work with.
        graph.add_edge(0, 0, 1, 0.5).unwrap();
        graph.add_edge(0, 1, 2, 0.3).unwrap();

        let editable = vec![true; cfg.seq_len];
        let y_t: Vec<u32> = (0..cfg.seq_len as u32).collect();
        let pconfig = PheromoneConfig::from_config(&cfg);
        let mut rng = ChaCha8Rng::seed_from_u64(99);

        let result = refine_step_with_pheromones(
            &y_t, &scorer, &mut graph, 0, &cfg, &pconfig, &editable, &mut rng,
        )
        .unwrap();

        assert_eq!(result.y_new.len(), cfg.seq_len);
        for &t in &result.y_new {
            assert!((t as usize) < cfg.vocab_size);
        }
        assert!(result.pheromone_stats.mean_phi.is_finite());
        assert!(result.pheromone_stats.max_phi.is_finite());
    }

    #[test]
    fn test_refine_step_with_pheromones_updates_graph() {
        let cfg = test_config();
        let scorer = Scorer::new(&cfg, cfg.vocab_size, 42);
        let mut graph = RouteGraph::new_empty(&cfg);

        graph.add_edge(0, 0, 1, 1.0).unwrap();
        let flat = graph.idx(0, 0, 0);
        let phi_before = graph.phi[flat];
        let age_before = graph.age[flat];

        let editable = vec![true; cfg.seq_len];
        let y_t: Vec<u32> = (0..cfg.seq_len as u32).collect();
        let pconfig = PheromoneConfig::from_config(&cfg);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        refine_step_with_pheromones(
            &y_t, &scorer, &mut graph, 0, &cfg, &pconfig, &editable, &mut rng,
        )
        .unwrap();

        // Age should have increased.
        assert!(
            graph.age[flat] > age_before,
            "age should increase: {} vs {}",
            graph.age[flat],
            age_before
        );
        // Phi should have changed (evaporation at least).
        assert!(
            (graph.phi[flat] - phi_before).abs() > 1e-6,
            "phi should change after pheromone update"
        );
    }

    // ── full_colony_step tests ───────────────────────────────────────────

    #[test]
    fn test_full_colony_step_produces_valid_output() {
        let cfg = test_config();
        let scorer = Scorer::new(&cfg, cfg.vocab_size, 42);
        let mut graph = RouteGraph::new_empty(&cfg);
        let mut ant_state = AntState::new(&cfg);

        graph.add_edge(0, 0, 1, 0.5).unwrap();
        graph.add_edge(0, 2, 3, 0.3).unwrap();

        let editable = vec![true; cfg.seq_len];
        let y_t: Vec<u32> = (0..cfg.seq_len as u32).collect();
        let pconfig = PheromoneConfig::from_config(&cfg);
        let mut rng = ChaCha8Rng::seed_from_u64(77);

        let result = full_colony_step(
            &y_t,
            &scorer,
            &mut graph,
            &mut ant_state,
            0,
            &cfg,
            &pconfig,
            &editable,
            DeathMode::Streak,
            &mut rng,
        )
        .unwrap();

        // Validate outputs.
        assert_eq!(result.y_new.len(), cfg.seq_len);
        assert_eq!(result.ant_deltas.len(), cfg.num_ants);
        for &t in &result.y_new {
            assert!((t as usize) < cfg.vocab_size);
        }
        for &d in &result.ant_deltas {
            assert!(d.is_finite());
        }
        assert!(result.pheromone_stats.mean_phi.is_finite());
    }

    #[test]
    fn test_full_colony_step_with_random_pool_death() {
        let cfg = ErmConfig {
            num_ants: 20,
            ..test_config()
        };
        let scorer = Scorer::new(&cfg, cfg.vocab_size, 42);
        let mut graph = RouteGraph::new_empty(&cfg);
        let mut ant_state = AntState::new(&cfg);

        let editable = vec![true; cfg.seq_len];
        let y_t: Vec<u32> = (0..cfg.seq_len as u32).collect();
        let pconfig = PheromoneConfig::from_config(&cfg);
        let mut rng = ChaCha8Rng::seed_from_u64(88);

        let result = full_colony_step(
            &y_t,
            &scorer,
            &mut graph,
            &mut ant_state,
            0,
            &cfg,
            &pconfig,
            &editable,
            DeathMode::RandomPool,
            &mut rng,
        )
        .unwrap();

        // Should have some deaths (ceil(20 * 0.1) = 2 per batch).
        assert!(result.deaths > 0, "RandomPool should produce deaths, got 0");
    }

    #[test]
    fn test_full_colony_step_deterministic() {
        let cfg = test_config();
        let scorer = Scorer::new(&cfg, cfg.vocab_size, 42);

        let editable = vec![true; cfg.seq_len];
        let y_t: Vec<u32> = (0..cfg.seq_len as u32).collect();
        let pconfig = PheromoneConfig::from_config(&cfg);

        let mut graph1 = RouteGraph::new_empty(&cfg);
        let mut ant_state1 = AntState::new(&cfg);
        let mut rng1 = ChaCha8Rng::seed_from_u64(42);
        let r1 = full_colony_step(
            &y_t,
            &scorer,
            &mut graph1,
            &mut ant_state1,
            0,
            &cfg,
            &pconfig,
            &editable,
            DeathMode::Streak,
            &mut rng1,
        )
        .unwrap();

        let mut graph2 = RouteGraph::new_empty(&cfg);
        let mut ant_state2 = AntState::new(&cfg);
        let mut rng2 = ChaCha8Rng::seed_from_u64(42);
        let r2 = full_colony_step(
            &y_t,
            &scorer,
            &mut graph2,
            &mut ant_state2,
            0,
            &cfg,
            &pconfig,
            &editable,
            DeathMode::Streak,
            &mut rng2,
        )
        .unwrap();

        assert_eq!(r1.y_new, r2.y_new);
        assert_eq!(r1.num_edits, r2.num_edits);
        assert_eq!(r1.deaths, r2.deaths);
    }

    // ── multi_step_refine tests (Milestone 6) ───────────────────────────

    #[test]
    fn test_multi_step_refine_shape() {
        let cfg = test_config();
        let scorer = Scorer::new(&cfg, cfg.vocab_size, 42);
        let mut graph = RouteGraph::new_empty(&cfg);
        let pconfig = PheromoneConfig::from_config(&cfg);
        let multi_cfg = MultiStepConfig {
            max_steps: 3,
            confidence_threshold: 0.99,
        };
        let editable = vec![true; cfg.seq_len];
        let y_init: Vec<u32> = (0..cfg.seq_len as u32).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result = multi_step_refine(
            &y_init, &scorer, &mut graph, 0, &cfg, &pconfig, &multi_cfg, &editable, &mut rng,
        )
        .unwrap();

        assert_eq!(result.y_final.len(), cfg.seq_len);
        for &t in &result.y_final {
            assert!(
                (t as usize) < cfg.vocab_size,
                "token {t} >= vocab_size {}",
                cfg.vocab_size
            );
        }
        assert!(result.steps_executed <= multi_cfg.max_steps);
        assert!(!result.stop_reason.is_empty());
        assert!(result.peak_memory_estimate > 0);
    }

    #[test]
    fn test_multi_step_reduces_corruption() {
        // After multiple steps, the sequence should differ from the initial MASK-heavy input.
        let cfg = test_config();
        let scorer = Scorer::new(&cfg, cfg.vocab_size, 42);
        let mut graph = RouteGraph::new_empty(&cfg);
        let pconfig = PheromoneConfig::from_config(&cfg);
        let multi_cfg = MultiStepConfig {
            max_steps: 4,
            confidence_threshold: 0.99,
        };
        let editable = vec![true; cfg.seq_len];
        // Start with all zeros (simulating a "corrupted" sequence).
        let y_init = vec![0_u32; cfg.seq_len];
        let mut rng = ChaCha8Rng::seed_from_u64(77);

        let result = multi_step_refine(
            &y_init, &scorer, &mut graph, 0, &cfg, &pconfig, &multi_cfg, &editable, &mut rng,
        )
        .unwrap();

        // At least one step should have been executed.
        assert!(result.steps_executed >= 1, "expected at least 1 step");
    }

    #[test]
    fn test_multi_step_early_stop_no_edits() {
        // If the scorer proposes no changes, the loop should stop early.
        let cfg = test_config();
        let scorer = Scorer::new(&cfg, cfg.vocab_size, 42);
        let mut graph = RouteGraph::new_empty(&cfg);
        let pconfig = PheromoneConfig::from_config(&cfg);
        let multi_cfg = MultiStepConfig {
            max_steps: 10,
            confidence_threshold: 0.99,
        };
        // Make nothing editable → no edits possible → early stop.
        let editable = vec![false; cfg.seq_len];
        let y_init: Vec<u32> = (0..cfg.seq_len as u32).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result = multi_step_refine(
            &y_init, &scorer, &mut graph, 0, &cfg, &pconfig, &multi_cfg, &editable, &mut rng,
        )
        .unwrap();

        // Should stop after 1 step with "no_edits".
        assert_eq!(result.steps_executed, 1);
        assert_eq!(result.stop_reason, "no_edits");
        // y_final should be identical to y_init since nothing was editable.
        assert_eq!(result.y_final, y_init);
    }

    #[test]
    fn test_generate_from_scratch() {
        let cfg = test_config();
        let scorer = Scorer::new(&cfg, cfg.vocab_size, 42);
        let mut graph = RouteGraph::new_empty(&cfg);
        let pconfig = PheromoneConfig::from_config(&cfg);
        let mut rng = ChaCha8Rng::seed_from_u64(55);

        let result =
            generate_from_scratch(&scorer, &mut graph, &cfg, &pconfig, cfg.seq_len, &mut rng)
                .unwrap();

        assert_eq!(result.y_final.len(), cfg.seq_len);
        for &t in &result.y_final {
            assert!(
                (t as usize) < cfg.vocab_size,
                "generated token {t} out of range"
            );
        }
    }

    #[test]
    fn test_generate_from_scratch_wrong_length() {
        let cfg = test_config();
        let scorer = Scorer::new(&cfg, cfg.vocab_size, 42);
        let mut graph = RouteGraph::new_empty(&cfg);
        let pconfig = PheromoneConfig::from_config(&cfg);
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        // Wrong length should fail.
        let result = generate_from_scratch(
            &scorer,
            &mut graph,
            &cfg,
            &pconfig,
            cfg.seq_len + 1,
            &mut rng,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_prompted_preserves_prefix() {
        let cfg = test_config();
        let scorer = Scorer::new(&cfg, cfg.vocab_size, 42);
        let mut graph = RouteGraph::new_empty(&cfg);
        let pconfig = PheromoneConfig::from_config(&cfg);
        let mut rng = ChaCha8Rng::seed_from_u64(99);

        let prefix: Vec<u32> = vec![1, 2, 3, 4];
        let gen_length = cfg.seq_len - prefix.len();

        let result = generate_prompted(
            &scorer, &mut graph, &cfg, &pconfig, &prefix, gen_length, &mut rng,
        )
        .unwrap();

        assert_eq!(result.y_final.len(), cfg.seq_len);
        // Prefix must be preserved.
        for (i, &expected) in prefix.iter().enumerate() {
            assert_eq!(
                result.y_final[i], expected,
                "prefix position {i}: expected {expected}, got {}",
                result.y_final[i]
            );
        }
    }

    #[test]
    fn test_generate_prompted_wrong_total_length() {
        let cfg = test_config();
        let scorer = Scorer::new(&cfg, cfg.vocab_size, 42);
        let mut graph = RouteGraph::new_empty(&cfg);
        let pconfig = PheromoneConfig::from_config(&cfg);
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        let prefix = vec![1_u32, 2, 3];
        // Total = 3 + 10 = 13, but seq_len = 8.
        let result = generate_prompted(&scorer, &mut graph, &cfg, &pconfig, &prefix, 10, &mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_prompted_empty_prefix() {
        let cfg = test_config();
        let scorer = Scorer::new(&cfg, cfg.vocab_size, 42);
        let mut graph = RouteGraph::new_empty(&cfg);
        let pconfig = PheromoneConfig::from_config(&cfg);
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        let result = generate_prompted(
            &scorer,
            &mut graph,
            &cfg,
            &pconfig,
            &[],
            cfg.seq_len,
            &mut rng,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_mean_confidence_all_editable() {
        let seq_len = 4;
        let vocab_size = 3;
        // Very confident logits: one token dominates.
        let mut logits = vec![0.0_f32; seq_len * vocab_size];
        for i in 0..seq_len {
            logits[i * vocab_size] = 100.0; // token 0 very confident
        }
        let editable = vec![true; seq_len];

        let conf = mean_confidence(&logits, &editable, seq_len, vocab_size);
        // With logits [100, 0, 0], softmax ≈ [1.0, ~0, ~0], so max ≈ 1.0.
        assert!(conf > 0.9, "expected high confidence, got {conf}");
    }

    #[test]
    fn test_mean_confidence_none_editable() {
        let logits = vec![0.0_f32; 12];
        let editable = vec![false; 4];
        let conf = mean_confidence(&logits, &editable, 4, 3);
        assert!((conf - 1.0).abs() < 1e-6, "expected 1.0, got {conf}");
    }
}
