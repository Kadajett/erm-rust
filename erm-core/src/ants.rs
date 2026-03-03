//! Ant colony data structures for the Emergent Route Model.
//!
//! ERM uses two ant populations per refinement step:
//!
//! - **Leaders** (10%): explore high-uncertainty, low-support positions.
//!   Higher temperature sampling. Can propose new edges.
//! - **Followers** (90%): exploit strong routes and high-confidence positions.
//!   Conservative sampling from top-k candidates.
//!
//! # Tensor shapes (default config)
//!
//! | Tensor      | Shape               | Size     |
//! |------------|---------------------|----------|
//! | `ant_pos`  | `[B, A, pmax]` i32  | 65.5 KB  |
//! | `ant_tok`  | `[B, A, pmax]` i32  | 65.5 KB  |
//! | `ant_gain` | `[B, A, pmax]` f32  | 131 KB   |
//! | `streak`   | `[B, A]` i32        | 8 KB     |
//! | `ant_type` | `[B, A]` u8         | 2 KB     |
//!
//! Total: < 1 MB — negligible.

use rand::Rng;

use serde::{Deserialize, Serialize};

use crate::config::ErmConfig;
use crate::error::{ErmError, ErmResult};
use crate::graph::{RouteGraph, EMPTY_SLOT};
use crate::merge::SimpleEditProposal;
use crate::types::{AntIdx, BatchIdx, PosIdx, TokenId};

/// Death/respawn mode for ant lifecycle management.
///
/// Controls how underperforming ants are identified and replaced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeathMode {
    /// Streak-based: ant dies after `K` consecutive no-improvement steps.
    Streak,
    /// Random pool: replace a fixed fraction (10%) of ants each step.
    RandomPool,
}

/// Type of an ant in the colony.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum AntType {
    /// Follower: exploits existing routes.
    Follower = 0,
    /// Leader: explores new routes.
    Leader = 1,
}

/// A single ant's edit proposal for one refinement step.
///
/// Each ant proposes edits at up to `pmax` positions within a single
/// batch element.
#[derive(Debug, Clone)]
pub struct EditProposal {
    /// Batch index this proposal belongs to.
    pub batch_idx: BatchIdx,
    /// Ant index within the colony.
    pub ant_idx: AntIdx,
    /// Proposed (position, token, predicted_gain) triples.
    /// Length ≤ `pmax`.
    pub edits: Vec<(PosIdx, TokenId, f32)>,
}

/// Proposal for a new edge in the route graph (leader-only).
#[derive(Debug, Clone)]
pub struct EdgeProposal {
    /// Batch index.
    pub batch_idx: BatchIdx,
    /// Source position (evidence comes from here).
    pub src: PosIdx,
    /// Destination position (prediction target).
    pub dst: PosIdx,
    /// Edge type (0 = local, 1 = long-range, 2 = concept).
    pub etype: u8,
}

/// Per-ant lifecycle state (streak counter and type).
///
/// Stored as flat arrays with shape `[B, A]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntState {
    /// Batch size.
    pub batch_size: usize,
    /// Total ants per sequence.
    pub num_ants: usize,
    /// Consecutive no-improvement steps per ant. Shape: `[B * A]`.
    pub streak: Vec<i32>,
    /// Ant type per ant. Shape: `[B * A]`.
    pub ant_type: Vec<AntType>,
}

impl AntState {
    /// Initialize ant state from config.
    ///
    /// Leaders are assigned to the first `num_leaders` indices in each batch,
    /// followers to the rest.
    #[must_use]
    pub fn new(config: &ErmConfig) -> Self {
        let b = config.batch_size;
        let a = config.num_ants;
        let n_leaders = config.num_leaders();
        let total = b * a;

        let mut ant_type = Vec::with_capacity(total);
        for _batch in 0..b {
            for ant in 0..a {
                if ant < n_leaders {
                    ant_type.push(AntType::Leader);
                } else {
                    ant_type.push(AntType::Follower);
                }
            }
        }

        Self {
            batch_size: b,
            num_ants: a,
            streak: vec![0; total],
            ant_type,
        }
    }

    /// Flat index into `[B, A]` arrays.
    #[inline]
    #[must_use]
    pub fn idx(&self, b: BatchIdx, k: AntIdx) -> usize {
        b * self.num_ants + k
    }

    /// Record an improvement (or lack thereof) for ant `(b, k)`.
    ///
    /// - `delta > epsilon`: streak resets to 0 (improvement).
    /// - Otherwise: streak increments.
    ///
    /// Returns `true` if the ant should "die" (streak >= `death_streak`).
    ///
    /// Use `config.effective_death_streak(step)` for warmstart-aware death threshold.
    pub fn record_delta(
        &mut self,
        b: BatchIdx,
        k: AntIdx,
        delta: f32,
        death_streak: usize,
    ) -> bool {
        let flat = self.idx(b, k);
        let epsilon = 1e-6_f32;

        if delta > epsilon {
            self.streak[flat] = 0;
            false
        } else {
            self.streak[flat] += 1;
            self.streak[flat] >= death_streak as i32
        }
    }

    /// Kill and respawn an ant: reset its streak and reassign type to maintain
    /// the leader/follower ratio.
    pub fn respawn(&mut self, b: BatchIdx, k: AntIdx, config: &ErmConfig) {
        let flat = self.idx(b, k);
        self.streak[flat] = 0;

        // Maintain 10/90 leader/follower ratio.
        if k < config.num_leaders() {
            self.ant_type[flat] = AntType::Leader;
        } else {
            self.ant_type[flat] = AntType::Follower;
        }
    }

    /// Count leaders for batch element `b`.
    #[must_use]
    pub fn count_leaders(&self, b: BatchIdx) -> usize {
        (0..self.num_ants)
            .filter(|&k| self.ant_type[self.idx(b, k)] == AntType::Leader)
            .count()
    }

    /// Count followers for batch element `b`.
    #[must_use]
    pub fn count_followers(&self, b: BatchIdx) -> usize {
        self.num_ants - self.count_leaders(b)
    }
}

/// Result of the conflict-free edit merge across all ants.
///
/// For each position, the edit with the highest predicted gain wins.
#[derive(Debug, Clone)]
pub struct MergedEdits {
    /// Best token per position. Shape: `[B * L]`.
    /// Sentinel value `-1` means "no edit proposed".
    pub best_tok: Vec<TokenId>,
    /// Best gain per position. Shape: `[B * L]`.
    pub best_gain: Vec<f32>,
    /// Number of positions actually edited.
    pub num_edited: usize,
}

/// Merge edit proposals from all ants into a conflict-free edit set.
///
/// For each position, the proposal with the highest `predicted_gain` wins.
/// At most `max_edits` positions are edited (top-M by gain).
/// Clamped (non-editable) positions are skipped.
///
/// # Arguments
///
/// - `proposals`: all ant proposals for one refinement step.
/// - `editable`: flat `[B * L]` boolean mask (`true` = editable).
/// - `config`: provides `max_edits()` and dimension info.
///
/// # Shape
///
/// Output `best_tok` and `best_gain` are flat `[B * L]`.
pub fn merge_edits(
    proposals: &[EditProposal],
    editable: &[bool],
    config: &ErmConfig,
) -> MergedEdits {
    let b = config.batch_size;
    let l = config.seq_len;
    let total = b * l;
    let max_edits = config.max_edits();

    let mut best_tok = vec![-1i32; total];
    let mut best_gain = vec![f32::NEG_INFINITY; total];

    // Phase 1: collect best per position.
    for proposal in proposals {
        let batch = proposal.batch_idx;
        for &(pos, tok, gain) in &proposal.edits {
            let flat = batch * l + pos;
            if flat < total && editable[flat] && gain > best_gain[flat] {
                best_gain[flat] = gain;
                best_tok[flat] = tok;
            }
        }
    }

    // Phase 2: enforce max edits per batch element (keep top-M by gain).
    for batch in 0..b {
        let start = batch * l;
        let end = start + l;

        // Collect edited positions with their gains.
        let mut edits: Vec<(usize, f32)> = (start..end)
            .filter(|&flat| best_tok[flat] != -1)
            .map(|flat| (flat, best_gain[flat]))
            .collect();

        if edits.len() > max_edits {
            // Sort descending by gain, keep top-M.
            edits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for &(flat, _) in &edits[max_edits..] {
                best_tok[flat] = -1;
                best_gain[flat] = f32::NEG_INFINITY;
            }
        }
    }

    let num_edited = best_tok.iter().filter(|&&t| t != -1).count();

    MergedEdits {
        best_tok,
        best_gain,
        num_edited,
    }
}

/// Apply merged edits to corrupted tokens, producing the next-step tokens.
///
/// - Positions where `best_tok[i] != -1` are updated.
/// - Others keep their value from `y_t`.
///
/// # Shape
///
/// Both input and output are flat `[B * L]`.
#[must_use]
pub fn apply_edits(y_t: &[TokenId], merged: &MergedEdits) -> Vec<TokenId> {
    y_t.iter()
        .zip(merged.best_tok.iter())
        .map(|(&current, &edit)| if edit != -1 { edit } else { current })
        .collect()
}

/// Configuration for follower ant sampling.
#[derive(Debug, Clone)]
pub struct FollowerConfig {
    /// Epsilon floor for route strength (prevents zero-strength positions from
    /// being permanently ignored).
    pub epsilon: f32,
    /// Sampling temperature for token selection (< 1.0 = more conservative).
    pub temperature: f32,
    /// Maximum positions each ant proposes edits at.
    pub pmax: usize,
    /// Top-k candidates per position.
    pub topk: usize,
}

impl Default for FollowerConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-4,
            temperature: 0.7,
            pmax: 8,
            topk: 8,
        }
    }
}

impl FollowerConfig {
    /// Build from the global ErmConfig.
    #[must_use]
    pub fn from_config(config: &ErmConfig) -> Self {
        Self {
            epsilon: 1e-4,
            temperature: 0.7,
            pmax: config.pmax,
            topk: config.topk,
        }
    }

    /// Return a copy with the given temperature.
    #[must_use]
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }
}

/// Compute follower temperature that decays with training progress.
///
/// `T_follower(step) = max(0.3, 1.0 - step / total_steps)`
///
/// Early training: high temperature (explore). Late: low temperature (exploit).
#[must_use]
pub fn follower_temperature_schedule(step: usize, total_steps: usize) -> f32 {
    if total_steps == 0 {
        return 0.7;
    }
    let progress = step as f32 / total_steps as f32;
    (1.0 - progress).max(0.3)
}

/// Compute leader temperature that scales with uncertainty.
///
/// `T_leader(step, mean_uncertainty) = max(0.5, 2.0 * mean_uncertainty)`
///
/// Leaders always explore, but more aggressively at uncertain positions.
#[must_use]
pub fn leader_temperature_schedule(mean_uncertainty: f32) -> f32 {
    (2.0 * mean_uncertainty).max(0.5)
}

/// Ant colony that produces edit proposals from follower ants.
///
/// Followers exploit existing route graph pheromone trails and scorer confidence
/// to select high-value edit positions, then sample replacement tokens from the
/// top-k logit candidates with low temperature.
pub struct AntColony;

impl AntColony {
    /// Sample follower edit proposals from scorer logits and route graph.
    ///
    /// For each follower ant, scores every editable position by:
    ///
    /// ```text
    /// score_i = conf_i * (route_strength_i + epsilon)
    /// ```
    ///
    /// where:
    /// - `conf_i = max(softmax(logits[i]))` (prediction confidence)
    /// - `route_strength_i = Σ_e φ[i,e]` for valid edges to position `i`
    /// - `epsilon` is the epsilon floor per AGENTS.md
    ///
    /// Then samples `pmax` positions per ant weighted by score, and for each
    /// position samples a token from the top-k logits with temperature scaling.
    ///
    /// # Arguments
    ///
    /// - `logits`: flat `[L * V]` logit values for one sequence.
    /// - `graph`: route graph providing pheromone strengths.
    /// - `batch_idx`: which batch element to look up in the graph.
    /// - `follower_config`: sampling parameters (epsilon, temperature, pmax, topk).
    /// - `editable`: boolean mask `[L]`, `true` = position may be edited.
    /// - `num_followers`: number of follower ants to sample proposals for.
    /// - `first_follower_id`: ant id offset for the first follower.
    /// - `seq_len`: sequence length `L`.
    /// - `vocab_size`: vocabulary size `V`.
    /// - `rng`: random number generator.
    ///
    /// # Returns
    ///
    /// A vector of [`SimpleEditProposal`]s (one per position per ant).
    ///
    /// # Errors
    ///
    /// Returns [`ErmError::ShapeMismatch`] if input dimensions don't match.
    ///
    /// # Shape reference
    ///
    /// | Tensor | Shape |
    /// |---|---|
    /// | `logits` (input) | `[L, V]` flat |
    /// | `editable` (input) | `[L]` |
    /// | output | `Vec<SimpleEditProposal>`, up to `num_followers * pmax` |
    #[allow(clippy::too_many_arguments)]
    pub fn sample_follower_proposals<R: Rng>(
        logits: &[f32],
        graph: &RouteGraph,
        batch_idx: usize,
        follower_config: &FollowerConfig,
        editable: &[bool],
        num_followers: usize,
        first_follower_id: usize,
        seq_len: usize,
        vocab_size: usize,
        rng: &mut R,
    ) -> ErmResult<Vec<SimpleEditProposal>> {
        let expected_logits = seq_len * vocab_size;
        if logits.len() != expected_logits {
            return Err(ErmError::ShapeMismatch {
                expected: format!("[L={seq_len}, V={vocab_size}] = {expected_logits}"),
                got: format!("{}", logits.len()),
            });
        }
        if editable.len() != seq_len {
            return Err(ErmError::ShapeMismatch {
                expected: format!("editable length = {seq_len}"),
                got: format!("{}", editable.len()),
            });
        }

        let epsilon = follower_config.epsilon;
        let temperature = follower_config.temperature;
        let pmax = follower_config.pmax;
        let topk = follower_config.topk.min(vocab_size);

        // Step 1: Compute per-position scores.
        // conf_i = max(softmax(logits[i]))
        // route_strength_i = sum of phi for valid neighbors of i
        // score_i = conf_i * (route_strength_i + epsilon) if editable[i]
        let mut scores = Vec::with_capacity(seq_len);
        let mut confs = Vec::with_capacity(seq_len);

        for (i, &is_editable) in editable.iter().enumerate() {
            if !is_editable {
                scores.push(0.0_f32);
                confs.push(0.0_f32);
                continue;
            }

            // Compute softmax max for numerical stability, then conf.
            let row_start = i * vocab_size;
            let row = &logits[row_start..row_start + vocab_size];

            let max_logit = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut exp_sum = 0.0_f32;
            for &v in row {
                exp_sum += (v - max_logit).exp();
            }

            // conf_i = max probability in softmax
            let mut max_prob = 0.0_f32;
            for &v in row {
                let p = (v - max_logit).exp() / exp_sum;
                if p > max_prob {
                    max_prob = p;
                }
            }

            confs.push(max_prob);

            // route_strength_i = sum of phi for valid edges at this position.
            let mut route_strength = 0.0_f32;
            if batch_idx < graph.batch_size && i < graph.seq_len {
                for e in 0..graph.emax {
                    let flat = graph.idx(batch_idx, i, e);
                    if graph.nbr_idx[flat] != EMPTY_SLOT {
                        route_strength += graph.phi[flat];
                    }
                }
            }

            let score = max_prob * (route_strength + epsilon);
            scores.push(score);
        }

        // Step 2: Build CDF for weighted sampling over editable positions.
        let editable_positions: Vec<usize> = (0..seq_len).filter(|&i| editable[i]).collect();
        if editable_positions.is_empty() {
            return Ok(Vec::new());
        }

        let score_sum: f32 = editable_positions.iter().map(|&i| scores[i]).sum();

        // Step 3: For each follower ant, sample pmax positions and propose tokens.
        let mut proposals = Vec::with_capacity(num_followers * pmax);

        for ant_offset in 0..num_followers {
            let ant_id = first_follower_id + ant_offset;

            // Sample pmax positions (with replacement for simplicity).
            let mut sampled_positions = Vec::with_capacity(pmax);

            for _ in 0..pmax {
                let pos = if score_sum > 0.0 {
                    // Weighted sampling.
                    let r: f32 = rng.gen::<f32>() * score_sum;
                    let mut cumulative = 0.0_f32;
                    let mut chosen = editable_positions[editable_positions.len() - 1];
                    for &p in &editable_positions {
                        cumulative += scores[p];
                        if cumulative >= r {
                            chosen = p;
                            break;
                        }
                    }
                    chosen
                } else {
                    // Uniform fallback.
                    let idx = rng.gen_range(0..editable_positions.len());
                    editable_positions[idx]
                };
                sampled_positions.push(pos);
            }

            // Deduplicate positions (keep first occurrence).
            sampled_positions.sort_unstable();
            sampled_positions.dedup();

            // For each sampled position, pick a token from top-k with temperature.
            for pos in sampled_positions {
                let row_start = pos * vocab_size;
                let row = &logits[row_start..row_start + vocab_size];

                // Find top-k indices and scores.
                let mut indexed: Vec<(u32, f32)> = row
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| (j as u32, v))
                    .collect();

                if topk < vocab_size {
                    indexed.select_nth_unstable_by(topk - 1, |a, b| {
                        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
                let top_slice = &mut indexed[..topk];

                // Apply temperature and sample from softmax over top-k.
                let max_val = top_slice
                    .iter()
                    .map(|(_, v)| *v)
                    .fold(f32::NEG_INFINITY, f32::max);

                let mut exp_vals: Vec<f32> = top_slice
                    .iter()
                    .map(|(_, v)| ((v - max_val) / temperature).exp())
                    .collect();

                let exp_sum: f32 = exp_vals.iter().sum();
                if exp_sum > 0.0 {
                    for v in &mut exp_vals {
                        *v /= exp_sum;
                    }
                }

                // Sample token from this distribution.
                let r: f32 = rng.gen();
                let mut cumulative = 0.0_f32;
                let mut chosen_idx = 0;
                for (j, &p) in exp_vals.iter().enumerate() {
                    cumulative += p;
                    if cumulative >= r {
                        chosen_idx = j;
                        break;
                    }
                }

                let token = top_slice[chosen_idx].0;
                let predicted_gain = confs[pos] * scores[pos];

                proposals.push(SimpleEditProposal {
                    position: pos,
                    token,
                    predicted_gain,
                    ant_id,
                });
            }
        }

        Ok(proposals)
    }
}

/// Configuration for leader ant sampling.
#[derive(Debug, Clone)]
pub struct LeaderConfig {
    /// Epsilon floor for route strength denominator.
    pub epsilon: f32,
    /// Sampling temperature for token selection (> 1.0 = more exploratory).
    pub temperature: f32,
    /// Maximum positions each leader proposes edits at.
    pub pmax: usize,
    /// Top-k candidates per position.
    pub topk: usize,
}

impl Default for LeaderConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-4,
            temperature: 1.5,
            pmax: 8,
            topk: 8,
        }
    }
}

impl LeaderConfig {
    /// Build from the global ErmConfig.
    #[must_use]
    pub fn from_config(config: &ErmConfig) -> Self {
        Self {
            epsilon: 1e-4,
            temperature: 1.5,
            pmax: config.pmax,
            topk: config.topk,
        }
    }

    /// Return a copy with the given temperature.
    #[must_use]
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }
}

impl AntColony {
    /// Sample leader edit proposals targeting high-uncertainty, low-pheromone positions.
    ///
    /// Leaders explore under-served regions of the sequence by scoring positions as:
    ///
    /// ```text
    /// score_i = u_i / (route_strength_i + ε)
    /// ```
    ///
    /// where `u_i` is the uncertainty signal and `route_strength_i` is the sum
    /// of pheromone at position `i`. This targets positions with high uncertainty
    /// and low pheromone support.
    ///
    /// Leaders also propose new edges (`EdgeProposal`) connecting high-uncertainty
    /// positions to other positions, enabling the graph to grow.
    ///
    /// # Arguments
    ///
    /// - `logits`: flat `[L * V]` logit values for one sequence.
    /// - `uncertainty`: flat `[L]` uncertainty values from scorer.
    /// - `graph`: route graph providing pheromone strengths.
    /// - `batch_idx`: which batch element in the graph to use.
    /// - `leader_config`: sampling parameters.
    /// - `editable`: boolean mask `[L]`, `true` = position can be edited.
    /// - `num_leaders`: number of leader ants.
    /// - `first_leader_id`: ant id offset for the first leader (typically 0).
    /// - `seq_len`: sequence length `L`.
    /// - `vocab_size`: vocabulary size `V`.
    /// - `rng`: random number generator.
    ///
    /// # Returns
    ///
    /// `(proposals, edge_proposals)` — edit proposals and new edge proposals.
    ///
    /// # Errors
    ///
    /// Returns [`ErmError::ShapeMismatch`] if input dimensions don't match.
    ///
    /// # Shape reference
    ///
    /// | Tensor | Shape |
    /// |---|---|
    /// | `logits` (input) | `[L, V]` flat |
    /// | `uncertainty` (input) | `[L]` |
    /// | `editable` (input) | `[L]` |
    /// | edit output | `Vec<SimpleEditProposal>`, up to `num_leaders * pmax` |
    /// | edge output | `Vec<EdgeProposal>`, up to `num_leaders` |
    #[allow(clippy::too_many_arguments)]
    pub fn sample_leader_proposals<R: Rng>(
        logits: &[f32],
        uncertainty: &[f32],
        graph: &RouteGraph,
        batch_idx: usize,
        leader_config: &LeaderConfig,
        editable: &[bool],
        num_leaders: usize,
        first_leader_id: usize,
        seq_len: usize,
        vocab_size: usize,
        rng: &mut R,
    ) -> ErmResult<(Vec<SimpleEditProposal>, Vec<EdgeProposal>)> {
        let expected_logits = seq_len * vocab_size;
        if logits.len() != expected_logits {
            return Err(ErmError::ShapeMismatch {
                expected: format!("[L={seq_len}, V={vocab_size}] = {expected_logits}"),
                got: format!("{}", logits.len()),
            });
        }
        if uncertainty.len() != seq_len {
            return Err(ErmError::ShapeMismatch {
                expected: format!("uncertainty length = {seq_len}"),
                got: format!("{}", uncertainty.len()),
            });
        }
        if editable.len() != seq_len {
            return Err(ErmError::ShapeMismatch {
                expected: format!("editable length = {seq_len}"),
                got: format!("{}", editable.len()),
            });
        }

        let epsilon = leader_config.epsilon;
        let temperature = leader_config.temperature;
        let pmax = leader_config.pmax;
        let topk = leader_config.topk.min(vocab_size);

        // Step 1: Compute per-position leader scores.
        // score_i = u_i / (route_strength_i + ε)
        let mut scores = Vec::with_capacity(seq_len);

        for (i, &is_editable) in editable.iter().enumerate() {
            if !is_editable {
                scores.push(0.0_f32);
                continue;
            }

            let u_i = uncertainty[i];

            // route_strength_i = sum of phi for valid edges at this position.
            let mut route_strength = 0.0_f32;
            if batch_idx < graph.batch_size && i < graph.seq_len {
                for e in 0..graph.emax {
                    let flat = graph.idx(batch_idx, i, e);
                    if graph.nbr_idx[flat] != EMPTY_SLOT {
                        route_strength += graph.phi[flat];
                    }
                }
            }

            let score = u_i / (route_strength + epsilon);
            scores.push(score);
        }

        // Step 2: Build CDF for weighted sampling.
        let editable_positions: Vec<usize> = (0..seq_len).filter(|&i| editable[i]).collect();
        if editable_positions.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        let score_sum: f32 = editable_positions.iter().map(|&i| scores[i]).sum();

        // Step 3: For each leader, sample positions and propose tokens + edges.
        let mut proposals = Vec::with_capacity(num_leaders * pmax);
        let mut edge_proposals = Vec::with_capacity(num_leaders);

        for leader_offset in 0..num_leaders {
            let ant_id = first_leader_id + leader_offset;

            // Sample pmax positions weighted by leader score.
            let mut sampled_positions = Vec::with_capacity(pmax);

            for _ in 0..pmax {
                let pos = if score_sum > 0.0 {
                    let r: f32 = rng.gen::<f32>() * score_sum;
                    let mut cumulative = 0.0_f32;
                    let mut chosen = editable_positions[editable_positions.len() - 1];
                    for &p in &editable_positions {
                        cumulative += scores[p];
                        if cumulative >= r {
                            chosen = p;
                            break;
                        }
                    }
                    chosen
                } else {
                    let idx = rng.gen_range(0..editable_positions.len());
                    editable_positions[idx]
                };
                sampled_positions.push(pos);
            }

            sampled_positions.sort_unstable();
            sampled_positions.dedup();

            // Propose edge from the highest-scored sampled position to a random other position.
            if let Some(&best_pos) = sampled_positions.first() {
                // Pick a random destination that's different from best_pos.
                let dst = loop {
                    let candidate = rng.gen_range(0..seq_len);
                    if candidate != best_pos {
                        break candidate;
                    }
                    if seq_len <= 1 {
                        break 0;
                    }
                };
                edge_proposals.push(EdgeProposal {
                    batch_idx,
                    src: best_pos,
                    dst,
                    etype: 1, // long-range
                });
            }

            // For each sampled position, pick a token from top-k with high temperature.
            for pos in sampled_positions {
                let row_start = pos * vocab_size;
                let row = &logits[row_start..row_start + vocab_size];

                let mut indexed: Vec<(u32, f32)> = row
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| (j as u32, v))
                    .collect();

                if topk < vocab_size {
                    indexed.select_nth_unstable_by(topk - 1, |a, b| {
                        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
                let top_slice = &mut indexed[..topk];

                let max_val = top_slice
                    .iter()
                    .map(|(_, v)| *v)
                    .fold(f32::NEG_INFINITY, f32::max);

                let mut exp_vals: Vec<f32> = top_slice
                    .iter()
                    .map(|(_, v)| ((v - max_val) / temperature).exp())
                    .collect();

                let exp_sum: f32 = exp_vals.iter().sum();
                if exp_sum > 0.0 {
                    for v in &mut exp_vals {
                        *v /= exp_sum;
                    }
                }

                let r: f32 = rng.gen();
                let mut cumulative = 0.0_f32;
                let mut chosen_idx = 0;
                for (j, &p) in exp_vals.iter().enumerate() {
                    cumulative += p;
                    if cumulative >= r {
                        chosen_idx = j;
                        break;
                    }
                }

                let token = top_slice[chosen_idx].0;
                let predicted_gain = uncertainty[pos] * scores[pos];

                proposals.push(SimpleEditProposal {
                    position: pos,
                    token,
                    predicted_gain,
                    ant_id,
                });
            }
        }

        Ok((proposals, edge_proposals))
    }
}

/// Apply death/respawn logic to the ant colony.
///
/// # Death Modes
///
/// - `DeathMode::Streak`: Ants with consecutive no-improvement streaks >= `K`
///   are killed and respawned. During warm-start, `K` is multiplied by
///   `warmstart_death_mult` to reduce churn.
/// - `DeathMode::RandomPool`: A fixed fraction (10%) of ants are randomly
///   replaced each step, regardless of performance.
///
/// # Arguments
///
/// - `ant_state`: mutable ant lifecycle state.
/// - `ant_deltas`: per-ant improvement deltas from the merge step. Shape: `[num_ants]`.
/// - `config`: ERM hyperparameters.
/// - `mode`: which death mode to use.
/// - `step`: current training step (used for warmstart-aware death streak).
/// - `rng`: random number generator.
///
/// # Returns
///
/// Number of ants that died and were respawned.
pub fn apply_death_respawn<R: Rng>(
    ant_state: &mut AntState,
    ant_deltas: &[f32],
    config: &ErmConfig,
    mode: DeathMode,
    step: usize,
    rng: &mut R,
) -> usize {
    let mut deaths = 0;
    let effective_streak = config.effective_death_streak(step);

    match mode {
        DeathMode::Streak => {
            for b in 0..ant_state.batch_size {
                for k in 0..ant_state.num_ants {
                    let delta = if k < ant_deltas.len() {
                        ant_deltas[k]
                    } else {
                        0.0
                    };
                    let should_die = ant_state.record_delta(b, k, delta, effective_streak);
                    if should_die {
                        ant_state.respawn(b, k, config);
                        deaths += 1;
                    }
                }
            }
        }
        DeathMode::RandomPool => {
            let pool_size = (ant_state.num_ants as f32 * 0.1).ceil() as usize;
            for b in 0..ant_state.batch_size {
                // Randomly select pool_size ants to replace.
                let mut replaced = 0;
                while replaced < pool_size {
                    let k = rng.gen_range(0..ant_state.num_ants);
                    ant_state.respawn(b, k, config);
                    replaced += 1;
                    deaths += 1;
                }
            }
        }
    }

    deaths
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> ErmConfig {
        ErmConfig {
            batch_size: 2,
            seq_len: 8,
            num_ants: 10,
            leader_fraction: 0.10,
            max_edits_per_step: 0.15,
            death_streak: 3,
            ..ErmConfig::default()
        }
    }

    #[test]
    fn test_ant_state_init() {
        let cfg = small_config();
        let state = AntState::new(&cfg);
        assert_eq!(state.streak.len(), cfg.batch_size * cfg.num_ants);
        assert!(state.streak.iter().all(|&s| s == 0));

        // Check leader/follower ratio.
        assert_eq!(state.count_leaders(0), cfg.num_leaders());
        assert_eq!(state.count_followers(0), cfg.num_followers());
    }

    #[test]
    fn test_streak_and_death() {
        let cfg = small_config(); // death_streak = 3
        let mut state = AntState::new(&cfg);

        // Ant (0, 5) has no improvement 3 times → dies.
        assert!(!state.record_delta(0, 5, 0.0, cfg.death_streak)); // streak=1
        assert!(!state.record_delta(0, 5, -1.0, cfg.death_streak)); // streak=2
        assert!(state.record_delta(0, 5, 0.0, cfg.death_streak)); // streak=3 → dies

        // Respawn.
        state.respawn(0, 5, &cfg);
        assert_eq!(state.streak[state.idx(0, 5)], 0);
    }

    #[test]
    fn test_streak_resets_on_improvement() {
        let cfg = small_config();
        let mut state = AntState::new(&cfg);

        state.record_delta(0, 3, 0.0, cfg.death_streak); // streak=1
        state.record_delta(0, 3, 0.0, cfg.death_streak); // streak=2
        state.record_delta(0, 3, 1.0, cfg.death_streak); // improvement → streak=0
        assert_eq!(state.streak[state.idx(0, 3)], 0);
    }

    #[test]
    fn test_merge_conflict_resolution() {
        let cfg = small_config();
        let editable = vec![true; cfg.batch_size * cfg.seq_len];

        let proposals = vec![
            EditProposal {
                batch_idx: 0,
                ant_idx: 0,
                edits: vec![(2, 10, 0.5)],
            },
            EditProposal {
                batch_idx: 0,
                ant_idx: 1,
                edits: vec![(2, 20, 0.8)], // higher gain → wins
            },
        ];

        let merged = merge_edits(&proposals, &editable, &cfg);
        assert_eq!(merged.best_tok[2], 20); // ant 1 wins
    }

    #[test]
    fn test_merge_respects_editable() {
        let cfg = small_config();
        let mut editable = vec![true; cfg.batch_size * cfg.seq_len];
        editable[3] = false; // position 3 not editable

        let proposals = vec![EditProposal {
            batch_idx: 0,
            ant_idx: 0,
            edits: vec![(3, 99, 10.0)],
        }];

        let merged = merge_edits(&proposals, &editable, &cfg);
        assert_eq!(merged.best_tok[3], -1); // rejected — not editable
    }

    #[test]
    fn test_merge_max_edits() {
        let cfg = ErmConfig {
            batch_size: 1,
            seq_len: 8,
            max_edits_per_step: 0.15, // ceil(0.15 * 8) = 2 max edits
            ..small_config()
        };
        let editable = vec![true; 8];

        // Propose 5 edits, all different positions.
        let proposals = vec![EditProposal {
            batch_idx: 0,
            ant_idx: 0,
            edits: vec![
                (0, 10, 1.0),
                (1, 20, 2.0),
                (2, 30, 3.0),
                (3, 40, 4.0),
                (4, 50, 5.0),
            ],
        }];

        let merged = merge_edits(&proposals, &editable, &cfg);
        // Only top-2 by gain should survive: positions 3 and 4.
        assert_eq!(merged.num_edited, 2);
        assert_eq!(merged.best_tok[4], 50); // gain=5.0 (highest)
        assert_eq!(merged.best_tok[3], 40); // gain=4.0 (second)
        assert_eq!(merged.best_tok[2], -1); // pruned
    }

    #[test]
    fn test_apply_edits() {
        let y_t = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let merged = MergedEdits {
            best_tok: vec![-1, -1, 99, -1, -1, 77, -1, -1],
            best_gain: vec![0.0; 8],
            num_edited: 2,
        };

        let y_next = apply_edits(&y_t, &merged);
        assert_eq!(y_next, vec![1, 2, 99, 4, 5, 77, 7, 8]);
    }

    #[test]
    fn test_merge_deterministic() {
        let cfg = small_config();
        let editable = vec![true; cfg.batch_size * cfg.seq_len];

        let proposals = vec![
            EditProposal {
                batch_idx: 0,
                ant_idx: 0,
                edits: vec![(1, 10, 0.5), (3, 30, 0.9)],
            },
            EditProposal {
                batch_idx: 0,
                ant_idx: 1,
                edits: vec![(1, 20, 0.3), (5, 50, 1.0)],
            },
        ];

        let m1 = merge_edits(&proposals, &editable, &cfg);
        let m2 = merge_edits(&proposals, &editable, &cfg);
        assert_eq!(m1.best_tok, m2.best_tok);
        assert_eq!(m1.num_edited, m2.num_edited);
    }

    // ── Follower ant sampler tests ──────────────────────────────────────

    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_follower_produces_valid_proposals() {
        let seq_len = 8;
        let vocab_size = 16;
        let cfg = ErmConfig {
            batch_size: 1,
            seq_len,
            vocab_size,
            emax: 4,
            ..ErmConfig::default()
        };
        let graph = RouteGraph::new_empty(&cfg);
        let editable = vec![true; seq_len];
        let follower_cfg = FollowerConfig {
            epsilon: 1e-4,
            temperature: 0.7,
            pmax: 4,
            topk: 4,
        };

        // Random logits.
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let logits: Vec<f32> = (0..seq_len * vocab_size)
            .map(|i| (i as f32 * 0.1) - 5.0)
            .collect();

        let proposals = AntColony::sample_follower_proposals(
            &logits,
            &graph,
            0,
            &follower_cfg,
            &editable,
            3, // 3 followers
            0, // first follower id
            seq_len,
            vocab_size,
            &mut rng,
        )
        .unwrap();

        // Check all proposals are valid.
        for p in &proposals {
            assert!(p.position < seq_len, "position {} out of range", p.position);
            assert!(
                (p.token as usize) < vocab_size,
                "token {} out of vocab range {}",
                p.token,
                vocab_size
            );
            assert!(p.predicted_gain.is_finite(), "gain must be finite");
            assert!(p.ant_id < 3, "ant_id {} out of range", p.ant_id);
        }

        // Should produce some proposals (not empty).
        assert!(
            !proposals.is_empty(),
            "should produce at least some proposals"
        );
    }

    #[test]
    fn test_follower_deterministic_with_seed() {
        let seq_len = 8;
        let vocab_size = 16;
        let cfg = ErmConfig {
            batch_size: 1,
            seq_len,
            vocab_size,
            emax: 4,
            ..ErmConfig::default()
        };
        let graph = RouteGraph::new_empty(&cfg);
        let editable = vec![true; seq_len];
        let follower_cfg = FollowerConfig {
            epsilon: 1e-4,
            temperature: 0.7,
            pmax: 4,
            topk: 4,
        };

        let logits: Vec<f32> = (0..seq_len * vocab_size)
            .map(|i| (i as f32 * 0.1) - 5.0)
            .collect();

        let mut rng1 = ChaCha8Rng::seed_from_u64(123);
        let p1 = AntColony::sample_follower_proposals(
            &logits,
            &graph,
            0,
            &follower_cfg,
            &editable,
            2,
            0,
            seq_len,
            vocab_size,
            &mut rng1,
        )
        .unwrap();

        let mut rng2 = ChaCha8Rng::seed_from_u64(123);
        let p2 = AntColony::sample_follower_proposals(
            &logits,
            &graph,
            0,
            &follower_cfg,
            &editable,
            2,
            0,
            seq_len,
            vocab_size,
            &mut rng2,
        )
        .unwrap();

        assert_eq!(p1.len(), p2.len());
        for (a, b) in p1.iter().zip(p2.iter()) {
            assert_eq!(a.position, b.position);
            assert_eq!(a.token, b.token);
            assert_eq!(a.ant_id, b.ant_id);
            assert!((a.predicted_gain - b.predicted_gain).abs() < 1e-9);
        }
    }

    #[test]
    fn test_follower_respects_non_editable() {
        let seq_len = 4;
        let vocab_size = 8;
        let cfg = ErmConfig {
            batch_size: 1,
            seq_len,
            vocab_size,
            emax: 2,
            ..ErmConfig::default()
        };
        let graph = RouteGraph::new_empty(&cfg);
        // Only position 2 is editable.
        let editable = vec![false, false, true, false];
        let follower_cfg = FollowerConfig {
            epsilon: 1e-4,
            temperature: 0.7,
            pmax: 4,
            topk: 4,
        };

        let logits: Vec<f32> = (0..seq_len * vocab_size).map(|i| i as f32).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(77);

        let proposals = AntColony::sample_follower_proposals(
            &logits,
            &graph,
            0,
            &follower_cfg,
            &editable,
            5,
            0,
            seq_len,
            vocab_size,
            &mut rng,
        )
        .unwrap();

        // All proposals must be at position 2.
        for p in &proposals {
            assert_eq!(p.position, 2, "only position 2 is editable");
        }
    }

    #[test]
    fn test_follower_no_editable_returns_empty() {
        let seq_len = 4;
        let vocab_size = 8;
        let cfg = ErmConfig {
            batch_size: 1,
            seq_len,
            vocab_size,
            emax: 2,
            ..ErmConfig::default()
        };
        let graph = RouteGraph::new_empty(&cfg);
        let editable = vec![false; seq_len];
        let follower_cfg = FollowerConfig::default();

        let logits = vec![0.0_f32; seq_len * vocab_size];
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        let proposals = AntColony::sample_follower_proposals(
            &logits,
            &graph,
            0,
            &follower_cfg,
            &editable,
            3,
            0,
            seq_len,
            vocab_size,
            &mut rng,
        )
        .unwrap();

        assert!(proposals.is_empty());
    }

    #[test]
    fn test_follower_shape_mismatch() {
        let cfg = ErmConfig {
            batch_size: 1,
            seq_len: 4,
            vocab_size: 8,
            emax: 2,
            ..ErmConfig::default()
        };
        let graph = RouteGraph::new_empty(&cfg);
        let editable = vec![true; 4];
        let follower_cfg = FollowerConfig::default();
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        // Wrong logits size.
        let bad_logits = vec![0.0_f32; 10];
        assert!(AntColony::sample_follower_proposals(
            &bad_logits,
            &graph,
            0,
            &follower_cfg,
            &editable,
            1,
            0,
            4,
            8,
            &mut rng,
        )
        .is_err());
    }

    // ── Leader ant sampler tests ────────────────────────────────────────

    #[test]
    fn test_leader_targets_high_uncertainty_low_pheromone() {
        let seq_len = 8;
        let vocab_size = 16;
        let cfg = ErmConfig {
            batch_size: 1,
            seq_len,
            vocab_size,
            emax: 4,
            ..ErmConfig::default()
        };
        let mut graph = RouteGraph::new_empty(&cfg);

        // Position 2 has strong pheromone, position 5 has none.
        graph.add_edge(0, 2, 0, 5.0).expect("add edge");
        graph.add_edge(0, 2, 1, 3.0).expect("add edge");

        let editable = vec![true; seq_len];
        let leader_cfg = LeaderConfig {
            epsilon: 1e-4,
            temperature: 1.5,
            pmax: 4,
            topk: 4,
        };

        let logits: Vec<f32> = (0..seq_len * vocab_size)
            .map(|i| (i as f32 * 0.1) - 5.0)
            .collect();

        // High uncertainty at positions 5 and 6, low elsewhere.
        let mut uncertainty = vec![0.1_f32; seq_len];
        uncertainty[5] = 0.9;
        uncertainty[6] = 0.85;

        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let (proposals, edge_proposals) = AntColony::sample_leader_proposals(
            &logits,
            &uncertainty,
            &graph,
            0,
            &leader_cfg,
            &editable,
            3,
            0,
            seq_len,
            vocab_size,
            &mut rng,
        )
        .expect("leader proposals");

        // Should produce some proposals.
        assert!(!proposals.is_empty(), "leaders should produce proposals");

        // Should produce edge proposals.
        assert!(
            !edge_proposals.is_empty(),
            "leaders should propose new edges"
        );

        // Validate all proposals.
        for p in &proposals {
            assert!(p.position < seq_len);
            assert!((p.token as usize) < vocab_size);
            assert!(p.predicted_gain.is_finite());
        }

        // Check edge proposals are valid.
        for ep in &edge_proposals {
            assert!(ep.src < seq_len);
            assert!(ep.dst < seq_len);
            assert_ne!(ep.src, ep.dst, "edge should not be self-loop");
        }
    }

    #[test]
    fn test_leader_deterministic_with_seed() {
        let seq_len = 8;
        let vocab_size = 16;
        let cfg = ErmConfig {
            batch_size: 1,
            seq_len,
            vocab_size,
            emax: 4,
            ..ErmConfig::default()
        };
        let graph = RouteGraph::new_empty(&cfg);
        let editable = vec![true; seq_len];
        let leader_cfg = LeaderConfig::from_config(&cfg);

        let logits: Vec<f32> = (0..seq_len * vocab_size)
            .map(|i| (i as f32 * 0.1) - 5.0)
            .collect();
        let uncertainty = vec![0.5_f32; seq_len];

        let mut rng1 = ChaCha8Rng::seed_from_u64(123);
        let (p1, e1) = AntColony::sample_leader_proposals(
            &logits,
            &uncertainty,
            &graph,
            0,
            &leader_cfg,
            &editable,
            2,
            0,
            seq_len,
            vocab_size,
            &mut rng1,
        )
        .expect("proposals");

        let mut rng2 = ChaCha8Rng::seed_from_u64(123);
        let (p2, e2) = AntColony::sample_leader_proposals(
            &logits,
            &uncertainty,
            &graph,
            0,
            &leader_cfg,
            &editable,
            2,
            0,
            seq_len,
            vocab_size,
            &mut rng2,
        )
        .expect("proposals");

        assert_eq!(p1.len(), p2.len());
        for (a, b) in p1.iter().zip(p2.iter()) {
            assert_eq!(a.position, b.position);
            assert_eq!(a.token, b.token);
            assert_eq!(a.ant_id, b.ant_id);
        }
        assert_eq!(e1.len(), e2.len());
    }

    #[test]
    fn test_leader_no_editable_returns_empty() {
        let seq_len = 4;
        let vocab_size = 8;
        let cfg = ErmConfig {
            batch_size: 1,
            seq_len,
            vocab_size,
            emax: 2,
            ..ErmConfig::default()
        };
        let graph = RouteGraph::new_empty(&cfg);
        let editable = vec![false; seq_len];
        let leader_cfg = LeaderConfig::default();
        let logits = vec![0.0_f32; seq_len * vocab_size];
        let uncertainty = vec![0.5_f32; seq_len];
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        let (proposals, edge_proposals) = AntColony::sample_leader_proposals(
            &logits,
            &uncertainty,
            &graph,
            0,
            &leader_cfg,
            &editable,
            3,
            0,
            seq_len,
            vocab_size,
            &mut rng,
        )
        .expect("proposals");

        assert!(proposals.is_empty());
        assert!(edge_proposals.is_empty());
    }

    // ── Death/respawn tests ─────────────────────────────────────────────

    #[test]
    fn test_death_mode_streak() {
        let cfg = ErmConfig {
            batch_size: 1,
            num_ants: 5,
            death_streak: 2,
            leader_fraction: 0.0,
            ..small_config()
        };
        let mut state = AntState::new(&cfg);
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        // All ants have zero delta → all should die after 2 steps.
        // step=999 → past any warmstart.
        let deltas = vec![0.0_f32; 5];
        let d1 = apply_death_respawn(&mut state, &deltas, &cfg, DeathMode::Streak, 999, &mut rng);
        assert_eq!(d1, 0, "no deaths after 1 step (streak=1)");

        let d2 = apply_death_respawn(&mut state, &deltas, &cfg, DeathMode::Streak, 999, &mut rng);
        assert_eq!(d2, 5, "all 5 ants should die after 2 steps");

        // After respawn, streaks should be reset.
        for k in 0..5 {
            assert_eq!(state.streak[state.idx(0, k)], 0);
        }
    }

    #[test]
    fn test_death_mode_random_pool() {
        let cfg = ErmConfig {
            batch_size: 1,
            num_ants: 20,
            leader_fraction: 0.10,
            ..small_config()
        };
        let mut state = AntState::new(&cfg);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let deltas = vec![0.0_f32; 20];
        let deaths = apply_death_respawn(
            &mut state,
            &deltas,
            &cfg,
            DeathMode::RandomPool,
            999,
            &mut rng,
        );

        // Should replace ~10% = 2 ants (ceil(20*0.1) = 2).
        assert_eq!(deaths, 2, "should replace ceil(10%) = 2 ants");
    }

    #[test]
    fn test_death_streak_resets_on_improvement() {
        let cfg = ErmConfig {
            batch_size: 1,
            num_ants: 3,
            death_streak: 3,
            leader_fraction: 0.0,
            ..small_config()
        };
        let mut state = AntState::new(&cfg);
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        // No improvement for 2 steps. step=999 → past warmstart.
        let zero_deltas = vec![0.0_f32; 3];
        apply_death_respawn(
            &mut state,
            &zero_deltas,
            &cfg,
            DeathMode::Streak,
            999,
            &mut rng,
        );
        apply_death_respawn(
            &mut state,
            &zero_deltas,
            &cfg,
            DeathMode::Streak,
            999,
            &mut rng,
        );

        // Now improve: streak should reset.
        let good_deltas = vec![1.0_f32; 3];
        let deaths = apply_death_respawn(
            &mut state,
            &good_deltas,
            &cfg,
            DeathMode::Streak,
            999,
            &mut rng,
        );
        assert_eq!(deaths, 0, "no deaths after improvement");

        // Two more zero steps should not kill (streak only at 2, need 3).
        apply_death_respawn(
            &mut state,
            &zero_deltas,
            &cfg,
            DeathMode::Streak,
            999,
            &mut rng,
        );
        let deaths2 = apply_death_respawn(
            &mut state,
            &zero_deltas,
            &cfg,
            DeathMode::Streak,
            999,
            &mut rng,
        );
        assert_eq!(deaths2, 0, "streak should be 2, not enough to die");
    }

    #[test]
    fn test_warmstart_relaxes_death_streak() {
        let cfg = ErmConfig {
            batch_size: 1,
            num_ants: 3,
            death_streak: 2,
            leader_fraction: 0.0,
            warmstart_steps: 100,
            warmstart_death_mult: 4,
            ..small_config()
        };
        let mut state = AntState::new(&cfg);
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        let zero_deltas = vec![0.0_f32; 3];

        // During warmstart (step=0), effective streak = 2*4 = 8.
        // After 2 steps with no improvement, ants should NOT die.
        apply_death_respawn(
            &mut state,
            &zero_deltas,
            &cfg,
            DeathMode::Streak,
            0,
            &mut rng,
        );
        let d = apply_death_respawn(
            &mut state,
            &zero_deltas,
            &cfg,
            DeathMode::Streak,
            1,
            &mut rng,
        );
        assert_eq!(
            d, 0,
            "ants should not die during warmstart with only 2 zero steps"
        );

        // After warmstart ends (step=100), with fresh state, death_streak = 2.
        let mut state2 = AntState::new(&cfg);
        apply_death_respawn(
            &mut state2,
            &zero_deltas,
            &cfg,
            DeathMode::Streak,
            100,
            &mut rng,
        );
        let d2 = apply_death_respawn(
            &mut state2,
            &zero_deltas,
            &cfg,
            DeathMode::Streak,
            101,
            &mut rng,
        );
        assert_eq!(
            d2, 3,
            "all ants should die after warmstart with death_streak=2"
        );
    }

    #[test]
    fn test_follower_temperature_schedule() {
        use super::follower_temperature_schedule;

        // At step 0, temperature = max(0.3, 1.0) = 1.0
        let t0 = follower_temperature_schedule(0, 1000);
        assert!((t0 - 1.0).abs() < 1e-5, "step 0 should be 1.0, got {t0}");

        // At halfway, temperature = max(0.3, 0.5) = 0.5
        let t500 = follower_temperature_schedule(500, 1000);
        assert!((t500 - 0.5).abs() < 1e-5, "step 500 should be 0.5, got {t500}");

        // At end, temperature = max(0.3, 0.0) = 0.3
        let t1000 = follower_temperature_schedule(1000, 1000);
        assert!((t1000 - 0.3).abs() < 1e-5, "step 1000 should be 0.3, got {t1000}");

        // total_steps=0 returns default 0.7
        let td = follower_temperature_schedule(50, 0);
        assert!((td - 0.7).abs() < 1e-5, "total_steps=0 should be 0.7, got {td}");
    }

    #[test]
    fn test_leader_temperature_schedule() {
        use super::leader_temperature_schedule;

        // Low uncertainty → floor at 0.5
        let tl = leader_temperature_schedule(0.1);
        assert!((tl - 0.5).abs() < 1e-5, "low unc should be 0.5, got {tl}");

        // Moderate uncertainty → 2 * 0.4 = 0.8
        let tm = leader_temperature_schedule(0.4);
        assert!((tm - 0.8).abs() < 1e-5, "mid unc should be 0.8, got {tm}");

        // High uncertainty → 2 * 1.0 = 2.0
        let th = leader_temperature_schedule(1.0);
        assert!((th - 2.0).abs() < 1e-5, "high unc should be 2.0, got {th}");
    }

    #[test]
    fn test_follower_config_with_temperature() {
        let cfg = FollowerConfig::default().with_temperature(0.42);
        assert!((cfg.temperature - 0.42).abs() < 1e-5);
    }

    #[test]
    fn test_leader_config_with_temperature() {
        let cfg = LeaderConfig::default().with_temperature(1.23);
        assert!((cfg.temperature - 1.23).abs() < 1e-5);
    }
}
