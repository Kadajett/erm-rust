//! Pheromone update loop for the Emergent Route Model.
//!
//! Implements the stigmergic feedback cycle:
//! 1. Build edge traces from ant proposals and route weights
//! 2. Evaporate all pheromone trails
//! 3. Deposit pheromone from successful ants (tanh-bounded)
//! 4. Update taint from harmful ants
//! 5. Decay taint over time
//! 6. Age all edges
//! 7. Prune weak or old edges
//!
//! # Key invariants
//!
//! - `φ ∈ [0, φ_max]` — pheromone is non-negative and bounded
//! - `τ ∈ [0, τ_max]` — taint is clamped
//! - Deposit uses `tanh` bounding to prevent runaway feedback loops
//!
//! # Tensor shapes
//!
//! | Structure       | Shape / Size                            |
//! |-----------------|-----------------------------------------|
//! | `EdgeTrace`     | per-ant: `Vec<(batch, dest, edge_idx, weight)>` |
//! | `PheromoneStats`| scalar summary of graph state           |

use crate::config::PheromoneConfig;
use crate::error::ErmResult;
use crate::graph::{RouteGraph, EMPTY_SLOT};
use crate::merge::SimpleEditProposal;

/// Running standard deviation accumulator for pheromone deposit normalization.
///
/// Tracks a running mean and variance of positive deltas using Welford's
/// online algorithm. The resulting `sigma()` is used to normalize the deposit:
/// `eta * tanh(Delta / (sigma + eps))` instead of raw `eta * tanh(relu(Delta))`.
///
/// Falls back to the unnormalized formula when sigma is zero (cold start).
#[derive(Debug, Clone)]
pub struct RunningDeltaStats {
    /// Number of positive deltas observed.
    count: u64,
    /// Running mean of positive deltas.
    mean: f64,
    /// Running M2 (sum of squared differences from the mean).
    m2: f64,
}

impl Default for RunningDeltaStats {
    fn default() -> Self {
        Self::new()
    }
}

impl RunningDeltaStats {
    /// Create a new accumulator with no observations.
    #[must_use]
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Record a positive delta value.
    pub fn push(&mut self, value: f32) {
        let v = value as f64;
        self.count += 1;
        let delta = v - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = v - self.mean;
        self.m2 += delta * delta2;
    }

    /// Current running standard deviation (population). Returns 0.0 if < 2 observations.
    #[must_use]
    pub fn sigma(&self) -> f32 {
        if self.count < 2 {
            0.0
        } else {
            (self.m2 / self.count as f64).sqrt() as f32
        }
    }

    /// Number of observations.
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }
}

/// A single ant's edge trace through the route graph.
///
/// Records which edges an ant used during its proposal step, along with
/// the edge weights from the route aggregation softmax.
///
/// # Fields
///
/// - `ant_id`: index of the ant in the colony.
/// - `entries`: `Vec<(batch, dest, edge_idx, weight)>` — each tuple records
///   a specific edge slot that contributed to this ant's edit proposals,
///   weighted by the softmax attention weight from route aggregation.
#[derive(Debug, Clone)]
pub struct EdgeTrace {
    /// Index of the ant that generated this trace.
    pub ant_id: usize,
    /// Edge entries: `(batch_idx, dest_pos, edge_slot, softmax_weight)`.
    /// Shape: variable length, up to `B * L * Emax` in the worst case.
    pub entries: Vec<(usize, usize, usize, f32)>,
}

/// Summary statistics of the pheromone graph after an update step.
///
/// Used for logging and monitoring convergence.
#[derive(Debug, Clone)]
pub struct PheromoneStats {
    /// Mean pheromone `φ` across all valid edges.
    pub mean_phi: f32,
    /// Maximum pheromone `φ` across all valid edges.
    pub max_phi: f32,
    /// Mean taint `τ` across all valid edges.
    pub mean_taint: f32,
    /// Number of edges with taint > 0.
    pub tainted_count: usize,
}

/// Build edge traces from ant proposals and route-aggregate edge weights.
///
/// For each ant's accepted proposal, records which edges at the edited
/// positions contributed to the route aggregation, weighted by the
/// softmax attention weights.
///
/// # Arguments
///
/// - `proposals`: edit proposals from the merge step.
/// - `edge_weights`: flat `[B, L, Emax]` softmax weights from route aggregation.
/// - `batch_size`: batch dimension `B`.
/// - `seq_len`: sequence length `L`.
/// - `emax`: maximum edges per destination `Emax`.
///
/// # Returns
///
/// A `Vec<EdgeTrace>`, one per unique ant that proposed edits.
/// Ants with no proposals produce no trace.
///
/// # Shape reference
///
/// | Input          | Shape             |
/// |----------------|-------------------|
/// | `edge_weights` | `[B, L, Emax]` flat |
/// | output         | `Vec<EdgeTrace>`  |
pub fn build_edge_traces(
    proposals: &[SimpleEditProposal],
    edge_weights: &[f32],
    batch_size: usize,
    seq_len: usize,
    emax: usize,
) -> Vec<EdgeTrace> {
    // Group proposals by ant_id.
    let mut ant_ids: Vec<usize> = proposals.iter().map(|p| p.ant_id).collect();
    ant_ids.sort_unstable();
    ant_ids.dedup();

    let expected_ew = batch_size * seq_len * emax;

    ant_ids
        .into_iter()
        .map(|ant_id| {
            let mut entries = Vec::new();

            for p in proposals.iter().filter(|p| p.ant_id == ant_id) {
                let pos = p.position;

                // For each batch element, record the edge weights at this position.
                for b in 0..batch_size {
                    let ew_base = (b * seq_len + pos) * emax;
                    if ew_base + emax > expected_ew {
                        continue;
                    }
                    for e in 0..emax {
                        let w = edge_weights[ew_base + e];
                        if w > 0.0 {
                            entries.push((b, pos, e, w));
                        }
                    }
                }
            }

            EdgeTrace { ant_id, entries }
        })
        .collect()
}

/// Update pheromone trails on the route graph.
///
/// Applies the full pheromone update cycle in order:
///
/// 1. **Evaporation**: `φ *= (1 - ρ)` for all valid edges
/// 2. **Deposit**: `φ += η * tanh(Δ / (σ + ε))` for edges used by improving ants
///    (where σ is the running std of positive deltas; falls back to `η * tanh(Δ)` when σ ≈ 0)
/// 3. **Taint**: `τ += ζ * max(-Δ_k, 0.0)` for edges used by harmful ants, clamped to `τ_max`
/// 4. **Taint decay**: `τ *= (1 - ρ_τ)` for all valid edges
/// 5. **Age increment**: `age += 1` for all valid edges
/// 6. **Enforce bounds**: `φ ∈ [0, φ_max]`, `τ ∈ [0, τ_max]`
///
/// # Arguments
///
/// - `graph`: mutable route graph to update in-place.
/// - `traces`: edge traces from `build_edge_traces`.
/// - `ant_deltas`: per-ant improvement deltas `Δ_k`. Shape: `[num_ants]`.
/// - `config`: pheromone hyperparameters.
///
/// # Returns
///
/// `PheromoneStats` summarizing the graph state after the update.
///
/// # Errors
///
/// Returns `ErmError` on dimension mismatches.
pub fn update_pheromones(
    graph: &mut RouteGraph,
    traces: &[EdgeTrace],
    ant_deltas: &[f32],
    config: &PheromoneConfig,
) -> ErmResult<PheromoneStats> {
    update_pheromones_full(graph, traces, ant_deltas, config, None, None, None)
}

/// Update pheromones with optional running delta normalization.
///
/// When `delta_stats` is `Some`, uses normalized deposit:
/// `η * tanh(Δ / (σ + ε))` where σ = running std. Falls back to
/// `η * tanh(Δ)` when σ ≈ 0 (cold start).
///
/// Positive deltas are also pushed into the running stats accumulator.
///
/// # Errors
///
/// Returns `ErmError` on dimension mismatches.
pub fn update_pheromones_with_stats(
    graph: &mut RouteGraph,
    traces: &[EdgeTrace],
    ant_deltas: &[f32],
    config: &PheromoneConfig,
    delta_stats: Option<&mut RunningDeltaStats>,
) -> ErmResult<PheromoneStats> {
    update_pheromones_full(graph, traces, ant_deltas, config, delta_stats, None, None)
}

/// Update pheromones with both running delta normalization and diversity pressure.
///
/// When `hidden` is `Some(&[f32])` with shape `[B, L, d]`, applies diversity
/// pressure after deposit: for each destination, if two incoming edges have
/// source hidden states with cosine similarity > 0.9, the weaker edge's
/// pheromone is penalized. This prevents the route graph from collapsing to
/// redundant connections — the Muon "orthogonalization" analog.
///
/// # Arguments
///
/// - `hidden`: optional flat `[B, L, d]` hidden states from scorer.
/// - `hidden_dim`: hidden dimension `d` (required when `hidden` is `Some`).
///
/// # Errors
///
/// Returns `ErmError` on dimension mismatches.
pub fn update_pheromones_with_diversity(
    graph: &mut RouteGraph,
    traces: &[EdgeTrace],
    ant_deltas: &[f32],
    config: &PheromoneConfig,
    delta_stats: Option<&mut RunningDeltaStats>,
    hidden: &[f32],
    hidden_dim: usize,
) -> ErmResult<PheromoneStats> {
    update_pheromones_full(
        graph,
        traces,
        ant_deltas,
        config,
        delta_stats,
        Some((hidden, hidden_dim)),
        None,
    )
}

/// Update pheromones with per-position credit assignment and diversity pressure.
///
/// When `position_deltas` is `Some`, the deposit for each edge uses the
/// improvement delta at the edge's **destination position** rather than
/// the aggregate delta for the edge's **ant**. This gives finer-grained
/// credit: edges routing to positions that improved get positive deposit,
/// edges routing to positions that worsened get zero deposit (taint still
/// uses per-ant deltas).
///
/// # Arguments
///
/// - `position_deltas`: flat `[B * L]` per-position improvement deltas.
///   Index as `position_deltas[batch_idx * seq_len + dest_pos]`.
///
/// # Errors
///
/// Returns `ErmError` on dimension mismatches.
#[allow(clippy::too_many_arguments)]
pub fn update_pheromones_with_position_credit(
    graph: &mut RouteGraph,
    traces: &[EdgeTrace],
    ant_deltas: &[f32],
    config: &PheromoneConfig,
    delta_stats: Option<&mut RunningDeltaStats>,
    hidden: &[f32],
    hidden_dim: usize,
    position_deltas: &[f32],
) -> ErmResult<PheromoneStats> {
    update_pheromones_full(
        graph,
        traces,
        ant_deltas,
        config,
        delta_stats,
        Some((hidden, hidden_dim)),
        Some(position_deltas),
    )
}

/// Build an optional elite mask from per-ant deltas.
///
/// Returns `None` when elite filtering is disabled (`elite_k == 0`) or when
/// `elite_k >= ant_deltas.len()` (all ants are effectively elite).
fn build_elite_mask(ant_deltas: &[f32], elite_k: usize) -> Option<Vec<bool>> {
    if elite_k == 0 || ant_deltas.is_empty() || elite_k >= ant_deltas.len() {
        return None;
    }

    let mut ranked: Vec<(usize, f32)> = ant_deltas.iter().copied().enumerate().collect();
    ranked.sort_unstable_by(|(i_a, d_a), (i_b, d_b)| d_b.total_cmp(d_a).then_with(|| i_a.cmp(i_b)));

    let mut mask = vec![false; ant_deltas.len()];
    for (ant_id, _) in ranked.into_iter().take(elite_k) {
        mask[ant_id] = true;
    }
    Some(mask)
}

/// Internal: full pheromone update with all optional features.
fn update_pheromones_full(
    graph: &mut RouteGraph,
    traces: &[EdgeTrace],
    ant_deltas: &[f32],
    config: &PheromoneConfig,
    mut delta_stats: Option<&mut RunningDeltaStats>,
    hidden: Option<(&[f32], usize)>,
    position_deltas: Option<&[f32]>,
) -> ErmResult<PheromoneStats> {
    let total = graph.total_elements();
    let rho = config.evaporation_rate;
    let eta = config.deposit_rate;
    let zeta = config.taint_rate;
    let rho_tau = config.taint_decay;
    let tau_max = config.taint_max;
    let phi_max = config.phi_max;
    let phi_min = config.phi_min.clamp(0.0, phi_max);
    let seq_len = graph.seq_len;
    let elite_mask = build_elite_mask(ant_deltas, config.elite_k);

    // Compute sigma from running stats (if provided).
    let sigma = delta_stats.as_ref().map_or(0.0_f32, |s| s.sigma());
    let norm_eps = 1e-6_f32;
    let use_log = config.use_log_deposit;

    // Deposit bounding function: log1p or tanh.
    // log1p(|Δ|/σ) has better dynamic range than tanh(Δ/σ):
    //   - small deltas get proportional credit
    //   - large deltas get diminishing but nonzero credit (never saturates to 1)
    let bounded_deposit = |positive_delta: f32| -> f32 {
        if use_log {
            if sigma > norm_eps {
                (1.0 + positive_delta / (sigma + norm_eps)).ln()
            } else {
                (1.0 + positive_delta).ln()
            }
        } else if sigma > norm_eps {
            (positive_delta / (sigma + norm_eps)).tanh()
        } else {
            positive_delta.tanh()
        }
    };

    // Step 1: Evaporation — φ *= (1 - ρ) for all valid edges.
    for flat in 0..total {
        if graph.nbr_idx[flat] != EMPTY_SLOT {
            graph.phi[flat] *= 1.0 - rho;
        }
    }

    // Step 2: Deposit — bounded (log1p or tanh) for edges used by elite ants.
    // Step 3: Taint — τ += ζ * max(-Δ_k, 0.0) for edges used by ant k.
    //
    // When position_deltas is Some, deposit uses per-position delta at
    // the edge's destination (finer credit assignment). When None, falls
    // back to per-ant aggregate delta (original behavior).
    // Taint always uses per-ant delta (harmful ant → penalize all its edges).
    for trace in traces {
        let ant_id = trace.ant_id;
        let ant_delta = if ant_id < ant_deltas.len() {
            ant_deltas[ant_id]
        } else {
            0.0
        };
        let is_elite = if let Some(mask) = elite_mask.as_ref() {
            ant_id < mask.len() && mask[ant_id]
        } else {
            true
        };

        // Taint from ant-level delta (always per-ant).
        let negative_ant_delta = (-ant_delta).max(0.0);
        let taint_deposit = zeta * negative_ant_delta;

        // Pre-compute ant-level deposit base (used when position_deltas is None,
        // or as fallback for out-of-bounds positions).
        let ant_positive_delta = ant_delta.max(0.0);

        // Update running stats with positive ant deltas.
        if is_elite && ant_positive_delta > 0.0 {
            if let Some(ref mut stats) = delta_stats {
                stats.push(ant_positive_delta);
            }
        }

        let ant_deposit_base = if is_elite {
            bounded_deposit(ant_positive_delta)
        } else {
            0.0
        };

        for &(b, dst, e, _weight) in &trace.entries {
            if b >= graph.batch_size || dst >= graph.seq_len || e >= graph.emax {
                continue;
            }
            let flat = graph.idx(b, dst, e);
            if graph.nbr_idx[flat] == EMPTY_SLOT {
                continue;
            }

            // Choose deposit base: per-position when available, per-ant otherwise.
            let deposit_base = if is_elite {
                if let Some(pos_deltas) = position_deltas {
                    let pos_idx = b * seq_len + dst;
                    if pos_idx < pos_deltas.len() {
                        let pos_delta = pos_deltas[pos_idx].max(0.0);
                        bounded_deposit(pos_delta)
                    } else {
                        ant_deposit_base
                    }
                } else {
                    ant_deposit_base
                }
            } else {
                0.0
            };

            // Per-edge learning rate decay: η / (1 + age).
            // Older edges receive smaller deposits (diminishing returns),
            // similar to learning rate decay in gradient-based optimizers.
            let edge_eta = eta / (1.0 + graph.age[flat] as f32);
            graph.phi[flat] += edge_eta * deposit_base;

            // Taint deposit (not age-decayed — harmful signals stay strong).
            graph.taint[flat] += taint_deposit;
        }
    }

    // Step 4: Taint decay — τ *= (1 - ρ_τ) for all valid edges.
    // Step 5: Age increment for all valid edges.
    // Step 6: Enforce bounds.
    for flat in 0..total {
        if graph.nbr_idx[flat] != EMPTY_SLOT {
            // Taint decay.
            graph.taint[flat] *= 1.0 - rho_tau;

            // Age increment.
            graph.age[flat] += 1;

            // Enforce φ ∈ [φ_min, φ_max] for active edges.
            graph.phi[flat] = graph.phi[flat].clamp(phi_min, phi_max);

            // Enforce τ ∈ [0, τ_max].
            graph.taint[flat] = graph.taint[flat].clamp(0.0, tau_max);

            debug_assert!(graph.phi[flat] >= 0.0);
            debug_assert!(graph.taint[flat] >= 0.0 && graph.taint[flat] <= tau_max);
        }
    }

    // Step 7 (optional): Diversity pressure using hidden states.
    //
    // For each destination node, compare the hidden states of its incoming
    // edges' source positions. If two edges have cosine similarity > 0.9,
    // penalize the weaker one (reduce φ by 20%). This prevents the route
    // graph from collapsing into redundant connections — analogous to
    // Muon's momentum orthogonalization which amplifies rare directions.
    if let Some((hidden, d)) = hidden {
        let b_size = graph.batch_size;
        let l = graph.seq_len;
        let emax = graph.emax;

        for bi in 0..b_size {
            for dst in 0..l {
                // Collect valid edges at this destination.
                let mut valid_edges: Vec<(usize, usize)> = Vec::new(); // (edge_slot, src_pos)
                for e in 0..emax {
                    let flat = graph.idx(bi, dst, e);
                    if graph.nbr_idx[flat] != EMPTY_SLOT {
                        valid_edges.push((e, graph.nbr_idx[flat] as usize));
                    }
                }

                // Compare all pairs of valid edges.
                for i in 0..valid_edges.len() {
                    for j in (i + 1)..valid_edges.len() {
                        let (ei, src_i) = valid_edges[i];
                        let (ej, src_j) = valid_edges[j];

                        // Compute cosine similarity between hidden[bi, src_i, :] and hidden[bi, src_j, :]
                        let base_i = (bi * l + src_i) * d;
                        let base_j = (bi * l + src_j) * d;

                        // Bounds check to avoid panics.
                        if base_i + d > hidden.len() || base_j + d > hidden.len() {
                            continue;
                        }

                        let mut dot = 0.0_f32;
                        let mut norm_i = 0.0_f32;
                        let mut norm_j = 0.0_f32;
                        for k in 0..d {
                            let hi = hidden[base_i + k];
                            let hj = hidden[base_j + k];
                            dot += hi * hj;
                            norm_i += hi * hi;
                            norm_j += hj * hj;
                        }

                        let denom = (norm_i * norm_j).sqrt();
                        if denom < 1e-8 {
                            continue;
                        }
                        let cosine_sim = dot / denom;

                        // If sources are very similar, penalize the weaker edge.
                        if cosine_sim > config.diversity_threshold {
                            let flat_i = graph.idx(bi, dst, ei);
                            let flat_j = graph.idx(bi, dst, ej);

                            if graph.phi[flat_i] < graph.phi[flat_j] {
                                graph.phi[flat_i] *= config.diversity_penalty;
                            } else {
                                graph.phi[flat_j] *= config.diversity_penalty;
                            }
                        }
                    }
                }
            }
        }
    }

    // Compute stats.
    let stats = compute_stats(graph);
    Ok(stats)
}

/// Prune edges that are too weak or too old.
///
/// An edge at `(b, dst, e)` is pruned if:
/// - Its composite score `φ - λ·τ < min_score`, OR
/// - Its age exceeds `max_age`
///
/// Uses swap-remove to maintain dense packing.
///
/// # Arguments
///
/// - `graph`: mutable route graph.
/// - `min_score`: minimum composite score threshold.
/// - `max_age`: maximum allowed edge age.
/// - `lambda`: taint penalty coefficient for composite score.
///
/// # Returns
///
/// Number of edges pruned.
pub fn prune_edges(graph: &mut RouteGraph, min_score: f32, max_age: u32, lambda: f32) -> usize {
    let mut pruned = 0;
    let b_size = graph.batch_size;
    let l = graph.seq_len;
    let emax = graph.emax;

    for b in 0..b_size {
        for dst in 0..l {
            // Collect slots to prune (iterate in reverse to avoid index shifting).
            let mut slots_to_prune = Vec::new();
            for e in 0..emax {
                let flat = graph.idx(b, dst, e);
                if graph.nbr_idx[flat] == EMPTY_SLOT {
                    continue;
                }
                let score = graph.phi[flat] - lambda * graph.taint[flat];
                let age = graph.age[flat] as u32;

                if score < min_score || age > max_age {
                    slots_to_prune.push(e);
                }
            }

            // Prune in reverse order to preserve indices during swap-remove.
            slots_to_prune.sort_unstable();
            slots_to_prune.reverse();
            for slot in slots_to_prune {
                if graph.remove_edge(b, dst, slot).is_ok() {
                    pruned += 1;
                }
            }
        }
    }

    pruned
}

/// Rescale pheromone values to prevent runaway growth (MuonClip analog).
///
/// If the maximum φ across all valid edges exceeds `threshold`, all φ
/// values are multiplied by `sqrt(threshold / max_phi)`. This keeps the
/// softmax in route aggregation from collapsing into a hard argmax while
/// preserving the relative ordering of edge strengths.
///
/// Analogous to QK-Clip in the Muon optimizer, which rescales query/key
/// projection weights when attention logits grow too large.
///
/// # Arguments
///
/// - `graph`: mutable route graph.
/// - `threshold`: maximum allowed φ before rescaling kicks in (e.g., 8.0).
///
/// # Returns
///
/// `true` if rescaling was applied, `false` otherwise.
pub fn pheromone_rescale(graph: &mut RouteGraph, threshold: f32) -> bool {
    let total = graph.total_elements();
    let mut max_phi = 0.0_f32;

    for flat in 0..total {
        if graph.nbr_idx[flat] != EMPTY_SLOT && graph.phi[flat] > max_phi {
            max_phi = graph.phi[flat];
        }
    }

    if max_phi > threshold {
        let gamma = (threshold / max_phi).sqrt();
        for flat in 0..total {
            if graph.nbr_idx[flat] != EMPTY_SLOT {
                graph.phi[flat] *= gamma;
            }
        }
        true
    } else {
        false
    }
}

/// Compute summary statistics of the pheromone graph.
fn compute_stats(graph: &RouteGraph) -> PheromoneStats {
    let total = graph.total_elements();
    let mut sum_phi = 0.0_f32;
    let mut max_phi = 0.0_f32;
    let mut sum_taint = 0.0_f32;
    let mut tainted_count = 0_usize;
    let mut valid_count = 0_usize;

    for flat in 0..total {
        if graph.nbr_idx[flat] != EMPTY_SLOT {
            valid_count += 1;
            sum_phi += graph.phi[flat];
            if graph.phi[flat] > max_phi {
                max_phi = graph.phi[flat];
            }
            sum_taint += graph.taint[flat];
            if graph.taint[flat] > 0.0 {
                tainted_count += 1;
            }
        }
    }

    let mean_phi = if valid_count > 0 {
        sum_phi / valid_count as f32
    } else {
        0.0
    };
    let mean_taint = if valid_count > 0 {
        sum_taint / valid_count as f32
    } else {
        0.0
    };

    PheromoneStats {
        mean_phi,
        max_phi,
        mean_taint,
        tainted_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ErmConfig;

    fn small_config() -> ErmConfig {
        ErmConfig {
            batch_size: 1,
            seq_len: 4,
            emax: 3,
            ..ErmConfig::default()
        }
    }

    fn small_pheromone_config() -> PheromoneConfig {
        PheromoneConfig {
            evaporation_rate: 0.1,
            deposit_rate: 0.5,
            taint_rate: 0.3,
            taint_decay: 0.05,
            taint_max: 5.0,
            phi_max: 100.0,
            phi_min: 1e-4,
            elite_k: 0,
            prune_min_score: -1.0,
            prune_max_age: 1000,
            route_lambda: 1.0,
            diversity_threshold: 0.9,
            diversity_penalty: 0.8,
            use_log_deposit: false, // tests were written for tanh mode
        }
    }

    #[test]
    fn test_evaporation() {
        let cfg = small_config();
        let mut graph = RouteGraph::new_empty(&cfg);
        graph.add_edge(0, 0, 1, 1.0).expect("add edge");
        graph.add_edge(0, 1, 2, 2.0).expect("add edge");

        let traces: Vec<EdgeTrace> = vec![];
        let ant_deltas: Vec<f32> = vec![];
        let pconfig = small_pheromone_config();

        let _stats = update_pheromones(&mut graph, &traces, &ant_deltas, &pconfig).expect("update");

        // φ *= (1 - 0.1) = 0.9 * original
        let flat0 = graph.idx(0, 0, 0);
        let flat1 = graph.idx(0, 1, 0);
        assert!(
            (graph.phi[flat0] - 0.9).abs() < 1e-5,
            "expected 0.9, got {}",
            graph.phi[flat0]
        );
        assert!(
            (graph.phi[flat1] - 1.8).abs() < 1e-5,
            "expected 1.8, got {}",
            graph.phi[flat1]
        );
    }

    #[test]
    fn test_phi_min_floor_applies_to_active_edges() {
        let cfg = small_config();
        let mut graph = RouteGraph::new_empty(&cfg);
        graph.add_edge(0, 0, 1, 0.02).expect("add edge");

        let traces: Vec<EdgeTrace> = vec![];
        let ant_deltas: Vec<f32> = vec![];
        let pconfig = PheromoneConfig {
            phi_min: 0.05,
            ..small_pheromone_config()
        };

        let _stats = update_pheromones(&mut graph, &traces, &ant_deltas, &pconfig).expect("update");

        let flat = graph.idx(0, 0, 0);
        assert!(
            (graph.phi[flat] - 0.05).abs() < 1e-6,
            "expected phi floor at 0.05, got {}",
            graph.phi[flat]
        );
    }

    #[test]
    fn test_elite_k_filters_positive_deposits() {
        let cfg = small_config();
        let mut graph = RouteGraph::new_empty(&cfg);
        graph.add_edge(0, 0, 1, 0.5).expect("add edge");
        graph.add_edge(0, 1, 2, 0.5).expect("add edge");

        let traces = vec![
            EdgeTrace {
                ant_id: 0,
                entries: vec![(0, 0, 0, 1.0)],
            },
            EdgeTrace {
                ant_id: 1,
                entries: vec![(0, 1, 0, 1.0)],
            },
        ];
        let ant_deltas = vec![1.0_f32, 0.8_f32];
        let pconfig = PheromoneConfig {
            elite_k: 1,
            phi_min: 0.0,
            ..small_pheromone_config()
        };

        let _stats = update_pheromones(&mut graph, &traces, &ant_deltas, &pconfig).expect("update");

        let flat_elite = graph.idx(0, 0, 0);
        let flat_non_elite = graph.idx(0, 1, 0);
        assert!(
            graph.phi[flat_elite] > graph.phi[flat_non_elite],
            "elite edge should have higher phi: elite={}, non_elite={}",
            graph.phi[flat_elite],
            graph.phi[flat_non_elite]
        );
        assert!(
            (graph.phi[flat_non_elite] - 0.45).abs() < 1e-4,
            "non-elite edge should only evaporate, got {}",
            graph.phi[flat_non_elite]
        );
    }

    #[test]
    fn test_non_elite_negative_ant_still_adds_taint() {
        let cfg = small_config();
        let mut graph = RouteGraph::new_empty(&cfg);
        graph.add_edge(0, 0, 1, 0.5).expect("add edge");
        graph.add_edge(0, 1, 2, 0.5).expect("add edge");

        let traces = vec![
            EdgeTrace {
                ant_id: 0,
                entries: vec![(0, 0, 0, 1.0)],
            },
            EdgeTrace {
                ant_id: 1,
                entries: vec![(0, 1, 0, 1.0)],
            },
        ];
        let ant_deltas = vec![1.0_f32, -1.0_f32];
        let pconfig = PheromoneConfig {
            elite_k: 1,
            phi_min: 0.0,
            ..small_pheromone_config()
        };

        let _stats = update_pheromones(&mut graph, &traces, &ant_deltas, &pconfig).expect("update");

        let flat_non_elite = graph.idx(0, 1, 0);
        assert!(
            graph.taint[flat_non_elite] > 0.0,
            "negative non-elite ant should still taint edge, got {}",
            graph.taint[flat_non_elite]
        );
    }

    #[test]
    fn test_elite_k_zero_disables_filtering() {
        let cfg = small_config();
        let mut graph = RouteGraph::new_empty(&cfg);
        graph.add_edge(0, 0, 1, 0.5).expect("add edge");
        graph.add_edge(0, 1, 2, 0.5).expect("add edge");

        let traces = vec![
            EdgeTrace {
                ant_id: 0,
                entries: vec![(0, 0, 0, 1.0)],
            },
            EdgeTrace {
                ant_id: 1,
                entries: vec![(0, 1, 0, 1.0)],
            },
        ];
        let ant_deltas = vec![1.0_f32, 0.8_f32];
        let pconfig = PheromoneConfig {
            elite_k: 0,
            phi_min: 0.0,
            ..small_pheromone_config()
        };

        let _stats = update_pheromones(&mut graph, &traces, &ant_deltas, &pconfig).expect("update");

        let flat_a = graph.idx(0, 0, 0);
        let flat_b = graph.idx(0, 1, 0);
        assert!(
            graph.phi[flat_a] > 0.45 && graph.phi[flat_b] > 0.45,
            "both ants should deposit when elite_k=0, got {}, {}",
            graph.phi[flat_a],
            graph.phi[flat_b]
        );
    }

    #[test]
    fn test_bounded_deposit_uses_tanh() {
        let cfg = small_config();
        let mut graph = RouteGraph::new_empty(&cfg);
        graph.add_edge(0, 0, 1, 0.5).expect("add edge");

        // Ant 0 has a large positive delta → deposit should be tanh-bounded.
        let traces = vec![EdgeTrace {
            ant_id: 0,
            entries: vec![(0, 0, 0, 1.0)],
        }];
        let ant_deltas = vec![100.0_f32]; // Very large delta.
        let pconfig = small_pheromone_config();

        let _stats = update_pheromones(&mut graph, &traces, &ant_deltas, &pconfig).expect("update");

        let flat = graph.idx(0, 0, 0);
        // After evaporation: 0.5 * 0.9 = 0.45
        // Deposit: 0.5 * tanh(100.0) ≈ 0.5 * 1.0 = 0.5
        // Total ≈ 0.95
        // The key test: deposit must be bounded ≤ η (0.5) regardless of delta.
        let phi = graph.phi[flat];
        assert!(phi < 1.0, "phi should be bounded by tanh, got {phi}");
        assert!(phi > 0.4, "phi should include deposit, got {phi}");
    }

    #[test]
    fn test_deposit_tanh_bounding_small_delta() {
        let cfg = small_config();
        let mut graph = RouteGraph::new_empty(&cfg);
        graph.add_edge(0, 0, 1, 0.5).expect("add edge");

        // Small delta → tanh(0.1) ≈ 0.0997
        let traces = vec![EdgeTrace {
            ant_id: 0,
            entries: vec![(0, 0, 0, 1.0)],
        }];
        let ant_deltas = vec![0.1_f32];
        let pconfig = small_pheromone_config();

        let _stats = update_pheromones(&mut graph, &traces, &ant_deltas, &pconfig).expect("update");

        let flat = graph.idx(0, 0, 0);
        // After evap: 0.45, deposit: 0.5 * tanh(0.1) ≈ 0.5 * 0.0997 ≈ 0.0498
        let expected = 0.45 + 0.5 * (0.1_f32).tanh();
        assert!(
            (graph.phi[flat] - expected).abs() < 1e-4,
            "expected {expected}, got {}",
            graph.phi[flat]
        );
    }

    #[test]
    fn test_log_deposit_small_delta() {
        let cfg = small_config();
        let mut graph = RouteGraph::new_empty(&cfg);
        graph.add_edge(0, 0, 1, 0.5).expect("add edge");

        let traces = vec![EdgeTrace {
            ant_id: 0,
            entries: vec![(0, 0, 0, 1.0)],
        }];
        let ant_deltas = vec![0.1_f32];
        let mut pconfig = small_pheromone_config();
        pconfig.use_log_deposit = true;

        let _stats = update_pheromones(&mut graph, &traces, &ant_deltas, &pconfig).expect("update");

        let flat = graph.idx(0, 0, 0);
        // After evap: 0.45, deposit: 0.5 * ln(1 + 0.1) ≈ 0.5 * 0.0953 ≈ 0.0477
        let expected = 0.45 + 0.5 * (1.0 + 0.1_f32).ln();
        assert!(
            (graph.phi[flat] - expected).abs() < 1e-4,
            "expected {expected}, got {}",
            graph.phi[flat]
        );
    }

    #[test]
    fn test_log_deposit_large_delta_no_saturation() {
        let cfg = small_config();
        let mut graph = RouteGraph::new_empty(&cfg);
        graph.add_edge(0, 0, 1, 0.5).expect("add edge");

        let traces = vec![EdgeTrace {
            ant_id: 0,
            entries: vec![(0, 0, 0, 1.0)],
        }];
        let ant_deltas = vec![100.0_f32];
        let mut pconfig = small_pheromone_config();
        pconfig.use_log_deposit = true;

        let _stats = update_pheromones(&mut graph, &traces, &ant_deltas, &pconfig).expect("update");

        let flat = graph.idx(0, 0, 0);
        // log1p never saturates to 1 — unlike tanh(100) ≈ 1.0, log(101) ≈ 4.62
        let deposit_base = (1.0 + 100.0_f32).ln();
        assert!(
            deposit_base > 1.0,
            "log deposit should exceed 1.0 for large delta, got {}",
            deposit_base
        );
        // phi should be > evap(0.5) + eta*1.0 = 0.45 + 0.5 = 0.95
        assert!(
            graph.phi[flat] > 0.95,
            "log deposit for large delta should exceed tanh-equivalent, got {}",
            graph.phi[flat]
        );
    }

    #[test]
    fn test_taint_clamping() {
        let cfg = small_config();
        let mut graph = RouteGraph::new_empty(&cfg);
        graph.add_edge(0, 0, 1, 1.0).expect("add edge");

        // Ant with very negative delta → large taint deposit.
        let traces = vec![EdgeTrace {
            ant_id: 0,
            entries: vec![(0, 0, 0, 1.0)],
        }];
        let ant_deltas = vec![-100.0_f32]; // Very negative.
        let pconfig = small_pheromone_config();

        let _stats = update_pheromones(&mut graph, &traces, &ant_deltas, &pconfig).expect("update");

        let flat = graph.idx(0, 0, 0);
        // Taint should be clamped to τ_max = 5.0.
        assert!(
            graph.taint[flat] <= pconfig.taint_max,
            "taint {} exceeds max {}",
            graph.taint[flat],
            pconfig.taint_max
        );
        assert!(
            graph.taint[flat] >= 0.0,
            "taint {} is negative",
            graph.taint[flat]
        );
    }

    #[test]
    fn test_taint_decay() {
        let cfg = small_config();
        let mut graph = RouteGraph::new_empty(&cfg);
        graph.add_edge(0, 0, 1, 1.0).expect("add edge");

        // Set initial taint.
        let flat = graph.idx(0, 0, 0);
        graph.taint[flat] = 3.0;

        let traces: Vec<EdgeTrace> = vec![];
        let ant_deltas: Vec<f32> = vec![];
        let pconfig = small_pheromone_config();

        let _stats = update_pheromones(&mut graph, &traces, &ant_deltas, &pconfig).expect("update");

        // Taint decay: 3.0 * (1 - 0.05) = 2.85
        assert!(
            (graph.taint[flat] - 2.85).abs() < 1e-4,
            "expected 2.85, got {}",
            graph.taint[flat]
        );
    }

    #[test]
    fn test_age_increment() {
        let cfg = small_config();
        let mut graph = RouteGraph::new_empty(&cfg);
        graph.add_edge(0, 0, 1, 1.0).expect("add edge");

        let flat = graph.idx(0, 0, 0);
        assert_eq!(graph.age[flat], 0);

        let traces: Vec<EdgeTrace> = vec![];
        let ant_deltas: Vec<f32> = vec![];
        let pconfig = small_pheromone_config();

        update_pheromones(&mut graph, &traces, &ant_deltas, &pconfig).expect("update");
        assert_eq!(graph.age[flat], 1);

        update_pheromones(&mut graph, &traces, &ant_deltas, &pconfig).expect("update");
        assert_eq!(graph.age[flat], 2);
    }

    #[test]
    fn test_phi_non_negative() {
        let cfg = small_config();
        let mut graph = RouteGraph::new_empty(&cfg);
        graph.add_edge(0, 0, 1, 0.001).expect("add edge");

        // Many evaporation steps should not make phi negative.
        let traces: Vec<EdgeTrace> = vec![];
        let ant_deltas: Vec<f32> = vec![];
        let pconfig = small_pheromone_config();

        for _ in 0..100 {
            update_pheromones(&mut graph, &traces, &ant_deltas, &pconfig).expect("update");
        }

        let flat = graph.idx(0, 0, 0);
        assert!(
            graph.phi[flat] >= 0.0,
            "phi should be non-negative, got {}",
            graph.phi[flat]
        );
    }

    #[test]
    fn test_pruning_by_score() {
        let cfg = small_config();
        let mut graph = RouteGraph::new_empty(&cfg);
        graph.add_edge(0, 0, 1, 0.1).expect("add edge");
        graph.add_edge(0, 0, 2, 2.0).expect("add edge");

        // Give first edge high taint → low score.
        let flat0 = graph.idx(0, 0, 0);
        graph.taint[flat0] = 4.0; // score = 0.1 - 1.0*4.0 = -3.9

        let pruned = prune_edges(&mut graph, -1.0, 1000, 1.0);
        assert_eq!(pruned, 1, "should prune 1 weak edge");
        assert_eq!(graph.edge_count(0, 0), 1, "should have 1 edge left");
    }

    #[test]
    fn test_pruning_by_age() {
        let cfg = small_config();
        let mut graph = RouteGraph::new_empty(&cfg);
        graph.add_edge(0, 0, 1, 1.0).expect("add edge");

        // Set old age.
        let flat = graph.idx(0, 0, 0);
        graph.age[flat] = 1001;

        let pruned = prune_edges(&mut graph, -1.0, 1000, 1.0);
        assert_eq!(pruned, 1, "should prune 1 old edge");
        assert_eq!(graph.edge_count(0, 0), 0, "should have no edges");
    }

    #[test]
    fn test_pruning_preserves_good_edges() {
        let cfg = small_config();
        let mut graph = RouteGraph::new_empty(&cfg);
        graph.add_edge(0, 0, 1, 5.0).expect("add edge");
        graph.add_edge(0, 0, 2, 3.0).expect("add edge");

        let pruned = prune_edges(&mut graph, -1.0, 1000, 1.0);
        assert_eq!(pruned, 0, "should not prune good edges");
        assert_eq!(graph.edge_count(0, 0), 2);
    }

    #[test]
    fn test_build_edge_traces() {
        let proposals = vec![
            SimpleEditProposal {
                position: 0,
                token: 5,
                predicted_gain: 0.5,
                ant_id: 0,
            },
            SimpleEditProposal {
                position: 1,
                token: 7,
                predicted_gain: 0.3,
                ant_id: 1,
            },
            SimpleEditProposal {
                position: 2,
                token: 3,
                predicted_gain: 0.8,
                ant_id: 0,
            },
        ];

        // edge_weights: [B=1, L=4, Emax=3]
        let mut edge_weights = vec![0.0_f32; 4 * 3];
        // Position 0: slot 0 has weight 0.6, slot 1 has weight 0.4
        edge_weights[0] = 0.6;
        edge_weights[1] = 0.4;
        // Position 1: slot 0 has weight 1.0
        edge_weights[3] = 1.0;
        // Position 2: slot 0 has weight 0.5, slot 2 has weight 0.5
        edge_weights[6] = 0.5;
        edge_weights[8] = 0.5;

        let traces = build_edge_traces(&proposals, &edge_weights, 1, 4, 3);

        // Should have 2 traces (ant 0 and ant 1).
        assert_eq!(traces.len(), 2);

        // Ant 0 should have entries from positions 0 and 2.
        let ant0_trace = traces.iter().find(|t| t.ant_id == 0).expect("ant 0 trace");
        assert_eq!(ant0_trace.entries.len(), 4); // 2 from pos 0, 2 from pos 2

        // Ant 1 should have entries from position 1.
        let ant1_trace = traces.iter().find(|t| t.ant_id == 1).expect("ant 1 trace");
        assert_eq!(ant1_trace.entries.len(), 1); // 1 from pos 1
    }

    #[test]
    fn test_stats_computed_correctly() {
        let cfg = small_config();
        let mut graph = RouteGraph::new_empty(&cfg);
        graph.add_edge(0, 0, 1, 2.0).expect("add edge");
        graph.add_edge(0, 1, 2, 4.0).expect("add edge");

        // Set some taint.
        let flat0 = graph.idx(0, 0, 0);
        graph.taint[flat0] = 1.0;

        let stats = compute_stats(&graph);
        assert!(
            (stats.mean_phi - 3.0).abs() < 1e-5,
            "mean_phi should be 3.0"
        );
        assert!((stats.max_phi - 4.0).abs() < 1e-5, "max_phi should be 4.0");
        assert!(
            (stats.mean_taint - 0.5).abs() < 1e-5,
            "mean_taint should be 0.5"
        );
        assert_eq!(stats.tainted_count, 1);
    }

    #[test]
    fn test_negative_delta_no_deposit() {
        let cfg = small_config();
        let mut graph = RouteGraph::new_empty(&cfg);
        graph.add_edge(0, 0, 1, 1.0).expect("add edge");

        // Negative delta → no deposit, only taint.
        let traces = vec![EdgeTrace {
            ant_id: 0,
            entries: vec![(0, 0, 0, 1.0)],
        }];
        let ant_deltas = vec![-0.5_f32];
        let pconfig = small_pheromone_config();

        update_pheromones(&mut graph, &traces, &ant_deltas, &pconfig).expect("update");

        let flat = graph.idx(0, 0, 0);
        // After evap: 1.0 * 0.9 = 0.9
        // Deposit: η * tanh(max(-0.5, 0)) = η * tanh(0) = 0
        // So phi should be exactly 0.9.
        assert!(
            (graph.phi[flat] - 0.9).abs() < 1e-5,
            "negative delta should not deposit, got phi={}",
            graph.phi[flat]
        );

        // Taint should be increased: ζ * max(0.5, 0) = 0.3 * 0.5 = 0.15
        // Then decayed: 0.15 * (1 - 0.05) = 0.1425
        let expected_taint = 0.3 * 0.5 * (1.0 - 0.05);
        assert!(
            (graph.taint[flat] - expected_taint).abs() < 1e-4,
            "expected taint {expected_taint}, got {}",
            graph.taint[flat]
        );
    }

    #[test]
    fn test_full_update_cycle() {
        let cfg = ErmConfig {
            batch_size: 2,
            seq_len: 4,
            emax: 3,
            ..ErmConfig::default()
        };
        let mut graph = RouteGraph::new_empty(&cfg);

        // Add some edges across both batch elements.
        graph.add_edge(0, 0, 1, 1.0).expect("add");
        graph.add_edge(0, 1, 2, 0.5).expect("add");
        graph.add_edge(1, 0, 3, 2.0).expect("add");
        graph.add_edge(1, 2, 1, 0.3).expect("add");

        let traces = vec![
            EdgeTrace {
                ant_id: 0,
                entries: vec![(0, 0, 0, 0.8), (1, 0, 0, 0.6)],
            },
            EdgeTrace {
                ant_id: 1,
                entries: vec![(0, 1, 0, 1.0)],
            },
        ];
        let ant_deltas = vec![0.5, -0.3];
        let pconfig = small_pheromone_config();

        let stats = update_pheromones(&mut graph, &traces, &ant_deltas, &pconfig).expect("update");

        // Verify stats are reasonable.
        assert!(stats.mean_phi > 0.0, "mean_phi should be positive");
        assert!(stats.max_phi > 0.0, "max_phi should be positive");
        assert!(stats.mean_phi.is_finite(), "mean_phi should be finite");
        assert!(stats.max_phi.is_finite(), "max_phi should be finite");
        assert!(stats.mean_taint.is_finite(), "mean_taint should be finite");

        // Verify all invariants hold.
        for flat in 0..graph.total_elements() {
            if graph.nbr_idx[flat] != EMPTY_SLOT {
                assert!(
                    graph.phi[flat] >= 0.0,
                    "phi must be >= 0, got {}",
                    graph.phi[flat]
                );
                assert!(
                    graph.phi[flat] <= pconfig.phi_max,
                    "phi must be <= phi_max, got {}",
                    graph.phi[flat]
                );
                assert!(
                    graph.taint[flat] >= 0.0,
                    "taint must be >= 0, got {}",
                    graph.taint[flat]
                );
                assert!(
                    graph.taint[flat] <= pconfig.taint_max,
                    "taint must be <= tau_max, got {}",
                    graph.taint[flat]
                );
            }
        }
    }

    #[test]
    fn test_diversity_pressure_penalizes_similar_sources() {
        let cfg = small_config();
        let mut graph = RouteGraph::new_empty(&cfg);
        // Destination 0 has two edges: from src=1 and src=2.
        graph.add_edge(0, 0, 1, 5.0).expect("add edge");
        graph.add_edge(0, 0, 2, 3.0).expect("add edge");

        let traces: Vec<EdgeTrace> = vec![];
        let ant_deltas: Vec<f32> = vec![];
        let pconfig = PheromoneConfig {
            diversity_threshold: 0.9,
            diversity_penalty: 0.8,
            ..small_pheromone_config()
        };

        // Build hidden states [B=1, L=4, d=4].
        // Make src=1 and src=2 have nearly identical hidden states (cosine ≈ 1.0).
        let d = 4;
        let mut hidden = vec![0.0_f32; 1 * 4 * d];
        // src=1: [1.0, 2.0, 3.0, 4.0]
        hidden[1 * d..2 * d].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        // src=2: [1.01, 2.01, 3.01, 4.01] — nearly identical
        hidden[2 * d..3 * d].copy_from_slice(&[1.01, 2.01, 3.01, 4.01]);

        let _stats = update_pheromones_with_diversity(
            &mut graph,
            &traces,
            &ant_deltas,
            &pconfig,
            None,
            &hidden,
            d,
        )
        .expect("update");

        // After evaporation: 5.0*0.9=4.5, 3.0*0.9=2.7.
        // Diversity: cosine(src1, src2) ≈ 1.0 > 0.9 → weaker (2.7) penalized by 0.8.
        let flat0 = graph.idx(0, 0, 0);
        let flat1 = graph.idx(0, 0, 1);

        let stronger = graph.phi[flat0];
        let weaker = graph.phi[flat1];

        // Stronger should be evaporated but NOT diversity-penalized: ~4.5.
        assert!(
            (stronger - 4.5).abs() < 0.1,
            "stronger edge should be ~4.5, got {stronger}"
        );
        // Weaker should be evaporated AND diversity-penalized: ~2.7 * 0.8 = ~2.16.
        assert!(
            (weaker - 2.16).abs() < 0.1,
            "weaker edge should be ~2.16, got {weaker}"
        );
    }

    #[test]
    fn test_diversity_pressure_skips_dissimilar_sources() {
        let cfg = small_config();
        let mut graph = RouteGraph::new_empty(&cfg);
        graph.add_edge(0, 0, 1, 5.0).expect("add edge");
        graph.add_edge(0, 0, 2, 3.0).expect("add edge");

        let traces: Vec<EdgeTrace> = vec![];
        let ant_deltas: Vec<f32> = vec![];
        let pconfig = PheromoneConfig {
            diversity_threshold: 0.9,
            diversity_penalty: 0.8,
            ..small_pheromone_config()
        };

        // Make src=1 and src=2 have orthogonal hidden states (cosine ≈ 0).
        let d = 4;
        let mut hidden = vec![0.0_f32; 1 * 4 * d];
        hidden[1 * d..2 * d].copy_from_slice(&[1.0, 0.0, 0.0, 0.0]);
        hidden[2 * d..3 * d].copy_from_slice(&[0.0, 1.0, 0.0, 0.0]);

        let _stats = update_pheromones_with_diversity(
            &mut graph,
            &traces,
            &ant_deltas,
            &pconfig,
            None,
            &hidden,
            d,
        )
        .expect("update");

        // No diversity penalty — only evaporation.
        let flat0 = graph.idx(0, 0, 0);
        let flat1 = graph.idx(0, 0, 1);
        assert!(
            (graph.phi[flat0] - 4.5).abs() < 0.01,
            "no penalty expected, got {}",
            graph.phi[flat0]
        );
        assert!(
            (graph.phi[flat1] - 2.7).abs() < 0.01,
            "no penalty expected, got {}",
            graph.phi[flat1]
        );
    }

    // ── Per-position credit assignment tests ──────────────────────────────────

    #[test]
    fn test_position_credit_deposits_per_position() {
        // Two edges at different destinations, with different position deltas.
        // Edge at dst=0 has high position delta, edge at dst=1 has zero.
        let cfg = ErmConfig {
            batch_size: 1,
            seq_len: 4,
            emax: 3,
            ..ErmConfig::default()
        };
        let mut graph = RouteGraph::new_empty(&cfg);
        graph.add_edge(0, 0, 1, 0.5).expect("add edge"); // edge to dst=0
        graph.add_edge(0, 1, 2, 0.5).expect("add edge"); // edge to dst=1

        // Both edges used by ant 0 — same ant, but different destinations.
        let traces = vec![EdgeTrace {
            ant_id: 0,
            entries: vec![(0, 0, 0, 1.0), (0, 1, 0, 1.0)],
        }];

        // Ant delta = 2.0 (aggregate), but position deltas differ.
        let ant_deltas = vec![2.0_f32];
        // Position 0 improved a lot (delta=3.0), position 1 did not (delta=0.0).
        let position_deltas = vec![3.0_f32, 0.0, 0.0, 0.0]; // [B*L] = [1*4]
        let pconfig = small_pheromone_config();
        let d = 4;
        let hidden = vec![0.0_f32; 1 * 4 * d];

        let _stats = update_pheromones_with_position_credit(
            &mut graph,
            &traces,
            &ant_deltas,
            &pconfig,
            None,
            &hidden,
            d,
            &position_deltas,
        )
        .expect("update");

        let flat0 = graph.idx(0, 0, 0); // edge to dst=0 (position delta=3.0)
        let flat1 = graph.idx(0, 1, 0); // edge to dst=1 (position delta=0.0)

        // Edge at dst=0 should have HIGHER phi than edge at dst=1.
        // After evap: 0.5 * 0.9 = 0.45 for both.
        // Deposit at dst=0: η * tanh(3.0) ≈ 0.5 * 0.9951 = ~0.498 → phi ≈ 0.948
        // Deposit at dst=1: η * tanh(0.0) = 0.5 * 0.0 = 0.0 → phi ≈ 0.45
        assert!(
            graph.phi[flat0] > graph.phi[flat1],
            "position with high delta ({}) should have more phi than zero-delta ({})",
            graph.phi[flat0],
            graph.phi[flat1]
        );
        assert!(
            graph.phi[flat0] > 0.9,
            "high-delta position should get substantial deposit, got {}",
            graph.phi[flat0]
        );
        assert!(
            (graph.phi[flat1] - 0.45).abs() < 0.01,
            "zero-delta position should only be evaporated, got {}",
            graph.phi[flat1]
        );
    }

    #[test]
    fn test_position_credit_taint_still_uses_ant_delta() {
        // Negative ant delta should still taint all edges regardless of position deltas.
        let cfg = small_config();
        let mut graph = RouteGraph::new_empty(&cfg);
        graph.add_edge(0, 0, 1, 1.0).expect("add edge");

        let traces = vec![EdgeTrace {
            ant_id: 0,
            entries: vec![(0, 0, 0, 1.0)],
        }];

        // Negative ant delta → taint should be applied.
        let ant_deltas = vec![-0.5_f32];
        // Position delta = 0 (no improvement at this position).
        let position_deltas = vec![0.0_f32; 4]; // [B*L] = [1*4]
        let pconfig = small_pheromone_config();
        let d = 4;
        let hidden = vec![0.0_f32; 1 * 4 * d];

        let _stats = update_pheromones_with_position_credit(
            &mut graph,
            &traces,
            &ant_deltas,
            &pconfig,
            None,
            &hidden,
            d,
            &position_deltas,
        )
        .expect("update");

        let flat = graph.idx(0, 0, 0);
        // Taint should be: ζ * max(0.5, 0) = 0.3 * 0.5 = 0.15
        // Then decayed: 0.15 * (1 - 0.05) = 0.1425
        let expected_taint = 0.3 * 0.5 * (1.0 - 0.05);
        assert!(
            (graph.taint[flat] - expected_taint).abs() < 1e-4,
            "taint should use ant delta, expected {expected_taint}, got {}",
            graph.taint[flat]
        );
    }

    #[test]
    fn test_position_credit_matches_ant_when_uniform() {
        // When all positions have the same delta as the ant aggregate,
        // per-position credit should produce the same result as per-ant.
        let cfg = small_config();

        let traces = vec![EdgeTrace {
            ant_id: 0,
            entries: vec![(0, 0, 0, 1.0)],
        }];
        let ant_deltas = vec![1.0_f32];
        let pconfig = small_pheromone_config();

        // Graph A: per-ant (no position deltas).
        let mut graph_a = RouteGraph::new_empty(&cfg);
        graph_a.add_edge(0, 0, 1, 1.0).expect("add edge");
        let _sa = update_pheromones(&mut graph_a, &traces, &ant_deltas, &pconfig).expect("update");

        // Graph B: per-position with uniform delta = 1.0 everywhere.
        let mut graph_b = RouteGraph::new_empty(&cfg);
        graph_b.add_edge(0, 0, 1, 1.0).expect("add edge");
        let position_deltas = vec![1.0_f32; 4]; // all positions = 1.0
        let d = 4;
        let hidden = vec![0.0_f32; 1 * 4 * d];
        let _sb = update_pheromones_with_position_credit(
            &mut graph_b,
            &traces,
            &ant_deltas,
            &pconfig,
            None,
            &hidden,
            d,
            &position_deltas,
        )
        .expect("update");

        let flat = graph_a.idx(0, 0, 0);
        assert!(
            (graph_a.phi[flat] - graph_b.phi[flat]).abs() < 1e-5,
            "uniform position delta should match ant delta: a={}, b={}",
            graph_a.phi[flat],
            graph_b.phi[flat]
        );
    }

    // ── Pheromone rescaling (MuonClip analog) tests ───────────────────────

    #[test]
    fn test_rescale_reduces_max_phi() {
        let cfg = small_config();
        let mut graph = RouteGraph::new_empty(&cfg);
        graph.add_edge(0, 0, 1, 9.0).expect("add edge");
        graph.add_edge(0, 1, 2, 3.0).expect("add edge");

        // Threshold = 4.0, max = 9.0 → should rescale.
        let rescaled = pheromone_rescale(&mut graph, 4.0);
        assert!(rescaled, "should rescale when max > threshold");

        let flat0 = graph.idx(0, 0, 0);
        let flat1 = graph.idx(0, 1, 0);

        // gamma = sqrt(4.0 / 9.0) = 2/3 ≈ 0.6667
        // phi[0] = 9.0 * 0.6667 ≈ 6.0
        // phi[1] = 3.0 * 0.6667 ≈ 2.0
        let gamma = (4.0_f32 / 9.0).sqrt();
        assert!(
            (graph.phi[flat0] - 9.0 * gamma).abs() < 1e-4,
            "expected {}, got {}",
            9.0 * gamma,
            graph.phi[flat0]
        );
        assert!(
            (graph.phi[flat1] - 3.0 * gamma).abs() < 1e-4,
            "expected {}, got {}",
            3.0 * gamma,
            graph.phi[flat1]
        );

        // sqrt rescaling reduces max but doesn't clamp to threshold in one pass.
        // 9.0 * sqrt(4/9) = 9 * 2/3 = 6.0, which is less than original 9.0
        assert!(
            graph.phi[flat0] < 9.0,
            "max phi should be reduced after rescale, got {}",
            graph.phi[flat0]
        );
    }

    #[test]
    fn test_rescale_preserves_relative_order() {
        let cfg = small_config();
        let mut graph = RouteGraph::new_empty(&cfg);
        graph.add_edge(0, 0, 1, 8.0).expect("add edge");
        graph.add_edge(0, 0, 2, 4.0).expect("add edge");
        graph.add_edge(0, 1, 3, 2.0).expect("add edge");

        pheromone_rescale(&mut graph, 5.0);

        let flat0 = graph.idx(0, 0, 0);
        let flat1 = graph.idx(0, 0, 1);
        let flat2 = graph.idx(0, 1, 0);

        assert!(
            graph.phi[flat0] > graph.phi[flat1],
            "relative order should be preserved"
        );
        assert!(
            graph.phi[flat1] > graph.phi[flat2],
            "relative order should be preserved"
        );
    }

    #[test]
    fn test_rescale_noop_below_threshold() {
        let cfg = small_config();
        let mut graph = RouteGraph::new_empty(&cfg);
        graph.add_edge(0, 0, 1, 3.0).expect("add edge");
        graph.add_edge(0, 1, 2, 2.0).expect("add edge");

        let rescaled = pheromone_rescale(&mut graph, 5.0);
        assert!(!rescaled, "should not rescale when max < threshold");

        // Values should be unchanged.
        let flat0 = graph.idx(0, 0, 0);
        let flat1 = graph.idx(0, 1, 0);
        assert!((graph.phi[flat0] - 3.0).abs() < 1e-6);
        assert!((graph.phi[flat1] - 2.0).abs() < 1e-6);
    }
}
