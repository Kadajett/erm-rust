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
/// 2. **Deposit**: `φ += η * tanh(max(Δ_k, 0.0))` for edges used by improving ants
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
    let total = graph.total_elements();
    let rho = config.evaporation_rate;
    let eta = config.deposit_rate;
    let zeta = config.taint_rate;
    let rho_tau = config.taint_decay;
    let tau_max = config.taint_max;
    let phi_max = config.phi_max;

    // Step 1: Evaporation — φ *= (1 - ρ) for all valid edges.
    for flat in 0..total {
        if graph.nbr_idx[flat] != EMPTY_SLOT {
            graph.phi[flat] *= 1.0 - rho;
        }
    }

    // Step 2: Deposit — φ += η * tanh(max(Δ_k, 0.0)) for edges used by ant k.
    // Step 3: Taint — τ += ζ * max(-Δ_k, 0.0) for edges used by ant k.
    for trace in traces {
        let ant_id = trace.ant_id;
        let delta = if ant_id < ant_deltas.len() {
            ant_deltas[ant_id]
        } else {
            0.0
        };

        let positive_delta = delta.max(0.0);
        let negative_delta = (-delta).max(0.0);

        // tanh-bounded deposit.
        let deposit = eta * positive_delta.tanh();
        let taint_deposit = zeta * negative_delta;

        for &(_, _, edge_idx, weight) in &trace.entries {
            // Use a flat index directly from the trace.
            // The trace stores (batch, dest, edge_slot, weight).
            // We need to convert to flat index.
            let &(b, dst, e, _) = &(
                trace.entries.iter().find(|x| x.2 == edge_idx).map(|x| x.0),
                trace.entries.iter().find(|x| x.2 == edge_idx).map(|x| x.1),
                edge_idx,
                weight,
            );
            // Actually, let's just iterate directly over the entries.
            let _ = (b, dst, e); // suppress unused
        }

        // Re-iterate cleanly.
        for &(b, dst, e, _weight) in &trace.entries {
            if b >= graph.batch_size || dst >= graph.seq_len || e >= graph.emax {
                continue;
            }
            let flat = graph.idx(b, dst, e);
            if graph.nbr_idx[flat] == EMPTY_SLOT {
                continue;
            }

            // Deposit pheromone (tanh-bounded).
            graph.phi[flat] += deposit;

            // Taint deposit.
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

            // Enforce φ ∈ [0, φ_max].
            graph.phi[flat] = graph.phi[flat].clamp(0.0, phi_max);

            // Enforce τ ∈ [0, τ_max].
            graph.taint[flat] = graph.taint[flat].clamp(0.0, tau_max);
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
            prune_min_score: -1.0,
            prune_max_age: 1000,
            route_lambda: 1.0,
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
}
