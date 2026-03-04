//! Graph-health telemetry for route-graph diagnostics.
//!
//! Computes scalar metrics from the dense-neighbor route graph so training can
//! detect collapse/stagnation early.

use std::collections::HashSet;

use erm_core::graph::{RouteGraph, EMPTY_SLOT};

/// Scalar graph-health metrics emitted to `metrics.jsonl`.
#[derive(Debug, Clone, Copy)]
pub struct GraphHealthMetrics {
    /// Number of active edges (`nbr_idx != EMPTY_SLOT`).
    pub active_edges: usize,
    /// Number of active leader-introduced edges.
    pub leader_edges: usize,
    /// Fraction `leader_edges / active_edges`.
    pub leader_edge_fraction: f32,
    /// Mean edge age across active edges.
    pub mean_age: f32,
    /// Maximum edge age across active edges.
    pub max_age: u32,
    /// Fraction of active edges at (or near) `phi_max`.
    pub phi_clamped_fraction: f32,
    /// Fraction of active edges at (or near) `taint_max`.
    pub taint_clamped_fraction: f32,
    /// Mean per-destination entropy of route edge weights.
    pub edge_weight_entropy_mean: f32,
    /// Mean per-destination top-1 edge share.
    pub top1_edge_share_mean: f32,
    /// Fraction of previous-step leader edges still present this step.
    pub leader_edge_survival_rate: f32,
}

impl Default for GraphHealthMetrics {
    fn default() -> Self {
        Self {
            active_edges: 0,
            leader_edges: 0,
            leader_edge_fraction: 0.0,
            mean_age: 0.0,
            max_age: 0,
            phi_clamped_fraction: 0.0,
            taint_clamped_fraction: 0.0,
            edge_weight_entropy_mean: 0.0,
            top1_edge_share_mean: 0.0,
            leader_edge_survival_rate: 0.0,
        }
    }
}

/// Compute graph-health telemetry from a route graph.
///
/// Inputs:
/// - `graph`: route graph arrays `[B, L, Emax]`
///
/// Output:
/// - `(metrics, current_leader_edges)` where `current_leader_edges` can be fed
///   into the next call as `prev_leader_edges` to compute survival.
#[must_use]
pub fn compute_graph_health_metrics(
    graph: &RouteGraph,
    route_lambda: f32,
    route_mu: f32,
    phi_max: f32,
    taint_max: f32,
    prev_leader_edges: Option<&HashSet<(usize, usize, usize)>>,
) -> (GraphHealthMetrics, HashSet<(usize, usize, usize)>) {
    let eps = 1e-6_f32;
    let mut metrics = GraphHealthMetrics::default();
    let mut current_leader_edges: HashSet<(usize, usize, usize)> = HashSet::new();

    let mut sum_age = 0.0_f32;
    let mut phi_clamped_count = 0_usize;
    let mut taint_clamped_count = 0_usize;

    let mut entropy_sum = 0.0_f32;
    let mut top1_sum = 0.0_f32;
    let mut destinations_with_edges = 0_usize;

    for b in 0..graph.batch_size {
        for dst in 0..graph.seq_len {
            let mut raw_scores: Vec<f32> = Vec::with_capacity(graph.emax);

            for e in 0..graph.emax {
                let flat = graph.idx(b, dst, e);
                let src = graph.nbr_idx[flat];
                if src == EMPTY_SLOT {
                    continue;
                }

                metrics.active_edges += 1;
                sum_age += graph.age[flat] as f32;
                if graph.age[flat] >= 0 {
                    metrics.max_age = metrics.max_age.max(graph.age[flat] as u32);
                }

                if graph.leader_edge[flat] {
                    metrics.leader_edges += 1;
                    current_leader_edges.insert((b, dst, src as usize));
                }

                if graph.phi[flat] >= phi_max - eps {
                    phi_clamped_count += 1;
                }
                if graph.taint[flat] >= taint_max - eps {
                    taint_clamped_count += 1;
                }

                let raw = (graph.phi[flat] + 1e-8_f32).ln()
                    - route_lambda * graph.taint[flat]
                    - route_mu * graph.age[flat] as f32;
                raw_scores.push(raw);
            }

            if raw_scores.is_empty() {
                continue;
            }
            destinations_with_edges += 1;

            let max_raw = raw_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_scores: Vec<f32> = Vec::with_capacity(raw_scores.len());
            let mut sum_exp = 0.0_f32;
            for &raw in &raw_scores {
                let ex = (raw - max_raw).exp();
                exp_scores.push(ex);
                sum_exp += ex;
            }

            if sum_exp <= 0.0 {
                continue;
            }

            let mut entropy = 0.0_f32;
            let mut top1 = 0.0_f32;
            for ex in exp_scores {
                let p = ex / sum_exp;
                if p > top1 {
                    top1 = p;
                }
                if p > 0.0 {
                    entropy -= p * p.ln();
                }
            }

            entropy_sum += entropy;
            top1_sum += top1;
        }
    }

    if metrics.active_edges > 0 {
        metrics.leader_edge_fraction = metrics.leader_edges as f32 / metrics.active_edges as f32;
        metrics.mean_age = sum_age / metrics.active_edges as f32;
        metrics.phi_clamped_fraction = phi_clamped_count as f32 / metrics.active_edges as f32;
        metrics.taint_clamped_fraction = taint_clamped_count as f32 / metrics.active_edges as f32;
    }

    if destinations_with_edges > 0 {
        metrics.edge_weight_entropy_mean = entropy_sum / destinations_with_edges as f32;
        metrics.top1_edge_share_mean = top1_sum / destinations_with_edges as f32;
    }

    metrics.leader_edge_survival_rate = match prev_leader_edges {
        Some(prev) if !prev.is_empty() => {
            let survived = current_leader_edges
                .iter()
                .filter(|edge| prev.contains(*edge))
                .count();
            survived as f32 / prev.len() as f32
        }
        _ => 0.0,
    };

    (metrics, current_leader_edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    use erm_core::config::ErmConfig;
    use erm_core::graph::RouteGraph;

    fn config() -> ErmConfig {
        ErmConfig {
            batch_size: 1,
            seq_len: 4,
            emax: 3,
            route_lambda: 1.0,
            route_mu: 0.01,
            phi_max: 10.0,
            taint_max: 5.0,
            ..ErmConfig::default()
        }
    }

    #[test]
    fn computes_basic_graph_health_metrics() {
        let cfg = config();
        let mut graph = RouteGraph::new_empty(&cfg);

        // Destination 0 has two incoming edges.
        let s0 = graph.add_edge_with_leader(0, 0, 1, 9.0, true).unwrap();
        let s1 = graph.add_edge_with_leader(0, 0, 2, 3.0, false).unwrap();
        // Destination 1 has one incoming edge.
        let s2 = graph.add_edge_with_leader(0, 1, 3, 1.0, true).unwrap();

        let f0 = graph.idx(0, 0, s0);
        graph.taint[f0] = cfg.taint_max;
        graph.age[f0] = 4;

        let f1 = graph.idx(0, 0, s1);
        graph.age[f1] = 2;

        let f2 = graph.idx(0, 1, s2);
        graph.phi[f2] = cfg.phi_max;
        graph.age[f2] = 1;

        let (metrics, leader_edges) = compute_graph_health_metrics(
            &graph,
            cfg.route_lambda,
            cfg.route_mu,
            cfg.phi_max,
            cfg.taint_max,
            None,
        );

        assert_eq!(metrics.active_edges, 3);
        assert_eq!(metrics.leader_edges, 2);
        assert!((metrics.leader_edge_fraction - (2.0 / 3.0)).abs() < 1e-6);
        assert_eq!(metrics.max_age, 4);
        assert!((metrics.mean_age - (7.0 / 3.0)).abs() < 1e-6);
        assert!((metrics.phi_clamped_fraction - (1.0 / 3.0)).abs() < 1e-6);
        assert!((metrics.taint_clamped_fraction - (1.0 / 3.0)).abs() < 1e-6);
        assert!(metrics.edge_weight_entropy_mean >= 0.0);
        assert!(metrics.top1_edge_share_mean > 0.0);
        assert!(metrics.top1_edge_share_mean <= 1.0);
        assert_eq!(metrics.leader_edge_survival_rate, 0.0);

        assert_eq!(leader_edges.len(), 2);
        assert!(leader_edges.contains(&(0, 0, 1)));
        assert!(leader_edges.contains(&(0, 1, 3)));
    }

    #[test]
    fn computes_leader_survival_rate() {
        let cfg = config();
        let mut graph = RouteGraph::new_empty(&cfg);
        graph.add_edge_with_leader(0, 0, 1, 1.0, true).unwrap();
        graph.add_edge_with_leader(0, 1, 2, 1.0, true).unwrap();

        let mut prev = HashSet::new();
        prev.insert((0, 0, 1));
        prev.insert((0, 1, 7)); // intentionally absent now

        let (metrics, _) = compute_graph_health_metrics(
            &graph,
            cfg.route_lambda,
            cfg.route_mu,
            cfg.phi_max,
            cfg.taint_max,
            Some(&prev),
        );

        assert!((metrics.leader_edge_survival_rate - 0.5).abs() < 1e-6);
    }
}
