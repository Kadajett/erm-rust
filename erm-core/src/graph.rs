//! Route graph in dense-neighbor format.
//!
//! The route graph `G` stores per-destination sparse edges. For each position
//! `(b, i)` in a batch, up to `Emax` incoming edges are stored.
//!
//! # Tensor shapes (default config)
//!
//! | Tensor     | Shape            | Dtype | Size    |
//! |-----------|------------------|-------|---------|
//! | `nbr_idx` | `[B, L, Emax]`   | i32   | 655 KB  |
//! | `phi`     | `[B, L, Emax]`   | f32   | 1.3 MB  |
//! | `taint`   | `[B, L, Emax]`   | f32   | 1.3 MB  |
//! | `age`     | `[B, L, Emax]`   | i32   | 655 KB  |
//!
//! Total ≈ 4 MB per batch — well within VRAM budget.
//!
//! # Invariants
//!
//! - `phi[b][i][e] >= 0` for all valid edges.
//! - `taint[b][i][e]` in `[0, tau_max]`.
//! - `nbr_idx[b][i][e] == -1` marks an empty slot.
//! - Valid `nbr_idx` values are in `[0, L)`.
//! - At most `Emax` active (non -1) edges per `(b, i)`.

use serde::{Deserialize, Serialize};

use crate::config::ErmConfig;
use crate::error::{ErmError, ErmResult};
use crate::types::{BatchIdx, EdgeSlot, PosIdx};

/// The EMPTY sentinel value for `nbr_idx` (no neighbor).
pub const EMPTY_SLOT: i32 = -1;

/// Dense-neighbor route graph.
///
/// All data is stored as flat `Vec`s with shape `[B, L, Emax]`, indexed
/// via the helper [`idx`] method.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteGraph {
    /// Batch size.
    pub batch_size: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Maximum edges per destination node.
    pub emax: usize,

    /// Neighbor source-position indices. Shape: `[B, L, Emax]`. `-1` = empty.
    pub nbr_idx: Vec<i32>,
    /// Pheromone strength. Shape: `[B, L, Emax]`. `>= 0`.
    pub phi: Vec<f32>,
    /// Taint level. Shape: `[B, L, Emax]`. In `[0, tau_max]`.
    pub taint: Vec<f32>,
    /// Edge age (steps since creation). Shape: `[B, L, Emax]`.
    pub age: Vec<i32>,
}

impl RouteGraph {
    /// Create an empty route graph.
    ///
    /// - `nbr_idx` initialized to [`EMPTY_SLOT`] (-1)
    /// - `phi` initialized to `phi_init` (small warm-start value)
    /// - `taint` initialized to `0.0`
    /// - `age` initialized to `0`
    #[must_use]
    pub fn new(config: &ErmConfig) -> Self {
        let b = config.batch_size;
        let l = config.seq_len;
        let e = config.emax;
        let total = b * l * e;

        Self {
            batch_size: b,
            seq_len: l,
            emax: e,
            nbr_idx: vec![EMPTY_SLOT; total],
            phi: vec![config.phi_init; total],
            taint: vec![0.0; total],
            age: vec![0; total],
        }
    }

    /// Total number of elements in each flat array.
    #[inline]
    #[must_use]
    pub fn total_elements(&self) -> usize {
        self.batch_size * self.seq_len * self.emax
    }

    /// Flat index into the `[B, L, Emax]` arrays.
    #[inline]
    #[must_use]
    pub fn idx(&self, b: BatchIdx, i: PosIdx, e: EdgeSlot) -> usize {
        debug_assert!(b < self.batch_size);
        debug_assert!(i < self.seq_len);
        debug_assert!(e < self.emax);
        (b * self.seq_len + i) * self.emax + e
    }

    /// Count active (non-empty) edges for destination `(b, i)`.
    #[must_use]
    pub fn edge_count(&self, b: BatchIdx, i: PosIdx) -> usize {
        (0..self.emax)
            .filter(|&e| self.nbr_idx[self.idx(b, i, e)] != EMPTY_SLOT)
            .count()
    }

    /// Find the first empty slot for destination `(b, i)`.
    ///
    /// Returns `None` if all `Emax` slots are occupied.
    #[must_use]
    pub fn first_empty_slot(&self, b: BatchIdx, i: PosIdx) -> Option<EdgeSlot> {
        (0..self.emax).find(|&e| self.nbr_idx[self.idx(b, i, e)] == EMPTY_SLOT)
    }

    /// Insert an edge `src → dst` at the first empty slot.
    ///
    /// # Errors
    ///
    /// Returns [`ErmError::GraphError`] if the destination's slots are full.
    pub fn add_edge(
        &mut self,
        b: BatchIdx,
        dst: PosIdx,
        src: PosIdx,
        phi_init: f32,
    ) -> ErmResult<EdgeSlot> {
        let slot = self
            .first_empty_slot(b, dst)
            .ok_or_else(|| ErmError::GraphError(format!("slots full at ({b}, {dst})")))?;

        let flat = self.idx(b, dst, slot);
        self.nbr_idx[flat] = src as i32;
        self.phi[flat] = phi_init;
        self.taint[flat] = 0.0;
        self.age[flat] = 0;

        Ok(slot)
    }

    /// Remove an edge by swap-removing: move the last valid slot into this slot.
    ///
    /// # Errors
    ///
    /// Returns [`ErmError::IndexOutOfBounds`] if the slot is already empty.
    pub fn remove_edge(&mut self, b: BatchIdx, dst: PosIdx, slot: EdgeSlot) -> ErmResult<()> {
        let flat = self.idx(b, dst, slot);
        if self.nbr_idx[flat] == EMPTY_SLOT {
            return Err(ErmError::IndexOutOfBounds(format!(
                "slot ({b}, {dst}, {slot}) is already empty"
            )));
        }

        // Find the last valid slot.
        let last_valid = (0..self.emax)
            .rev()
            .find(|&e| self.nbr_idx[self.idx(b, dst, e)] != EMPTY_SLOT);

        if let Some(last) = last_valid {
            if last != slot {
                let flat_last = self.idx(b, dst, last);
                self.nbr_idx[flat] = self.nbr_idx[flat_last];
                self.phi[flat] = self.phi[flat_last];
                self.taint[flat] = self.taint[flat_last];
                self.age[flat] = self.age[flat_last];
            }
            // Clear the (now-vacant) last slot.
            let flat_last = self.idx(b, dst, last);
            self.nbr_idx[flat_last] = EMPTY_SLOT;
            self.phi[flat_last] = 0.0;
            self.taint[flat_last] = 0.0;
            self.age[flat_last] = 0;
        }

        Ok(())
    }

    /// Prune the weakest edge at `(b, dst)` by the score `phi - lambda * taint`.
    ///
    /// Only prunes if all slots are full. Returns the removed neighbor index, or
    /// `None` if pruning was not needed.
    pub fn prune_weakest(
        &mut self,
        b: BatchIdx,
        dst: PosIdx,
        lambda: f32,
    ) -> ErmResult<Option<i32>> {
        if self.edge_count(b, dst) < self.emax {
            return Ok(None);
        }

        // Find the slot with the minimum score.
        let mut min_score = f32::INFINITY;
        let mut min_slot = 0;
        for e in 0..self.emax {
            let flat = self.idx(b, dst, e);
            if self.nbr_idx[flat] == EMPTY_SLOT {
                continue;
            }
            let score = self.phi[flat] - lambda * self.taint[flat];
            if score < min_score {
                min_score = score;
                min_slot = e;
            }
        }

        let removed_nbr = self.nbr_idx[self.idx(b, dst, min_slot)];
        self.remove_edge(b, dst, min_slot)?;
        Ok(Some(removed_nbr))
    }

    /// Compute the route-aggregate message for all positions in a batch.
    ///
    /// For each destination position `(b, i)`, aggregates incoming neighbour
    /// hidden states, weighted by a softmax over pheromone / taint / age scores:
    ///
    /// ```text
    /// w_raw[b,i,e] = log(φ[b,i,e] + ε) - λ · τ[b,i,e] - μ · age[b,i,e]
    /// w[b,i,:]     = softmax(w_raw[b,i,:])   (−∞ mask on EMPTY_SLOT entries)
    /// r[b,i,:]     = Σ_e  w[b,i,e] · h[b, nbr[b,i,e], :]
    /// ```
    ///
    /// EMPTY_SLOT neighbours receive zero weight and contribute zero to `r`.
    ///
    /// **Does not materialise `h_nbr [B, L, Emax, d]`** — iterates one
    /// destination at a time to keep peak memory at `O(B · Emax · d)`.
    ///
    /// # Arguments
    ///
    /// * `hidden` — source embeddings, **flat** layout `[B, L, d]`
    ///   (row-major: index = `(b * L + i) * d + k`).
    /// * `d` — hidden dimension.
    /// * `epsilon` — additive constant for `log(phi + eps)`.  Use `1e-6`.
    /// * `lambda` — taint penalty coefficient.
    /// * `mu` — age penalty coefficient.
    ///
    /// # Returns
    ///
    /// `(r, edge_weights)` where:
    ///
    /// * `r` — route messages, flat `[B, L, d]`.
    /// * `edge_weights` — per-edge softmax weights, flat `[B, L, Emax]`.
    ///   Empty slots have weight `0.0`.  Weights sum to `≈1.0` for every
    ///   `(b, i)` that has at least one valid neighbour; rows with no valid
    ///   neighbours are all zeros.
    ///
    /// # Errors
    ///
    /// Returns [`ErmError::ShapeMismatch`] if `hidden.len() != B * L * d`.
    ///
    /// # Shape reference
    ///
    /// | Tensor | Shape | Notes |
    /// |---|---|---|
    /// | `hidden` (input) | `[B, L, d]` | row-major flat |
    /// | `r` (output) | `[B, L, d]` | row-major flat |
    /// | `edge_weights` (output) | `[B, L, Emax]` | row-major flat |
    pub fn route_aggregate(
        &self,
        hidden: &[f32],
        d: usize,
        epsilon: f32,
        lambda: f32,
        mu: f32,
    ) -> ErmResult<(Vec<f32>, Vec<f32>)> {
        let b = self.batch_size;
        let l = self.seq_len;
        let e = self.emax;

        let expected_hidden = b * l * d;
        if hidden.len() != expected_hidden {
            return Err(ErmError::ShapeMismatch {
                expected: format!("[B={b}, L={l}, d={d}] = {expected_hidden}"),
                got: format!("{}", hidden.len()),
            });
        }

        let mut r = vec![0.0_f32; b * l * d];
        let mut edge_weights = vec![0.0_f32; b * l * e];

        for bi in 0..b {
            for i in 0..l {
                // --- Compute raw scores for each edge slot ---
                // w_raw[e] = log(phi + eps) - lambda * taint - mu * age
                // EMPTY_SLOT → -inf (masked out)
                let mut w_raw = [f32::NEG_INFINITY; 64]; // stack array; Emax ≤ 64
                let w_raw = &mut w_raw[..e];
                let mut any_valid = false;

                // `ei` indexes both `w_raw` and the graph arrays via `self.idx`
                // so an enumerate-iterator refactor would not reduce clarity.
                #[allow(clippy::needless_range_loop)]
                for ei in 0..e {
                    let flat = self.idx(bi, i, ei);
                    let nbr = self.nbr_idx[flat];
                    if nbr == EMPTY_SLOT {
                        w_raw[ei] = f32::NEG_INFINITY;
                    } else {
                        let phi_v = self.phi[flat];
                        let taint_v = self.taint[flat];
                        let age_v = self.age[flat] as f32;
                        w_raw[ei] = (phi_v + epsilon).ln() - lambda * taint_v - mu * age_v;
                        any_valid = true;
                    }
                }

                if !any_valid {
                    // All slots empty — r stays zero, weights stay zero.
                    continue;
                }

                // --- Numerically stable softmax over valid edges ---
                let max_w = w_raw
                    .iter()
                    .copied()
                    .filter(|v| v.is_finite())
                    .fold(f32::NEG_INFINITY, f32::max);

                let mut exp_sum = 0.0_f32;
                let mut exps = [0.0_f32; 64];
                let exps = &mut exps[..e];
                for ei in 0..e {
                    if w_raw[ei].is_finite() {
                        let v = (w_raw[ei] - max_w).exp();
                        exps[ei] = v;
                        exp_sum += v;
                    }
                }

                // Normalise → softmax weights
                let ew_base = (bi * l + i) * e;
                for ei in 0..e {
                    edge_weights[ew_base + ei] = if exp_sum > 0.0 {
                        exps[ei] / exp_sum
                    } else {
                        0.0
                    };
                }

                // --- Weighted sum of neighbour embeddings ---
                let r_base = (bi * l + i) * d;
                for ei in 0..e {
                    let w = edge_weights[ew_base + ei];
                    if w == 0.0 {
                        continue;
                    }
                    let flat = self.idx(bi, i, ei);
                    let nbr = self.nbr_idx[flat] as usize; // safe: w>0 ⇒ not EMPTY_SLOT
                    let h_base = (bi * l + nbr) * d;
                    for k in 0..d {
                        r[r_base + k] += w * hidden[h_base + k];
                    }
                }
            }
        }

        Ok((r, edge_weights))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ErmConfig;

    fn test_config() -> ErmConfig {
        ErmConfig {
            batch_size: 2,
            seq_len: 4,
            emax: 3,
            ..ErmConfig::default()
        }
    }

    /// Default RouteAggregate hyperparams used in tests.
    const EPS: f32 = 1e-6;
    const LAMBDA: f32 = 1.0;
    const MU: f32 = 0.01;

    #[test]
    fn test_init_shapes() {
        let cfg = test_config();
        let g = RouteGraph::new(&cfg);
        let expected = 2 * 4 * 3;
        assert_eq!(g.nbr_idx.len(), expected);
        assert_eq!(g.phi.len(), expected);
        assert_eq!(g.taint.len(), expected);
        assert_eq!(g.age.len(), expected);
    }

    #[test]
    fn test_init_values() {
        let cfg = test_config();
        let g = RouteGraph::new(&cfg);
        // All nbr_idx should be EMPTY_SLOT
        assert!(g.nbr_idx.iter().all(|&v| v == EMPTY_SLOT));
        // All phi should be phi_init
        assert!(g.phi.iter().all(|&v| (v - cfg.phi_init).abs() < 1e-9));
        // All taint = 0
        assert!(g.taint.iter().all(|&v| v == 0.0));
        // All age = 0
        assert!(g.age.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_add_and_count_edges() {
        let cfg = test_config();
        let mut g = RouteGraph::new(&cfg);

        // Initially empty.
        assert_eq!(g.edge_count(0, 0), 0);

        // Add three edges (fill all slots for (0, 0), emax=3).
        g.add_edge(0, 0, 1, 0.1).unwrap();
        g.add_edge(0, 0, 2, 0.2).unwrap();
        g.add_edge(0, 0, 3, 0.3).unwrap();
        assert_eq!(g.edge_count(0, 0), 3);

        // Fourth should fail.
        assert!(g.add_edge(0, 0, 3, 0.4).is_err());
    }

    #[test]
    fn test_remove_edge_swap() {
        let cfg = test_config();
        let mut g = RouteGraph::new(&cfg);

        g.add_edge(0, 1, 0, 1.0).unwrap(); // slot 0 → nbr 0
        g.add_edge(0, 1, 2, 2.0).unwrap(); // slot 1 → nbr 2
        g.add_edge(0, 1, 3, 3.0).unwrap(); // slot 2 → nbr 3

        // Remove slot 0 — should swap-remove: slot 2's data moves to slot 0.
        g.remove_edge(0, 1, 0).unwrap();
        assert_eq!(g.edge_count(0, 1), 2);

        // Slot 0 should now contain what was in slot 2 (nbr 3).
        let flat0 = g.idx(0, 1, 0);
        assert_eq!(g.nbr_idx[flat0], 3);

        // Slot 2 should be empty.
        let flat2 = g.idx(0, 1, 2);
        assert_eq!(g.nbr_idx[flat2], EMPTY_SLOT);
    }

    #[test]
    fn test_prune_weakest() {
        let cfg = test_config();
        let mut g = RouteGraph::new(&cfg);

        // Fill all slots for (0, 2) with varying phi and taint.
        g.add_edge(0, 2, 0, 1.0).unwrap();
        g.add_edge(0, 2, 1, 0.5).unwrap();
        g.add_edge(0, 2, 3, 2.0).unwrap();

        // Give slot 1 (nbr 1) high taint → worst score with λ=1.0.
        let flat1 = g.idx(0, 2, 1);
        g.taint[flat1] = 4.0; // score = 0.5 - 1.0*4.0 = -3.5 (worst)

        let removed = g.prune_weakest(0, 2, 1.0).unwrap();
        assert_eq!(removed, Some(1)); // neighbor 1 had the lowest score
        assert_eq!(g.edge_count(0, 2), 2);
    }

    #[test]
    fn test_prune_not_needed() {
        let cfg = test_config();
        let mut g = RouteGraph::new(&cfg);
        g.add_edge(0, 0, 1, 0.5).unwrap();
        // Only 1 of 3 slots used — no prune needed.
        let removed = g.prune_weakest(0, 0, 1.0).unwrap();
        assert_eq!(removed, None);
    }

    #[test]
    fn test_serde_roundtrip() {
        let cfg = test_config();
        let mut g = RouteGraph::new(&cfg);
        g.add_edge(0, 0, 1, 0.5).unwrap();
        g.add_edge(1, 3, 2, 1.0).unwrap();

        let json = serde_json::to_string(&g).unwrap();
        let g2: RouteGraph = serde_json::from_str(&json).unwrap();

        assert_eq!(g.nbr_idx, g2.nbr_idx);
        assert_eq!(g.phi, g2.phi);
        assert_eq!(g.taint, g2.taint);
        assert_eq!(g.age, g2.age);
    }

    #[test]
    fn test_valid_nbr_idx_invariant() {
        let cfg = test_config();
        let mut g = RouteGraph::new(&cfg);
        g.add_edge(0, 0, 1, 0.1).unwrap();
        g.add_edge(0, 0, 3, 0.2).unwrap();

        // Check that all non-empty nbr_idx are in [0, L).
        for &idx in &g.nbr_idx {
            assert!(idx == EMPTY_SLOT || (idx >= 0 && (idx as usize) < g.seq_len));
        }
    }

    // ── RouteAggregate unit tests ────────────────────────────────────────────

    #[test]
    fn test_route_aggregate_empty_graph() {
        // With all slots empty, r must be zeros. Edge weights must also be zeros.
        let cfg = test_config();
        let g = RouteGraph::new(&cfg);
        let b = cfg.batch_size;
        let l = cfg.seq_len;
        let d = 8_usize;

        let hidden = vec![1.0_f32; b * l * d];
        let (r, ew) = g.route_aggregate(&hidden, d, EPS, LAMBDA, MU).unwrap();

        assert_eq!(r.len(), b * l * d);
        assert_eq!(ew.len(), b * l * cfg.emax);
        assert!(
            r.iter().all(|&v| v == 0.0),
            "r must be zero for empty graph"
        );
        assert!(
            ew.iter().all(|&v| v == 0.0),
            "edge_weights must be zero for empty graph"
        );
    }

    #[test]
    fn test_route_aggregate_output_shapes() {
        // Verify r shape [B, L, d] and edge_weights shape [B, L, Emax].
        let cfg = ErmConfig {
            batch_size: 3,
            seq_len: 5,
            emax: 4,
            ..ErmConfig::default()
        };
        let mut g = RouteGraph::new(&cfg);
        g.add_edge(0, 1, 0, 0.5).unwrap();

        let d = 16_usize;
        let hidden = vec![0.5_f32; cfg.batch_size * cfg.seq_len * d];
        let (r, ew) = g.route_aggregate(&hidden, d, EPS, LAMBDA, MU).unwrap();

        assert_eq!(r.len(), cfg.batch_size * cfg.seq_len * d);
        assert_eq!(ew.len(), cfg.batch_size * cfg.seq_len * cfg.emax);
    }

    #[test]
    fn test_route_aggregate_single_edge() {
        // Manually verify the math for a single edge.
        let cfg = ErmConfig {
            batch_size: 1,
            seq_len: 3,
            emax: 2,
            ..ErmConfig::default()
        };
        let mut g = RouteGraph::new(&cfg);
        g.add_edge(0, 1, 0, 1.0).unwrap();

        let d = 2_usize;
        let bi: usize = 0;
        let l: usize = 3;
        let mut hidden = vec![0.0_f32; l * d];
        hidden[0] = 10.0; // [b=0, pos=0, k=0]
        hidden[1] = 20.0; // [b=0, pos=0, k=1]

        let (r, ew) = g.route_aggregate(&hidden, d, EPS, LAMBDA, MU).unwrap();

        let r_base = (bi * l + 1) * d;
        assert!((r[r_base] - 10.0).abs() < 1e-4);
        assert!((r[r_base + 1] - 20.0).abs() < 1e-4);

        let emax = cfg.emax;
        let ew_base = (bi * l + 1) * emax;
        assert!((ew[ew_base] - 1.0).abs() < 1e-4);
        assert_eq!(ew[ew_base + 1], 0.0);
    }

    #[test]
    fn test_route_aggregate_empty_slot_zero_weight() {
        let cfg = ErmConfig {
            batch_size: 1,
            seq_len: 4,
            emax: 3,
            ..ErmConfig::default()
        };
        let mut g = RouteGraph::new(&cfg);
        g.add_edge(0, 2, 1, 0.5).unwrap();

        let bi: usize = 0;
        let l: usize = 4;
        let emax: usize = 3;
        let d = 4_usize;
        let hidden = vec![1.0_f32; l * d];
        let (_, ew) = g.route_aggregate(&hidden, d, EPS, LAMBDA, MU).unwrap();

        let ew_base = (bi * l + 2) * emax;
        assert!(ew[ew_base] > 0.0);
        assert_eq!(ew[ew_base + 1], 0.0);
        assert_eq!(ew[ew_base + 2], 0.0);
    }

    #[test]
    fn test_route_aggregate_weights_sum_one() {
        let cfg = ErmConfig {
            batch_size: 1,
            seq_len: 5,
            emax: 4,
            ..ErmConfig::default()
        };
        let mut g = RouteGraph::new(&cfg);
        g.add_edge(0, 3, 0, 1.0).unwrap();
        g.add_edge(0, 3, 1, 0.5).unwrap();
        g.add_edge(0, 3, 2, 2.0).unwrap();

        let bi: usize = 0;
        let l: usize = 5;
        let emax: usize = 4;
        let d = 8_usize;
        let hidden = vec![1.0_f32; l * d];
        let (_, ew) = g.route_aggregate(&hidden, d, EPS, LAMBDA, MU).unwrap();

        let ew_base = (bi * l + 3) * emax;
        let weight_sum: f32 = ew[ew_base..ew_base + emax].iter().sum();
        assert!((weight_sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_route_aggregate_deterministic() {
        let cfg = test_config();
        let mut g = RouteGraph::new(&cfg);
        g.add_edge(0, 0, 1, 0.8).unwrap();
        g.add_edge(0, 0, 3, 0.3).unwrap();

        let d = 6_usize;
        let hidden: Vec<f32> = (0..cfg.batch_size * cfg.seq_len * d)
            .map(|i| i as f32 * 0.1)
            .collect();

        let (r1, ew1) = g.route_aggregate(&hidden, d, EPS, LAMBDA, MU).unwrap();
        let (r2, ew2) = g.route_aggregate(&hidden, d, EPS, LAMBDA, MU).unwrap();

        assert_eq!(r1, r2);
        assert_eq!(ew1, ew2);
    }

    #[test]
    fn test_route_aggregate_no_nan_no_inf() {
        // Verify no NaN or Inf appears in output even with varied phi/taint/age.
        let cfg = ErmConfig {
            batch_size: 2,
            seq_len: 6,
            emax: 4,
            ..ErmConfig::default()
        };
        let mut g = RouteGraph::new(&cfg);

        // Add edges with extreme phi/taint/age values.
        g.add_edge(0, 0, 1, 0.0001).unwrap(); // very small phi
        g.add_edge(0, 0, 2, 100.0).unwrap(); // large phi
        let flat = g.idx(0, 0, 0);
        g.taint[flat] = 4.9; // near taint_max
        g.age[flat] = 10000; // very old edge

        g.add_edge(1, 3, 0, 0.05).unwrap();
        g.add_edge(1, 3, 5, 0.05).unwrap();
        let flat2 = g.idx(1, 3, 1);
        g.age[flat2] = 999;

        let d = 8_usize;
        let hidden: Vec<f32> = (0..cfg.batch_size * cfg.seq_len * d)
            .map(|i| (i as f32) * 0.01 - 0.5)
            .collect();

        let (r, ew) = g.route_aggregate(&hidden, d, EPS, LAMBDA, MU).unwrap();

        for (i, &v) in r.iter().enumerate() {
            assert!(!v.is_nan(), "NaN in r at index {i}");
            assert!(!v.is_infinite(), "Inf in r at index {i}");
        }
        for (i, &v) in ew.iter().enumerate() {
            assert!(!v.is_nan(), "NaN in edge_weights at index {i}");
            assert!(!v.is_infinite(), "Inf in edge_weights at index {i}");
        }
    }

    #[test]
    fn test_route_aggregate_shape_mismatch() {
        let cfg = test_config();
        let g = RouteGraph::new(&cfg);

        // Wrong hidden length.
        let hidden = vec![1.0_f32; 5];
        assert!(g.route_aggregate(&hidden, 8, EPS, LAMBDA, MU).is_err());
    }

    #[test]
    fn test_route_aggregate_timing() {
        use std::time::Instant;

        let cfg = ErmConfig {
            batch_size: 8,
            seq_len: 128,
            emax: 16,
            hidden_dim: 256,
            ..ErmConfig::default()
        };
        let mut g = RouteGraph::new(&cfg);

        for bi in 0..cfg.batch_size {
            for i in 0..cfg.seq_len {
                let src1 = if i > 0 { i - 1 } else { 0 };
                let src2 = if i + 1 < cfg.seq_len { i + 1 } else { i };
                let _ = g.add_edge(bi, i, src1, 1.0);
                let _ = g.add_edge(bi, i, src2, 0.5);
            }
        }

        let d = 256_usize;
        let hidden = vec![0.01_f32; cfg.batch_size * cfg.seq_len * d];

        let t0 = Instant::now();
        let (r, ew) = g.route_aggregate(&hidden, d, EPS, LAMBDA, MU).unwrap();
        let elapsed = t0.elapsed();

        assert_eq!(r.len(), cfg.batch_size * cfg.seq_len * d);
        assert_eq!(ew.len(), cfg.batch_size * cfg.seq_len * cfg.emax);
        assert!(elapsed.as_secs() < 2);
        println!("route_aggregate B=8 L=128 Emax=16 d=256: {elapsed:?}");
    }
}

// ── Proptest-based property tests ────────────────────────────────────────────

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::config::ErmConfig;
    use proptest::prelude::*;

    /// Default hyperparams for route_aggregate tests.
    const EPS: f32 = 1e-6;
    const LAMBDA: f32 = 1.0;
    const MU: f32 = 0.01;

    /// Strategy: generate a RouteGraph with random edges and matching hidden state.
    ///
    /// Fixed dims: B=2, L=8, Emax=4, d=16 to keep tests fast.
    fn route_aggregate_inputs() -> impl Strategy<Value = (RouteGraph, Vec<f32>, usize)> {
        let b = 2_usize;
        let l = 8_usize;
        let emax = 4_usize;
        let d = 16_usize;

        // Generate hidden values in a reasonable range.
        let hidden_strategy = proptest::collection::vec(-10.0_f32..10.0_f32, b * l * d);

        // Generate per-edge data: for each (b, i) slot, decide whether to fill.
        let edge_fill = proptest::collection::vec(proptest::bool::ANY, b * l * emax);
        let phi_vals = proptest::collection::vec(0.001_f32..10.0_f32, b * l * emax);
        let taint_vals = proptest::collection::vec(0.0_f32..5.0_f32, b * l * emax);
        let age_vals = proptest::collection::vec(0_i32..1000_i32, b * l * emax);

        (hidden_strategy, edge_fill, phi_vals, taint_vals, age_vals).prop_map(
            move |(hidden, fill, phi, taint, age)| {
                let cfg = ErmConfig {
                    batch_size: b,
                    seq_len: l,
                    emax,
                    ..ErmConfig::default()
                };
                let mut g = RouteGraph::new(&cfg);

                // Fill edges according to the random fill vector.
                for bi in 0..b {
                    for i in 0..l {
                        for e in 0..emax {
                            let flat = g.idx(bi, i, e);
                            if fill[flat] {
                                // Pick a valid source (any position in [0, L) except i itself).
                                let src = if e % l != i { e % l } else { (e + 1) % l };
                                g.nbr_idx[flat] = src as i32;
                                g.phi[flat] = phi[flat];
                                g.taint[flat] = taint[flat];
                                g.age[flat] = age[flat];
                            }
                        }
                    }
                }

                (g, hidden, d)
            },
        )
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        /// Output shapes always match [B, L, d] and [B, L, Emax].
        #[test]
        fn prop_route_aggregate_output_shapes((ref g, ref hidden, d) in route_aggregate_inputs()) {
            let (r, ew) = g.route_aggregate(hidden, d, EPS, LAMBDA, MU).unwrap();
            prop_assert_eq!(r.len(), g.batch_size * g.seq_len * d);
            prop_assert_eq!(ew.len(), g.batch_size * g.seq_len * g.emax);
        }

        /// No NaN or Inf in output.
        #[test]
        fn prop_route_aggregate_no_nan_inf((ref g, ref hidden, d) in route_aggregate_inputs()) {
            let (r, ew) = g.route_aggregate(hidden, d, EPS, LAMBDA, MU).unwrap();
            for &v in r.iter() {
                prop_assert!(!v.is_nan(), "NaN in r");
                prop_assert!(!v.is_infinite(), "Inf in r");
            }
            for &v in ew.iter() {
                prop_assert!(!v.is_nan(), "NaN in ew");
                prop_assert!(!v.is_infinite(), "Inf in ew");
            }
        }

        /// Edge weights sum to ≈1.0 for destinations with valid neighbors;
        /// exactly 0.0 for destinations with no neighbors.
        #[test]
        fn prop_route_aggregate_weights_sum((ref g, ref hidden, d) in route_aggregate_inputs()) {
            let (_, ew) = g.route_aggregate(hidden, d, EPS, LAMBDA, MU).unwrap();

            for bi in 0..g.batch_size {
                for i in 0..g.seq_len {
                    let ew_base = (bi * g.seq_len + i) * g.emax;
                    let has_valid = (0..g.emax).any(|e| {
                        g.nbr_idx[g.idx(bi, i, e)] != EMPTY_SLOT
                    });
                    let weight_sum: f32 = ew[ew_base..ew_base + g.emax].iter().sum();

                    if has_valid {
                        prop_assert!(
                            (weight_sum - 1.0).abs() < 1e-4,
                            "weight sum {weight_sum} != 1.0 at ({bi}, {i})"
                        );
                    } else {
                        prop_assert!(
                            weight_sum == 0.0,
                            "weight sum {weight_sum} != 0.0 at empty ({bi}, {i})"
                        );
                    }
                }
            }
        }

        /// EMPTY_SLOT neighbors always get zero weight.
        #[test]
        fn prop_route_aggregate_empty_slot_zero_weight((ref g, ref hidden, d) in route_aggregate_inputs()) {
            let (_, ew) = g.route_aggregate(hidden, d, EPS, LAMBDA, MU).unwrap();

            for bi in 0..g.batch_size {
                for i in 0..g.seq_len {
                    for e in 0..g.emax {
                        let flat = g.idx(bi, i, e);
                        if g.nbr_idx[flat] == EMPTY_SLOT {
                            let ew_idx = (bi * g.seq_len + i) * g.emax + e;
                            let w = ew[ew_idx];
                            prop_assert!(
                                w == 0.0,
                                "EMPTY_SLOT at ({}, {}, {}) has weight {}",
                                bi, i, e, w
                            );
                        }
                    }
                }
            }
        }

        /// Deterministic: same input always produces the same output.
        #[test]
        fn prop_route_aggregate_deterministic((ref g, ref hidden, d) in route_aggregate_inputs()) {
            let (r1, ew1) = g.route_aggregate(hidden, d, EPS, LAMBDA, MU).unwrap();
            let (r2, ew2) = g.route_aggregate(hidden, d, EPS, LAMBDA, MU).unwrap();
            prop_assert_eq!(&r1, &r2);
            prop_assert_eq!(&ew1, &ew2);
        }
    }
}
