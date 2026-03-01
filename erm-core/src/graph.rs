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

    /// Placeholder for the differentiable route aggregation kernel.
    ///
    /// # Shape
    ///
    /// Given hidden states `h: [B, L, d]`, computes `r: [B, L, d]` where each
    /// `r[b, i, :]` is a weighted sum of neighbor hidden states gated by
    /// pheromone, taint, and age.
    ///
    /// # Panics
    ///
    /// Always panics — this stub will be replaced in Phase 2.
    pub fn route_aggregate(&self, _hidden: &[f32], _d: usize) -> Vec<f32> {
        todo!("RouteAggregate kernel not yet implemented — Phase 2")
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
}
