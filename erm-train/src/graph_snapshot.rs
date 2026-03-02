//! Graph snapshot capture for visualization.
//!
//! Saves RouteGraph state at regular intervals during training for later
//! rendering into animations.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

use erm_core::ants::AntState;
use erm_core::error::ErmResult;
use erm_core::graph::RouteGraph;

/// A snapshot of the graph state at a specific training step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSnapshot {
    /// Training step number.
    pub step: usize,
    /// Loss at this step.
    pub loss: f32,
    /// Number of edits made.
    pub edits: usize,
    /// Mean pheromone level.
    pub mean_phi: f32,
    /// Number of pruned edges.
    pub pruned: usize,
    /// Number of inserted edges.
    pub inserted: usize,
    /// Number of ant deaths.
    pub deaths: usize,
    /// The graph structure (full state).
    pub graph: RouteGraph,
    /// Ant colony state.
    pub ant_state: AntState,
}

impl GraphSnapshot {
    /// Create a new snapshot from current training state.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        step: usize,
        loss: f32,
        edits: usize,
        mean_phi: f32,
        pruned: usize,
        inserted: usize,
        deaths: usize,
        graph: RouteGraph,
        ant_state: AntState,
    ) -> Self {
        Self {
            step,
            loss,
            edits,
            mean_phi,
            pruned,
            inserted,
            deaths,
            graph,
            ant_state,
        }
    }

    /// Save snapshot to a JSON file.
    ///
    /// # Errors
    /// Returns error if directory creation or file write fails.
    pub fn save<P: AsRef<Path>>(&self, dir: P) -> ErmResult<()> {
        let dir = dir.as_ref();
        fs::create_dir_all(dir)?;
        let path = dir.join(format!("step_{:05}.json", self.step));
        let json = serde_json::to_string_pretty(self)?;
        fs::write(&path, json)?;
        Ok(())
    }

    /// Load a snapshot from a JSON file.
    ///
    /// # Errors
    /// Returns error if file read or JSON parse fails.
    pub fn load<P: AsRef<Path>>(path: P) -> ErmResult<Self> {
        let json = fs::read_to_string(path)?;
        let snapshot = serde_json::from_str(&json)?;
        Ok(snapshot)
    }

    /// Get summary statistics for the graph.
    #[must_use]
    pub fn graph_stats(&self) -> GraphStats {
        let mut total_edges = 0;
        let mut total_phi = 0.0_f32;
        let mut max_phi = 0.0_f32;
        let mut leader_edges = 0;

        let b = self.graph.batch_size;
        let l = self.graph.seq_len;
        let e = self.graph.emax;

        for bi in 0..b {
            for li in 0..l {
                for ei in 0..e {
                    let idx = self.graph.idx(bi, li, ei);
                    if self.graph.nbr_idx[idx] >= 0 {
                        total_edges += 1;
                        let phi = self.graph.phi[idx];
                        total_phi += phi;
                        max_phi = max_phi.max(phi);
                        if self.graph.leader_edge[idx] {
                            leader_edges += 1;
                        }
                    }
                }
            }
        }

        GraphStats {
            total_edges,
            avg_phi: if total_edges > 0 {
                total_phi / total_edges as f32
            } else {
                0.0
            },
            max_phi,
            leader_edges,
            follower_edges: total_edges - leader_edges,
        }
    }
}

/// Summary statistics for a graph snapshot.
#[derive(Debug, Clone, Copy)]
pub struct GraphStats {
    /// Total number of active edges.
    pub total_edges: usize,
    /// Average pheromone level across edges.
    pub avg_phi: f32,
    /// Maximum pheromone level.
    pub max_phi: f32,
    /// Number of leader-introduced edges.
    pub leader_edges: usize,
    /// Number of follower edges.
    pub follower_edges: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use erm_core::ants::AntState;
    use erm_core::config::ErmConfig;
    use erm_core::graph::RouteGraph;

    fn test_config() -> ErmConfig {
        ErmConfig {
            vocab_size: 100,
            seq_len: 16,
            batch_size: 2,
            hidden_dim: 64,
            num_blocks: 2,
            emax: 4,
            topk: 5,
            num_ants: 10,
            phi_init: 0.01,
            phi_max: 100.0,
            death_streak: 100,
            max_edits_per_step: 0.15,
            leader_fraction: 0.2,
            ..ErmConfig::default()
        }
    }

    #[test]
    fn test_snapshot_save_load_roundtrip() {
        let config = test_config();
        let graph = RouteGraph::new(&config);
        let ant_state = AntState::new(&config);

        let snapshot = GraphSnapshot::new(100, 3.5, 20, 0.5, 2, 5, 100, graph, ant_state);

        // Save to temp dir
        let temp_dir = std::env::temp_dir().join("erm_test_snapshots");
        let _ = std::fs::remove_dir_all(&temp_dir);

        snapshot.save(&temp_dir).expect("save should succeed");

        // Load back
        let loaded =
            GraphSnapshot::load(temp_dir.join("step_00100.json")).expect("load should succeed");

        assert_eq!(loaded.step, 100);
        assert_eq!(loaded.loss, 3.5);
        assert_eq!(loaded.edits, 20);

        // Cleanup
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_graph_stats() {
        let config = test_config();
        let mut graph = RouteGraph::new(&config);

        // Add some test edges with known pheromone
        graph.nbr_idx[0] = 0;
        graph.phi[0] = 0.5;
        graph.leader_edge[0] = true;

        graph.nbr_idx[1] = 1;
        graph.phi[1] = 1.5;
        graph.leader_edge[1] = false;

        let ant_state = AntState::new(&config);
        let snapshot = GraphSnapshot::new(0, 0.0, 0, 0.0, 0, 0, 0, graph, ant_state);

        let stats = snapshot.graph_stats();
        assert_eq!(stats.total_edges, 2);
        assert_eq!(stats.leader_edges, 1);
        assert_eq!(stats.follower_edges, 1);
        assert!((stats.avg_phi - 1.0).abs() < 0.001);
        assert!((stats.max_phi - 1.5).abs() < 0.001);
    }
}
