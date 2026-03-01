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

use serde::{Deserialize, Serialize};

use crate::config::ErmConfig;
use crate::types::{AntIdx, BatchIdx, PosIdx, TokenId};

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
    pub fn record_delta(&mut self, b: BatchIdx, k: AntIdx, delta: f32, config: &ErmConfig) -> bool {
        let flat = self.idx(b, k);
        let epsilon = 1e-6_f32;

        if delta > epsilon {
            self.streak[flat] = 0;
            false
        } else {
            self.streak[flat] += 1;
            self.streak[flat] >= config.death_streak as i32
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
        assert!(!state.record_delta(0, 5, 0.0, &cfg)); // streak=1
        assert!(!state.record_delta(0, 5, -1.0, &cfg)); // streak=2
        assert!(state.record_delta(0, 5, 0.0, &cfg)); // streak=3 → dies

        // Respawn.
        state.respawn(0, 5, &cfg);
        assert_eq!(state.streak[state.idx(0, 5)], 0);
    }

    #[test]
    fn test_streak_resets_on_improvement() {
        let cfg = small_config();
        let mut state = AntState::new(&cfg);

        state.record_delta(0, 3, 0.0, &cfg); // streak=1
        state.record_delta(0, 3, 0.0, &cfg); // streak=2
        state.record_delta(0, 3, 1.0, &cfg); // improvement → streak=0
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
}
