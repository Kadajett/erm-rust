//! Conflict-free merge of ant edit proposals.
//!
//! Given a set of [`EditProposal`]s from follower ants, this module selects
//! the best candidate token per position (highest `predicted_gain`), caps total
//! edits at `M = floor(0.15 * L)`, skips non-editable positions, and produces
//! the refined token sequence `y_{t-1}`.
//!
//! # Algorithm
//!
//! 1. Per position: keep the proposal with the highest `predicted_gain`.
//! 2. Discard any proposal at a non-editable position.
//! 3. Sort surviving proposals by gain descending; keep at most `max_edits`.
//! 4. Apply accepted edits to `y_t` to produce `y_{t-1}`.

use crate::error::{ErmError, ErmResult};

/// A single edit proposal from one ant at one position.
///
/// Produced by the follower ant sampler. Each ant may propose edits at up to
/// `pmax` positions.
#[derive(Debug, Clone)]
pub struct SimpleEditProposal {
    /// Sequence position to edit (0-indexed within the sequence).
    pub position: usize,
    /// Proposed replacement token id.
    pub token: u32,
    /// Predicted quality gain for this edit.
    pub predicted_gain: f32,
    /// Index of the ant that proposed this edit.
    pub ant_id: usize,
}

/// Merge edit proposals into a conflict-free edit set and apply to `y_t`.
///
/// Per position, the proposal with the highest `predicted_gain` wins.
/// At most `max_edits` positions are edited (top-M by gain).
/// Non-editable positions and proposals with negative gain are skipped.
///
/// # Arguments
///
/// - `proposals`: all edit proposals from follower ants.
/// - `y_t`: current token sequence, length `seq_len`. Values are `u32` token ids.
/// - `editable`: boolean mask of length `seq_len`. `true` = position may be edited.
/// - `seq_len`: sequence length `L`.
/// - `max_edits`: maximum number of edits allowed (typically `floor(0.15 * L)`).
///
/// # Returns
///
/// `y_{t-1}`: the refined token sequence of length `seq_len`.
///
/// # Errors
///
/// Returns [`ErmError::ShapeMismatch`] if `y_t` or `editable` length ≠ `seq_len`.
///
/// # Shape reference
///
/// | Tensor | Shape |
/// |---|---|
/// | `y_t` (input) | `[L]` |
/// | `editable` (input) | `[L]` |
/// | output | `[L]` |
pub fn merge_proposals(
    proposals: &[SimpleEditProposal],
    y_t: &[u32],
    editable: &[bool],
    seq_len: usize,
    max_edits: usize,
) -> ErmResult<Vec<u32>> {
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

    // Phase 1: collect best proposal per position.
    // (position, token, gain, ant_id)
    let mut best: Vec<Option<(u32, f32, usize)>> = vec![None; seq_len];

    for p in proposals {
        if p.position >= seq_len {
            continue;
        }
        if !editable[p.position] {
            continue;
        }
        // Only accept non-negative gain proposals.
        if p.predicted_gain < 0.0 {
            continue;
        }

        match &best[p.position] {
            Some((_, existing_gain, _)) if *existing_gain >= p.predicted_gain => {
                // Existing proposal is at least as good; keep it.
            }
            _ => {
                best[p.position] = Some((p.token, p.predicted_gain, p.ant_id));
            }
        }
    }

    // Phase 2: collect all accepted edits, sorted descending by gain.
    let mut edits: Vec<(usize, u32, f32)> = best
        .iter()
        .enumerate()
        .filter_map(|(pos, opt)| opt.map(|(tok, gain, _)| (pos, tok, gain)))
        .collect();

    // Sort descending by gain; cap at max_edits.
    edits.sort_unstable_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    edits.truncate(max_edits);

    // Phase 3: apply edits.
    let mut y_new: Vec<u32> = y_t.to_vec();
    for &(pos, tok, _) in &edits {
        y_new[pos] = tok;
    }

    Ok(y_new)
}

/// Compute per-ant improvement deltas after a merge step.
///
/// For each ant, sums the logit improvement at positions where that ant's
/// proposal was accepted. This bridges Phase 3 → Phase 4 (pheromone deposit).
///
/// `delta_k = Σ_{i ∈ accepted_edits_by_ant_k} (logit_after[i][token] - logit_before[i][token])`
///
/// # Arguments
///
/// - `proposals`: all edit proposals from this step.
/// - `y_t`: original tokens before merge, length `seq_len`.
/// - `y_new`: tokens after merge, length `seq_len`.
/// - `logits_before`: scorer logits before merge, flat `[L * V]`.
/// - `logits_after`: scorer logits after merge, flat `[L * V]`.
/// - `vocab_size`: vocabulary dimension `V`.
/// - `num_ants`: total number of ants `A`.
///
/// # Returns
///
/// Per-ant delta vector of length `num_ants`.
///
/// # Errors
///
/// Returns [`ErmError::ShapeMismatch`] if logit dimensions don't match.
///
/// # Shape reference
///
/// | Tensor | Shape |
/// |---|---|
/// | `logits_before` | `[L, V]` flat |
/// | `logits_after` | `[L, V]` flat |
/// | output | `[A]` |
pub fn compute_ant_deltas(
    proposals: &[SimpleEditProposal],
    y_t: &[u32],
    y_new: &[u32],
    logits_before: &[f32],
    logits_after: &[f32],
    vocab_size: usize,
    num_ants: usize,
) -> ErmResult<Vec<f32>> {
    let seq_len = y_t.len();
    let expected_logits = seq_len * vocab_size;

    if logits_before.len() != expected_logits {
        return Err(ErmError::ShapeMismatch {
            expected: format!("logits_before [L={seq_len}, V={vocab_size}] = {expected_logits}"),
            got: format!("{}", logits_before.len()),
        });
    }
    if logits_after.len() != expected_logits {
        return Err(ErmError::ShapeMismatch {
            expected: format!("logits_after [L={seq_len}, V={vocab_size}] = {expected_logits}"),
            got: format!("{}", logits_after.len()),
        });
    }
    if y_new.len() != seq_len {
        return Err(ErmError::ShapeMismatch {
            expected: format!("y_new length = {seq_len}"),
            got: format!("{}", y_new.len()),
        });
    }

    let mut deltas = vec![0.0_f32; num_ants];

    // Build a map: position → winning ant_id (only for positions that changed).
    let mut winning_ant: Vec<Option<usize>> = vec![None; seq_len];
    for p in proposals {
        if p.position >= seq_len {
            continue;
        }
        // A position was accepted if y_new differs from y_t and the token matches.
        if y_new[p.position] != y_t[p.position] && y_new[p.position] == p.token {
            // Among proposals at this position, pick the one whose token matches y_new.
            // This should already be the winner from merge.
            match winning_ant[p.position] {
                Some(existing_ant) => {
                    // If multiple proposals had the same token, the one with higher gain wins.
                    // We need to compare gains — find existing proposal's gain.
                    let existing_gain = proposals
                        .iter()
                        .filter(|q| {
                            q.position == p.position
                                && q.ant_id == existing_ant
                                && q.token == p.token
                        })
                        .map(|q| q.predicted_gain)
                        .fold(f32::NEG_INFINITY, f32::max);
                    if p.predicted_gain > existing_gain {
                        winning_ant[p.position] = Some(p.ant_id);
                    }
                }
                None => {
                    winning_ant[p.position] = Some(p.ant_id);
                }
            }
        }
    }

    // Compute delta for each winning ant.
    for (pos, opt_ant) in winning_ant.iter().enumerate() {
        if let Some(ant_id) = opt_ant {
            if *ant_id < num_ants {
                let token = y_new[pos] as usize;
                if token < vocab_size {
                    let logit_old = logits_before[pos * vocab_size + token];
                    let logit_new = logits_after[pos * vocab_size + token];
                    deltas[*ant_id] += logit_new - logit_old;
                }
            }
        }
    }

    Ok(deltas)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_conflict_resolution() {
        let y_t = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let editable = vec![true; 8];

        let proposals = vec![
            SimpleEditProposal {
                position: 2,
                token: 10,
                predicted_gain: 0.5,
                ant_id: 0,
            },
            SimpleEditProposal {
                position: 2,
                token: 20,
                predicted_gain: 0.8, // higher gain → wins
                ant_id: 1,
            },
        ];

        let result = merge_proposals(&proposals, &y_t, &editable, 8, 4).unwrap();
        assert_eq!(result[2], 20); // ant 1 wins
    }

    #[test]
    fn test_merge_edit_cap() {
        let seq_len = 8;
        let y_t = vec![0; seq_len];
        let editable = vec![true; seq_len];
        let max_edits = 2;

        // 5 proposals at different positions.
        let proposals = vec![
            SimpleEditProposal {
                position: 0,
                token: 10,
                predicted_gain: 1.0,
                ant_id: 0,
            },
            SimpleEditProposal {
                position: 1,
                token: 20,
                predicted_gain: 2.0,
                ant_id: 0,
            },
            SimpleEditProposal {
                position: 2,
                token: 30,
                predicted_gain: 3.0,
                ant_id: 0,
            },
            SimpleEditProposal {
                position: 3,
                token: 40,
                predicted_gain: 4.0,
                ant_id: 0,
            },
            SimpleEditProposal {
                position: 4,
                token: 50,
                predicted_gain: 5.0,
                ant_id: 0,
            },
        ];

        let result = merge_proposals(&proposals, &y_t, &editable, seq_len, max_edits).unwrap();

        // Only top-2 by gain should survive: positions 3 (gain=4) and 4 (gain=5).
        let num_changed = result
            .iter()
            .zip(y_t.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(num_changed, 2);
        assert_eq!(result[4], 50); // gain=5.0 highest
        assert_eq!(result[3], 40); // gain=4.0 second
        assert_eq!(result[2], 0); // pruned
    }

    #[test]
    fn test_merge_non_editable_untouched() {
        let y_t = vec![0, 1, 2, 3];
        let mut editable = vec![true; 4];
        editable[2] = false; // position 2 not editable

        let proposals = vec![SimpleEditProposal {
            position: 2,
            token: 99,
            predicted_gain: 10.0,
            ant_id: 0,
        }];

        let result = merge_proposals(&proposals, &y_t, &editable, 4, 4).unwrap();
        assert_eq!(result[2], 2); // unchanged
    }

    #[test]
    fn test_merge_no_edit_stability() {
        // All negative gains → y unchanged.
        let y_t = vec![10, 20, 30, 40];
        let editable = vec![true; 4];

        let proposals = vec![
            SimpleEditProposal {
                position: 0,
                token: 99,
                predicted_gain: -1.0,
                ant_id: 0,
            },
            SimpleEditProposal {
                position: 1,
                token: 88,
                predicted_gain: -0.5,
                ant_id: 1,
            },
        ];

        let result = merge_proposals(&proposals, &y_t, &editable, 4, 4).unwrap();
        assert_eq!(result, y_t);
    }

    #[test]
    fn test_merge_empty_proposals() {
        let y_t = vec![1, 2, 3];
        let editable = vec![true; 3];
        let result = merge_proposals(&[], &y_t, &editable, 3, 3).unwrap();
        assert_eq!(result, y_t);
    }

    #[test]
    fn test_merge_shape_mismatch_yt() {
        let y_t = vec![1, 2]; // wrong length
        let editable = vec![true; 4];
        assert!(merge_proposals(&[], &y_t, &editable, 4, 2).is_err());
    }

    #[test]
    fn test_merge_shape_mismatch_editable() {
        let y_t = vec![1, 2, 3, 4];
        let editable = vec![true; 2]; // wrong length
        assert!(merge_proposals(&[], &y_t, &editable, 4, 2).is_err());
    }

    #[test]
    fn test_ant_deltas_correct_length() {
        let num_ants = 4;
        let seq_len = 4;
        let vocab_size = 8;

        let y_t = vec![0, 1, 2, 3];
        let y_new = vec![0, 5, 2, 3]; // only position 1 changed

        let proposals = vec![SimpleEditProposal {
            position: 1,
            token: 5,
            predicted_gain: 0.5,
            ant_id: 2,
        }];

        let logits_before = vec![0.0_f32; seq_len * vocab_size];
        let mut logits_after = vec![0.0_f32; seq_len * vocab_size];
        // logits_after[pos=1, token=5] = 2.0
        logits_after[vocab_size + 5] = 2.0;

        let deltas = compute_ant_deltas(
            &proposals,
            &y_t,
            &y_new,
            &logits_before,
            &logits_after,
            vocab_size,
            num_ants,
        )
        .unwrap();

        assert_eq!(deltas.len(), num_ants);
        // Ant 2 should have delta = 2.0 - 0.0 = 2.0.
        assert!((deltas[2] - 2.0).abs() < 1e-6);
        // Other ants should be 0.
        assert!((deltas[0]).abs() < 1e-6);
        assert!((deltas[1]).abs() < 1e-6);
        assert!((deltas[3]).abs() < 1e-6);
    }

    #[test]
    fn test_ant_deltas_finite_values() {
        let num_ants = 3;
        let seq_len = 2;
        let vocab_size = 4;

        let y_t = vec![0, 1];
        let y_new = vec![2, 1]; // position 0 changed

        let proposals = vec![SimpleEditProposal {
            position: 0,
            token: 2,
            predicted_gain: 1.0,
            ant_id: 1,
        }];

        let logits_before = vec![1.0_f32; seq_len * vocab_size];
        let logits_after = vec![0.5_f32; seq_len * vocab_size];

        let deltas = compute_ant_deltas(
            &proposals,
            &y_t,
            &y_new,
            &logits_before,
            &logits_after,
            vocab_size,
            num_ants,
        )
        .unwrap();

        for &d in &deltas {
            assert!(d.is_finite(), "delta must be finite, got {d}");
        }
    }

    #[test]
    fn test_ant_deltas_no_changes() {
        let num_ants = 2;
        let seq_len = 3;
        let vocab_size = 4;

        let y_t = vec![0, 1, 2];
        let y_new = vec![0, 1, 2]; // no changes

        let proposals = vec![SimpleEditProposal {
            position: 0,
            token: 5,
            predicted_gain: 0.1,
            ant_id: 0,
        }];

        let logits_before = vec![1.0_f32; seq_len * vocab_size];
        let logits_after = vec![2.0_f32; seq_len * vocab_size];

        let deltas = compute_ant_deltas(
            &proposals,
            &y_t,
            &y_new,
            &logits_before,
            &logits_after,
            vocab_size,
            num_ants,
        )
        .unwrap();

        // No positions changed, so all deltas should be 0.
        for &d in &deltas {
            assert!((d).abs() < 1e-9, "expected zero delta, got {d}");
        }
    }

    #[test]
    fn test_ant_deltas_shape_mismatch() {
        let y_t = vec![0, 1];
        let y_new = vec![0, 1];
        let logits_before = vec![0.0; 5]; // wrong size
        let logits_after = vec![0.0; 8];

        assert!(
            compute_ant_deltas(&[], &y_t, &y_new, &logits_before, &logits_after, 4, 2).is_err()
        );
    }
}
