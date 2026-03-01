//! Top-k extraction from logits.
//!
//! Extracts the top-k token ids and their corresponding scores from scorer
//! logits. Uses a simple partial-sort approach per position — no GPU needed.
//!
//! # Shape convention
//!
//! Input logits are flat `[B * L * V]` (row-major `[B, L, V]`).
//! Outputs are flat `[B * L * k]` (row-major `[B, L, k]`):
//! - `topk_ids`: token ids of the k highest logits per position.
//! - `topk_scores`: corresponding logit values, descending.

use crate::error::{ErmError, ErmResult};

/// Extract the top-k tokens and scores from logits.
///
/// For each position `(b, i)`, finds the `k` tokens with the highest logit
/// values using a simple partial sort.
///
/// # Arguments
///
/// - `logits`: flat `[B * L * V]` logit values (row-major `[B, L, V]`).
/// - `batch_size`: batch dimension `B`.
/// - `seq_len`: sequence dimension `L`.
/// - `vocab_size`: vocabulary dimension `V`.
/// - `k`: number of top tokens to extract per position.
///
/// # Returns
///
/// `(topk_ids, topk_scores)` where:
/// - `topk_ids`: flat `[B * L * k]` of `u32` token indices, descending by score.
/// - `topk_scores`: flat `[B * L * k]` of `f32` logit values, descending.
///
/// # Errors
///
/// Returns [`ErmError::ShapeMismatch`] if `logits.len() != B * L * V`.
/// Returns [`ErmError::InvalidConfig`] if `k > V` or `k == 0`.
///
/// # Shape reference
///
/// | Tensor | Shape |
/// |---|---|
/// | `logits` (input) | `[B, L, V]` flat |
/// | `topk_ids` (output) | `[B, L, k]` flat |
/// | `topk_scores` (output) | `[B, L, k]` flat |
pub fn extract_topk(
    logits: &[f32],
    batch_size: usize,
    seq_len: usize,
    vocab_size: usize,
    k: usize,
) -> ErmResult<(Vec<u32>, Vec<f32>)> {
    let expected = batch_size * seq_len * vocab_size;
    if logits.len() != expected {
        return Err(ErmError::ShapeMismatch {
            expected: format!("[B={batch_size}, L={seq_len}, V={vocab_size}] = {expected}"),
            got: format!("{}", logits.len()),
        });
    }
    if k == 0 {
        return Err(ErmError::InvalidConfig("k must be > 0".to_string()));
    }
    if k > vocab_size {
        return Err(ErmError::InvalidConfig(format!(
            "k={k} exceeds vocab_size={vocab_size}"
        )));
    }

    let total_positions = batch_size * seq_len;
    let mut topk_ids = Vec::with_capacity(total_positions * k);
    let mut topk_scores = Vec::with_capacity(total_positions * k);

    for pos in 0..total_positions {
        let start = pos * vocab_size;
        let row = &logits[start..start + vocab_size];

        // Build (index, value) pairs and partial-sort to find top-k.
        let mut indexed: Vec<(u32, f32)> = row
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as u32, v))
            .collect();

        // Partial sort: move top-k to the front via select_nth_unstable_by.
        // Then sort the top-k slice in descending order.
        if k < vocab_size {
            indexed.select_nth_unstable_by(k - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        let top_slice = &mut indexed[..k];
        top_slice
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for &(id, score) in top_slice.iter() {
            topk_ids.push(id);
            topk_scores.push(score);
        }
    }

    Ok((topk_ids, topk_scores))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topk_correct_shape() {
        let b = 2;
        let l = 4;
        let v = 8;
        let k = 3;
        let logits = vec![0.0_f32; b * l * v];

        let (ids, scores) = extract_topk(&logits, b, l, v, k).unwrap();
        assert_eq!(ids.len(), b * l * k);
        assert_eq!(scores.len(), b * l * k);
    }

    #[test]
    fn test_topk_values_are_top() {
        let v = 6;
        // Single position: logits = [1.0, 5.0, 3.0, 0.5, 4.0, 2.0]
        let logits = vec![1.0, 5.0, 3.0, 0.5, 4.0, 2.0];
        let k = 3;

        let (ids, scores) = extract_topk(&logits, 1, 1, v, k).unwrap();

        // Top-3 should be indices 1(5.0), 4(4.0), 2(3.0) in descending order.
        assert_eq!(ids, vec![1, 4, 2]);
        assert_eq!(scores, vec![5.0, 4.0, 3.0]);
    }

    #[test]
    fn test_topk_handles_ties() {
        let v = 5;
        // Logits with ties: [3.0, 3.0, 3.0, 1.0, 1.0]
        let logits = vec![3.0, 3.0, 3.0, 1.0, 1.0];
        let k = 3;

        let (ids, scores) = extract_topk(&logits, 1, 1, v, k).unwrap();

        // All top-3 should have score 3.0.
        assert_eq!(scores.len(), 3);
        for &s in &scores {
            assert!((s - 3.0).abs() < 1e-9);
        }
        // Ids should be from {0, 1, 2}.
        for &id in &ids {
            assert!(id <= 2, "expected tie token from {{0,1,2}}, got {id}");
        }
    }

    #[test]
    fn test_topk_k_equals_v() {
        let v = 4;
        let logits = vec![2.0, 0.0, 3.0, 1.0];
        let k = 4;

        let (ids, scores) = extract_topk(&logits, 1, 1, v, k).unwrap();
        assert_eq!(ids.len(), 4);
        assert_eq!(scores, vec![3.0, 2.0, 1.0, 0.0]);
    }

    #[test]
    fn test_topk_multi_position() {
        let b = 1;
        let l = 2;
        let v = 4;
        let k = 2;
        // Position 0: [0.1, 0.4, 0.3, 0.2] → top-2: ids 1, 2
        // Position 1: [0.9, 0.1, 0.0, 0.5] → top-2: ids 0, 3
        let logits = vec![0.1, 0.4, 0.3, 0.2, 0.9, 0.1, 0.0, 0.5];

        let (ids, scores) = extract_topk(&logits, b, l, v, k).unwrap();
        assert_eq!(ids.len(), l * k);

        // Position 0: top-2 are (1, 0.4) and (2, 0.3).
        assert_eq!(ids[0], 1);
        assert_eq!(ids[1], 2);
        assert!((scores[0] - 0.4).abs() < 1e-6);
        assert!((scores[1] - 0.3).abs() < 1e-6);

        // Position 1: top-2 are (0, 0.9) and (3, 0.5).
        assert_eq!(ids[2], 0);
        assert_eq!(ids[3], 3);
    }

    #[test]
    fn test_topk_shape_mismatch() {
        let logits = vec![0.0; 10];
        assert!(extract_topk(&logits, 2, 3, 4, 2).is_err());
    }

    #[test]
    fn test_topk_k_zero() {
        let logits = vec![0.0; 8];
        assert!(extract_topk(&logits, 1, 1, 8, 0).is_err());
    }

    #[test]
    fn test_topk_k_exceeds_vocab() {
        let logits = vec![0.0; 4];
        assert!(extract_topk(&logits, 1, 1, 4, 5).is_err());
    }

    #[test]
    fn test_topk_descending_order() {
        let v = 10;
        let logits: Vec<f32> = (0..v).map(|i| i as f32).collect();
        let k = 5;

        let (_, scores) = extract_topk(&logits, 1, 1, v, k).unwrap();
        for w in scores.windows(2) {
            assert!(w[0] >= w[1], "scores not descending: {:?}", scores);
        }
    }
}
