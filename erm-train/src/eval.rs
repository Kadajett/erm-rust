//! Evaluation metrics for the Emergent Route Model.
//!
//! - [`evaluate_denoising`] — masked token accuracy over a test set.
//! - [`evaluate_generation`] — token entropy and unique-token ratio.

use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use erm_core::config::ErmConfig;
use erm_core::corruption::corrupt;
use erm_core::error::ErmResult;
use erm_core::scorer::Scorer;

use crate::dataset::TextDataset;

/// Evaluation metrics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvalMetrics {
    /// Fraction of masked positions where top-1 logit matches ground-truth.
    pub masked_token_accuracy: Option<f32>,
    /// Shannon entropy of token distribution.
    pub token_entropy: Option<f32>,
    /// Fraction of unique token ids.
    pub unique_token_ratio: Option<f32>,
    /// Number of positions evaluated.
    pub num_positions: usize,
    /// Average denoising cross-entropy loss.
    pub avg_loss: Option<f32>,
}

/// Evaluate denoising accuracy on a dataset.
///
/// # Errors
///
/// Propagates errors from corruption or scorer forward pass.
pub fn evaluate_denoising(
    scorer: &Scorer,
    dataset: &TextDataset,
    config: &ErmConfig,
    num_batches: usize,
    t: Option<usize>,
    rng: &mut ChaCha8Rng,
) -> ErmResult<EvalMetrics> {
    let t_val = t.unwrap_or(config.refinement_steps);
    let v = scorer.vocab_size;
    let batch_size = config.batch_size;

    let mut total_correct = 0usize;
    let mut total_corrupted = 0usize;
    let mut total_loss = 0.0_f32;
    let mut loss_terms = 0usize;

    for _ in 0..num_batches {
        let batch = dataset.get_batch(batch_size, rng);
        let b = batch.batch_size;
        let l = batch.seq_len;

        let x_i32: Vec<i32> = batch.tokens.iter().map(|&tok| tok as i32).collect();
        let corruption = corrupt(&x_i32, t_val, config, rng)?;
        let y_t_u32: Vec<u32> = corruption.y_t.iter().map(|&tok| tok as u32).collect();

        let output = scorer.forward(&y_t_u32, b)?;
        let logits = &output.logits;

        for (pos, &original) in x_i32.iter().enumerate().take(b * l) {
            let corrupted = corruption.y_t[pos];

            if corrupted == original {
                continue;
            }

            if original < 0 || (original as usize) >= v {
                continue;
            }
            let target = original as usize;

            let logit_start = pos * v;
            let logit_slice = &logits[logit_start..logit_start + v];

            let pred = argmax(logit_slice);
            if pred == target {
                total_correct += 1;
            }

            let ce = cross_entropy(logit_slice, target);
            total_loss += ce;
            loss_terms += 1;

            total_corrupted += 1;
        }
    }

    let masked_token_accuracy = if total_corrupted > 0 {
        Some(total_correct as f32 / total_corrupted as f32)
    } else {
        Some(0.0)
    };

    let avg_loss = if loss_terms > 0 {
        Some(total_loss / loss_terms as f32)
    } else {
        None
    };

    Ok(EvalMetrics {
        masked_token_accuracy,
        token_entropy: None,
        unique_token_ratio: None,
        num_positions: total_corrupted,
        avg_loss,
    })
}

/// Evaluate generation diversity from fully-masked inputs.
///
/// # Errors
///
/// Propagates errors from scorer forward pass.
pub fn evaluate_generation(
    scorer: &Scorer,
    config: &ErmConfig,
    num_batches: usize,
    _rng: &mut ChaCha8Rng,
) -> ErmResult<EvalMetrics> {
    let batch_size = config.batch_size;
    let l = config.seq_len;
    let v = scorer.vocab_size;
    let mask_id = config.mask_token_id() as u32;

    let fully_masked: Vec<u32> = vec![mask_id; batch_size * l];

    let mut token_counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    let mut total_tokens = 0usize;

    for _ in 0..num_batches {
        let output = scorer.forward(&fully_masked, batch_size)?;
        let logits = &output.logits;

        for logit_slice in logits.chunks_exact(v).take(batch_size * l) {
            let pred = argmax(logit_slice) as u32;
            *token_counts.entry(pred).or_insert(0) += 1;
            total_tokens += 1;
        }
    }

    let unique_count = token_counts.len();
    let unique_ratio = if total_tokens > 0 {
        unique_count as f32 / total_tokens as f32
    } else {
        0.0
    };

    let entropy = if total_tokens > 0 {
        let mut h = 0.0_f32;
        for &count in token_counts.values() {
            let p = count as f32 / total_tokens as f32;
            if p > 0.0 {
                h -= p * p.log2();
            }
        }
        h
    } else {
        0.0
    };

    Ok(EvalMetrics {
        masked_token_accuracy: None,
        token_entropy: Some(entropy),
        unique_token_ratio: Some(unique_ratio),
        num_positions: total_tokens,
        avg_loss: None,
    })
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
            if v > bv {
                (i, v)
            } else {
                (bi, bv)
            }
        })
        .0
}

fn cross_entropy(logits: &[f32], target: usize) -> f32 {
    if logits.is_empty() || target >= logits.len() {
        return 0.0;
    }
    let max_v = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&x| (x - max_v).exp()).sum();
    if exp_sum <= 0.0 {
        return 0.0;
    }
    (logits[target] - max_v) - exp_sum.ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use erm_core::tokenizer::CharTokenizer;
    use rand::SeedableRng;

    fn small_config() -> ErmConfig {
        ErmConfig {
            vocab_size: 16,
            seq_len: 8,
            hidden_dim: 8,
            num_blocks: 1,
            num_heads: 2,
            mlp_expansion: 2,
            dropout: 0.0,
            batch_size: 2,
            refinement_steps: 2,
            mask_rate_max: 0.6,
            mask_rate_min: 0.1,
            replace_rate_max: 0.05,
            replace_rate_min: 0.01,
            ..ErmConfig::default()
        }
    }

    fn make_scorer_and_dataset(cfg: &ErmConfig) -> (Scorer, TextDataset) {
        let text = "abcdefghijklmnopqrstuvwxyz ".repeat(30);
        let tokenizer = CharTokenizer::from_text(&text);
        let vocab = tokenizer.vocab_size();
        let erm_cfg = ErmConfig {
            vocab_size: vocab,
            ..*cfg
        };
        let scorer = Scorer::new(&erm_cfg, vocab, 7);
        let ds = TextDataset::from_text(&text, &tokenizer, cfg.seq_len).unwrap();
        (scorer, ds)
    }

    #[test]
    fn test_eval_metrics_serde_roundtrip() {
        let m = EvalMetrics {
            masked_token_accuracy: Some(0.75),
            token_entropy: Some(3.2),
            unique_token_ratio: Some(0.5),
            num_positions: 100,
            avg_loss: Some(1.5),
        };
        let json = serde_json::to_string(&m).unwrap();
        let m2: EvalMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(m, m2);
    }

    #[test]
    fn test_evaluate_denoising_returns_metrics() {
        let cfg = small_config();
        let (scorer, ds) = make_scorer_and_dataset(&cfg);
        let erm_cfg = ErmConfig {
            vocab_size: scorer.vocab_size,
            ..cfg
        };
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let metrics = evaluate_denoising(&scorer, &ds, &erm_cfg, 2, None, &mut rng).unwrap();
        assert!(metrics.masked_token_accuracy.is_some());
        let acc = metrics.masked_token_accuracy.unwrap();
        assert!((0.0..=1.0).contains(&acc));
        assert!(metrics.num_positions > 0);
    }

    #[test]
    fn test_evaluate_generation_returns_metrics() {
        let cfg = small_config();
        let text = "abcdefghijklmnopqrstuvwxyz ".repeat(30);
        let tokenizer = CharTokenizer::from_text(&text);
        let vocab = tokenizer.vocab_size();
        let erm_cfg = ErmConfig {
            vocab_size: vocab,
            ..cfg
        };
        let scorer = Scorer::new(&erm_cfg, vocab, 99);
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let metrics = evaluate_generation(&scorer, &erm_cfg, 2, &mut rng).unwrap();
        assert!(metrics.token_entropy.is_some());
        assert!(metrics.unique_token_ratio.is_some());
        let entropy = metrics.token_entropy.unwrap_or(0.0);
        let ratio = metrics.unique_token_ratio.unwrap_or(0.0);
        assert!(entropy >= 0.0);
        assert!((0.0..=1.0).contains(&ratio));
        assert!(metrics.num_positions > 0);
    }

    #[test]
    fn test_argmax_basic() {
        let logits = vec![0.1_f32, 0.5, 0.3, 0.9, 0.0];
        assert_eq!(argmax(&logits), 3);
    }
}
