//! Training loop for the ERM discrete denoiser.
//!
//! Milestone 1 implements forward-pass + loss computation only (no gradient
//! updates). The goal is proving the pipeline works end-to-end:
//!
//! 1. Sample a batch of token sequences.
//! 2. Sample a refinement step `t ~ Uniform[1, T]`.
//! 3. Corrupt `x → y_t` using the corruption schedule.
//! 4. Forward `y_t` through the scorer to get logits `[B, L, V]`.
//! 5. Compute `L_denoise = mean cross-entropy over corrupted positions`.
//! 6. Return the loss value.

use rand::Rng;
use rand_chacha::ChaCha8Rng;

use erm_core::config::ErmConfig;
use erm_core::corruption::corrupt;
use erm_core::error::ErmResult;
use erm_core::scorer::Scorer;

use crate::dataset::DataBatch;

/// Configuration for the training loop.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Total number of training steps.
    pub total_steps: usize,
    /// Log every N steps.
    pub log_every: usize,
    /// ERM hyperparameters.
    pub erm: ErmConfig,
    /// RNG seed.
    pub seed: u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            total_steps: 1000,
            log_every: 100,
            erm: ErmConfig::default(),
            seed: 42,
        }
    }
}

/// Result of a single training step.
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Training step number.
    pub step: usize,
    /// Denoising loss (mean CE over corrupted positions).
    pub loss: f32,
    /// Number of corrupted positions in the batch.
    pub num_corrupted: usize,
    /// Refinement step `t` used for corruption.
    pub t: usize,
}

/// Perform a single training step (forward pass + loss computation).
///
/// # Arguments
///
/// - `scorer`: the neural scorer network.
/// - `batch`: a batch of token sequences.
/// - `t`: refinement step for corruption (1 = lightest, T = heaviest).
///   If `None`, samples uniformly from `[1, T]`.
/// - `config`: ERM configuration.
/// - `rng`: random number generator.
///
/// # Returns
///
/// A [`StepResult`] with the denoising loss.
///
/// # Errors
///
/// Propagates errors from corruption or scorer forward pass.
pub fn train_step(
    scorer: &Scorer,
    batch: &DataBatch,
    t: Option<usize>,
    config: &ErmConfig,
    rng: &mut ChaCha8Rng,
) -> ErmResult<StepResult> {
    let b = batch.batch_size;
    let l = batch.seq_len;
    let v = scorer.vocab_size;

    // Sample refinement step t
    let t_val = t.unwrap_or_else(|| rng.gen_range(1..=config.refinement_steps));

    // Convert u32 tokens to i32 for the corruption module
    let x_i32: Vec<i32> = batch.tokens.iter().map(|&t| t as i32).collect();

    // Corrupt x → y_t
    let corruption = corrupt(&x_i32, t_val, config, rng)?;
    let y_t_flat = &corruption.y_t;

    // Convert y_t back to u32 for the scorer
    let y_t_u32: Vec<u32> = y_t_flat.iter().map(|&t| t as u32).collect();

    // Forward through scorer → logits [B, L, V]
    let output = scorer.forward(&y_t_u32, b)?;
    let logits = &output.logits;

    // Compute L_denoise = mean cross-entropy over corrupted positions
    let mut total_loss = 0.0_f32;
    let mut num_corrupted = 0_usize;

    for pos in 0..(b * l) {
        // Only count positions that were corrupted (y_t != x)
        if y_t_flat[pos] == x_i32[pos] {
            continue;
        }

        let target = x_i32[pos];
        // Skip if target is outside the output vocab range
        if target < 0 || (target as usize) >= v {
            continue;
        }
        let target_usize = target as usize;

        // Extract logits for this position: [V]
        let logit_start = pos * v;
        let logit_end = logit_start + v;
        let pos_logits = &logits[logit_start..logit_end];

        // Cross-entropy: -log(softmax(logits)[target])
        let ce = cross_entropy_loss(pos_logits, target_usize);
        total_loss += ce;
        num_corrupted += 1;
    }

    // Mean over corrupted positions (per AGENTS.md: mean, not sum)
    let loss = if num_corrupted > 0 {
        total_loss / num_corrupted as f32
    } else {
        0.0
    };

    Ok(StepResult {
        step: 0, // caller sets this
        loss,
        num_corrupted,
        t: t_val,
    })
}

/// Compute cross-entropy loss for a single position.
///
/// `CE = -log(softmax(logits)[target])`
///
/// Numerically stable: subtracts max logit before exponentiation.
fn cross_entropy_loss(logits: &[f32], target: usize) -> f32 {
    if logits.is_empty() || target >= logits.len() {
        return 0.0;
    }

    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&x| (x - max_val).exp()).sum();

    if exp_sum <= 0.0 {
        return 0.0;
    }

    // log_softmax[target] = (logits[target] - max_val) - log(sum_exp)
    let log_softmax_target = (logits[target] - max_val) - exp_sum.ln();
    -log_softmax_target
}

#[cfg(test)]
mod tests {
    use super::*;
    use erm_core::config::ErmConfig;
    use erm_core::tokenizer::CharTokenizer;
    use rand::SeedableRng;

    use crate::dataset::TextDataset;

    fn small_config() -> ErmConfig {
        ErmConfig {
            vocab_size: 32,
            seq_len: 16,
            hidden_dim: 16,
            num_blocks: 2,
            num_heads: 2,
            mlp_expansion: 4,
            dropout: 0.0,
            batch_size: 2,
            refinement_steps: 3,
            mask_rate_max: 0.8,
            mask_rate_min: 0.15,
            replace_rate_max: 0.1,
            replace_rate_min: 0.02,
            ..ErmConfig::default()
        }
    }

    fn make_dataset_and_scorer() -> (TextDataset, Scorer, ErmConfig) {
        let text = "the quick brown fox jumps over the lazy dog. ".repeat(100);
        let tokenizer = CharTokenizer::from_text(&text);
        let vocab = tokenizer.vocab_size();
        let cfg = ErmConfig {
            vocab_size: vocab,
            ..small_config()
        };
        let ds = TextDataset::from_text(&text, &tokenizer, cfg.seq_len).unwrap();
        let scorer = Scorer::new(&cfg, vocab, 42);
        (ds, scorer, cfg)
    }

    #[test]
    fn test_train_step_returns_finite_loss() {
        let (ds, scorer, cfg) = make_dataset_and_scorer();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let batch = ds.get_batch(2, &mut rng);

        let result = train_step(&scorer, &batch, Some(3), &cfg, &mut rng).unwrap();

        assert!(result.loss.is_finite(), "loss should be finite, got {}", result.loss);
        assert!(result.loss >= 0.0, "loss should be non-negative, got {}", result.loss);
    }

    #[test]
    fn test_train_step_loss_varies_with_corruption() {
        let (ds, scorer, cfg) = make_dataset_and_scorer();

        // Light corruption (t=1)
        let mut rng1 = ChaCha8Rng::seed_from_u64(100);
        let batch1 = ds.get_batch(2, &mut rng1);
        let result1 = train_step(&scorer, &batch1, Some(1), &cfg, &mut rng1).unwrap();

        // Heavy corruption (t=T)
        let mut rng2 = ChaCha8Rng::seed_from_u64(100);
        let batch2 = ds.get_batch(2, &mut rng2);
        let result2 = train_step(
            &scorer, &batch2, Some(cfg.refinement_steps), &cfg, &mut rng2,
        ).unwrap();

        // Both should be finite
        assert!(result1.loss.is_finite());
        assert!(result2.loss.is_finite());

        // More corruption → more corrupted positions
        assert!(
            result2.num_corrupted >= result1.num_corrupted,
            "heavier corruption should corrupt more positions: t=1 corrupted {}, t=T corrupted {}",
            result1.num_corrupted,
            result2.num_corrupted,
        );
    }

    #[test]
    fn test_train_step_deterministic() {
        let (ds, scorer, cfg) = make_dataset_and_scorer();

        let mut rng1 = ChaCha8Rng::seed_from_u64(77);
        let batch1 = ds.get_batch(2, &mut rng1);
        let r1 = train_step(&scorer, &batch1, Some(2), &cfg, &mut rng1).unwrap();

        let mut rng2 = ChaCha8Rng::seed_from_u64(77);
        let batch2 = ds.get_batch(2, &mut rng2);
        let r2 = train_step(&scorer, &batch2, Some(2), &cfg, &mut rng2).unwrap();

        assert_eq!(r1.loss, r2.loss);
        assert_eq!(r1.num_corrupted, r2.num_corrupted);
    }

    #[test]
    fn test_cross_entropy_basic() {
        // Uniform logits → CE = log(V)
        let logits = vec![0.0f32; 10];
        let ce = cross_entropy_loss(&logits, 0);
        let expected = (10.0_f32).ln();
        assert!(
            (ce - expected).abs() < 1e-5,
            "uniform logits CE: expected {expected}, got {ce}"
        );
    }

    #[test]
    fn test_cross_entropy_confident() {
        // One very high logit → CE ≈ 0
        let mut logits = vec![0.0f32; 10];
        logits[3] = 100.0;
        let ce = cross_entropy_loss(&logits, 3);
        assert!(ce < 0.01, "confident prediction CE should be near 0, got {ce}");
    }

    #[test]
    fn test_cross_entropy_wrong_prediction() {
        // High logit at wrong position → high CE
        let mut logits = vec![0.0f32; 10];
        logits[0] = 100.0;
        let ce = cross_entropy_loss(&logits, 5);
        assert!(ce > 1.0, "wrong prediction CE should be high, got {ce}");
    }

    #[test]
    fn test_train_step_auto_sample_t() {
        let (ds, scorer, cfg) = make_dataset_and_scorer();
        let mut rng = ChaCha8Rng::seed_from_u64(55);
        let batch = ds.get_batch(2, &mut rng);

        // Pass t=None to auto-sample
        let result = train_step(&scorer, &batch, None, &cfg, &mut rng).unwrap();
        assert!(result.loss.is_finite());
        assert!(result.t >= 1 && result.t <= cfg.refinement_steps);
    }
}
