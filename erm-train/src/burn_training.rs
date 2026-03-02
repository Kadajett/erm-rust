//! Burn-based training step with autodiff gradient computation.
//!
//! Provides a single training step that:
//! 1. Corrupts input tokens using the existing corruption module.
//! 2. Forwards through the [`BurnScorer`] to produce logits.
//! 3. Computes cross-entropy loss over corrupted positions.
//! 4. Backpropagates to update model weights via Adam.
//!
//! This module uses burn's autodiff backend for automatic differentiation,
//! enabling GPU-accelerated training via the wgpu backend.

use burn::optim::decay::WeightDecayConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use erm_core::burn_scorer::{BurnScorer, BurnScorerConfig};
use erm_core::config::ErmConfig;
use erm_core::corruption::corrupt;
use erm_core::error::ErmResult;

use rand::Rng;
use rand_chacha::ChaCha8Rng;

use crate::dataset::DataBatch;

/// State for the burn-based training loop.
pub struct BurnTrainer<B: AutodiffBackend> {
    /// The burn scorer model.
    pub scorer: BurnScorer<B>,
    /// Adam optimizer.
    optimizer: burn::optim::adaptor::OptimizerAdaptor<burn::optim::Adam, BurnScorer<B>, B>,
    /// Learning rate.
    lr: f64,
    /// Device.
    device: B::Device,
}

impl<B: AutodiffBackend> BurnTrainer<B> {
    /// Create a new burn trainer.
    ///
    /// # Arguments
    ///
    /// - `config`: ERM configuration.
    /// - `device`: burn device to place tensors on.
    pub fn new(config: &ErmConfig, device: B::Device) -> Self {
        let scorer_cfg = BurnScorerConfig::from_erm(config);
        let scorer = scorer_cfg.init::<B>(&device);

        let optimizer = AdamConfig::new()
            .with_weight_decay(Some(WeightDecayConfig::new(config.weight_decay as f32)))
            .init();

        Self {
            scorer,
            optimizer,
            lr: config.learning_rate,
            device,
        }
    }

    /// Perform a single training step: forward + loss + backward + optimizer step.
    ///
    /// # Arguments
    ///
    /// - `batch`: a batch of token sequences (from `TextDataset`).
    /// - `t`: refinement step for corruption (1 = lightest, T = heaviest).
    ///   If `None`, samples uniformly from `[1, T]`.
    /// - `config`: ERM configuration.
    /// - `rng`: random number generator for corruption.
    ///
    /// # Returns
    ///
    /// The scalar loss value as f32.
    ///
    /// # Errors
    ///
    /// Propagates errors from corruption.
    pub fn train_step(
        &mut self,
        batch: &DataBatch,
        t: Option<usize>,
        config: &ErmConfig,
        rng: &mut ChaCha8Rng,
    ) -> ErmResult<f32> {
        let b = batch.batch_size;
        let l = batch.seq_len;

        // Sample refinement step t
        let t_val = t.unwrap_or_else(|| rng.gen_range(1..=config.refinement_steps));

        // Corrupt x → y_t using existing corruption module
        let x_i32: Vec<i32> = batch.tokens.iter().map(|&t| t as i32).collect();
        let corruption = corrupt(&x_i32, t_val, config, rng)?;
        let y_t_flat = &corruption.y_t;

        // Build mask of corrupted positions and target labels
        let mut mask_vals = Vec::with_capacity(b * l);
        let mut target_vals = Vec::with_capacity(b * l);
        for pos in 0..(b * l) {
            if y_t_flat[pos] != x_i32[pos] && x_i32[pos] >= 0 {
                mask_vals.push(1.0_f32);
                target_vals.push(x_i32[pos] as i64);
            } else {
                mask_vals.push(0.0_f32);
                target_vals.push(0); // won't be used, masked out
            }
        }

        let num_corrupted: usize = mask_vals.iter().filter(|&&v| v > 0.5).count();
        if num_corrupted == 0 {
            return Ok(0.0);
        }

        // Convert to burn tensors
        let y_t_i64: Vec<i64> = y_t_flat.iter().map(|&v| v as i64).collect();
        let tokens_tensor =
            Tensor::<B, 2, Int>::from_data(TensorData::new(y_t_i64, [b, l]), &self.device);

        // Forward pass
        let (logits, _uncertainty) = self.scorer.forward(tokens_tensor);

        // logits: [B, L, V] → reshape to [B*L, V]
        let v = logits.dims()[2];
        let logits_flat = logits.reshape([b * l, v]);

        // Targets: [B*L]
        let targets_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::new(target_vals, [b * l]), &self.device);

        // Mask: [B*L]
        let mask_tensor =
            Tensor::<B, 1>::from_data(TensorData::new(mask_vals, [b * l]), &self.device);

        // Compute per-position cross-entropy: log_softmax + nll
        let log_probs = burn::tensor::activation::log_softmax(logits_flat, 1); // [B*L, V]

        // Gather target log-probs: [B*L]
        let targets_2d = targets_tensor.clone().unsqueeze_dim(1); // [B*L, 1]
        let target_log_probs = log_probs.gather(1, targets_2d).reshape([b * l]); // [B*L]

        // Masked mean of -log_probs
        let neg_log_probs = target_log_probs.neg(); // [B*L]
        let masked_loss = neg_log_probs * mask_tensor.clone(); // [B*L]
        let loss = masked_loss.sum() / mask_tensor.sum(); // scalar

        // Extract loss value before backward
        let loss_val = loss.clone().into_data().as_slice::<f32>().unwrap()[0];

        // Backward + optimizer step
        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &self.scorer);
        self.scorer = self
            .optimizer
            .step(self.lr, self.scorer.clone(), grads_params);

        Ok(loss_val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_autodiff::Autodiff;
    use burn_ndarray::NdArray;
    use erm_core::config::ErmConfig;
    use erm_core::tokenizer::CharTokenizer;
    use rand::SeedableRng;

    use crate::dataset::TextDataset;

    type TestAutodiffBackend = Autodiff<NdArray<f32>>;

    fn small_config() -> ErmConfig {
        ErmConfig {
            vocab_size: 32,
            seq_len: 16,
            hidden_dim: 16,
            num_blocks: 2,
            num_heads: 2,
            mlp_expansion: 4,
            dropout: 0.0,
            batch_size: 4,
            refinement_steps: 3,
            mask_rate_max: 0.8,
            mask_rate_min: 0.15,
            replace_rate_max: 0.1,
            replace_rate_min: 0.02,
            learning_rate: 1e-3,
            weight_decay: 0.0,
            ..ErmConfig::default()
        }
    }

    fn make_dataset(cfg: &ErmConfig) -> (TextDataset, usize) {
        let text = "the quick brown fox jumps over the lazy dog. ".repeat(100);
        let tokenizer = CharTokenizer::from_text(&text);
        let vocab = tokenizer.vocab_size();
        let ds = TextDataset::from_text(&text, &tokenizer, cfg.seq_len).unwrap();
        (ds, vocab)
    }

    #[test]
    fn test_burn_train_step_returns_finite_loss() {
        let mut cfg = small_config();
        let (ds, vocab) = make_dataset(&cfg);
        cfg.vocab_size = vocab;
        let device = Default::default();
        let mut trainer = BurnTrainer::<TestAutodiffBackend>::new(&cfg, device);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let batch = ds.get_batch(cfg.batch_size, &mut rng);

        let loss = trainer.train_step(&batch, Some(3), &cfg, &mut rng).unwrap();

        assert!(loss.is_finite(), "loss should be finite, got {loss}");
        assert!(loss >= 0.0, "loss should be non-negative, got {loss}");
    }

    #[test]
    fn test_burn_loss_decreases_over_steps() {
        let mut cfg = small_config();
        let (ds, vocab) = make_dataset(&cfg);
        cfg.vocab_size = vocab;
        let device = Default::default();
        let mut trainer = BurnTrainer::<TestAutodiffBackend>::new(&cfg, device);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let mut losses = Vec::new();
        for _ in 0..10 {
            let batch = ds.get_batch(cfg.batch_size, &mut rng);
            let loss = trainer
                .train_step(&batch, Some(cfg.refinement_steps), &cfg, &mut rng)
                .unwrap();
            losses.push(loss);
        }

        // Average of first 3 should be higher than average of last 3
        let early_avg: f32 = losses[..3].iter().sum::<f32>() / 3.0;
        let late_avg: f32 = losses[7..].iter().sum::<f32>() / 3.0;

        assert!(
            late_avg < early_avg,
            "loss should decrease: early_avg={early_avg:.4}, late_avg={late_avg:.4}, all={losses:?}"
        );
    }
}
