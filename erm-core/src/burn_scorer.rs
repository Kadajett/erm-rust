//! Burn-based neural scorer for ERM (GPU-accelerated).
//!
//! This is a parallel implementation of [`crate::scorer::Scorer`] using the
//! [burn](https://burn.dev) deep learning framework, enabling GPU acceleration
//! via the wgpu backend.
//!
//! # Architecture
//!
//! Identical to the Vec<f32> scorer:
//!
//! ```text
//! y_t: [B, L]  ──→  token_emb [B, L, d]
//!                        + pos_emb [L, d]
//!                    ──→ 6 feed-forward blocks:
//!                        linear(d → 4d) → ReLU → linear(4d → d) + residual
//!                    ──→ logit_head: linear(d → V)  → logits [B, L, V]
//!                    ──→ uncertainty_head: linear(d → 1) → sigmoid → u [B, L]
//! ```

use burn::nn;
use burn::prelude::*;

use crate::config::ErmConfig;

/// Configuration for [`BurnScorer`]. Mirrors the relevant parts of [`ErmConfig`].
///
/// Implements `burn::config::Config` via derive so it can produce modules.
#[derive(Config, Debug)]
pub struct BurnScorerConfig {
    /// Total vocabulary size (including MASK sentinel for embedding).
    pub total_vocab: usize,
    /// Output vocabulary size (number of classes for logit head).
    pub vocab_size: usize,
    /// Hidden dimension `d`.
    pub hidden_dim: usize,
    /// Maximum sequence length `L`.
    pub seq_len: usize,
    /// Number of feed-forward blocks.
    pub num_blocks: usize,
    /// MLP expansion factor (inner dim = expansion * d).
    pub mlp_expansion: usize,
}

impl BurnScorerConfig {
    /// Create from an [`ErmConfig`].
    pub fn from_erm(config: &ErmConfig) -> Self {
        Self {
            total_vocab: config.total_vocab_size(),
            vocab_size: config.total_vocab_size(),
            hidden_dim: config.hidden_dim,
            seq_len: config.seq_len,
            num_blocks: config.num_blocks,
            mlp_expansion: config.mlp_expansion,
        }
    }

    /// Initialize the [`BurnScorer`] module on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> BurnScorer<B> {
        let d = self.hidden_dim;
        let d_inner = d * self.mlp_expansion;

        let token_embed = nn::EmbeddingConfig::new(self.total_vocab, d).init(device);
        let pos_embed = nn::EmbeddingConfig::new(self.seq_len, d).init(device);

        let blocks: Vec<FeedForwardBlock<B>> = (0..self.num_blocks)
            .map(|_| FeedForwardBlock {
                linear1: nn::LinearConfig::new(d, d_inner).init(device),
                linear2: nn::LinearConfig::new(d_inner, d).init(device),
            })
            .collect();

        let logit_head = nn::LinearConfig::new(d, self.vocab_size).init(device);
        let uncertainty_head = nn::LinearConfig::new(d, 1).init(device);

        BurnScorer {
            token_embed,
            pos_embed,
            blocks,
            logit_head,
            uncertainty_head,
            hidden_dim: d,
            seq_len: self.seq_len,
            vocab_size: self.vocab_size,
        }
    }
}

/// A single feed-forward block with residual connection (burn version).
///
/// Architecture: `linear(d → 4d) → ReLU → linear(4d → d) + residual`.
#[derive(Module, Debug)]
pub struct FeedForwardBlock<B: Backend> {
    /// Up-projection: `d → 4d`.
    linear1: nn::Linear<B>,
    /// Down-projection: `4d → d`.
    linear2: nn::Linear<B>,
}

impl<B: Backend> FeedForwardBlock<B> {
    /// Forward pass: `x + down(relu(up(x)))`.
    ///
    /// Input shape: `[..., d]`. Output shape: `[..., d]`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden = self.linear1.forward(x.clone());
        let hidden = burn::tensor::activation::relu(hidden);
        let out = self.linear2.forward(hidden);
        x + out // residual
    }
}

/// Burn-based neural scorer for ERM.
///
/// Drop-in replacement for [`crate::scorer::Scorer`] that uses burn tensors
/// for GPU acceleration. The architecture is identical.
#[derive(Module, Debug)]
pub struct BurnScorer<B: Backend> {
    /// Token embedding table. Shape: `[V_total, d]`.
    token_embed: nn::Embedding<B>,
    /// Position embedding table. Shape: `[L, d]`.
    pos_embed: nn::Embedding<B>,
    /// Feed-forward blocks with residual connections.
    blocks: Vec<FeedForwardBlock<B>>,
    /// Logit output head: `d → V`.
    logit_head: nn::Linear<B>,
    /// Uncertainty output head: `d → 1`.
    uncertainty_head: nn::Linear<B>,
    /// Hidden dimension (not a parameter, stored for shape checks).
    #[module(skip)]
    hidden_dim: usize,
    /// Sequence length (not a parameter, stored for shape checks).
    #[module(skip)]
    seq_len: usize,
    /// Vocabulary size (not a parameter, stored for shape checks).
    #[module(skip)]
    vocab_size: usize,
}

impl<B: Backend> BurnScorer<B> {
    /// Forward pass through the scorer.
    ///
    /// # Arguments
    ///
    /// - `tokens`: corrupted token ids. Shape: `[B, L]` (Int tensor).
    ///
    /// # Returns
    ///
    /// A tuple of:
    /// - `logits`: `[B, L, V]` — per-position token distributions.
    /// - `uncertainty`: `[B, L]` — per-position uncertainty (sigmoid output).
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let device = tokens.device();
        let [batch_size, seq_len] = tokens.dims();

        // Token embedding: [B, L] → [B, L, d]
        let tok_emb = self.token_embed.forward(tokens);

        // Position indices: [L] → broadcast to [B, L]
        let pos_ids = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device)
            .unsqueeze::<2>() // [1, L]
            .repeat_dim(0, batch_size); // [B, L]
        let pos_emb = self.pos_embed.forward(pos_ids);

        // h = tok_emb + pos_emb : [B, L, d]
        let mut h = tok_emb + pos_emb;

        // Feed-forward blocks
        for block in &self.blocks {
            h = block.forward(h);
        }

        // Logit head: [B, L, d] → [B, L, V]
        let logits = self.logit_head.forward(h.clone());

        // Uncertainty head: [B, L, d] → [B, L, 1] → sigmoid → reshape → [B, L]
        let unc_raw = self.uncertainty_head.forward(h);
        let unc = burn::tensor::activation::sigmoid(unc_raw).reshape([batch_size, seq_len]);

        (logits, unc)
    }

    /// Get the vocabulary size this scorer was built for.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get the hidden dimension.
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Get the sequence length.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn test_config() -> ErmConfig {
        ErmConfig {
            batch_size: 2,
            seq_len: 16,
            vocab_size: 64,
            hidden_dim: 32,
            num_blocks: 2,
            num_heads: 2,
            mlp_expansion: 4,
            dropout: 0.0,
            ..ErmConfig::default()
        }
    }

    #[test]
    fn test_burn_scorer_forward_shapes() {
        let cfg = test_config();
        let scorer_cfg = BurnScorerConfig::from_erm(&cfg);
        let device = Default::default();
        let scorer = scorer_cfg.init::<TestBackend>(&device);

        // Input: [B=2, L=16]
        let tokens = Tensor::<TestBackend, 2, Int>::zeros([2, 16], &device);

        let (logits, uncertainty) = scorer.forward(tokens);

        // logits: [2, 16, total_vocab=65]
        assert_eq!(logits.dims(), [2, 16, cfg.total_vocab_size()]);
        // uncertainty: [2, 16]
        assert_eq!(uncertainty.dims(), [2, 16]);
    }

    #[test]
    fn test_burn_scorer_no_nan() {
        let cfg = ErmConfig {
            batch_size: 1,
            seq_len: 8,
            vocab_size: 32,
            hidden_dim: 16,
            num_blocks: 2,
            num_heads: 2,
            mlp_expansion: 4,
            dropout: 0.0,
            ..ErmConfig::default()
        };
        let scorer_cfg = BurnScorerConfig::from_erm(&cfg);
        let device = Default::default();
        let scorer = scorer_cfg.init::<TestBackend>(&device);

        let tokens = Tensor::<TestBackend, 2, Int>::zeros([1, 8], &device);
        let (logits, uncertainty) = scorer.forward(tokens);

        let logits_data = logits.into_data();
        for &v in logits_data.as_slice::<f32>().unwrap() {
            assert!(!v.is_nan(), "NaN in logits");
            assert!(!v.is_infinite(), "Inf in logits");
        }

        let unc_data = uncertainty.into_data();
        for &v in unc_data.as_slice::<f32>().unwrap() {
            assert!(!v.is_nan(), "NaN in uncertainty");
            assert!((0.0..=1.0).contains(&v), "uncertainty {v} out of [0,1]");
        }
    }

    #[test]
    fn test_burn_scorer_different_inputs_different_outputs() {
        let cfg = ErmConfig {
            seq_len: 4,
            hidden_dim: 8,
            num_blocks: 2,
            mlp_expansion: 4,
            vocab_size: 10,
            ..ErmConfig::default()
        };
        let scorer_cfg = BurnScorerConfig::from_erm(&cfg);
        let device = Default::default();
        let scorer = scorer_cfg.init::<TestBackend>(&device);

        let t1 = Tensor::<TestBackend, 2, Int>::from_data([[2, 3, 4, 5]], &device);
        let t2 = Tensor::<TestBackend, 2, Int>::from_data([[5, 4, 3, 2]], &device);

        let (logits1, _) = scorer.forward(t1);
        let (logits2, _) = scorer.forward(t2);

        let d1 = logits1.into_data();
        let d2 = logits2.into_data();
        assert_ne!(d1.as_slice::<f32>().unwrap(), d2.as_slice::<f32>().unwrap());
    }

    #[test]
    fn test_burn_scorer_deterministic() {
        let cfg = test_config();
        let scorer_cfg = BurnScorerConfig::from_erm(&cfg);
        let device = Default::default();
        let scorer = scorer_cfg.init::<TestBackend>(&device);

        let tokens = Tensor::<TestBackend, 2, Int>::from_data([[3; 16], [3; 16]], &device);

        let (l1, u1) = scorer.forward(tokens.clone());
        let (l2, u2) = scorer.forward(tokens);

        assert_eq!(
            l1.into_data().as_slice::<f32>().unwrap(),
            l2.into_data().as_slice::<f32>().unwrap()
        );
        assert_eq!(
            u1.into_data().as_slice::<f32>().unwrap(),
            u2.into_data().as_slice::<f32>().unwrap()
        );
    }
}
