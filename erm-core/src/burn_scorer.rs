//! Burn-based neural scorer for ERM (GPU-accelerated).
//!
//! This is a parallel implementation of [`crate::scorer::Scorer`] using the
//! [burn](https://burn.dev) deep learning framework, enabling GPU acceleration
//! via the wgpu backend.
//!
//! # Architecture
//!
//! ```text
//! y_t: [B, L]  ──→  token_emb [B, L, d]
//!                        + fourier_pos_emb [L, d] (fixed sinusoidal + learned projection)
//!                    ──→ interleaved blocks (num_blocks total):
//!                        even: linear(d → 4d) → ReLU → linear(4d → d) + residual
//!                        odd:  multi-head self-attention(d, num_heads) + residual
//!                    ──→ logit_head: linear(d → V)  → logits [B, L, V]
//!                    ──→ uncertainty_head: linear(d → 1) → sigmoid → u [B, L]
//! ```

use burn::nn;
use burn::nn::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig};
use burn::prelude::*;
use burn::tensor::TensorData;

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
    /// Number of attention heads for self-attention layers.
    pub num_heads: usize,
    /// MLP expansion factor (inner dim = expansion * d).
    pub mlp_expansion: usize,
    /// Dropout probability for attention layers.
    #[config(default = 0.0)]
    pub dropout: f64,
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
            num_heads: config.num_heads,
            mlp_expansion: config.mlp_expansion,
            dropout: config.dropout,
        }
    }

    /// Initialize the [`BurnScorer`] module on the given device.
    ///
    /// Creates `num_blocks` interleaved layers: even-indexed layers are
    /// feed-forward blocks, odd-indexed layers are self-attention blocks.
    /// This gives roughly half FF and half attention layers.
    pub fn init<B: Backend>(&self, device: &B::Device) -> BurnScorer<B> {
        let d = self.hidden_dim;
        let d_inner = d * self.mlp_expansion;

        let token_embed = nn::EmbeddingConfig::new(self.total_vocab, d).init(device);

        // Build fixed Fourier position embedding table [L, d]
        let fourier_table = build_fourier_pos_table(self.seq_len, d);
        let fourier_pos_embed =
            Tensor::<B, 2>::from_data(TensorData::new(fourier_table, [self.seq_len, d]), device);

        // Learned projection from Fourier features to hidden dim
        let pos_proj = nn::LinearConfig::new(d, d).init(device);

        // Interleaved blocks: even = FF, odd = self-attention.
        let mut ff_blocks = Vec::new();
        let mut attn_blocks = Vec::new();
        let mut block_order = Vec::new(); // false = FF, true = attention

        for i in 0..self.num_blocks {
            if i % 2 == 0 {
                // Feed-forward block
                ff_blocks.push(FeedForwardBlock {
                    linear1: nn::LinearConfig::new(d, d_inner).init(device),
                    linear2: nn::LinearConfig::new(d_inner, d).init(device),
                });
                block_order.push(false);
            } else {
                // Self-attention block
                attn_blocks.push(
                    MultiHeadAttentionConfig::new(d, self.num_heads)
                        .with_dropout(self.dropout)
                        .init(device),
                );
                block_order.push(true);
            }
        }

        let logit_head = nn::LinearConfig::new(d, self.vocab_size).init(device);
        let uncertainty_head = nn::LinearConfig::new(d, 1).init(device);
        let output_dropout = nn::DropoutConfig::new(self.dropout).init();

        BurnScorer {
            token_embed,
            fourier_pos_embed,
            pos_proj,
            ff_blocks,
            attn_blocks,
            block_order,
            logit_head,
            uncertainty_head,
            output_dropout,
            hidden_dim: d,
            seq_len: self.seq_len,
            vocab_size: self.vocab_size,
        }
    }
}

/// Build a fixed sinusoidal position embedding table.
///
/// Uses the standard transformer formula from "Attention Is All You Need":
///   PE(pos, 2i)   = sin(pos / 10000^(2i/d))
///   PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
///
/// This overcomes spectral bias (Tancik et al. 2020) by providing
/// multi-frequency positional features that let the network learn
/// fine-grained position-dependent patterns.
fn build_fourier_pos_table(seq_len: usize, d: usize) -> Vec<f32> {
    let mut table = vec![0.0f32; seq_len * d];
    for pos in 0..seq_len {
        for i in 0..d / 2 {
            let freq = 1.0 / (10000.0_f64.powf(2.0 * i as f64 / d as f64));
            let angle = pos as f64 * freq;
            table[pos * d + 2 * i] = angle.sin() as f32;
            table[pos * d + 2 * i + 1] = angle.cos() as f32;
        }
        // If d is odd, fill the last element with sin at highest frequency
        if d % 2 == 1 {
            let freq = 1.0 / (10000.0_f64.powf(2.0 * (d / 2) as f64 / d as f64));
            let angle = pos as f64 * freq;
            table[pos * d + d - 1] = angle.sin() as f32;
        }
    }
    table
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
/// for GPU acceleration. Uses fixed Fourier position embeddings with a
/// learned projection to overcome spectral bias. Interleaves feed-forward
/// and self-attention blocks for both local and global pattern learning.
#[derive(Module, Debug)]
pub struct BurnScorer<B: Backend> {
    /// Token embedding table. Shape: `[V_total, d]`.
    token_embed: nn::Embedding<B>,
    /// Fixed Fourier position embedding table. Shape: `[L, d]`.
    /// Not a learned parameter — sinusoidal features at multiple frequencies.
    #[module(skip)]
    fourier_pos_embed: Tensor<B, 2>,
    /// Learned projection from Fourier position features to hidden dim.
    pos_proj: nn::Linear<B>,
    /// Feed-forward blocks (even-indexed layers).
    ff_blocks: Vec<FeedForwardBlock<B>>,
    /// Self-attention blocks (odd-indexed layers).
    attn_blocks: Vec<MultiHeadAttention<B>>,
    /// Block execution order: `false` = FF, `true` = attention.
    #[module(skip)]
    block_order: Vec<bool>,
    /// Logit output head: `d → V`.
    logit_head: nn::Linear<B>,
    /// Uncertainty output head: `d → 1`.
    uncertainty_head: nn::Linear<B>,
    /// Dropout applied to hidden states before logit head projection.
    /// Prevents overfitting to a small set of output directions (mode collapse).
    output_dropout: nn::Dropout,
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
        let (logits, unc, _hidden) = self.forward_with_hidden(tokens);
        (logits, unc)
    }

    /// Forward pass returning logits, uncertainty, AND encoder hidden states.
    ///
    /// The hidden states `h` are the encoder outputs after all feed-forward
    /// blocks, before the logit/uncertainty heads. Shape: `[B, L, d]`.
    ///
    /// Use this when you need hidden states for route aggregation.
    pub fn forward_with_hidden(
        &self,
        tokens: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 3>, Tensor<B, 2>, Tensor<B, 3>) {
        let [batch_size, seq_len] = tokens.dims();

        // Token embedding: [B, L] → [B, L, d]
        let tok_emb = self.token_embed.forward(tokens);

        // Fourier position embedding: [L, d] → project → [L, d] → broadcast [B, L, d]
        let pos_feats = self.fourier_pos_embed.clone().unsqueeze::<3>(); // [1, L, d]
        let pos_emb = self.pos_proj.forward(pos_feats); // [1, L, d]
        let pos_emb = pos_emb.repeat_dim(0, batch_size); // [B, L, d]

        // h = tok_emb + pos_emb : [B, L, d]
        let mut h = tok_emb + pos_emb;

        // Interleaved feed-forward and self-attention blocks
        let mut ff_idx = 0;
        let mut attn_idx = 0;
        for &is_attn in &self.block_order {
            if is_attn {
                // Self-attention with residual connection
                let input = MhaInput::self_attn(h.clone());
                let output = self.attn_blocks[attn_idx].forward(input);
                h = h + output.context; // residual
                attn_idx += 1;
            } else {
                h = self.ff_blocks[ff_idx].forward(h);
                ff_idx += 1;
            }
        }

        // Apply output dropout before logit head to prevent mode collapse.
        // Use dropped hidden for logits only; keep clean h for uncertainty + hidden output.
        let h_dropped = self.output_dropout.forward(h.clone());

        // Logit head: [B, L, d] → [B, L, V]
        let logits = self.logit_head.forward(h_dropped);

        // Uncertainty head: [B, L, d] → [B, L, 1] → sigmoid → reshape → [B, L]
        let unc_raw = self.uncertainty_head.forward(h.clone());
        let unc = burn::tensor::activation::sigmoid(unc_raw).reshape([batch_size, seq_len]);

        (logits, unc, h)
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

    #[test]
    fn test_fourier_pos_table_values() {
        let table = build_fourier_pos_table(4, 8);
        // Check position 0 has sin(0)=0, cos(0)=1 at lowest frequency
        assert!((table[0] - 0.0).abs() < 1e-6, "sin(0) should be 0");
        assert!((table[1] - 1.0).abs() < 1e-6, "cos(0) should be 1");
        // Check no NaN/Inf
        for &v in &table {
            assert!(!v.is_nan(), "NaN in Fourier table");
            assert!(!v.is_infinite(), "Inf in Fourier table");
        }
    }
}
