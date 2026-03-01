//! Neural Scorer `Sθ` for the Emergent Route Model.
//!
//! The scorer is a lightweight transformer-encoder that takes corrupted tokens
//! and an optional route-conditioned message, then produces:
//!
//! - **logits** `[B, L, V]` — per-position token distributions
//! - **uncertainty** `[B, L]` — per-position uncertainty signal for leader targeting
//!
//! # Architecture
//!
//! ```text
//! y_t: [B, L]  ──→  token_emb [B, L, d]
//!                        + pos_emb [L, d]
//!                        + route_msg r [B, L, d]
//!                    ──→ TransformerEncoder (B blocks, pre-norm)
//!                    ──→ logit_head: Linear(d → V)  → logits [B, L, V]
//!                    ──→ uncertainty_head: Linear(d → 1) → sigmoid → u [B, L]
//! ```
//!
//! # VRAM estimates (default config: B=8, L=128, V=16384, d=256)
//!
//! - Scorer weights: ~48 MB (embedding table dominates)
//! - Logits tensor: 33.6 MB per forward pass (f16) — biggest single allocation
//! - Hidden states: 2.1 MB per step

use burn::nn::{
    Dropout, DropoutConfig, Embedding, EmbeddingConfig, Gelu, LayerNorm, LayerNormConfig, Linear,
    LinearConfig,
};
use burn::prelude::*;

use crate::config::ErmConfig;

/// Configuration for a single transformer encoder block.
///
/// Pre-norm style: LayerNorm → Attention → residual → LayerNorm → MLP → residual.
///
/// We implement self-attention manually (Q/K/V projections + scaled dot product)
/// rather than using burn's built-in TransformerEncoder, to maintain full control
/// over the attention mechanism (no causal mask — this is an encoder, not AR).
#[derive(Module, Debug)]
pub struct ScorerBlock<B: Backend> {
    /// Layer norm before self-attention.
    ln1: LayerNorm<B>,
    /// Query projection. Shape: `[d, d]`.
    wq: Linear<B>,
    /// Key projection. Shape: `[d, d]`.
    wk: Linear<B>,
    /// Value projection. Shape: `[d, d]`.
    wv: Linear<B>,
    /// Output projection after multi-head attention. Shape: `[d, d]`.
    wo: Linear<B>,
    /// Attention dropout.
    attn_dropout: Dropout,
    /// Layer norm before MLP.
    ln2: LayerNorm<B>,
    /// MLP up-projection. Shape: `[d, 4d]`.
    mlp_up: Linear<B>,
    /// MLP down-projection. Shape: `[4d, d]`.
    mlp_down: Linear<B>,
    /// GELU activation.
    gelu: Gelu,
    /// MLP dropout.
    mlp_dropout: Dropout,
    /// Number of attention heads (stored for reshaping).
    #[module(skip)]
    num_heads: usize,
}

/// Configuration to build a [`ScorerBlock`].
pub struct ScorerBlockConfig {
    /// Hidden dimension `d`.
    pub d_model: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// MLP expansion factor.
    pub mlp_expansion: usize,
    /// Dropout rate.
    pub dropout: f64,
}

impl ScorerBlockConfig {
    /// Initialize a [`ScorerBlock`] on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> ScorerBlock<B> {
        let d = self.d_model;
        let ff = d * self.mlp_expansion;

        ScorerBlock {
            ln1: LayerNormConfig::new(d).init(device),
            wq: LinearConfig::new(d, d).with_bias(false).init(device),
            wk: LinearConfig::new(d, d).with_bias(false).init(device),
            wv: LinearConfig::new(d, d).with_bias(false).init(device),
            wo: LinearConfig::new(d, d).with_bias(false).init(device),
            attn_dropout: DropoutConfig::new(self.dropout).init(),
            ln2: LayerNormConfig::new(d).init(device),
            mlp_up: LinearConfig::new(d, ff).init(device),
            mlp_down: LinearConfig::new(ff, d).init(device),
            gelu: Gelu::new(),
            mlp_dropout: DropoutConfig::new(self.dropout).init(),
            num_heads: self.num_heads,
        }
    }
}

impl<B: Backend> ScorerBlock<B> {
    /// Forward pass for a single encoder block.
    ///
    /// # Shape
    ///
    /// - Input: `[B, L, d]`
    /// - Output: `[B, L, d]`
    ///
    /// No causal mask — this is a bidirectional encoder.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, d_model] = x.dims();
        let head_dim = d_model / self.num_heads;

        // Pre-norm self-attention
        let normed = self.ln1.forward(x.clone());

        // Q, K, V projections: [B, L, d]
        let q = self.wq.forward(normed.clone());
        let k = self.wk.forward(normed.clone());
        let v = self.wv.forward(normed);

        // Reshape to [B, num_heads, L, head_dim]
        let q = q
            .reshape([batch, seq_len, self.num_heads, head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch, seq_len, self.num_heads, head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, seq_len, self.num_heads, head_dim])
            .swap_dims(1, 2);

        // Scaled dot-product attention: softmax(QK^T / sqrt(head_dim)) * V
        let scale = (head_dim as f64).sqrt();
        let attn_weights = q.matmul(k.swap_dims(2, 3)) / scale; // [B, h, L, L]
        let attn_weights = burn::tensor::activation::softmax(attn_weights, 3);
        let attn_weights = self.attn_dropout.forward(attn_weights);

        // [B, h, L, head_dim] → [B, L, d]
        let attn_out = attn_weights
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([batch, seq_len, d_model]);
        let attn_out = self.wo.forward(attn_out);

        // Residual
        let x = x + attn_out;

        // Pre-norm MLP
        let normed = self.ln2.forward(x.clone());
        let mlp_out = self.mlp_up.forward(normed);
        let mlp_out = self.gelu.forward(mlp_out);
        let mlp_out = self.mlp_down.forward(mlp_out);
        let mlp_out = self.mlp_dropout.forward(mlp_out);

        // Residual
        x + mlp_out
    }
}

/// The full neural scorer network.
///
/// Contains token embedding, position embedding, B transformer blocks,
/// and output heads for logits and uncertainty.
#[derive(Module, Debug)]
pub struct Scorer<B: Backend> {
    /// Token embedding table. Shape: `[V+1, d]` (includes MASK sentinel).
    token_emb: Embedding<B>,
    /// Learned position embedding. Shape: `[L, d]`.
    pos_emb: Embedding<B>,
    /// Transformer encoder blocks.
    blocks: Vec<ScorerBlock<B>>,
    /// Final layer norm.
    final_ln: LayerNorm<B>,
    /// Logit output head. Shape: `[d, V]` (no MASK in output — predict real tokens).
    logit_head: Linear<B>,
    /// Uncertainty output head. Shape: `[d, 1]`.
    uncertainty_head: Linear<B>,
}

/// Configuration to build a [`Scorer`].
pub struct ScorerConfig {
    /// Full ERM config.
    pub erm: ErmConfig,
}

impl ScorerConfig {
    /// Initialize the scorer on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Scorer<B> {
        let c = &self.erm;

        // V+1 embeddings (real vocab + MASK sentinel)
        let token_emb = EmbeddingConfig::new(c.total_vocab_size(), c.hidden_dim).init(device);
        let pos_emb = EmbeddingConfig::new(c.seq_len, c.hidden_dim).init(device);

        let block_cfg = ScorerBlockConfig {
            d_model: c.hidden_dim,
            num_heads: c.num_heads,
            mlp_expansion: c.mlp_expansion,
            dropout: c.dropout,
        };

        let blocks: Vec<_> = (0..c.num_blocks).map(|_| block_cfg.init(device)).collect();

        let final_ln = LayerNormConfig::new(c.hidden_dim).init(device);

        // Output logits over real vocab (V tokens, not V+1).
        let logit_head = LinearConfig::new(c.hidden_dim, c.vocab_size)
            .with_bias(false)
            .init(device);

        let uncertainty_head = LinearConfig::new(c.hidden_dim, 1).init(device);

        Scorer {
            token_emb,
            pos_emb,
            blocks,
            final_ln,
            logit_head,
            uncertainty_head,
        }
    }
}

impl<B: Backend> Scorer<B> {
    /// Forward pass.
    ///
    /// # Arguments
    ///
    /// - `y_t`: corrupted token ids `[B, L]` (i32)
    /// - `route_msg`: route-conditioned message `[B, L, d]` (from RouteAggregate).
    ///   Pass zeros if no route graph is wired yet (Phase 1).
    ///
    /// # Returns
    ///
    /// `(logits, uncertainty)` where:
    /// - `logits`: `[B, L, V]` (f32) — unnormalized token scores
    /// - `uncertainty`: `[B, L]` (f32) — sigmoid output in `[0, 1]`
    pub fn forward(
        &self,
        y_t: Tensor<B, 2, Int>,
        route_msg: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let [batch, seq_len] = y_t.dims();
        let device = y_t.device();

        // Token embedding: [B, L, d]
        let tok_emb = self.token_emb.forward(y_t);

        // Position ids: [L] → [1, L] → broadcast to [B, L]
        let pos_ids = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device)
            .unsqueeze::<2>()
            .repeat_dim(0, batch);
        let p_emb = self.pos_emb.forward(pos_ids);

        // Combine: token_emb + pos_emb + route_msg
        let mut h = tok_emb + p_emb + route_msg;

        // Transformer blocks
        for block in &self.blocks {
            h = block.forward(h);
        }

        // Final layer norm
        h = self.final_ln.forward(h);

        // Output heads
        let logits = self.logit_head.forward(h.clone()); // [B, L, V]
        let u_raw = self.uncertainty_head.forward(h); // [B, L, 1]
                                                      // Reshape [B, L, 1] → [B, L] instead of squeeze (which requires exactly one dim of size 1).
        let u_flat: Tensor<B, 2> = u_raw.reshape([batch, seq_len]);
        let uncertainty = burn::tensor::activation::sigmoid(u_flat); // [B, L]

        (logits, uncertainty)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_scorer_forward_shapes() {
        let cfg = ErmConfig {
            batch_size: 2,
            seq_len: 16,
            vocab_size: 64,
            hidden_dim: 32,
            num_blocks: 2,
            num_heads: 2,
            mlp_expansion: 4,
            dropout: 0.0,
            ..ErmConfig::default()
        };
        let device = Default::default();
        let scorer = ScorerConfig { erm: cfg.clone() }.init::<TestBackend>(&device);

        // Create input: [B=2, L=16]
        let y_t = Tensor::<TestBackend, 2, Int>::zeros([2, 16], &device);
        let route_msg = Tensor::<TestBackend, 3>::zeros([2, 16, 32], &device);

        let (logits, uncertainty) = scorer.forward(y_t, route_msg);

        assert_eq!(logits.dims(), [2, 16, 64]);
        assert_eq!(uncertainty.dims(), [2, 16]);
    }

    #[test]
    fn test_scorer_no_nan() {
        let cfg = ErmConfig {
            batch_size: 1,
            seq_len: 8,
            vocab_size: 32,
            hidden_dim: 16,
            num_blocks: 1,
            num_heads: 2,
            mlp_expansion: 4,
            dropout: 0.0,
            ..ErmConfig::default()
        };
        let device = Default::default();
        let scorer = ScorerConfig { erm: cfg }.init::<TestBackend>(&device);

        let y_t = Tensor::<TestBackend, 2, Int>::zeros([1, 8], &device);
        let route_msg = Tensor::<TestBackend, 3>::zeros([1, 8, 16], &device);

        let (logits, uncertainty) = scorer.forward(y_t, route_msg);

        // Check no NaN in logits
        let logits_data = logits.to_data();
        for &v in logits_data.as_slice::<f32>().unwrap() {
            assert!(!v.is_nan(), "NaN found in logits");
            assert!(!v.is_infinite(), "Inf found in logits");
        }

        // Check no NaN in uncertainty
        let u_data = uncertainty.to_data();
        for &v in u_data.as_slice::<f32>().unwrap() {
            assert!(!v.is_nan(), "NaN found in uncertainty");
            assert!((0.0..=1.0).contains(&v), "uncertainty {v} out of [0,1]");
        }
    }
}
