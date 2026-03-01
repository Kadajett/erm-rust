//! Neural Scorer `Sθ` for the Emergent Route Model (Vec<f32> implementation).
//!
//! The scorer is a lightweight feed-forward network that takes corrupted tokens
//! and produces:
//!
//! - **logits** `[B, L, V]` — per-position token distributions
//! - **uncertainty** `[B, L]` — per-position uncertainty signal for leader targeting
//!
//! # Architecture (v1 — Milestone 1)
//!
//! ```text
//! y_t: [B, L]  ──→  token_emb [B, L, d]
//!                        + pos_emb [L, d]
//!                    ──→ 6 feed-forward blocks:
//!                        linear(d → 4d) → ReLU → linear(4d → d) + residual
//!                    ──→ logit_head: linear(d → V)  → logits [B, L, V]
//!                    ──→ uncertainty_head: linear(d → 1) → sigmoid → u [B, L]
//! ```
//!
//! All math is plain `Vec<f32>`. No burn tensors.

use rand::Rng;
use rand_chacha::ChaCha8Rng;

use crate::config::ErmConfig;
use crate::error::{ErmError, ErmResult};

/// A single feed-forward block with residual connection.
///
/// Architecture: `linear(d → 4d) → ReLU → linear(4d → d) + residual`.
#[derive(Debug, Clone)]
pub struct FeedForwardBlock {
    /// Up-projection weights. Shape: `[4d, d]` (row-major: `w[out][in]`).
    pub w_up: Vec<f32>,
    /// Up-projection bias. Shape: `[4d]`.
    pub b_up: Vec<f32>,
    /// Down-projection weights. Shape: `[d, 4d]` (row-major: `w[out][in]`).
    pub w_down: Vec<f32>,
    /// Down-projection bias. Shape: `[d]`.
    pub b_down: Vec<f32>,
    /// Input dimension `d`.
    pub d_in: usize,
    /// Inner dimension `4d`.
    pub d_inner: usize,
}

impl FeedForwardBlock {
    /// Initialize a feed-forward block with small random weights.
    ///
    /// Uses Xavier-like initialization: `N(0, 1/sqrt(fan_in))`.
    pub fn new(d: usize, expansion: usize, rng: &mut ChaCha8Rng) -> Self {
        let d_inner = d * expansion;
        let scale_up = 1.0 / (d as f32).sqrt();
        let scale_down = 1.0 / (d_inner as f32).sqrt();

        let w_up = random_vec(d_inner * d, scale_up, rng);
        let b_up = vec![0.0; d_inner];
        let w_down = random_vec(d * d_inner, scale_down, rng);
        let b_down = vec![0.0; d];

        Self {
            w_up,
            b_up,
            w_down,
            b_down,
            d_in: d,
            d_inner,
        }
    }

    /// Forward pass for a single position vector.
    ///
    /// Input: `x` of length `d`.
    /// Output: `x + down(relu(up(x)))` of length `d`.
    fn forward_vec(&self, x: &[f32]) -> Vec<f32> {
        // up: [4d] = W_up @ x + b_up
        let mut hidden = vec![0.0f32; self.d_inner];
        for i in 0..self.d_inner {
            let mut sum = self.b_up[i];
            let row_start = i * self.d_in;
            for j in 0..self.d_in {
                sum += self.w_up[row_start + j] * x[j];
            }
            // ReLU
            hidden[i] = sum.max(0.0);
        }

        // down: [d] = W_down @ hidden + b_down
        let mut out = vec![0.0f32; self.d_in];
        for i in 0..self.d_in {
            let mut sum = self.b_down[i];
            let row_start = i * self.d_inner;
            for j in 0..self.d_inner {
                sum += self.w_down[row_start + j] * hidden[j];
            }
            // Residual connection
            out[i] = x[i] + sum;
        }

        out
    }
}

/// The full neural scorer network (Vec<f32> implementation).
///
/// Contains token embedding, position embedding, feed-forward blocks,
/// and output heads for logits and uncertainty.
#[derive(Debug, Clone)]
pub struct Scorer {
    /// Token embedding table. Shape: `[V_total, d]` (row-major).
    /// `V_total = vocab_size` (includes PAD, MASK, and all chars).
    pub token_emb: Vec<f32>,
    /// Position embedding table. Shape: `[L, d]` (row-major).
    pub pos_emb: Vec<f32>,
    /// Feed-forward blocks.
    pub blocks: Vec<FeedForwardBlock>,
    /// Logit head weights. Shape: `[V, d]` (row-major: `w[out][in]`).
    pub logit_w: Vec<f32>,
    /// Logit head bias. Shape: `[V]`.
    pub logit_b: Vec<f32>,
    /// Uncertainty head weights. Shape: `[1, d]`.
    pub uncertainty_w: Vec<f32>,
    /// Uncertainty head bias. Shape: `[1]`.
    pub uncertainty_b: Vec<f32>,
    /// Vocabulary size for output logits `V` (number of real token classes).
    pub vocab_size: usize,
    /// Total vocabulary size for embedding lookup (includes special tokens).
    pub total_vocab: usize,
    /// Hidden dimension `d`.
    pub hidden_dim: usize,
    /// Maximum sequence length `L`.
    pub seq_len: usize,
}

/// Output of the scorer forward pass.
#[derive(Debug, Clone)]
pub struct ScorerOutput {
    /// Logits. Shape: flat `[B * L * V]` (row-major `[B, L, V]`).
    pub logits: Vec<f32>,
    /// Uncertainty. Shape: flat `[B * L]`.
    pub uncertainty: Vec<f32>,
}

impl Scorer {
    /// Initialize the scorer with small random weights.
    ///
    /// # Arguments
    ///
    /// - `config`: ERM configuration (provides dimensions).
    /// - `vocab_size`: total vocabulary size for the embedding table
    ///   (should match tokenizer's `vocab_size()`).
    /// - `seed`: RNG seed for deterministic initialization.
    pub fn new(config: &ErmConfig, vocab_size: usize, seed: u64) -> Self {
        use rand::SeedableRng;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let d = config.hidden_dim;
        let l = config.seq_len;
        let expansion = config.mlp_expansion;

        // Token embedding: [vocab_size, d]
        let emb_scale = 1.0 / (d as f32).sqrt();
        let token_emb = random_vec(vocab_size * d, emb_scale, &mut rng);

        // Position embedding: [L, d]
        let pos_emb = random_vec(l * d, emb_scale, &mut rng);

        // Feed-forward blocks
        let blocks: Vec<FeedForwardBlock> = (0..config.num_blocks)
            .map(|_| FeedForwardBlock::new(d, expansion, &mut rng))
            .collect();

        // Logit head: [V, d] where V = vocab_size (predicting over all tokens)
        let logit_scale = 1.0 / (d as f32).sqrt();
        let logit_w = random_vec(vocab_size * d, logit_scale, &mut rng);
        let logit_b = vec![0.0; vocab_size];

        // Uncertainty head: [1, d]
        let uncertainty_w = random_vec(d, logit_scale, &mut rng);
        let uncertainty_b = vec![0.0; 1];

        Self {
            token_emb,
            pos_emb,
            blocks,
            logit_w,
            logit_b,
            uncertainty_w,
            uncertainty_b,
            vocab_size,
            total_vocab: vocab_size,
            hidden_dim: d,
            seq_len: l,
        }
    }

    /// Forward pass through the scorer.
    ///
    /// # Arguments
    ///
    /// - `y_t`: corrupted token ids. Shape: flat `[B * L]`.
    ///   Values must be in `[0, total_vocab)`.
    /// - `batch_size`: the batch dimension `B`.
    ///
    /// # Returns
    ///
    /// A [`ScorerOutput`] with logits `[B, L, V]` and uncertainty `[B, L]`.
    ///
    /// # Errors
    ///
    /// Returns [`ErmError::ShapeMismatch`] if input dimensions don't match.
    pub fn forward(&self, y_t: &[u32], batch_size: usize) -> ErmResult<ScorerOutput> {
        let l = self.seq_len;
        let d = self.hidden_dim;
        let v = self.vocab_size;

        let expected_len = batch_size * l;
        if y_t.len() != expected_len {
            return Err(ErmError::ShapeMismatch {
                expected: format!("[{batch_size} * {l}] = {expected_len}"),
                got: format!("{}", y_t.len()),
            });
        }

        let mut logits = Vec::with_capacity(batch_size * l * v);
        let mut uncertainty = Vec::with_capacity(batch_size * l);

        for b in 0..batch_size {
            for pos in 0..l {
                let tok_id = y_t[b * l + pos] as usize;

                // Token embedding lookup: [d]
                let tok_start = if tok_id < self.total_vocab {
                    tok_id * d
                } else {
                    0 // fallback to id 0 for out-of-range
                };
                let pos_start = pos * d;

                // h = token_emb[tok_id] + pos_emb[pos]
                let mut h = vec![0.0f32; d];
                for i in 0..d {
                    h[i] = self.token_emb[tok_start + i] + self.pos_emb[pos_start + i];
                }

                // Pass through feed-forward blocks
                for block in &self.blocks {
                    h = block.forward_vec(&h);
                }

                // Logit head: [V] = W_logit @ h + b_logit
                for vi in 0..v {
                    let row_start = vi * d;
                    let mut sum = self.logit_b[vi];
                    for j in 0..d {
                        sum += self.logit_w[row_start + j] * h[j];
                    }
                    logits.push(sum);
                }

                // Uncertainty head: scalar = sigmoid(W_unc @ h + b_unc)
                let mut u_raw = self.uncertainty_b[0];
                for j in 0..d {
                    u_raw += self.uncertainty_w[j] * h[j];
                }
                let u = sigmoid(u_raw);
                uncertainty.push(u);
            }
        }

        Ok(ScorerOutput {
            logits,
            uncertainty,
        })
    }

    /// Collect all weights into a single flat vector (for serialization/inspection).
    #[must_use]
    pub fn num_parameters(&self) -> usize {
        let mut count = 0;
        count += self.token_emb.len();
        count += self.pos_emb.len();
        for block in &self.blocks {
            count += block.w_up.len();
            count += block.b_up.len();
            count += block.w_down.len();
            count += block.b_down.len();
        }
        count += self.logit_w.len();
        count += self.logit_b.len();
        count += self.uncertainty_w.len();
        count += self.uncertainty_b.len();
        count
    }
}

/// Sigmoid activation: `1 / (1 + exp(-x))`.
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// Generate a vector of random floats from `N(0, scale)`.
fn random_vec(len: usize, scale: f32, rng: &mut ChaCha8Rng) -> Vec<f32> {
    use rand_chacha::ChaCha8Rng as _;
    (0..len)
        .map(|_| {
            // Box-Muller approximation using uniform samples.
            let u1: f32 = rng.gen::<f32>().max(1e-10);
            let u2: f32 = rng.gen();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            z * scale
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> ErmConfig {
        ErmConfig {
            batch_size: 2,
            seq_len: 16,
            vocab_size: 64,
            hidden_dim: 32,
            num_blocks: 6,
            num_heads: 2,
            mlp_expansion: 4,
            dropout: 0.0,
            ..ErmConfig::default()
        }
    }

    #[test]
    fn test_scorer_forward_shapes() {
        let cfg = test_config();
        let vocab = 64;
        let scorer = Scorer::new(&cfg, vocab, 42);

        // Input: [B=2, L=16] as flat u32 ids
        let y_t = vec![0u32; 2 * 16];
        let out = scorer.forward(&y_t, 2).unwrap();

        // logits: [B, L, V] = [2, 16, 64]
        assert_eq!(out.logits.len(), 2 * 16 * 64);
        // uncertainty: [B, L] = [2, 16]
        assert_eq!(out.uncertainty.len(), 2 * 16);
    }

    #[test]
    fn test_scorer_no_nan_no_inf() {
        let cfg = ErmConfig {
            batch_size: 1,
            seq_len: 8,
            vocab_size: 32,
            hidden_dim: 16,
            num_blocks: 6,
            num_heads: 2,
            mlp_expansion: 4,
            dropout: 0.0,
            ..ErmConfig::default()
        };
        let scorer = Scorer::new(&cfg, 32, 123);

        let y_t = vec![0u32; 8];
        let out = scorer.forward(&y_t, 1).unwrap();

        for (i, &v) in out.logits.iter().enumerate() {
            assert!(!v.is_nan(), "NaN in logits at index {i}");
            assert!(!v.is_infinite(), "Inf in logits at index {i}");
        }

        for (i, &v) in out.uncertainty.iter().enumerate() {
            assert!(!v.is_nan(), "NaN in uncertainty at index {i}");
            assert!(
                (0.0..=1.0).contains(&v),
                "uncertainty[{i}] = {v} out of [0,1]"
            );
        }
    }

    #[test]
    fn test_scorer_deterministic() {
        let cfg = test_config();
        let scorer = Scorer::new(&cfg, 64, 42);

        let y_t = vec![3u32; 2 * 16];
        let out1 = scorer.forward(&y_t, 2).unwrap();
        let out2 = scorer.forward(&y_t, 2).unwrap();

        assert_eq!(out1.logits, out2.logits);
        assert_eq!(out1.uncertainty, out2.uncertainty);
    }

    #[test]
    fn test_scorer_shape_mismatch() {
        let cfg = test_config();
        let scorer = Scorer::new(&cfg, 64, 42);

        // Wrong input length
        let y_t = vec![0u32; 5];
        assert!(scorer.forward(&y_t, 2).is_err());
    }

    #[test]
    fn test_scorer_different_inputs_different_outputs() {
        let cfg = ErmConfig {
            seq_len: 4,
            hidden_dim: 8,
            num_blocks: 2,
            mlp_expansion: 4,
            ..ErmConfig::default()
        };
        let scorer = Scorer::new(&cfg, 10, 99);

        let y1 = vec![2u32, 3, 4, 5];
        let y2 = vec![5u32, 4, 3, 2];
        let out1 = scorer.forward(&y1, 1).unwrap();
        let out2 = scorer.forward(&y2, 1).unwrap();

        assert_ne!(out1.logits, out2.logits);
    }

    #[test]
    fn test_scorer_num_parameters() {
        let cfg = ErmConfig {
            seq_len: 8,
            hidden_dim: 16,
            num_blocks: 2,
            mlp_expansion: 4,
            ..ErmConfig::default()
        };
        let scorer = Scorer::new(&cfg, 10, 0);
        let n = scorer.num_parameters();
        // Should be positive and reasonable.
        assert!(n > 0);
        // token_emb: 10*16=160, pos_emb: 8*16=128, 2 blocks each with
        // w_up: 64*16=1024, b_up: 64, w_down: 16*64=1024, b_down: 16 → 2128 per block
        // logit_w: 10*16=160, logit_b: 10, unc_w: 16, unc_b: 1
        // Total: 160 + 128 + 2*2128 + 160 + 10 + 16 + 1 = 4731
        assert_eq!(n, 4731);
    }

    #[test]
    fn test_sigmoid_basic() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(100.0) > 0.99);
        assert!(sigmoid(-100.0) < 0.01);
    }
}
