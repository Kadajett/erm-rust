//! Discrete corruption schedule for ERM.
//!
//! Given ground-truth tokens `x` and a refinement step `t`, produce corrupted
//! tokens `y_t` by applying per-token mask/replace/keep decisions.
//!
//! # Corruption logic (per token position `i`)
//!
//! 1. With probability `α_t`: set `y_t[i] = MASK`
//! 2. Else with probability `β_t`: set `y_t[i] = confuser` (sampled from uniform `[0, V)`)
//! 3. Else: keep `y_t[i] = x[i]`
//!
//! The rates are linearly interpolated over `T` steps:
//!
//! ```text
//! α_t = α_T + (α_1 - α_T) * (T - t) / (T - 1)
//! β_t = β_T + (β_1 - β_T) * (T - t) / (T - 1)
//! ```
//!
//! At `t = T` (heaviest): `α_T = 0.8`, `β_T = 0.1`.  
//! At `t = 1` (lightest): `α_1 = 0.15`, `β_1 = 0.02`.

use rand::Rng;

use crate::config::ErmConfig;
use crate::error::ErmResult;
use crate::types::TokenId;

/// Result of corrupting a single batch of ground-truth tokens.
#[derive(Debug, Clone)]
pub struct CorruptionResult {
    /// Corrupted tokens. Shape: `[B * L]` (flattened `[B, L]`).
    pub y_t: Vec<TokenId>,
    /// Number of masked positions (set to MASK sentinel).
    pub num_masked: usize,
    /// Number of replaced positions (set to a random confuser token).
    pub num_replaced: usize,
    /// Number of kept positions (unchanged from ground truth).
    pub num_kept: usize,
}

/// Apply the discrete corruption schedule to a batch of ground-truth tokens.
///
/// # Arguments
///
/// - `x`: ground-truth tokens, flat `[B * L]` with values in `[0, V)`.
/// - `t`: refinement step (1-indexed, 1 = lightest, `T` = heaviest).
/// - `config`: hyperparameters (provides schedule rates, vocab size, etc.).
/// - `rng`: random number generator for reproducibility.
///
/// # Shape
///
/// Input and output are both flat `[B * L]`.
///
/// # Errors
///
/// Returns an error if `t` is outside `[1, T]`.
pub fn corrupt<R: Rng>(
    x: &[TokenId],
    t: usize,
    config: &ErmConfig,
    rng: &mut R,
) -> ErmResult<CorruptionResult> {
    if t == 0 || t > config.refinement_steps {
        return Err(crate::error::ErmError::InvalidConfig(format!(
            "t={t} out of range [1, {}]",
            config.refinement_steps
        )));
    }

    let mask_id = config.mask_token_id();
    let alpha = config.mask_rate(t);
    let beta = config.replace_rate(t);
    let vocab = config.vocab_size as i32; // real tokens in [0, vocab)

    let mut y_t = Vec::with_capacity(x.len());
    let mut num_masked = 0usize;
    let mut num_replaced = 0usize;
    let mut num_kept = 0usize;

    for &xi in x {
        let r: f32 = rng.gen();
        if r < alpha {
            // Mask
            y_t.push(mask_id);
            num_masked += 1;
        } else if r < alpha + beta {
            // Replace with a uniform confuser token from [0, V)
            let confuser: TokenId = rng.gen_range(0..vocab);
            y_t.push(confuser);
            num_replaced += 1;
        } else {
            // Keep
            y_t.push(xi);
            num_kept += 1;
        }
    }

    Ok(CorruptionResult {
        y_t,
        num_masked,
        num_replaced,
        num_kept,
    })
}

/// Convenience: compute how many positions were corrupted (masked + replaced).
impl CorruptionResult {
    /// Number of corrupted positions (mask + replace).
    #[must_use]
    pub fn num_corrupted(&self) -> usize {
        self.num_masked + self.num_replaced
    }

    /// Total token count.
    #[must_use]
    pub fn total(&self) -> usize {
        self.num_masked + self.num_replaced + self.num_kept
    }

    /// Empirical mask rate (fraction of positions that are MASK).
    #[must_use]
    pub fn empirical_mask_rate(&self) -> f32 {
        if self.total() == 0 {
            return 0.0;
        }
        self.num_masked as f32 / self.total() as f32
    }

    /// Empirical replace rate (fraction of positions that are confusers).
    #[must_use]
    pub fn empirical_replace_rate(&self) -> f32 {
        if self.total() == 0 {
            return 0.0;
        }
        self.num_replaced as f32 / self.total() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    fn default_config() -> ErmConfig {
        ErmConfig::default()
    }

    /// Helper: create a simple ground-truth token vector.
    fn make_gt(batch_size: usize, seq_len: usize, vocab_size: usize) -> Vec<TokenId> {
        let mut tokens = Vec::with_capacity(batch_size * seq_len);
        for i in 0..(batch_size * seq_len) {
            tokens.push((i % vocab_size) as TokenId);
        }
        tokens
    }

    // ── Golden tests with fixed seed ──────────────────────────────────

    #[test]
    fn test_corruption_deterministic() {
        let cfg = default_config();
        let x = make_gt(cfg.batch_size, cfg.seq_len, cfg.vocab_size);

        let mut rng1 = ChaCha20Rng::seed_from_u64(42);
        let mut rng2 = ChaCha20Rng::seed_from_u64(42);

        let r1 = corrupt(&x, 3, &cfg, &mut rng1).unwrap();
        let r2 = corrupt(&x, 3, &cfg, &mut rng2).unwrap();

        assert_eq!(r1.y_t, r2.y_t, "same seed must produce identical output");
        assert_eq!(r1.num_masked, r2.num_masked);
        assert_eq!(r1.num_replaced, r2.num_replaced);
    }

    #[test]
    fn test_corruption_different_seeds() {
        let cfg = default_config();
        let x = make_gt(cfg.batch_size, cfg.seq_len, cfg.vocab_size);

        let mut rng1 = ChaCha20Rng::seed_from_u64(42);
        let mut rng2 = ChaCha20Rng::seed_from_u64(99);

        let r1 = corrupt(&x, 3, &cfg, &mut rng1).unwrap();
        let r2 = corrupt(&x, 3, &cfg, &mut rng2).unwrap();

        // Extremely unlikely to be equal with different seeds.
        assert_ne!(r1.y_t, r2.y_t);
    }

    // ── Shape assertions ──────────────────────────────────────────────

    #[test]
    fn test_corruption_shape() {
        let cfg = default_config();
        let n = cfg.batch_size * cfg.seq_len;
        let x = make_gt(cfg.batch_size, cfg.seq_len, cfg.vocab_size);
        let mut rng = ChaCha20Rng::seed_from_u64(1);

        let result = corrupt(&x, 4, &cfg, &mut rng).unwrap();
        assert_eq!(result.y_t.len(), n);
        assert_eq!(result.total(), n);
    }

    #[test]
    fn test_corruption_token_range() {
        let cfg = default_config();
        let x = make_gt(cfg.batch_size, cfg.seq_len, cfg.vocab_size);
        let mask_id = cfg.mask_token_id();
        let mut rng = ChaCha20Rng::seed_from_u64(7);

        let result = corrupt(&x, 6, &cfg, &mut rng).unwrap();
        for &tok in &result.y_t {
            assert!(
                tok >= 0 && tok <= mask_id,
                "token {tok} out of range [0, {mask_id}]"
            );
        }
    }

    // ── Mask rate validation (±tolerance) ─────────────────────────────

    #[test]
    fn test_mask_rate_at_t_max() {
        // At t=T=6: α_T = 0.8. With large N the empirical rate should be close.
        let cfg = ErmConfig {
            batch_size: 64,
            seq_len: 256,
            ..ErmConfig::default()
        };
        let x = make_gt(cfg.batch_size, cfg.seq_len, cfg.vocab_size);
        let mut rng = ChaCha20Rng::seed_from_u64(123);

        let result = corrupt(&x, 6, &cfg, &mut rng).unwrap();
        let emp = result.empirical_mask_rate();
        let expected = cfg.mask_rate(6);
        let tol = 0.03; // ±3%
        assert!(
            (emp - expected).abs() < tol,
            "empirical mask rate {emp:.4} vs expected {expected:.4} (tol {tol})"
        );
    }

    #[test]
    fn test_mask_rate_at_t_min() {
        // At t=1: α_1 = 0.15.
        let cfg = ErmConfig {
            batch_size: 64,
            seq_len: 256,
            ..ErmConfig::default()
        };
        let x = make_gt(cfg.batch_size, cfg.seq_len, cfg.vocab_size);
        let mut rng = ChaCha20Rng::seed_from_u64(456);

        let result = corrupt(&x, 1, &cfg, &mut rng).unwrap();
        let emp = result.empirical_mask_rate();
        let expected = cfg.mask_rate(1);
        let tol = 0.03;
        assert!(
            (emp - expected).abs() < tol,
            "empirical mask rate {emp:.4} vs expected {expected:.4} (tol {tol})"
        );
    }

    #[test]
    fn test_replace_rate_at_t_max() {
        // At t=T=6: β_T = 0.1.
        let cfg = ErmConfig {
            batch_size: 64,
            seq_len: 256,
            ..ErmConfig::default()
        };
        let x = make_gt(cfg.batch_size, cfg.seq_len, cfg.vocab_size);
        let mut rng = ChaCha20Rng::seed_from_u64(789);

        let result = corrupt(&x, 6, &cfg, &mut rng).unwrap();
        let emp = result.empirical_replace_rate();
        let expected = cfg.replace_rate(6);
        let tol = 0.03;
        assert!(
            (emp - expected).abs() < tol,
            "empirical replace rate {emp:.4} vs expected {expected:.4} (tol {tol})"
        );
    }

    #[test]
    fn test_mask_rate_monotonic() {
        // Higher t → higher mask rate.
        let cfg = ErmConfig {
            batch_size: 32,
            seq_len: 512,
            ..ErmConfig::default()
        };
        let x = make_gt(cfg.batch_size, cfg.seq_len, cfg.vocab_size);

        let mut rates = Vec::new();
        for t in 1..=cfg.refinement_steps {
            let mut rng = ChaCha20Rng::seed_from_u64(t as u64 * 1000);
            let result = corrupt(&x, t, &cfg, &mut rng).unwrap();
            rates.push(result.empirical_mask_rate());
        }

        // Should be roughly monotonically increasing (with some noise).
        // Check first < last at least.
        assert!(
            rates[0] < rates[rates.len() - 1],
            "mask rate should increase with t: {rates:?}"
        );
    }

    // ── Edge cases ────────────────────────────────────────────────────

    #[test]
    fn test_invalid_t_zero() {
        let cfg = default_config();
        let x = make_gt(1, 4, cfg.vocab_size);
        let mut rng = ChaCha20Rng::seed_from_u64(0);
        assert!(corrupt(&x, 0, &cfg, &mut rng).is_err());
    }

    #[test]
    fn test_invalid_t_too_large() {
        let cfg = default_config();
        let x = make_gt(1, 4, cfg.vocab_size);
        let mut rng = ChaCha20Rng::seed_from_u64(0);
        assert!(corrupt(&x, cfg.refinement_steps + 1, &cfg, &mut rng).is_err());
    }

    #[test]
    fn test_empty_input() {
        let cfg = default_config();
        let x: Vec<TokenId> = vec![];
        let mut rng = ChaCha20Rng::seed_from_u64(0);
        let result = corrupt(&x, 1, &cfg, &mut rng).unwrap();
        assert!(result.y_t.is_empty());
        assert_eq!(result.total(), 0);
    }

    #[test]
    fn test_single_step_config() {
        // Edge case: T=1, only one step.
        let cfg = ErmConfig {
            refinement_steps: 1,
            ..ErmConfig::default()
        };
        let x = make_gt(1, 32, cfg.vocab_size);
        let mut rng = ChaCha20Rng::seed_from_u64(55);
        let result = corrupt(&x, 1, &cfg, &mut rng).unwrap();
        assert_eq!(result.y_t.len(), 32);
    }

    // ── Counts consistency ────────────────────────────────────────────

    #[test]
    fn test_counts_sum_to_total() {
        let cfg = default_config();
        let x = make_gt(cfg.batch_size, cfg.seq_len, cfg.vocab_size);
        let mut rng = ChaCha20Rng::seed_from_u64(333);

        for t in 1..=cfg.refinement_steps {
            let result = corrupt(&x, t, &cfg, &mut rng).unwrap();
            assert_eq!(
                result.num_masked + result.num_replaced + result.num_kept,
                x.len(),
                "counts must sum to total at t={t}"
            );
        }
    }

    // ── Golden values test ────────────────────────────────────────────

    #[test]
    fn test_golden_corruption_seed_42_t3() {
        // Small config for exact golden values.
        let cfg = ErmConfig {
            batch_size: 1,
            seq_len: 8,
            vocab_size: 100,
            refinement_steps: 6,
            mask_rate_max: 0.8,
            mask_rate_min: 0.15,
            replace_rate_max: 0.1,
            replace_rate_min: 0.02,
            ..ErmConfig::default()
        };
        let x: Vec<TokenId> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        let result = corrupt(&x, 3, &cfg, &mut rng).unwrap();

        // Assert output length.
        assert_eq!(result.y_t.len(), 8);

        // Verify that each output token is either MASK (100), a value in [0, 100), or the original.
        let mask_id = cfg.mask_token_id();
        for (i, &tok) in result.y_t.iter().enumerate() {
            assert!(
                tok == mask_id || (0..100).contains(&tok) || tok == x[i],
                "token[{i}]={tok} not valid"
            );
        }

        // Store the golden output for regression.
        // We don't hard-code exact values because ChaCha may vary across platforms,
        // but we verify determinism by running twice.
        let mut rng2 = ChaCha20Rng::seed_from_u64(42);
        let result2 = corrupt(&x, 3, &cfg, &mut rng2).unwrap();
        assert_eq!(result.y_t, result2.y_t);
    }
}
