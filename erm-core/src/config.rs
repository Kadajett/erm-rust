//! Configuration for the Emergent Route Model.
//!
//! [`ErmConfig`] holds every hyperparameter referenced in the architecture spec.
//! All fields have sensible defaults for an RTX 3050 (4 GB VRAM) starting point.

use serde::{Deserialize, Serialize};

/// Complete hyperparameter configuration for ERM.
///
/// Instantiate with [`ErmConfig::default()`] for the recommended starting point,
/// or deserialize from JSON/TOML.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ErmConfig {
    // ── Vocabulary & sequence ───────────────────────────────────────────
    /// Vocabulary size (excluding MASK sentinel). MASK id = `vocab_size`.
    pub vocab_size: usize,
    /// Maximum sequence length.
    pub seq_len: usize,

    // ── Scorer network ─────────────────────────────────────────────────
    /// Hidden dimension `d`.
    pub hidden_dim: usize,
    /// Number of transformer encoder blocks.
    pub num_blocks: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// MLP expansion factor (feed-forward inner dim = `mlp_expansion * hidden_dim`).
    pub mlp_expansion: usize,
    /// Dropout probability used inside the transformer.
    pub dropout: f64,

    // ── Route graph ────────────────────────────────────────────────────
    /// Maximum edges per destination node in the dense-neighbor format.
    pub emax: usize,

    // ── Colony ──────────────────────────────────────────────────────────
    /// Total ants per sequence.
    pub num_ants: usize,
    /// Top-k candidates per position.
    pub topk: usize,
    /// Maximum positions proposed per ant.
    pub pmax: usize,
    /// Number of refinement steps `T`.
    pub refinement_steps: usize,
    /// Training batch size.
    pub batch_size: usize,

    // ── Corruption schedule ────────────────────────────────────────────
    /// Mask rate at step `T` (heaviest corruption). `α_T`.
    pub mask_rate_max: f32,
    /// Mask rate at step `1` (lightest corruption). `α_1`.
    pub mask_rate_min: f32,
    /// Replace rate at step `T`. `β_T`.
    pub replace_rate_max: f32,
    /// Replace rate at step `1`. `β_1`.
    pub replace_rate_min: f32,

    // ── Pheromone / stigmergy ──────────────────────────────────────────
    /// Evaporation rate `ρ`.
    pub pheromone_evap: f32,
    /// Deposit rate `η`.
    pub pheromone_eta: f32,
    /// Taint deposit rate `ζ`.
    pub taint_zeta: f32,
    /// Maximum taint value `τ_max`.
    pub taint_max: f32,
    /// Taint decay rate `ρ_τ`.
    pub taint_decay: f32,
    /// Maximum pheromone value `φ_max` (clip ceiling).
    pub phi_max: f32,
    /// Initial pheromone for new edges.
    pub phi_init: f32,

    // ── RouteAggregate hyperparams ─────────────────────────────────────
    /// Epsilon added to phi before log in weight computation.
    pub route_epsilon: f32,
    /// Taint penalty coefficient `λ` in route weight formula.
    pub route_lambda: f32,
    /// Age penalty coefficient `μ` in route weight formula.
    pub route_mu: f32,

    // ── Ant lifecycle ──────────────────────────────────────────────────
    /// Consecutive no-improvement steps before an ant "dies". `K`.
    pub death_streak: usize,
    /// Maximum fraction of positions edited per step (`M = ceil(max_edits_per_step * seq_len)`).
    pub max_edits_per_step: f32,
    /// Fraction of ants that are leaders (rest are followers).
    pub leader_fraction: f32,

    // ── Pruning ──────────────────────────────────────────────────────────
    /// Minimum composite score `φ - λ·τ` below which edges are pruned.
    pub prune_min_score: f32,
    /// Maximum edge age (in refinement steps) before pruning.
    pub prune_max_age: u32,

    // ── Optimizer (for reference; consumed by erm-train) ───────────────
    /// Learning rate.
    pub learning_rate: f64,
    /// Weight decay for AdamW.
    pub weight_decay: f64,
    /// LR warmup steps.
    pub warmup_steps: usize,
}

impl Default for ErmConfig {
    fn default() -> Self {
        Self {
            vocab_size: 16_384,
            seq_len: 128,

            hidden_dim: 256,
            num_blocks: 6,
            num_heads: 4,
            mlp_expansion: 4,
            dropout: 0.1,

            emax: 16,

            num_ants: 256,
            topk: 8,
            pmax: 8,
            refinement_steps: 6,
            batch_size: 8,

            mask_rate_max: 0.8,
            mask_rate_min: 0.15,
            replace_rate_max: 0.1,
            replace_rate_min: 0.02,

            pheromone_evap: 0.1,
            pheromone_eta: 0.5,
            taint_zeta: 0.3,
            taint_max: 5.0,
            taint_decay: 0.05,
            phi_max: 100.0,
            phi_init: 0.05,

            route_epsilon: 1e-6,
            route_lambda: 1.0,
            route_mu: 0.01,

            prune_min_score: -1.0,
            prune_max_age: 1000,

            death_streak: 5,
            max_edits_per_step: 0.15,
            leader_fraction: 0.10,

            learning_rate: 1e-3,
            weight_decay: 0.01,
            warmup_steps: 1000,
        }
    }
}

impl ErmConfig {
    /// The MASK sentinel token id (one past the last real vocab id).
    #[must_use]
    pub fn mask_token_id(&self) -> i32 {
        self.vocab_size as i32
    }

    /// Total vocabulary size including the MASK sentinel (`V + 1`).
    #[must_use]
    pub fn total_vocab_size(&self) -> usize {
        self.vocab_size + 1
    }

    /// Number of leader ants.
    #[must_use]
    pub fn num_leaders(&self) -> usize {
        ((self.num_ants as f32) * self.leader_fraction).ceil() as usize
    }

    /// Number of follower ants.
    #[must_use]
    pub fn num_followers(&self) -> usize {
        self.num_ants - self.num_leaders()
    }

    /// Maximum edits per refinement step.
    #[must_use]
    pub fn max_edits(&self) -> usize {
        ((self.seq_len as f32) * self.max_edits_per_step).ceil() as usize
    }

    /// Compute the mask rate `α_t` for a given refinement step `t` (1-indexed).
    ///
    /// Linear interpolation: `α_t = α_T + (α_1 - α_T) * (T - t) / (T - 1)`.
    /// At `t = T`: returns `α_T` (heaviest). At `t = 1`: returns `α_1` (lightest).
    #[must_use]
    pub fn mask_rate(&self, t: usize) -> f32 {
        let big_t = self.refinement_steps as f32;
        let t_f = t as f32;
        if self.refinement_steps <= 1 {
            return self.mask_rate_max;
        }
        self.mask_rate_max
            + (self.mask_rate_min - self.mask_rate_max) * (big_t - t_f) / (big_t - 1.0)
    }

    /// Compute the replace rate `β_t` for a given refinement step `t` (1-indexed).
    ///
    /// Linear interpolation analogous to [`mask_rate`](Self::mask_rate).
    #[must_use]
    pub fn replace_rate(&self, t: usize) -> f32 {
        let big_t = self.refinement_steps as f32;
        let t_f = t as f32;
        if self.refinement_steps <= 1 {
            return self.replace_rate_max;
        }
        self.replace_rate_max
            + (self.replace_rate_min - self.replace_rate_max) * (big_t - t_f) / (big_t - 1.0)
    }
}

/// Pheromone-specific configuration extracted from [`ErmConfig`].
///
/// Groups the stigmergy hyperparameters for ergonomic passing to
/// pheromone update functions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PheromoneConfig {
    /// Evaporation rate `ρ`. Applied to ALL edges each step.
    pub evaporation_rate: f32,
    /// Deposit rate `η`. Scales the bounded deposit `η * tanh(relu(Δ))`.
    pub deposit_rate: f32,
    /// Taint deposit rate `ζ`. Scales taint from negative deltas.
    pub taint_rate: f32,
    /// Taint decay rate `ρ_τ`. Applied to ALL edges each step.
    pub taint_decay: f32,
    /// Maximum taint value `τ_max`. Taint is clamped to `[0, τ_max]`.
    pub taint_max: f32,
    /// Maximum pheromone value `φ_max`. Pheromone is clamped to `[0, φ_max]`.
    pub phi_max: f32,
    /// Minimum composite score `φ - λ·τ` for pruning.
    pub prune_min_score: f32,
    /// Maximum edge age before pruning.
    pub prune_max_age: u32,
    /// Taint penalty coefficient `λ` used in composite score for pruning.
    pub route_lambda: f32,
}

impl Default for PheromoneConfig {
    fn default() -> Self {
        Self {
            evaporation_rate: 0.1,
            deposit_rate: 0.5,
            taint_rate: 0.3,
            taint_decay: 0.05,
            taint_max: 5.0,
            phi_max: 100.0,
            prune_min_score: -1.0,
            prune_max_age: 1000,
            route_lambda: 1.0,
        }
    }
}

impl PheromoneConfig {
    /// Build from the global [`ErmConfig`].
    #[must_use]
    pub fn from_config(config: &ErmConfig) -> Self {
        Self {
            evaporation_rate: config.pheromone_evap,
            deposit_rate: config.pheromone_eta,
            taint_rate: config.taint_zeta,
            taint_decay: config.taint_decay,
            taint_max: config.taint_max,
            phi_max: config.phi_max,
            prune_min_score: config.prune_min_score,
            prune_max_age: config.prune_max_age,
            route_lambda: config.route_lambda,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_serde_roundtrip() {
        let cfg = ErmConfig::default();
        let json = serde_json::to_string_pretty(&cfg).expect("serialize");
        let cfg2: ErmConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(cfg, cfg2);
    }

    #[test]
    fn test_default_values() {
        let cfg = ErmConfig::default();
        assert_eq!(cfg.vocab_size, 16_384);
        assert_eq!(cfg.seq_len, 128);
        assert_eq!(cfg.hidden_dim, 256);
        assert_eq!(cfg.num_blocks, 6);
        assert_eq!(cfg.emax, 16);
        assert_eq!(cfg.mask_token_id(), 16_384);
        assert_eq!(cfg.total_vocab_size(), 16_385);
    }

    #[test]
    fn test_leader_follower_counts() {
        let cfg = ErmConfig::default();
        assert_eq!(cfg.num_leaders(), 26);
        assert_eq!(cfg.num_followers(), 230);
        assert_eq!(cfg.num_leaders() + cfg.num_followers(), cfg.num_ants);
    }

    #[test]
    fn test_max_edits() {
        let cfg = ErmConfig::default();
        // ceil(0.15 * 128) = ceil(19.2) = 20
        assert_eq!(cfg.max_edits(), 20);
    }

    #[test]
    fn test_mask_rate_schedule() {
        let cfg = ErmConfig::default();
        // At t=T=6 (heaviest): α_T = 0.8
        let rate_t = cfg.mask_rate(6);
        assert!((rate_t - 0.8).abs() < 1e-6, "at t=T: {rate_t}");

        // At t=1 (lightest): α_1 = 0.15
        let rate_1 = cfg.mask_rate(1);
        assert!((rate_1 - 0.15).abs() < 1e-6, "at t=1: {rate_1}");

        // Monotonically decreasing from t=T to t=1
        for t in 2..=6 {
            assert!(cfg.mask_rate(t) >= cfg.mask_rate(t - 1));
        }
    }

    #[test]
    fn test_replace_rate_schedule() {
        let cfg = ErmConfig::default();
        let rate_t = cfg.replace_rate(6);
        assert!((rate_t - 0.1).abs() < 1e-6);
        let rate_1 = cfg.replace_rate(1);
        assert!((rate_1 - 0.02).abs() < 1e-6);
    }
}
