//! Configuration for the Emergent Route Model.
//!
//! [`ErmConfig`] holds every hyperparameter referenced in the architecture spec.
//! All fields have sensible defaults for an RTX 3050 (4 GB VRAM) starting point.

use serde::{Deserialize, Serialize};

/// Canonical mask-id policy used by ERM configs.
///
/// `ExtraSentinel` preserves current ERM behavior:
/// - `mask_token_id = vocab_size` (one-past-real-vocab)
/// - `total_vocab_size = vocab_size + 1`
///
/// This is the selected HF compatibility policy for now.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum MaskTokenPolicy {
    /// Keep ERM's one-past-vocab sentinel semantics.
    #[default]
    ExtraSentinel,
}

/// Complete hyperparameter configuration for ERM.
///
/// Instantiate with [`ErmConfig::default()`] for the recommended starting point,
/// or deserialize from JSON/TOML.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ErmConfig {
    // ── Vocabulary & sequence ───────────────────────────────────────────
    /// Vocabulary size (excluding MASK sentinel). MASK id = `vocab_size`.
    pub vocab_size: usize,
    /// Mask token-id policy. Canonical value: `extra_sentinel`.
    pub mask_token_policy: MaskTokenPolicy,
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
    /// Minimum pheromone value `φ_min` (floor for active edges).
    pub phi_min: f32,
    /// Initial pheromone for new edges.
    pub phi_init: f32,
    /// ACS local update rate `xi` for traversed edges.
    /// `0.0` disables local update.
    pub local_update_xi: f32,
    /// Number of ants allowed to deposit pheromone each step.
    /// `0` disables elite filtering (all ants may deposit).
    pub elite_k: usize,
    /// Age-based deposit schedule mode.
    /// Options: `"reciprocal"` (legacy `eta/(1+age)`) or `"half_life"`.
    pub age_eta_schedule: String,
    /// Half-life (in steps) for age-based deposit decay when
    /// `age_eta_schedule == "half_life"`.
    /// `<= 0` falls back to reciprocal schedule.
    pub age_half_life: f32,

    // ── RouteAggregate hyperparams ─────────────────────────────────────
    /// Epsilon added to phi before log in weight computation.
    pub route_epsilon: f32,
    /// Taint penalty coefficient `λ` in route weight formula.
    pub route_lambda: f32,
    /// Age penalty coefficient `μ` in route weight formula.
    pub route_mu: f32,
    /// Leader utility coefficient `κ` in route weight formula.
    /// `0.0` disables utility weighting.
    pub route_kappa_utility: f32,
    /// Step schedule mode for pheromone/route controls.
    /// Options: `"fixed"` (backward-compatible) or `"linear"`.
    pub pheromone_schedule_mode: String,
    /// Evaporation multiplier at the first refinement step (`t = T`).
    pub schedule_evap_mult_start: f32,
    /// Evaporation multiplier at the last refinement step (`t = 1`).
    pub schedule_evap_mult_end: f32,
    /// Route lambda multiplier at the first refinement step (`t = T`).
    pub schedule_route_lambda_mult_start: f32,
    /// Route lambda multiplier at the last refinement step (`t = 1`).
    pub schedule_route_lambda_mult_end: f32,
    /// Diversity-penalty multiplier at the first refinement step (`t = T`).
    pub schedule_diversity_penalty_mult_start: f32,
    /// Diversity-penalty multiplier at the last refinement step (`t = 1`).
    pub schedule_diversity_penalty_mult_end: f32,
    /// Enable parallel dense pheromone passes (evaporation + decay/age/clamp).
    ///
    /// Deposit remains sequential to preserve deterministic semantics.
    pub pheromone_parallel_dense_updates: bool,

    // ── Active-set refinement ──────────────────────────────────────────
    /// Enable confidence-based active-set refinement.
    ///
    /// When enabled, high-confidence corrupted positions are temporarily frozen
    /// so ants focus edits on low-confidence positions.
    pub active_set_mode: bool,
    /// Freeze threshold on max-softmax confidence in `[0, 1]`.
    ///
    /// Corrupted positions with confidence >= threshold are frozen unless
    /// needed to satisfy `min_active_positions`.
    pub freeze_confidence_threshold: f32,
    /// Minimum number of active corrupted positions per sequence when
    /// `active_set_mode` is enabled.
    pub min_active_positions: usize,
    /// Restrict corruption/editing to answer spans for QA-formatted data.
    ///
    /// Intended for corpora formatted as:
    /// `Question: ...` then `Answer: ...` (or `Output: ...`).
    /// When enabled, tokens before the detected answer marker are kept fixed.
    pub reasoning_answer_only_mode: bool,
    /// Fallback answer start as a fraction of sequence length in `[0, 1]`
    /// when no explicit answer marker is detected.
    pub reasoning_answer_fallback_start_frac: f32,
    /// Enable completion-style corruption on plain text.
    ///
    /// When enabled, a random suffix is masked and the prefix stays visible.
    /// This trains prompt→completion behavior instead of generic copy denoising.
    pub completion_mode: bool,
    /// Minimum completion target length as fraction of sequence length in `[0, 1]`.
    pub completion_target_min_frac: f32,
    /// Maximum completion target length as fraction of sequence length in `[0, 1]`.
    pub completion_target_max_frac: f32,

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

    // ── Leader utility EMA ─────────────────────────────────────────────
    /// EMA smoothing factor `γ` for leader edge utility tracking.
    /// `U(e) = (1 - γ) * U(e) + γ * relu(Δ)`.
    pub leader_ema_gamma: f32,

    // ── Warm-start ──────────────────────────────────────────────────────
    /// Number of warm-start steps during which death/respawn is relaxed.
    /// If > 0, `death_streak` is multiplied by `warmstart_death_mult`
    /// for the first `warmstart_steps` training steps.
    pub warmstart_steps: usize,
    /// Death streak multiplier during warm-start (default 4).
    pub warmstart_death_mult: usize,

    // ── Optimizer (for reference; consumed by erm-train) ───────────────
    /// Learning rate.
    pub learning_rate: f64,
    /// Weight decay for AdamW.
    pub weight_decay: f64,
    /// LR warmup steps.
    pub warmup_steps: usize,
    /// Use Muon optimizer (Newton-Schulz + SGD) instead of AdamW.
    /// Muon uses 1× parameter memory vs AdamW's 2× and preserves
    /// gradient matrix structure to help break out of mode collapse.
    pub use_muon: bool,

    // ── Diffusion schedule ─────────────────────────────────────────────
    /// Number of diffusion denoising steps `T` per training iteration.
    ///
    /// At each step `t` (from T down to 1), a noisy candidate is built and
    /// the colony proposes parallel edits (coarse at high t, fine at low t).
    pub diffusion_steps: usize,
    /// Noise schedule type for γ(t) weight in diffusion loss.
    /// Options: "linear", "cosine", "sqrt".
    pub noise_schedule: String,
    /// Minimum γ(t) weight (at t=1, lightest noise).
    pub gamma_min: f32,
    /// Maximum γ(t) weight (at t=T, heaviest noise).
    pub gamma_max: f32,

    // ── Dataset / streaming ────────────────────────────────────────────
    /// Use paragraph/sentence spans for book data (vs. sliding window).
    pub use_paragraph_spans: bool,
    /// Tokenizer type: "char" (legacy) or "bpe".
    pub tokenizer_type: String,
    /// BPE vocabulary size target (number of merge operations).
    pub bpe_vocab_size: usize,
    /// Path to pre-built BPE vocabulary file (empty = train from corpus).
    pub bpe_vocab_path: String,

    // ── Regularization / stability ──────────────────────────────────────
    /// Entropy regularization weight in diffusion loss. Penalizes peaked
    /// predictions to prevent mode collapse. 0.0 disables.
    pub entropy_weight: f64,
    /// Maximum gradient norm for clipping. Prevents exploding gradients.
    pub grad_clip_norm: f64,
    /// Use spectral corruption schedule (frequency-aware) instead of uniform.
    pub use_spectral_corruption: bool,

    // ── Experiment identity ────────────────────────────────────────────
    /// Experiment identifier (e.g., "exp-a"). Used in metrics.jsonl headers.
    pub exp_id: String,
    /// Path to write metrics JSONL file.
    pub metrics_path: String,
}

impl Default for ErmConfig {
    fn default() -> Self {
        Self {
            vocab_size: 0, // auto-detected from tokenizer
            mask_token_policy: MaskTokenPolicy::ExtraSentinel,
            seq_len: 256,

            hidden_dim: 512,
            num_blocks: 3,
            num_heads: 8,
            mlp_expansion: 4,
            dropout: 0.3,

            emax: 16,

            num_ants: 128,
            topk: 32,
            pmax: 6,
            refinement_steps: 4,
            batch_size: 1,

            mask_rate_max: 0.8,
            mask_rate_min: 0.15,
            replace_rate_max: 0.1,
            replace_rate_min: 0.02,

            pheromone_evap: 0.1,
            pheromone_eta: 0.7,
            taint_zeta: 0.3,
            taint_max: 5.0,
            taint_decay: 0.05,
            phi_max: 100.0,
            phi_min: 1e-4,
            phi_init: 0.05,
            local_update_xi: 0.0,
            elite_k: 0,
            age_eta_schedule: "reciprocal".to_string(),
            age_half_life: 0.0,

            route_epsilon: 1e-6,
            route_lambda: 1.0,
            route_mu: 0.01,
            route_kappa_utility: 0.0,
            pheromone_schedule_mode: "fixed".to_string(),
            schedule_evap_mult_start: 1.0,
            schedule_evap_mult_end: 1.0,
            schedule_route_lambda_mult_start: 1.0,
            schedule_route_lambda_mult_end: 1.0,
            schedule_diversity_penalty_mult_start: 1.0,
            schedule_diversity_penalty_mult_end: 1.0,
            pheromone_parallel_dense_updates: false,

            active_set_mode: false,
            freeze_confidence_threshold: 0.9,
            min_active_positions: 8,
            reasoning_answer_only_mode: false,
            reasoning_answer_fallback_start_frac: 0.5,
            completion_mode: false,
            completion_target_min_frac: 0.2,
            completion_target_max_frac: 0.8,

            prune_min_score: -1.0,
            prune_max_age: 1000,

            leader_ema_gamma: 0.3,

            death_streak: 5,
            max_edits_per_step: 0.18,
            leader_fraction: 0.12,

            warmstart_steps: 0,
            warmstart_death_mult: 4,

            learning_rate: 5e-4,
            weight_decay: 0.01,
            warmup_steps: 100,
            use_muon: true,

            diffusion_steps: 6,
            noise_schedule: "cosine".to_string(),
            gamma_min: 0.5,
            gamma_max: 2.0,

            use_paragraph_spans: true,
            tokenizer_type: "bpe".to_string(),
            bpe_vocab_size: 4096,
            bpe_vocab_path: "/workspace/erm-rust/data/tokenizers/merged/merged_vocab.json"
                .to_string(),

            entropy_weight: 0.01,
            grad_clip_norm: 1.0,
            use_spectral_corruption: false,

            exp_id: String::new(),
            metrics_path: String::new(),
        }
    }
}

impl ErmConfig {
    /// Scheduled pheromone/route parameters for one refinement step.
    #[must_use]
    pub fn pheromone_step_schedule(&self, t: usize, total_steps: usize) -> PheromoneStepSchedule {
        let base_evap = self.pheromone_evap.clamp(0.0, 1.0);
        let base_lambda = self.route_lambda.max(0.0);
        let base_diversity_penalty = PheromoneConfig::default().diversity_penalty;

        if self.pheromone_schedule_mode == "linear" {
            let t_total = total_steps.max(1) as f32;
            let t_clamped = (t as f32).clamp(1.0, t_total);
            // s=0 at t=T (coarse), s=1 at t=1 (fine).
            let s = if total_steps <= 1 {
                1.0
            } else {
                (t_total - t_clamped) / (t_total - 1.0)
            };

            let evap_mult = lerp(
                self.schedule_evap_mult_start,
                self.schedule_evap_mult_end,
                s,
            );
            let lambda_mult = lerp(
                self.schedule_route_lambda_mult_start,
                self.schedule_route_lambda_mult_end,
                s,
            );
            let diversity_mult = lerp(
                self.schedule_diversity_penalty_mult_start,
                self.schedule_diversity_penalty_mult_end,
                s,
            );

            return PheromoneStepSchedule {
                evaporation_rate: (base_evap * evap_mult).clamp(0.0, 1.0),
                route_lambda: (base_lambda * lambda_mult).max(0.0),
                diversity_penalty: (base_diversity_penalty * diversity_mult).clamp(0.0, 1.0),
            };
        }

        PheromoneStepSchedule {
            evaporation_rate: base_evap,
            route_lambda: base_lambda,
            diversity_penalty: base_diversity_penalty,
        }
    }

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

    /// Effective death streak for a given training step.
    ///
    /// During warm-start (step < `warmstart_steps`), the death streak is
    /// multiplied by `warmstart_death_mult` to reduce ant churn while the
    /// scorer is still learning.
    #[must_use]
    pub fn effective_death_streak(&self, step: usize) -> usize {
        if self.warmstart_steps > 0 && step < self.warmstart_steps {
            self.death_streak * self.warmstart_death_mult
        } else {
            self.death_streak
        }
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

    /// Compute γ(t) — the loss weight at diffusion step t (1-indexed, 1=cleanest).
    ///
    /// Supports three schedules:
    /// - `"linear"`: linear from γ_min (t=1) to γ_max (t=T).
    /// - `"cosine"`: cosine schedule (smooth ramp).
    /// - `"sqrt"`: square-root schedule.
    ///
    /// Used in the denoising loss: `L = E_t[ γ(t) * CE(x | z_t) ]`.
    #[must_use]
    pub fn gamma(&self, t: usize) -> f32 {
        let big_t = self.diffusion_steps.max(1) as f32;
        let t_f = (t as f32).clamp(1.0, big_t);
        // Normalize s ∈ [0, 1]: s=0 at t=1, s=1 at t=T.
        let s = (t_f - 1.0) / (big_t - 1.0).max(1.0);
        match self.noise_schedule.as_str() {
            "cosine" => {
                // Cosine: 0.5 * (1 - cos(π * s)) rescaled to [γ_min, γ_max].
                let cos_s = 0.5 * (1.0 - (std::f32::consts::PI * s).cos());
                self.gamma_min + (self.gamma_max - self.gamma_min) * cos_s
            }
            "sqrt" => self.gamma_min + (self.gamma_max - self.gamma_min) * s.sqrt(),
            _ => {
                // "linear" (default)
                self.gamma_min + (self.gamma_max - self.gamma_min) * s
            }
        }
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

/// Step-level schedule outputs for pheromone and route controls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PheromoneStepSchedule {
    /// Evaporation rate `ρ` to use at this step.
    pub evaporation_rate: f32,
    /// Route taint penalty coefficient `λ` to use at this step.
    pub route_lambda: f32,
    /// Diversity penalty multiplier to use at this step.
    pub diversity_penalty: f32,
}

fn lerp(start: f32, end: f32, s: f32) -> f32 {
    start + (end - start) * s
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
    /// Minimum pheromone value `φ_min` for active edges.
    pub phi_min: f32,
    /// Initial pheromone baseline for newly inserted edges.
    pub phi_init: f32,
    /// ACS local pheromone update rate.
    /// `0.0` disables local update.
    pub local_update_xi: f32,
    /// Number of ants allowed to deposit each step.
    /// `0` disables elite filtering (all ants may deposit).
    pub elite_k: usize,
    /// Age-based deposit schedule mode (`"reciprocal"` or `"half_life"`).
    pub age_eta_schedule: String,
    /// Age half-life for `"half_life"` schedule.
    pub age_half_life: f32,
    /// Minimum composite score `φ - λ·τ` for pruning.
    pub prune_min_score: f32,
    /// Maximum edge age before pruning.
    pub prune_max_age: u32,
    /// Taint penalty coefficient `λ` used in composite score for pruning.
    pub route_lambda: f32,
    /// Cosine similarity threshold for diversity pressure. Edges whose
    /// source hidden states exceed this threshold are considered redundant.
    pub diversity_threshold: f32,
    /// Pheromone multiplier for the weaker of two redundant edges.
    /// e.g. 0.8 means "reduce by 20%".
    pub diversity_penalty: f32,
    /// Use log-bounded deposit instead of tanh.
    /// `log1p(|Δ|/σ)` has better dynamic range: small improvements get
    /// proportional credit, large improvements get diminishing-but-nonzero credit.
    pub use_log_deposit: bool,
    /// Enable parallel dense pheromone passes (evaporation + decay/age/clamp).
    ///
    /// Deposit remains sequential to preserve deterministic semantics.
    pub parallel_dense_updates: bool,
}

impl Default for PheromoneConfig {
    fn default() -> Self {
        Self {
            evaporation_rate: 0.1,
            deposit_rate: 0.7,
            taint_rate: 0.3,
            taint_decay: 0.05,
            taint_max: 5.0,
            phi_max: 100.0,
            phi_min: 1e-4,
            phi_init: 0.05,
            local_update_xi: 0.0,
            elite_k: 0,
            age_eta_schedule: "reciprocal".to_string(),
            age_half_life: 0.0,
            prune_min_score: -1.0,
            prune_max_age: 1000,
            route_lambda: 1.0,
            diversity_threshold: 0.9,
            diversity_penalty: 0.8,
            use_log_deposit: true,
            parallel_dense_updates: false,
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
            phi_min: config.phi_min,
            phi_init: config.phi_init,
            local_update_xi: config.local_update_xi,
            elite_k: config.elite_k,
            age_eta_schedule: config.age_eta_schedule.clone(),
            age_half_life: config.age_half_life,
            prune_min_score: config.prune_min_score,
            prune_max_age: config.prune_max_age,
            route_lambda: config.route_lambda,
            diversity_threshold: 0.9,
            diversity_penalty: 0.8,
            use_log_deposit: true,
            parallel_dense_updates: config.pheromone_parallel_dense_updates,
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
        assert_eq!(cfg.vocab_size, 0);
        assert_eq!(cfg.mask_token_policy, MaskTokenPolicy::ExtraSentinel);
        assert_eq!(cfg.seq_len, 256);
        assert_eq!(cfg.hidden_dim, 512);
        assert_eq!(cfg.num_blocks, 3);
        assert_eq!(cfg.emax, 16);
        assert_eq!(cfg.route_kappa_utility, 0.0);
        assert!((cfg.phi_min - 1e-4).abs() < 1e-8);
        assert_eq!(cfg.local_update_xi, 0.0);
        assert_eq!(cfg.elite_k, 0);
        assert_eq!(cfg.age_eta_schedule, "reciprocal");
        assert!((cfg.age_half_life - 0.0).abs() < 1e-8);
        assert_eq!(cfg.pheromone_schedule_mode, "fixed");
        assert!(!cfg.pheromone_parallel_dense_updates);
        assert!(!cfg.active_set_mode);
        assert!((cfg.freeze_confidence_threshold - 0.9).abs() < 1e-8);
        assert_eq!(cfg.min_active_positions, 8);
        assert!(!cfg.reasoning_answer_only_mode);
        assert!((cfg.reasoning_answer_fallback_start_frac - 0.5).abs() < 1e-8);
        assert!(!cfg.completion_mode);
        assert!((cfg.completion_target_min_frac - 0.2).abs() < 1e-8);
        assert!((cfg.completion_target_max_frac - 0.8).abs() < 1e-8);
        assert_eq!(cfg.mask_token_id(), 0);
        assert_eq!(cfg.total_vocab_size(), 1);
    }

    #[test]
    fn test_pheromone_step_schedule_fixed_mode_uses_base_values() {
        let cfg = ErmConfig {
            pheromone_evap: 0.2,
            route_lambda: 1.3,
            pheromone_schedule_mode: "fixed".to_string(),
            ..ErmConfig::default()
        };
        let s = cfg.pheromone_step_schedule(4, 6);
        assert!((s.evaporation_rate - 0.2).abs() < 1e-8);
        assert!((s.route_lambda - 1.3).abs() < 1e-8);
        assert!((s.diversity_penalty - PheromoneConfig::default().diversity_penalty).abs() < 1e-8);
    }

    #[test]
    fn test_pheromone_step_schedule_linear_mode_interpolates() {
        let cfg = ErmConfig {
            pheromone_evap: 0.1,
            route_lambda: 2.0,
            pheromone_schedule_mode: "linear".to_string(),
            schedule_evap_mult_start: 1.5,
            schedule_evap_mult_end: 0.5,
            schedule_route_lambda_mult_start: 0.5,
            schedule_route_lambda_mult_end: 1.5,
            schedule_diversity_penalty_mult_start: 1.0,
            schedule_diversity_penalty_mult_end: 0.5,
            ..ErmConfig::default()
        };

        // t=T (coarse) => start multipliers.
        let s_t = cfg.pheromone_step_schedule(6, 6);
        assert!((s_t.evaporation_rate - 0.15).abs() < 1e-8);
        assert!((s_t.route_lambda - 1.0).abs() < 1e-8);
        assert!((s_t.diversity_penalty - 0.8).abs() < 1e-8);

        // t=1 (fine) => end multipliers.
        let s_1 = cfg.pheromone_step_schedule(1, 6);
        assert!((s_1.evaporation_rate - 0.05).abs() < 1e-8);
        assert!((s_1.route_lambda - 3.0).abs() < 1e-8);
        assert!((s_1.diversity_penalty - 0.4).abs() < 1e-8);
    }

    #[test]
    fn test_leader_follower_counts() {
        let cfg = ErmConfig::default();
        // ceil(128 * 0.12) = ceil(15.36) = 16
        assert_eq!(cfg.num_leaders(), 16);
        assert_eq!(cfg.num_followers(), 112);
        assert_eq!(cfg.num_leaders() + cfg.num_followers(), cfg.num_ants);
    }

    #[test]
    fn test_max_edits() {
        let cfg = ErmConfig::default();
        // ceil(0.18 * 256) = ceil(46.08) = 47
        assert_eq!(cfg.max_edits(), 47);
    }

    #[test]
    fn test_mask_rate_schedule() {
        let cfg = ErmConfig::default();
        // At t=T=4 (heaviest): α_T = 0.8
        let rate_t = cfg.mask_rate(4);
        assert!((rate_t - 0.8).abs() < 1e-6, "at t=T: {rate_t}");

        // At t=1 (lightest): α_1 = 0.15
        let rate_1 = cfg.mask_rate(1);
        assert!((rate_1 - 0.15).abs() < 1e-6, "at t=1: {rate_1}");

        // Monotonically decreasing from t=T to t=1
        for t in 2..=4 {
            assert!(cfg.mask_rate(t) >= cfg.mask_rate(t - 1));
        }
    }

    #[test]
    fn test_replace_rate_schedule() {
        let cfg = ErmConfig::default();
        let rate_t = cfg.replace_rate(4);
        assert!((rate_t - 0.1).abs() < 1e-6);
        let rate_1 = cfg.replace_rate(1);
        assert!((rate_1 - 0.02).abs() < 1e-6);
    }

    #[test]
    fn test_pheromone_config_from_config_includes_new_fields() {
        let cfg = ErmConfig {
            phi_max: 10.0,
            phi_min: 0.01,
            phi_init: 0.07,
            local_update_xi: 0.02,
            elite_k: 7,
            age_eta_schedule: "half_life".to_string(),
            age_half_life: 32.0,
            pheromone_parallel_dense_updates: true,
            ..ErmConfig::default()
        };
        let p = PheromoneConfig::from_config(&cfg);
        assert!((p.phi_max - 10.0).abs() < 1e-8);
        assert!((p.phi_min - 0.01).abs() < 1e-8);
        assert!((p.phi_init - 0.07).abs() < 1e-8);
        assert!((p.local_update_xi - 0.02).abs() < 1e-8);
        assert_eq!(p.elite_k, 7);
        assert_eq!(p.age_eta_schedule, "half_life");
        assert!((p.age_half_life - 32.0).abs() < 1e-8);
        assert!(p.parallel_dense_updates);
    }
}
