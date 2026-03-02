//! Training orchestrator for the Emergent Route Model.
//!
//! Manages the two-phase training loop:
//!
//! 1. **Warm-start phase** — plain denoiser training with no colony feedback.
//! 2. **Colony phase** — full ant-colony training (placeholder until colony wired in).
//!
//! Checkpoints are saved as a directory containing:
//! - `config.json`: serialised [`TrainingConfig`].
//! - `scorer_weights.bin`: raw little-endian f32 bytes.
//! - `step.json`: last completed step + phase.
//!
//! # No `unwrap()` in library code.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use erm_core::config::ErmConfig;
use erm_core::error::{ErmError, ErmResult};
use erm_core::scorer::Scorer;

use crate::dataset::{DataBatch, TextDataset};
use crate::training::train_step;

// ── TrainingConfig ─────────────────────────────────────────────────────────

/// Complete training configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainingConfig {
    /// ERM model hyperparameters.
    pub erm: ErmConfig,
    /// Number of warm-start (plain denoiser) training steps.
    pub warm_start_steps: usize,
    /// Number of colony-phase training steps.
    pub colony_steps: usize,
    /// Log loss every N steps.
    pub log_every: usize,
    /// Save checkpoint every N steps (0 = disabled).
    pub checkpoint_every: usize,
    /// Master RNG seed for determinism.
    pub seed: u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            erm: ErmConfig::default(),
            warm_start_steps: 1_000,
            colony_steps: 5_000,
            log_every: 100,
            checkpoint_every: 500,
            seed: 42,
        }
    }
}

// ── LossRecord ─────────────────────────────────────────────────────────────

/// A single loss log entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossRecord {
    /// Training step.
    pub step: usize,
    /// Phase: `"warm_start"` or `"colony"`.
    pub phase: String,
    /// Denoising cross-entropy loss.
    pub loss: f32,
    /// Number of corrupted positions in the batch.
    pub num_corrupted: usize,
    /// Refinement step `t` used.
    pub t: usize,
}

// ── CheckpointMeta ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointMeta {
    step: usize,
    phase: String,
}

// ── Orchestrator ───────────────────────────────────────────────────────────

/// Training orchestrator managing both training phases.
pub struct Orchestrator {
    /// Training configuration.
    pub config: TrainingConfig,
    /// The neural scorer.
    pub scorer: Scorer,
    /// Loss log for all completed steps.
    pub loss_log: Vec<LossRecord>,
    /// Global step counter.
    pub global_step: usize,
    /// Internal RNG.
    rng: ChaCha8Rng,
}

impl Orchestrator {
    /// Create a new orchestrator from a training configuration.
    pub fn new(config: TrainingConfig, vocab_size: usize) -> Self {
        let scorer = Scorer::new(&config.erm, vocab_size, config.seed);
        let rng = ChaCha8Rng::seed_from_u64(config.seed);
        Self {
            config,
            scorer,
            loss_log: Vec::new(),
            global_step: 0,
            rng,
        }
    }

    /// Run the warm-start phase.
    ///
    /// # Errors
    ///
    /// Propagates errors from training steps or checkpoint saves.
    pub fn run_warm_start(
        &mut self,
        dataset: &TextDataset,
        checkpoint_dir: Option<&str>,
    ) -> ErmResult<()> {
        let steps = self.config.warm_start_steps;
        let batch_size = self.config.erm.batch_size;
        for _ in 0..steps {
            let batch = dataset.get_batch(batch_size, &mut self.rng);
            self.step_phase("warm_start", &batch, checkpoint_dir)?;
        }
        Ok(())
    }

    /// Run the colony phase (placeholder; uses denoiser loop).
    ///
    /// # Errors
    ///
    /// Propagates errors from training steps or checkpoint saves.
    pub fn run_colony(
        &mut self,
        dataset: &TextDataset,
        checkpoint_dir: Option<&str>,
    ) -> ErmResult<()> {
        let steps = self.config.colony_steps;
        let batch_size = self.config.erm.batch_size;
        for _ in 0..steps {
            let batch = dataset.get_batch(batch_size, &mut self.rng);
            self.step_phase("colony", &batch, checkpoint_dir)?;
        }
        Ok(())
    }

    /// Run both phases sequentially.
    ///
    /// # Errors
    ///
    /// Propagates errors from either phase.
    pub fn run_all(
        &mut self,
        dataset: &TextDataset,
        checkpoint_dir: Option<&str>,
    ) -> ErmResult<()> {
        self.run_warm_start(dataset, checkpoint_dir)?;
        self.run_colony(dataset, checkpoint_dir)?;
        Ok(())
    }

    /// Save a checkpoint to `dir`.
    ///
    /// # Errors
    ///
    /// Returns an error if any file write fails.
    pub fn save_checkpoint(&self, dir: &str, phase: &str) -> ErmResult<()> {
        std::fs::create_dir_all(dir).map_err(|e| {
            ErmError::InvalidConfig(format!("cannot create checkpoint dir {dir}: {e}"))
        })?;

        let config_json = serde_json::to_string_pretty(&self.config)?;
        std::fs::write(format!("{dir}/config.json"), config_json)
            .map_err(|e| ErmError::InvalidConfig(format!("cannot write config.json: {e}")))?;

        let weights_bytes = scorer_to_bytes(&self.scorer);
        std::fs::write(format!("{dir}/scorer_weights.bin"), weights_bytes).map_err(|e| {
            ErmError::InvalidConfig(format!("cannot write scorer_weights.bin: {e}"))
        })?;

        let meta = CheckpointMeta {
            step: self.global_step,
            phase: phase.to_string(),
        };
        let meta_json = serde_json::to_string_pretty(&meta)?;
        std::fs::write(format!("{dir}/step.json"), meta_json)
            .map_err(|e| ErmError::InvalidConfig(format!("cannot write step.json: {e}")))?;

        Ok(())
    }

    /// Load a checkpoint from `dir`.
    ///
    /// # Errors
    ///
    /// Returns an error if any file cannot be read or weights don't match.
    pub fn load_checkpoint(dir: &str) -> ErmResult<(Self, String)> {
        let config_json = std::fs::read_to_string(format!("{dir}/config.json"))
            .map_err(|e| ErmError::InvalidConfig(format!("cannot read config.json: {e}")))?;
        let config: TrainingConfig = serde_json::from_str(&config_json)?;

        let weight_bytes = std::fs::read(format!("{dir}/scorer_weights.bin"))
            .map_err(|e| ErmError::InvalidConfig(format!("cannot read scorer_weights.bin: {e}")))?;

        let meta_json = std::fs::read_to_string(format!("{dir}/step.json"))
            .map_err(|e| ErmError::InvalidConfig(format!("cannot read step.json: {e}")))?;
        let meta: CheckpointMeta = serde_json::from_str(&meta_json)?;

        let vocab_size = config.erm.total_vocab_size();
        let mut scorer = Scorer::new(&config.erm, vocab_size, config.seed);
        bytes_to_scorer(&weight_bytes, &mut scorer)?;

        let rng = ChaCha8Rng::seed_from_u64(config.seed.wrapping_add(meta.step as u64));
        let orch = Self {
            config,
            scorer,
            loss_log: Vec::new(),
            global_step: meta.step,
            rng,
        };
        Ok((orch, meta.phase))
    }

    // ── Internal ──────────────────────────────────────────────────────────

    fn step_phase(
        &mut self,
        phase: &str,
        batch: &DataBatch,
        checkpoint_dir: Option<&str>,
    ) -> ErmResult<()> {
        let mut result = train_step(&self.scorer, batch, None, &self.config.erm, &mut self.rng)?;
        result.step = self.global_step;

        if self.global_step.is_multiple_of(self.config.log_every) {
            self.loss_log.push(LossRecord {
                step: self.global_step,
                phase: phase.to_string(),
                loss: result.loss,
                num_corrupted: result.num_corrupted,
                t: result.t,
            });
        }

        self.global_step += 1;

        if let Some(dir) = checkpoint_dir {
            if self.config.checkpoint_every > 0
                && self
                    .global_step
                    .is_multiple_of(self.config.checkpoint_every)
            {
                self.save_checkpoint(dir, phase)?;
            }
        }
        Ok(())
    }
}

// ── Weight serialisation ───────────────────────────────────────────────────

/// Serialise scorer weights to raw little-endian f32 bytes.
///
/// Order: token_emb, pos_emb, then per block (w_up, b_up, w_down, b_down),
/// then logit_w, logit_b, uncertainty_w, uncertainty_b.
fn scorer_to_bytes(scorer: &Scorer) -> Vec<u8> {
    let mut all: Vec<f32> = Vec::new();
    all.extend_from_slice(&scorer.token_emb);
    all.extend_from_slice(&scorer.pos_emb);
    for block in &scorer.blocks {
        all.extend_from_slice(&block.w_up);
        all.extend_from_slice(&block.b_up);
        all.extend_from_slice(&block.w_down);
        all.extend_from_slice(&block.b_down);
    }
    all.extend_from_slice(&scorer.logit_w);
    all.extend_from_slice(&scorer.logit_b);
    all.extend_from_slice(&scorer.uncertainty_w);
    all.extend_from_slice(&scorer.uncertainty_b);

    let mut bytes = Vec::with_capacity(all.len() * 4);
    for v in all {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Deserialise raw LE f32 bytes into scorer weight vecs.
///
/// # Errors
///
/// Returns [`ErmError::ShapeMismatch`] if byte count doesn't match expected params.
#[allow(unused_assignments)]
fn bytes_to_scorer(bytes: &[u8], scorer: &mut Scorer) -> ErmResult<()> {
    if !bytes.len().is_multiple_of(4) {
        return Err(ErmError::ShapeMismatch {
            expected: "multiple of 4 bytes".to_string(),
            got: format!("{} bytes", bytes.len()),
        });
    }
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let expected_params = scorer.num_parameters();
    if floats.len() != expected_params {
        return Err(ErmError::ShapeMismatch {
            expected: format!("{expected_params} parameters"),
            got: format!("{} floats", floats.len()),
        });
    }

    let mut cursor = 0usize;
    macro_rules! fill {
        ($slice:expr) => {{
            let n = $slice.len();
            $slice.copy_from_slice(&floats[cursor..cursor + n]);
            cursor += n;
        }};
    }

    fill!(scorer.token_emb);
    fill!(scorer.pos_emb);
    for block in &mut scorer.blocks {
        fill!(block.w_up);
        fill!(block.b_up);
        fill!(block.w_down);
        fill!(block.b_down);
    }
    fill!(scorer.logit_w);
    fill!(scorer.logit_b);
    fill!(scorer.uncertainty_w);
    fill!(scorer.uncertainty_b);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use erm_core::tokenizer::CharTokenizer;

    fn small_config() -> TrainingConfig {
        TrainingConfig {
            erm: ErmConfig {
                vocab_size: 32,
                seq_len: 8,
                hidden_dim: 8,
                num_blocks: 1,
                num_heads: 2,
                mlp_expansion: 2,
                dropout: 0.0,
                batch_size: 2,
                refinement_steps: 2,
                mask_rate_max: 0.5,
                mask_rate_min: 0.1,
                replace_rate_max: 0.05,
                replace_rate_min: 0.01,
                ..ErmConfig::default()
            },
            warm_start_steps: 3,
            colony_steps: 2,
            log_every: 1,
            checkpoint_every: 0,
            seed: 1,
        }
    }

    fn make_dataset(cfg: &TrainingConfig) -> TextDataset {
        let text = "abcdefghijklmnopqrstuvwxyz ".repeat(50);
        let tokenizer = CharTokenizer::from_text(&text);
        TextDataset::from_text(&text, &tokenizer, cfg.erm.seq_len).unwrap()
    }

    #[test]
    fn test_training_config_serde() {
        let cfg = TrainingConfig::default();
        let json = serde_json::to_string_pretty(&cfg).unwrap();
        let cfg2: TrainingConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg, cfg2);
    }

    #[test]
    fn test_warm_start_runs() {
        let cfg = small_config();
        let ds = make_dataset(&cfg);
        let vocab = cfg.erm.total_vocab_size();
        let mut orch = Orchestrator::new(cfg.clone(), vocab);
        orch.run_warm_start(&ds, None).unwrap();
        assert_eq!(orch.global_step, cfg.warm_start_steps);
    }

    #[test]
    fn test_loss_log_populated() {
        let cfg = small_config();
        let ds = make_dataset(&cfg);
        let vocab = cfg.erm.total_vocab_size();
        let mut orch = Orchestrator::new(cfg, vocab);
        orch.run_warm_start(&ds, None).unwrap();
        assert!(!orch.loss_log.is_empty());
        for rec in &orch.loss_log {
            assert!(rec.loss.is_finite());
            assert_eq!(rec.phase, "warm_start");
        }
    }

    #[test]
    fn test_colony_runs() {
        let cfg = small_config();
        let ds = make_dataset(&cfg);
        let vocab = cfg.erm.total_vocab_size();
        let mut orch = Orchestrator::new(cfg.clone(), vocab);
        orch.run_colony(&ds, None).unwrap();
        assert_eq!(orch.global_step, cfg.colony_steps);
    }

    #[test]
    fn test_run_all() {
        let cfg = small_config();
        let ds = make_dataset(&cfg);
        let vocab = cfg.erm.total_vocab_size();
        let total = cfg.warm_start_steps + cfg.colony_steps;
        let mut orch = Orchestrator::new(cfg, vocab);
        orch.run_all(&ds, None).unwrap();
        assert_eq!(orch.global_step, total);
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let cfg = small_config();
        let ds = make_dataset(&cfg);
        let vocab = cfg.erm.total_vocab_size();
        let mut orch = Orchestrator::new(cfg, vocab);
        orch.run_warm_start(&ds, None).unwrap();

        let dir = format!(
            "/tmp/erm_ckpt_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );
        orch.save_checkpoint(&dir, "warm_start").unwrap();

        let (loaded, phase) = Orchestrator::load_checkpoint(&dir).unwrap();
        assert_eq!(phase, "warm_start");
        assert_eq!(loaded.global_step, orch.global_step);
        assert_eq!(loaded.config, orch.config);
        assert_eq!(loaded.scorer.token_emb, orch.scorer.token_emb);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_scorer_bytes_roundtrip() {
        let cfg = small_config();
        let vocab = cfg.erm.total_vocab_size();
        let scorer = Scorer::new(&cfg.erm, vocab, 99);
        let bytes = scorer_to_bytes(&scorer);
        let mut scorer2 = Scorer::new(&cfg.erm, vocab, 0);
        bytes_to_scorer(&bytes, &mut scorer2).unwrap();
        assert_eq!(scorer.token_emb, scorer2.token_emb);
        assert_eq!(scorer.logit_w, scorer2.logit_w);
    }

    #[test]
    fn test_bytes_to_scorer_wrong_length() {
        let cfg = small_config();
        let vocab = cfg.erm.total_vocab_size();
        let mut scorer = Scorer::new(&cfg.erm, vocab, 0);
        let bad = vec![0u8; 4];
        assert!(bytes_to_scorer(&bad, &mut scorer).is_err());
    }
}
