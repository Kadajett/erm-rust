//! Colony training orchestrator: manages warm-start checkpoint loading
//! and colony phase execution.
//!
//! The orchestrator coordinates:
//! 1. Loading warm-start weights from a denoiser checkpoint into the burn scorer
//! 2. Running the colony training phase with logging
//! 3. Checkpoint saving/loading for the colony phase
//!
//! # Checkpoint format
//!
//! Colony checkpoints are saved as a directory containing:
//! - `config.json`: serialised [`ColonyCheckpointMeta`]
//! - `graph.json`: serialised [`RouteGraph`]
//! - `ant_state.json`: serialised [`AntState`]
//! - `step.json`: step counter and loss history summary

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use burn::tensor::backend::AutodiffBackend;

use erm_core::config::ErmConfig;
use erm_core::error::{ErmError, ErmResult};
use erm_core::graph::RouteGraph;

use crate::colony_training::{ColonyStepResult, ColonyTrainer};
use crate::dataset::TextDataset;

/// Colony training configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ColonyTrainingConfig {
    /// ERM model hyperparameters.
    pub erm: ErmConfig,
    /// Number of colony training steps.
    pub colony_steps: usize,
    /// Log every N steps.
    pub log_every: usize,
    /// Save checkpoint every N steps (0 = disabled).
    pub checkpoint_every: usize,
    /// Master RNG seed.
    pub seed: u64,
}

impl Default for ColonyTrainingConfig {
    fn default() -> Self {
        Self {
            erm: ErmConfig::default(),
            colony_steps: 5_000,
            log_every: 100,
            checkpoint_every: 500,
            seed: 42,
        }
    }
}

/// A single log entry from colony training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColonyLogRecord {
    /// Training step.
    pub step: usize,
    /// Denoising loss.
    pub loss: f32,
    /// Number of colony edits.
    pub num_edits: usize,
    /// Mean pheromone across active edges.
    pub mean_phi: f32,
    /// Edges pruned in this step.
    pub edges_pruned: usize,
    /// Edges inserted by leaders in this step.
    pub edges_inserted: usize,
    /// Ant deaths in this step.
    pub deaths: usize,
}

/// Metadata for colony checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ColonyCheckpointMeta {
    step: usize,
    config: ColonyTrainingConfig,
}

/// Colony training orchestrator.
///
/// Manages the colony training phase, logging, and checkpointing.
pub struct ColonyOrchestrator<B: AutodiffBackend> {
    /// The colony trainer (burn scorer + colony state).
    pub trainer: ColonyTrainer<B>,
    /// Colony training configuration.
    pub config: ColonyTrainingConfig,
    /// Loss/stats log.
    pub log: Vec<ColonyLogRecord>,
    /// Current step counter.
    pub step: usize,
    /// Internal RNG.
    rng: ChaCha8Rng,
}

impl<B: AutodiffBackend> ColonyOrchestrator<B> {
    /// Create a new colony orchestrator.
    ///
    /// # Arguments
    ///
    /// - `config`: colony training configuration.
    /// - `device`: burn device for the scorer.
    pub fn new(config: ColonyTrainingConfig, device: B::Device) -> Self {
        let trainer = ColonyTrainer::<B>::new(&config.erm, device);
        let rng = ChaCha8Rng::seed_from_u64(config.seed);

        Self {
            trainer,
            config,
            log: Vec::new(),
            step: 0,
            rng,
        }
    }

    /// Run the colony training phase.
    ///
    /// Iterates for `colony_steps`, logging every `log_every` steps.
    ///
    /// # Arguments
    ///
    /// - `dataset`: training data.
    /// - `checkpoint_dir`: optional directory for periodic checkpoints.
    ///
    /// # Returns
    ///
    /// Final colony step result from the last step.
    ///
    /// # Errors
    ///
    /// Propagates errors from the colony training step or checkpointing.
    pub fn run_colony_phase(
        &mut self,
        dataset: &TextDataset,
        checkpoint_dir: Option<&str>,
    ) -> ErmResult<Option<ColonyStepResult>> {
        let total_steps = self.config.colony_steps;
        let batch_size = self.config.erm.batch_size;
        let mut last_result = None;

        for _ in 0..total_steps {
            let batch = dataset.get_batch(batch_size, &mut self.rng);
            let result = self
                .trainer
                .colony_train_step(&batch, None, &mut self.rng)?;

            if self.step == 0 || (self.step + 1).is_multiple_of(self.config.log_every) {
                let record = ColonyLogRecord {
                    step: self.step,
                    loss: result.loss,
                    num_edits: result.num_edits,
                    mean_phi: result.pheromone_stats.mean_phi,
                    edges_pruned: result.edges_pruned,
                    edges_inserted: result.edges_inserted,
                    deaths: result.deaths,
                };
                eprintln!(
                    "[colony step {:>6}] loss={:.4} edits={} mean_φ={:.4} pruned={} inserted={} deaths={}",
                    record.step, record.loss, record.num_edits,
                    record.mean_phi, record.edges_pruned, record.edges_inserted, record.deaths
                );
                self.log.push(record);
            }

            // Checkpoint.
            if let Some(dir) = checkpoint_dir {
                if self.config.checkpoint_every > 0
                    && (self.step + 1).is_multiple_of(self.config.checkpoint_every)
                {
                    self.save_checkpoint(dir)?;
                }
            }

            last_result = Some(result);
            self.step += 1;
        }

        Ok(last_result)
    }

    /// Save a colony checkpoint.
    ///
    /// Saves the graph and ant state (JSON), plus metadata.
    /// Note: burn model weights are NOT saved here — use burn's built-in
    /// serialisation for the scorer if needed.
    ///
    /// # Errors
    ///
    /// Returns error if directory creation or file writes fail.
    pub fn save_checkpoint(&self, dir: &str) -> ErmResult<()> {
        std::fs::create_dir_all(dir).map_err(|e| {
            ErmError::InvalidConfig(format!("cannot create checkpoint dir {dir}: {e}"))
        })?;

        // Save config + step metadata.
        let meta = ColonyCheckpointMeta {
            step: self.step,
            config: self.config.clone(),
        };
        let meta_json = serde_json::to_string_pretty(&meta)?;
        std::fs::write(format!("{dir}/colony_meta.json"), meta_json)
            .map_err(|e| ErmError::InvalidConfig(format!("cannot write colony_meta.json: {e}")))?;

        // Save route graph.
        let graph_json = serde_json::to_string(&self.trainer.graph)?;
        std::fs::write(format!("{dir}/graph.json"), graph_json)
            .map_err(|e| ErmError::InvalidConfig(format!("cannot write graph.json: {e}")))?;

        // Save ant state.
        let ant_json = serde_json::to_string(&self.trainer.ant_state)?;
        std::fs::write(format!("{dir}/ant_state.json"), ant_json)
            .map_err(|e| ErmError::InvalidConfig(format!("cannot write ant_state.json: {e}")))?;

        // Save loss log.
        let log_json = serde_json::to_string_pretty(&self.log)?;
        std::fs::write(format!("{dir}/colony_log.json"), log_json)
            .map_err(|e| ErmError::InvalidConfig(format!("cannot write colony_log.json: {e}")))?;

        Ok(())
    }

    /// Load colony state (graph + ant state) from a checkpoint directory.
    ///
    /// This restores the graph and ant state into the trainer. The burn
    /// scorer weights must be loaded separately (e.g., from a warm-start
    /// checkpoint).
    ///
    /// # Errors
    ///
    /// Returns error if files cannot be read or parsed.
    pub fn load_colony_checkpoint(&mut self, dir: &str) -> ErmResult<()> {
        // Load metadata.
        let meta_json = std::fs::read_to_string(format!("{dir}/colony_meta.json"))
            .map_err(|e| ErmError::InvalidConfig(format!("cannot read colony_meta.json: {e}")))?;
        let meta: ColonyCheckpointMeta = serde_json::from_str(&meta_json)?;
        self.step = meta.step;

        // Load graph.
        let graph_json = std::fs::read_to_string(format!("{dir}/graph.json"))
            .map_err(|e| ErmError::InvalidConfig(format!("cannot read graph.json: {e}")))?;
        let graph: RouteGraph = serde_json::from_str(&graph_json)?;
        self.trainer.graph = graph;

        // Load ant state.
        let ant_json = std::fs::read_to_string(format!("{dir}/ant_state.json"))
            .map_err(|e| ErmError::InvalidConfig(format!("cannot read ant_state.json: {e}")))?;
        let ant_state = serde_json::from_str(&ant_json)?;
        self.trainer.ant_state = ant_state;

        // Load log if present.
        let log_path = format!("{dir}/colony_log.json");
        if let Ok(log_json) = std::fs::read_to_string(&log_path) {
            if let Ok(log) = serde_json::from_str(&log_json) {
                self.log = log;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_autodiff::Autodiff;
    use burn_ndarray::NdArray;
    use erm_core::tokenizer::CharTokenizer;

    type TestAutodiffBackend = Autodiff<NdArray<f32>>;

    fn small_config() -> ColonyTrainingConfig {
        ColonyTrainingConfig {
            erm: ErmConfig {
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
                learning_rate: 1e-3,
                weight_decay: 0.0,
                num_ants: 10,
                leader_fraction: 0.10,
                pmax: 4,
                topk: 4,
                emax: 4,
                max_edits_per_step: 0.15,
                ..ErmConfig::default()
            },
            colony_steps: 2,
            log_every: 1,
            checkpoint_every: 0,
            seed: 42,
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
    fn test_colony_orchestrator_runs() {
        let mut config = small_config();
        let (ds, vocab) = make_dataset(&config.erm);
        config.erm.vocab_size = vocab;
        let device = Default::default();
        let mut orch = ColonyOrchestrator::<TestAutodiffBackend>::new(config.clone(), device);

        let result = orch.run_colony_phase(&ds, None).unwrap();
        assert!(result.is_some());
        assert_eq!(orch.step, config.colony_steps);
        assert!(!orch.log.is_empty());
    }

    #[test]
    fn test_colony_orchestrator_checkpoint_roundtrip() {
        let mut config = small_config();
        let (ds, vocab) = make_dataset(&config.erm);
        config.erm.vocab_size = vocab;
        let device = Default::default();
        let mut orch = ColonyOrchestrator::<TestAutodiffBackend>::new(config.clone(), device);

        // Run a couple steps.
        orch.run_colony_phase(&ds, None).unwrap();

        // Save checkpoint.
        let dir = format!(
            "/tmp/erm_colony_ckpt_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );
        orch.save_checkpoint(&dir).unwrap();

        // Load into a fresh orchestrator.
        let device2 = Default::default();
        let mut orch2 = ColonyOrchestrator::<TestAutodiffBackend>::new(config, device2);
        orch2.load_colony_checkpoint(&dir).unwrap();

        assert_eq!(orch2.step, orch.step);
        assert_eq!(orch2.trainer.graph.nbr_idx, orch.trainer.graph.nbr_idx);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_colony_orchestrator_log_has_records() {
        let mut config = small_config();
        config.log_every = 1;
        let (ds, vocab) = make_dataset(&config.erm);
        config.erm.vocab_size = vocab;
        let device = Default::default();
        let mut orch = ColonyOrchestrator::<TestAutodiffBackend>::new(config, device);

        orch.run_colony_phase(&ds, None).unwrap();

        // Should have log records.
        assert!(
            !orch.log.is_empty(),
            "log should have records after colony phase"
        );
        for rec in &orch.log {
            assert!(rec.loss.is_finite(), "logged loss should be finite");
        }
    }
}
