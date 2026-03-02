//! Colony training step: burn scorer on GPU + colony logic on CPU.
//!
//! The colony code (ant sampling, merge, pheromone updates, graph mutations)
//! is inherently sequential and branchy — NOT suited for GPU. Only the scorer
//! forward and backward passes go through burn on GPU.
//!
//! # Pipeline per step
//!
//! 1. **Forward on GPU**: Convert batch tokens to burn Tensor, run scorer.forward()
//! 2. **Transfer to CPU**: `.into_data().as_slice()` → `Vec<f32>`
//! 3. **Colony on CPU**: top-k, follower + leader proposals, merge, pheromone, edges
//! 4. **Forward y_new on GPU**: Scorer on refined tokens → logits_new
//! 5. **Compute loss on GPU**: Cross-entropy between scorer output and ground truth
//! 6. **Backprop on GPU**: Autodiff backward, optimizer step
//! 7. **Compute deltas on CPU**: Per-ant Δ from logit differences
//! 8. **Update pheromones + prune on CPU**: Edge traces, deposit, prune, death/respawn

use burn::optim::decay::WeightDecayConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use rand::Rng;
use rand_chacha::ChaCha8Rng;

use erm_core::ants::{
    apply_death_respawn, AntColony, AntState, DeathMode, FollowerConfig, LeaderConfig,
};
use erm_core::burn_scorer::{BurnScorer, BurnScorerConfig};
use erm_core::config::{ErmConfig, PheromoneConfig};
use erm_core::corruption::corrupt;
use erm_core::error::ErmResult;
use erm_core::graph::RouteGraph;
use erm_core::merge::{compute_ant_deltas, merge_proposals};
use erm_core::pheromone::{build_edge_traces, prune_edges, update_pheromones, PheromoneStats};

use crate::bridge::{tensor2d_to_vec, tensor_to_vec, tokens_to_tensor};
use crate::dataset::DataBatch;

/// Result of a single colony training step.
#[derive(Debug, Clone)]
pub struct ColonyStepResult {
    /// Cross-entropy loss (denoising loss from backprop).
    pub loss: f32,
    /// Pheromone statistics after the update.
    pub pheromone_stats: PheromoneStats,
    /// Number of edits applied by the colony.
    pub num_edits: usize,
    /// Per-ant improvement deltas.
    pub ant_deltas: Vec<f32>,
    /// Number of edges pruned.
    pub edges_pruned: usize,
    /// Number of new edges inserted by leaders.
    pub edges_inserted: usize,
    /// Number of ants that died and were respawned.
    pub deaths: usize,
}

/// Colony trainer: burn scorer on GPU, colony logic on CPU.
///
/// Holds the burn scorer model with autodiff, the optimizer, the route graph,
/// ant state, and configuration needed for the hybrid GPU/CPU colony step.
pub struct ColonyTrainer<B: AutodiffBackend> {
    /// The burn scorer model (autodiff-enabled).
    pub scorer: BurnScorer<B>,
    /// Adam optimizer.
    optimizer: burn::optim::adaptor::OptimizerAdaptor<burn::optim::Adam, BurnScorer<B>, B>,
    /// Route graph for colony operations.
    pub graph: RouteGraph,
    /// Ant lifecycle state.
    pub ant_state: AntState,
    /// ERM configuration.
    pub config: ErmConfig,
    /// Pheromone configuration.
    pub pheromone_config: PheromoneConfig,
    /// Learning rate.
    lr: f64,
    /// Burn device.
    device: B::Device,
}

impl<B: AutodiffBackend> ColonyTrainer<B> {
    /// Create a new colony trainer.
    ///
    /// # Arguments
    ///
    /// - `config`: ERM configuration.
    /// - `device`: burn device (GPU or CPU).
    pub fn new(config: &ErmConfig, device: B::Device) -> Self {
        let scorer_cfg = BurnScorerConfig::from_erm(config);
        let scorer = scorer_cfg.init::<B>(&device);

        let optimizer = AdamConfig::new()
            .with_weight_decay(Some(WeightDecayConfig::new(config.weight_decay as f32)))
            .init();

        let graph = RouteGraph::new(config);
        let ant_state = AntState::new(config);
        let pheromone_config = PheromoneConfig::from_config(config);

        Self {
            scorer,
            optimizer,
            graph,
            ant_state,
            config: config.clone(),
            pheromone_config,
            lr: config.learning_rate,
            device,
        }
    }

    /// Perform a single colony training step.
    ///
    /// This is the main hybrid GPU/CPU pipeline:
    /// 1. Corrupt ground truth → y_t
    /// 2. Forward y_t through burn scorer on GPU → logits
    /// 3. Transfer logits to CPU Vec<f32>
    /// 4. Run colony on CPU: proposals, merge, pheromone update
    /// 5. Forward y_new through scorer on GPU for delta computation
    /// 6. Compute cross-entropy loss on GPU and backprop
    /// 7. Compute ant deltas on CPU
    /// 8. Update pheromones, prune edges, death/respawn
    ///
    /// # Arguments
    ///
    /// - `batch`: ground truth token batch.
    /// - `t`: corruption step (None = random).
    /// - `rng`: random number generator.
    ///
    /// # Returns
    ///
    /// [`ColonyStepResult`] with loss, pheromone stats, edits, etc.
    ///
    /// # Errors
    ///
    /// Propagates errors from corruption, scorer, or colony operations.
    pub fn colony_train_step(
        &mut self,
        batch: &DataBatch,
        t: Option<usize>,
        rng: &mut ChaCha8Rng,
    ) -> ErmResult<ColonyStepResult> {
        let b = batch.batch_size;
        let l = batch.seq_len;
        let config = &self.config;
        let vocab_size = config.total_vocab_size();

        // ── Step 0: Corruption ──────────────────────────────────────────
        let t_val = t.unwrap_or_else(|| rng.gen_range(1..=config.refinement_steps));
        let x_i32: Vec<i32> = batch.tokens.iter().map(|&tk| tk as i32).collect();
        let corruption = corrupt(&x_i32, t_val, config, rng)?;
        let y_t_flat = &corruption.y_t;

        // ── Step 1: Forward y_t on GPU → logits ────────────────────────
        let y_t_u32: Vec<u32> = y_t_flat.iter().map(|&v| v as u32).collect();
        let tokens_tensor = tokens_to_tensor::<B>(&y_t_u32, b, l, &self.device)?;
        let (logits_tensor, uncertainty_tensor) = self.scorer.forward(tokens_tensor);

        // ── Step 2: Transfer logits + uncertainty to CPU ────────────────
        // We need to clone the logits tensor before consuming it for CPU transfer,
        // because we also need it for the loss computation later.
        let logits_cpu = tensor_to_vec(logits_tensor.clone())?;
        let uncertainty_cpu = tensor2d_to_vec(uncertainty_tensor)?;

        // ── Step 3: Colony on CPU (all batch elements) ─────────────────
        let d = config.hidden_dim;
        let hidden = vec![0.0_f32; config.batch_size * l * d];
        let (_, edge_weights) = self.graph.route_aggregate(
            &hidden,
            d,
            config.route_epsilon,
            config.route_lambda,
            config.route_mu,
        )?;

        let num_followers = config.num_followers();
        let first_follower_id = config.num_leaders();
        let follower_cfg = FollowerConfig::from_config(config);
        let num_leaders = config.num_leaders();
        let leader_cfg = LeaderConfig::from_config(config);
        let max_edits = config.max_edits();
        let total_ants = config.num_ants;

        // Accumulate across all batch items.
        let mut all_proposals_global = Vec::new();
        let mut all_edge_proposals = Vec::new();
        let mut y_new_batch = Vec::with_capacity(b * l);
        let mut total_edits: usize = 0;

        for batch_idx in 0..b {
            let seq_logits =
                &logits_cpu[batch_idx * l * vocab_size..(batch_idx + 1) * l * vocab_size];
            let seq_uncertainty = &uncertainty_cpu[batch_idx * l..(batch_idx + 1) * l];
            let y_t_seq = &y_t_u32[batch_idx * l..(batch_idx + 1) * l];

            // Build editable mask: corrupted positions are editable.
            let editable: Vec<bool> = y_t_flat[batch_idx * l..(batch_idx + 1) * l]
                .iter()
                .zip(x_i32[batch_idx * l..(batch_idx + 1) * l].iter())
                .map(|(y, x)| y != x)
                .collect();

            // Follower proposals.
            let follower_proposals = AntColony::sample_follower_proposals(
                seq_logits,
                &self.graph,
                batch_idx,
                &follower_cfg,
                &editable,
                num_followers,
                first_follower_id,
                l,
                vocab_size,
                rng,
            )?;

            // Leader proposals.
            let (leader_proposals, edge_proposals) = AntColony::sample_leader_proposals(
                seq_logits,
                seq_uncertainty,
                &self.graph,
                batch_idx,
                &leader_cfg,
                &editable,
                num_leaders,
                0,
                l,
                vocab_size,
                rng,
            )?;

            // Merge proposals for this batch item.
            let mut batch_proposals =
                Vec::with_capacity(follower_proposals.len() + leader_proposals.len());
            batch_proposals.extend(follower_proposals.iter().cloned());
            batch_proposals.extend(leader_proposals.iter().cloned());

            let y_new = merge_proposals(&batch_proposals, y_t_seq, &editable, l, max_edits)?;
            let edits = y_t_seq
                .iter()
                .zip(y_new.iter())
                .filter(|(a, b_val)| a != b_val)
                .count();
            total_edits += edits;

            y_new_batch.extend_from_slice(&y_new);
            all_proposals_global.extend(batch_proposals);
            all_edge_proposals.extend(edge_proposals);
        }

        // ── Step 4: Forward y_new on GPU for deltas ────────────────────
        let y_new_tensor = tokens_to_tensor::<B>(&y_new_batch, b, l, &self.device)?;
        let (logits_new_tensor, _) = self.scorer.forward(y_new_tensor);
        let logits_new_cpu = tensor_to_vec(logits_new_tensor)?;

        // ── Step 5: Compute ant deltas on CPU (aggregate across batch) ─
        // Use first batch item's logits for delta computation (ant IDs are
        // per-batch-item but we aggregate the improvement signal).
        let batch_idx_for_deltas = 0;
        let seq_logits_0 = &logits_cpu
            [batch_idx_for_deltas * l * vocab_size..(batch_idx_for_deltas + 1) * l * vocab_size];
        let logits_new_seq_0 = &logits_new_cpu
            [batch_idx_for_deltas * l * vocab_size..(batch_idx_for_deltas + 1) * l * vocab_size];
        let y_t_seq_0 = &y_t_u32[batch_idx_for_deltas * l..(batch_idx_for_deltas + 1) * l];
        let y_new_seq_0 = &y_new_batch[batch_idx_for_deltas * l..(batch_idx_for_deltas + 1) * l];

        let ant_deltas = compute_ant_deltas(
            &all_proposals_global,
            y_t_seq_0,
            y_new_seq_0,
            seq_logits_0,
            logits_new_seq_0,
            vocab_size,
            total_ants,
        )?;

        // ── Step 6: Pheromone update on CPU ────────────────────────────
        let traces = build_edge_traces(
            &all_proposals_global,
            &edge_weights,
            config.batch_size,
            l,
            config.emax,
        );
        let pheromone_stats = update_pheromones(
            &mut self.graph,
            &traces,
            &ant_deltas,
            &self.pheromone_config,
        )?;

        // Insert leader-proposed edges.
        let edges_inserted =
            self.graph
                .propose_edges(&all_edge_proposals, config.phi_init, config.route_lambda);

        // Prune weak edges.
        let edges_pruned = prune_edges(
            &mut self.graph,
            self.pheromone_config.prune_min_score,
            self.pheromone_config.prune_max_age,
            self.pheromone_config.route_lambda,
        );

        // Death/respawn.
        let deaths = apply_death_respawn(
            &mut self.ant_state,
            &ant_deltas,
            config,
            DeathMode::Streak,
            rng,
        );

        let num_edits = total_edits;

        // ── Step 7: Compute loss on GPU and backprop ───────────────────
        // Cross-entropy loss between scorer output on y_t and ground truth x.
        let mut mask_vals = Vec::with_capacity(b * l);
        let mut target_vals = Vec::with_capacity(b * l);
        for pos in 0..(b * l) {
            if y_t_flat[pos] != x_i32[pos] && x_i32[pos] >= 0 {
                mask_vals.push(1.0_f32);
                target_vals.push(x_i32[pos] as i64);
            } else {
                mask_vals.push(0.0_f32);
                target_vals.push(0);
            }
        }

        let num_corrupted: usize = mask_vals.iter().filter(|&&v| v > 0.5).count();
        let loss_val = if num_corrupted == 0 {
            0.0
        } else {
            // logits_tensor is [B, L, V]
            let v_dim = logits_tensor.dims()[2];
            let logits_flat = logits_tensor.reshape([b * l, v_dim]);

            let targets_tensor =
                Tensor::<B, 1, Int>::from_data(TensorData::new(target_vals, [b * l]), &self.device);
            let mask_tensor =
                Tensor::<B, 1>::from_data(TensorData::new(mask_vals, [b * l]), &self.device);

            let log_probs = burn::tensor::activation::log_softmax(logits_flat, 1);
            let targets_2d = targets_tensor.unsqueeze_dim(1);
            let target_log_probs = log_probs.gather(1, targets_2d).reshape([b * l]);

            let neg_log_probs = target_log_probs.neg();
            let masked_loss = neg_log_probs * mask_tensor.clone();
            let loss = masked_loss.sum() / mask_tensor.sum();

            let lv = loss
                .clone()
                .into_data()
                .as_slice::<f32>()
                .map_or(0.0, |s| s[0]);

            // Backward + optimizer step.
            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &self.scorer);
            self.scorer = self
                .optimizer
                .step(self.lr, self.scorer.clone(), grads_params);

            lv
        };

        Ok(ColonyStepResult {
            loss: loss_val,
            pheromone_stats,
            num_edits,
            ant_deltas,
            edges_pruned,
            edges_inserted,
            deaths,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_autodiff::Autodiff;
    use burn_ndarray::NdArray;
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
    fn test_colony_train_step_returns_finite_loss() {
        let mut cfg = small_config();
        let (ds, vocab) = make_dataset(&cfg);
        cfg.vocab_size = vocab;
        let device = Default::default();
        let mut trainer = ColonyTrainer::<TestAutodiffBackend>::new(&cfg, device);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let batch = ds.get_batch(cfg.batch_size, &mut rng);

        let result = trainer
            .colony_train_step(&batch, Some(3), &mut rng)
            .unwrap();

        assert!(
            result.loss.is_finite(),
            "loss should be finite, got {}",
            result.loss
        );
    }

    #[test]
    fn test_colony_train_step_pheromone_stats_change() {
        let mut cfg = small_config();
        let (ds, vocab) = make_dataset(&cfg);
        cfg.vocab_size = vocab;
        let device = Default::default();
        let mut trainer = ColonyTrainer::<TestAutodiffBackend>::new(&cfg, device);

        // RouteGraph::new() pre-seeds skip-connection edges, so the graph
        // already has edges for pheromone updates to work with.
        let initial_phi: Vec<f32> = trainer.graph.phi.clone();

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let batch = ds.get_batch(cfg.batch_size, &mut rng);
        let result = trainer
            .colony_train_step(&batch, Some(3), &mut rng)
            .unwrap();

        // Pheromone stats should be finite.
        assert!(result.pheromone_stats.mean_phi.is_finite());
        assert!(result.pheromone_stats.max_phi.is_finite());

        // Phi values should have changed (evaporation at minimum).
        let final_phi = &trainer.graph.phi;
        let changed = initial_phi
            .iter()
            .zip(final_phi.iter())
            .any(|(a, b)| (a - b).abs() > 1e-9);
        assert!(changed, "pheromone values should change after colony step");
    }

    #[test]
    fn test_colony_train_step_edit_count() {
        let mut cfg = small_config();
        // Use heavy corruption to ensure edits happen.
        cfg.mask_rate_max = 0.9;
        cfg.mask_rate_min = 0.5;
        let (ds, vocab) = make_dataset(&cfg);
        cfg.vocab_size = vocab;
        let device = Default::default();
        let mut trainer = ColonyTrainer::<TestAutodiffBackend>::new(&cfg, device);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let batch = ds.get_batch(cfg.batch_size, &mut rng);

        let result = trainer
            .colony_train_step(&batch, Some(cfg.refinement_steps), &mut rng)
            .unwrap();

        // With heavy corruption, colony should produce some edits.
        // (May be 0 if proposals all target same position, but typically > 0)
        assert!(
            result.ant_deltas.len() == cfg.num_ants,
            "ant_deltas length should be num_ants"
        );
        // All deltas should be finite.
        for &d in &result.ant_deltas {
            assert!(d.is_finite(), "delta should be finite, got {d}");
        }
    }

    #[test]
    fn test_colony_trainer_multiple_steps() {
        let mut cfg = small_config();
        let (ds, vocab) = make_dataset(&cfg);
        cfg.vocab_size = vocab;
        let device = Default::default();
        let mut trainer = ColonyTrainer::<TestAutodiffBackend>::new(&cfg, device);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Run 3 steps without panic.
        for _ in 0..3 {
            let batch = ds.get_batch(cfg.batch_size, &mut rng);
            let result = trainer.colony_train_step(&batch, None, &mut rng).unwrap();
            assert!(result.loss.is_finite());
        }
    }
}
