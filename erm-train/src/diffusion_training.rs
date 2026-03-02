//! Diffusion-style colony training with coarse-to-fine parallel refinement.
//!
//! This module implements Mercury-style diffusion training layered on top of
//! the ERM ant colony. The key insight: at each noise level `t`, the colony
//! proposes parallel edits, then those refined tokens feed the next (cleaner)
//! level. Loss is accumulated across all T levels with γ(t) weighting.
//!
//! # Per-training-step pipeline
//!
//! ```text
//! For t = T..1 (coarse → fine):
//!   1. Build z_t = corrupt(x, t)   ← masking/replacing tokens
//!   2. forward_with_hidden(z_t) → logits [B,L,V], hidden [B,L,d]
//!   3. Ant colony proposes parallel edits (more/coarser at high t)
//!   4. Merge → y_{t-1}
//!   5. Accumulate: loss += γ(t) * CE(x | z_t)
//! Backprop accumulated loss; update scorer + pheromones
//! ```
//!
//! # Loss
//!
//! `L = E_t[ γ(t) * CE(x | z_t) ]`
//! γ(t) follows the configured schedule (cosine/linear/sqrt).

use burn::optim::decay::WeightDecayConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::AutodiffBackend;

use rand_chacha::ChaCha8Rng;

use erm_core::ants::{
    apply_death_respawn, AntColony, AntState, DeathMode, FollowerConfig, LeaderConfig,
};
use erm_core::burn_scorer::{BurnScorer, BurnScorerConfig};
use erm_core::config::{ErmConfig, PheromoneConfig};
use erm_core::corruption::corrupt;
use erm_core::error::{ErmError, ErmResult};
use erm_core::graph::RouteGraph;
use erm_core::merge::{compute_ant_deltas, merge_proposals};
use erm_core::pheromone::{
    build_edge_traces, prune_edges, update_pheromones_with_stats, PheromoneStats,
    RunningDeltaStats,
};

use crate::bridge::{tensor2d_to_vec, tensor_to_vec, tokens_to_tensor};
use crate::streaming_dataset::TokenBatch;

/// Result of one diffusion training step (summed/averaged over T refinement levels).
#[derive(Debug, Clone)]
pub struct DiffusionStepResult {
    /// Weighted-average diffusion loss across T steps.
    pub loss: f32,
    /// Total colony edits across all T levels.
    pub total_edits: usize,
    /// Pheromone stats from the final (finest) refinement level.
    pub pheromone_stats: PheromoneStats,
    /// Per-ant improvement deltas from the final level.
    pub ant_deltas: Vec<f32>,
    /// Number of ant deaths.
    pub deaths: usize,
    /// Number of edges pruned.
    pub edges_pruned: usize,
    /// Number of edges inserted by leaders.
    pub edges_inserted: usize,
}

/// Diffusion colony trainer.
///
/// Runs a T-step coarse-to-fine diffusion loop per training iteration:
/// GPU scorer for forward/backward, CPU colony for proposals/pheromones.
pub struct DiffusionTrainer<B: AutodiffBackend> {
    /// The burn scorer model with autodiff.
    pub scorer: BurnScorer<B>,
    /// Adam optimizer.
    optimizer: burn::optim::adaptor::OptimizerAdaptor<burn::optim::Adam, BurnScorer<B>, B>,
    /// Route graph for colony pheromone routing.
    pub graph: RouteGraph,
    /// Ant lifecycle state.
    pub ant_state: AntState,
    /// ERM configuration.
    pub config: ErmConfig,
    /// Pheromone configuration.
    pub pheromone_config: PheromoneConfig,
    /// Running delta statistics for normalized pheromone deposit.
    pub delta_stats: RunningDeltaStats,
    /// Current training step.
    pub step: usize,
    /// Learning rate.
    lr: f64,
    /// Burn device.
    device: B::Device,
}

impl<B: AutodiffBackend> DiffusionTrainer<B> {
    /// Create a new diffusion trainer.
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
            delta_stats: RunningDeltaStats::new(),
            step: 0,
            lr: config.learning_rate,
            device,
        }
    }

    /// Run one diffusion training step over T refinement levels.
    ///
    /// Iterates from heaviest noise (t=T) to lightest (t=1), accumulating the
    /// diffusion loss `L = Σ_t γ(t) * CE(x | z_t)`. Colony runs on CPU at each
    /// level, with edit scale proportional to noise level (coarse at high t).
    ///
    /// # Returns
    ///
    /// [`DiffusionStepResult`] with loss, edits, pheromone stats, and colony info.
    pub fn diffusion_step(
        &mut self,
        batch: &TokenBatch,
        rng: &mut ChaCha8Rng,
    ) -> ErmResult<DiffusionStepResult> {
        let cfg = &self.config.clone();
        let b = batch.batch_size;
        let l = batch.seq_len;
        let big_t = cfg.diffusion_steps.max(1);
        let vocab_size = cfg.total_vocab_size();

        let x_i32: Vec<i32> = batch.tokens.iter().map(|&v| v as i32).collect();

        // Accumulate weighted loss across T levels on GPU.
        let mut accumulated_loss: Option<Tensor<B, 1>> = None;
        let mut total_edits = 0usize;
        let mut last_pheromone_stats = PheromoneStats {
            mean_phi: 0.0,
            max_phi: 0.0,
            mean_taint: 0.0,
            tainted_count: 0,
        };
        let mut last_ant_deltas: Vec<f32> = vec![0.0; cfg.num_ants];
        let mut total_deaths = 0usize;
        let mut total_pruned = 0usize;
        let mut total_inserted = 0usize;

        // Iterate from heavy noise (t=big_t) down to fine (t=1).
        for t in (1..=big_t).rev() {
            // 1. Corrupt clean tokens at noise level t → z_t.
            let corruption = corrupt(&x_i32, t, cfg, rng)?;
            let z_t_u32: Vec<u32> = corruption.y_t.iter().map(|&v| v as u32).collect();

            // 2. forward_with_hidden(z_t) → logits [B,L,V], hidden [B,L,d].
            let z_t_tensor = tokens_to_tensor::<B>(&z_t_u32, b, l, &self.device)?;
            let (logits_tensor, unc_tensor, hidden_tensor) =
                self.scorer.forward_with_hidden(z_t_tensor);

            // 3. Diffusion loss: γ(t) * CE(x | z_t).
            let x_tensor = tokens_to_tensor::<B>(&batch.tokens, b, l, &self.device)?;
            let gamma = cfg.gamma(t);
            let step_loss = diffusion_ce_loss::<B>(
                logits_tensor.clone(),
                x_tensor,
                gamma,
                b,
                l,
                vocab_size,
            );
            accumulated_loss = Some(match accumulated_loss {
                None => step_loss,
                Some(acc) => acc + step_loss,
            });

            // 4. CPU colony step.
            // Transfer what we need to CPU.
            let logits_cpu = tensor_to_vec(logits_tensor)?;
            let uncertainty_cpu = tensor2d_to_vec(unc_tensor)?;
            let hidden_cpu = tensor_to_vec(hidden_tensor)?;

            // Scale edit budget: coarse (more edits) at high t, fine (fewer) at low t.
            let edit_scale = t as f32 / big_t as f32;
            let effective_max_edits =
                ((cfg.max_edits() as f32 * edit_scale.max(0.1)).ceil() as usize).max(1);

            // Route aggregation using hidden states.
            let d = cfg.hidden_dim;
            let (_, edge_weights) = self.graph.route_aggregate(
                &hidden_cpu,
                d,
                cfg.route_epsilon,
                cfg.route_lambda,
                cfg.route_mu,
            )?;

            let num_followers = cfg.num_followers();
            let first_follower_id = cfg.num_leaders();
            let follower_cfg = FollowerConfig::from_config(cfg);
            let leader_cfg = LeaderConfig::from_config(cfg);
            let num_leaders = cfg.num_leaders();

            let mut y_new_batch = Vec::with_capacity(b * l);
            let mut all_proposals = Vec::new();
            let mut all_edge_proposals = Vec::new();

            for batch_idx in 0..b {
                let seq_logits =
                    &logits_cpu[batch_idx * l * vocab_size..(batch_idx + 1) * l * vocab_size];
                let seq_uncertainty =
                    &uncertainty_cpu[batch_idx * l..(batch_idx + 1) * l];
                let z_t_seq = &z_t_u32[batch_idx * l..(batch_idx + 1) * l];

                // Editable mask: positions corrupted from clean tokens.
                let editable: Vec<bool> = corruption.y_t
                    [batch_idx * l..(batch_idx + 1) * l]
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

                let mut batch_proposals = Vec::with_capacity(
                    follower_proposals.len() + leader_proposals.len(),
                );
                batch_proposals.extend(follower_proposals);
                batch_proposals.extend(leader_proposals);

                let y_new = merge_proposals(
                    &batch_proposals,
                    z_t_seq,
                    &editable,
                    l,
                    effective_max_edits,
                )?;

                let edits = z_t_seq
                    .iter()
                    .zip(y_new.iter())
                    .filter(|(a, b_v)| a != b_v)
                    .count();
                total_edits += edits;

                y_new_batch.extend_from_slice(&y_new);
                all_proposals.extend(batch_proposals);
                all_edge_proposals.extend(edge_proposals);
            }

            // 5. Compute ant deltas (needs logits_new → forward on y_new).
            let y_new_tensor = tokens_to_tensor::<B>(&y_new_batch, b, l, &self.device)?;
            let (logits_new_tensor, _) = self.scorer.forward(y_new_tensor);
            let logits_new_cpu = tensor_to_vec(logits_new_tensor)?;

            let mut ant_deltas = vec![0.0_f32; cfg.num_ants];
            for batch_idx in 0..b {
                let logits_b =
                    &logits_cpu[batch_idx * l * vocab_size..(batch_idx + 1) * l * vocab_size];
                let logits_new_b =
                    &logits_new_cpu[batch_idx * l * vocab_size..(batch_idx + 1) * l * vocab_size];
                let z_t_b = &z_t_u32[batch_idx * l..(batch_idx + 1) * l];
                let y_new_b = &y_new_batch[batch_idx * l..(batch_idx + 1) * l];

                let batch_deltas = compute_ant_deltas(
                    &all_proposals,
                    z_t_b,
                    y_new_b,
                    logits_b,
                    logits_new_b,
                    vocab_size,
                    cfg.num_ants,
                )?;
                for (acc, &d) in ant_deltas.iter_mut().zip(batch_deltas.iter()) {
                    *acc += d;
                }
            }
            if b > 1 {
                let b_f = b as f32;
                for d in &mut ant_deltas {
                    *d /= b_f;
                }
            }

            // 6. Pheromone update (CPU).
            let traces = build_edge_traces(
                &all_proposals,
                &edge_weights,
                b,
                l,
                cfg.emax,
            );
            let pstats = update_pheromones_with_stats(
                &mut self.graph,
                &traces,
                &ant_deltas,
                &self.pheromone_config,
                Some(&mut self.delta_stats),
            )?;

            // Prune edges.
            let pruned = prune_edges(
                &mut self.graph,
                self.pheromone_config.prune_min_score,
                self.pheromone_config.prune_max_age,
                self.pheromone_config.route_lambda,
            );
            total_pruned += pruned;

            // Death/respawn.
            let deaths = apply_death_respawn(
                &mut self.ant_state,
                &ant_deltas,
                cfg,
                DeathMode::Streak,
                self.step,
                rng,
            );
            total_deaths += deaths;

            // Insert leader edges into graph.
            let inserted = insert_leader_edges(
                &mut self.graph,
                &all_edge_proposals,
                cfg,
            );
            total_inserted += inserted;

            // Keep final-level stats.
            last_pheromone_stats = pstats;
            last_ant_deltas = ant_deltas;
        }

        // 7. Backprop accumulated loss.
        let final_loss = if let Some(loss_tensor) = accumulated_loss {
            let raw_loss: f32 = loss_tensor
                .clone()
                .mean()
                .into_data()
                .as_slice::<f32>()
                .map(|s| s[0])
                .unwrap_or(0.0);

            let norm_loss = loss_tensor.mean() / (big_t as f32);
            let grads = norm_loss.backward();
            let grad_params = GradientsParams::from_grads(grads, &self.scorer);
            self.scorer = self.optimizer.step(self.lr, self.scorer.clone(), grad_params);

            raw_loss / big_t as f32
        } else {
            0.0
        };

        self.step += 1;

        Ok(DiffusionStepResult {
            loss: final_loss,
            total_edits,
            pheromone_stats: last_pheromone_stats,
            ant_deltas: last_ant_deltas,
            deaths: total_deaths,
            edges_pruned: total_pruned,
            edges_inserted: total_inserted,
        })
    }

    /// Save a checkpoint: scorer.bin, graph.json, ant_state.json, config.json.
    pub fn save_checkpoint(&self, dir: &str) -> ErmResult<()> {
        std::fs::create_dir_all(dir).map_err(|e| {
            ErmError::InvalidConfig(format!("cannot create checkpoint dir {dir}: {e}"))
        })?;

        // Scorer weights (burn binary format).
        let scorer_path = format!("{dir}/scorer");
        BinFileRecorder::<FullPrecisionSettings>::default()
            .record(self.scorer.clone().into_record(), scorer_path.into())
            .map_err(|e| ErmError::InvalidConfig(format!("scorer save failed: {e}")))?;

        // Graph state.
        let graph_json = serde_json::to_string_pretty(&self.graph)
            .map_err(|e| ErmError::InvalidConfig(format!("graph serialize: {e}")))?;
        std::fs::write(format!("{dir}/graph.json"), graph_json)
            .map_err(|e| ErmError::InvalidConfig(format!("write graph.json: {e}")))?;

        // Ant state.
        let ant_json = serde_json::to_string_pretty(&self.ant_state)
            .map_err(|e| ErmError::InvalidConfig(format!("ant_state serialize: {e}")))?;
        std::fs::write(format!("{dir}/ant_state.json"), ant_json)
            .map_err(|e| ErmError::InvalidConfig(format!("write ant_state.json: {e}")))?;

        // Config.
        let cfg_json = serde_json::to_string_pretty(&self.config)
            .map_err(|e| ErmError::InvalidConfig(format!("config serialize: {e}")))?;
        std::fs::write(format!("{dir}/config.json"), cfg_json)
            .map_err(|e| ErmError::InvalidConfig(format!("write config.json: {e}")))?;

        // Step counter.
        std::fs::write(
            format!("{dir}/step.json"),
            serde_json::json!({"step": self.step}).to_string(),
        )
        .map_err(|e| ErmError::InvalidConfig(format!("write step.json: {e}")))?;

        eprintln!(
            "[diffusion_training] checkpoint saved: dir={dir} step={}",
            self.step
        );
        Ok(())
    }

    /// Load scorer + step counter from a checkpoint directory.
    pub fn load_checkpoint(&mut self, dir: &str) -> ErmResult<()> {
        let scorer_path = format!("{dir}/scorer");
        let record = BinFileRecorder::<FullPrecisionSettings>::default()
            .load(scorer_path.into(), &self.device)
            .map_err(|e| ErmError::InvalidConfig(format!("scorer load failed: {e}")))?;
        self.scorer = self.scorer.clone().load_record(record);

        if let Ok(step_json) = std::fs::read_to_string(format!("{dir}/step.json")) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&step_json) {
                if let Some(s) = v.get("step").and_then(|x| x.as_u64()) {
                    self.step = s as usize;
                }
            }
        }
        Ok(())
    }
}

// ── Diffusion loss ────────────────────────────────────────────────────────────

/// Compute `γ(t) * CE(x | z_t)` on GPU.
///
/// `logits`: `[B, L, V]`, `targets`: `[B, L]` int tensor of clean ids.
/// Returns a scalar `[1]` tensor.
fn diffusion_ce_loss<B: AutodiffBackend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
    gamma: f32,
    b: usize,
    l: usize,
    v: usize,
) -> Tensor<B, 1> {
    // [B,L,V] → [B*L, V]
    let logits_flat = logits.reshape([b * l, v]);
    // [B,L] → [B*L]
    let targets_flat = targets.reshape([b * l]);
    // log-softmax for numerical stability
    let log_sm = burn::tensor::activation::log_softmax(logits_flat, 1);
    // gather log-prob at each target
    let targets_idx = targets_flat.unsqueeze_dim::<2>(1); // [B*L, 1]
    let log_prob = log_sm.gather(1, targets_idx); // [B*L, 1]
    let nll = -log_prob.reshape([b * l]).mean(); // scalar
    nll * gamma
}

// ── Leader edge insertion ─────────────────────────────────────────────────────

fn insert_leader_edges(
    graph: &mut RouteGraph,
    edge_proposals: &[erm_core::ants::EdgeProposal],
    cfg: &ErmConfig,
) -> usize {
    let mut inserted = 0;
    for ep in edge_proposals {
        if graph.add_edge(ep.batch_idx, ep.dst, ep.src, cfg.phi_init).is_ok() {
            inserted += 1;
        }
    }
    inserted
}

/// Diffusion inference: K iterative coarse-to-fine steps from a masked start.
///
/// No AR loop. Each step uses the scorer to fill the most uncertain masked
/// positions, guided by the route graph.
///
/// # Arguments
///
/// - `scorer`: trained scorer (non-autodiff backend).
/// - `graph`: route graph.
/// - `cfg`: ERM config.
/// - `k_steps`: number of refinement steps.
/// - `prompt_tokens`: optional fixed prefix (not denoised).
/// - `seq_len`: total output sequence length.
/// - `rng`: RNG.
/// - `device`: burn device.
///
/// # Returns
///
/// Final token sequence of length `seq_len`.
pub fn diffusion_infer<B: burn::tensor::backend::Backend>(
    scorer: &BurnScorer<B>,
    _graph: &mut RouteGraph,
    cfg: &ErmConfig,
    k_steps: usize,
    prompt_tokens: Option<&[u32]>,
    seq_len: usize,
    rng: &mut ChaCha8Rng,
    device: &B::Device,
) -> ErmResult<Vec<u32>> {
    let mask_id = cfg.mask_token_id() as u32;
    let prompt_len = prompt_tokens.map(|p| p.len()).unwrap_or(0);
    let vocab_size = cfg.total_vocab_size();

    // Initialise: prefix + masks for the rest.
    let mut y: Vec<u32> = Vec::with_capacity(seq_len);
    if let Some(prefix) = prompt_tokens {
        y.extend_from_slice(prefix);
    }
    while y.len() < seq_len {
        y.push(mask_id);
    }

    let b = 1usize;
    let l = seq_len;

    for step_idx in 0..k_steps {
        // Map step to noise level: first step = heaviest, last = lightest.
        let t_frac = (k_steps - step_idx) as f32 / k_steps as f32;
        let edit_scale = t_frac.max(0.05);

        // How many positions to fill this round.
        let to_fill =
            ((cfg.max_edits() as f32 * edit_scale).ceil() as usize).max(1);

        // Count remaining masks.
        let masked: Vec<usize> = y[prompt_len..]
            .iter()
            .enumerate()
            .filter(|(_, &t)| t == mask_id)
            .map(|(i, _)| i + prompt_len)
            .collect();
        if masked.is_empty() {
            break;
        }

        // Forward pass.
        let y_tensor = tokens_to_tensor::<B>(&y, b, l, device)?;
        let (logits_tensor, _unc, _hidden) = scorer.forward_with_hidden(y_tensor);
        let logits_cpu = tensor_to_vec(logits_tensor)?;

        // Fill top `to_fill` masked positions greedily.
        let mut candidates: Vec<(usize, u32, f32)> = masked
            .iter()
            .map(|&pos| {
                let pos_logits = &logits_cpu[pos * vocab_size..(pos + 1) * vocab_size];
                let best_tok = pos_logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, &v)| (i, v))
                    .unwrap_or((0, 0.0));
                (pos, best_tok.0 as u32, best_tok.1)
            })
            .collect();

        // Sort by confidence descending, fill top `to_fill`.
        candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        for (pos, tok, _) in candidates.iter().take(to_fill) {
            y[*pos] = *tok;
        }

        let _ = rng; // rng available for future stochastic sampling
    }

    // Fill any remaining masks with greedy decode.
    let y_tensor = tokens_to_tensor::<B>(&y, b, l, device)?;
    let (logits_tensor, _, _) = scorer.forward_with_hidden(y_tensor);
    let logits_cpu = tensor_to_vec(logits_tensor)?;
    for pos in prompt_len..l {
        if y[pos] == mask_id {
            let pos_logits = &logits_cpu[pos * vocab_size..(pos + 1) * vocab_size];
            let best = pos_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            y[pos] = best as u32;
        }
    }

    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use erm_core::config::ErmConfig;

    type TestBackend = burn_autodiff::Autodiff<NdArray<f32>>;

    fn small_cfg() -> ErmConfig {
        ErmConfig {
            vocab_size: 32,
            seq_len: 8,
            hidden_dim: 16,
            num_blocks: 2,
            num_heads: 2,
            mlp_expansion: 2,
            batch_size: 1,
            num_ants: 8,
            topk: 4,
            diffusion_steps: 2,
            noise_schedule: "linear".to_string(),
            gamma_min: 0.5,
            gamma_max: 2.0,
            ..ErmConfig::default()
        }
    }

    #[test]
    fn test_diffusion_step_runs() {
        use rand::SeedableRng;
        let cfg = small_cfg();
        let device: <TestBackend as Backend>::Device = Default::default();
        let mut trainer = DiffusionTrainer::<TestBackend>::new(&cfg, device);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let batch = TokenBatch {
            tokens: vec![5u32; 8],
            batch_size: 1,
            seq_len: 8,
        };

        let result = trainer.diffusion_step(&batch, &mut rng);
        assert!(
            result.is_ok(),
            "diffusion step should succeed: {:?}",
            result.err()
        );
        let r = result.unwrap();
        assert!(r.loss.is_finite(), "loss must be finite");
        assert!(r.loss >= 0.0, "loss must be non-negative");
    }

    #[test]
    fn test_gamma_linear() {
        let cfg = ErmConfig {
            diffusion_steps: 4,
            noise_schedule: "linear".to_string(),
            gamma_min: 0.5,
            gamma_max: 2.0,
            ..ErmConfig::default()
        };
        assert!((cfg.gamma(1) - 0.5).abs() < 1e-4);
        assert!((cfg.gamma(4) - 2.0).abs() < 1e-4);
        for t in 2..=4 {
            assert!(cfg.gamma(t) >= cfg.gamma(t - 1));
        }
    }

    #[test]
    fn test_gamma_cosine() {
        let cfg = ErmConfig {
            diffusion_steps: 6,
            noise_schedule: "cosine".to_string(),
            gamma_min: 0.5,
            gamma_max: 2.0,
            ..ErmConfig::default()
        };
        let g1 = cfg.gamma(1);
        let gt = cfg.gamma(6);
        assert!((g1 - 0.5).abs() < 1e-4, "gamma(1)={g1}");
        assert!((gt - 2.0).abs() < 1e-4, "gamma(T)={gt}");
    }

    #[test]
    fn test_diffusion_infer() {
        use burn_ndarray::NdArray;
        use rand::SeedableRng;

        type InfBackend = NdArray<f32>;
        let cfg = ErmConfig {
            vocab_size: 32,
            seq_len: 8,
            hidden_dim: 16,
            num_blocks: 2,
            num_heads: 2,
            mlp_expansion: 2,
            diffusion_steps: 2,
            ..ErmConfig::default()
        };
        let scorer_cfg = BurnScorerConfig::from_erm(&cfg);
        let device: <InfBackend as Backend>::Device = Default::default();
        let scorer = scorer_cfg.init::<InfBackend>(&device);
        let mut graph = RouteGraph::new(&cfg);
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        let result = diffusion_infer(&scorer, &mut graph, &cfg, 4, None, 8, &mut rng, &device);
        assert!(result.is_ok(), "infer failed: {:?}", result.err());
        assert_eq!(result.unwrap().len(), 8);
    }
}
