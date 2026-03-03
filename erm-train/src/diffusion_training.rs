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
    apply_death_respawn, follower_temperature_schedule, leader_temperature_schedule, AntColony,
    AntState, DeathMode, FollowerConfig, LeaderConfig,
};
use erm_core::burn_scorer::{BurnScorer, BurnScorerConfig};
use erm_core::config::{ErmConfig, PheromoneConfig};
use erm_core::corruption::corrupt;
use erm_core::error::{ErmError, ErmResult};
use erm_core::graph::RouteGraph;
use erm_core::merge::{compute_ant_deltas, compute_position_deltas, merge_proposals};
use erm_core::pheromone::{
    build_edge_traces, pheromone_rescale, prune_edges, update_pheromones_with_position_credit,
    PheromoneStats, RunningDeltaStats,
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
    /// Current learning rate.
    pub lr: f64,
    /// Current follower temperature.
    pub follower_temp: f32,
    /// Current leader temperature.
    pub leader_temp: f32,
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
    /// Total planned training steps (for temperature schedule). 0 = use fixed temps.
    pub total_steps: usize,
}

impl<B: AutodiffBackend> DiffusionTrainer<B> {
    /// Create a new diffusion trainer.
    ///
    /// Syncs `refinement_steps` to `diffusion_steps` so the corruption schedule
    /// validates correctly against the diffusion step count.
    pub fn new(config: &ErmConfig, device: B::Device) -> Self {
        // Ensure refinement_steps = diffusion_steps so corrupt() validates correctly.
        let mut config = config.clone();
        config.refinement_steps = config.diffusion_steps;

        let scorer_cfg = BurnScorerConfig::from_erm(&config);
        let scorer = scorer_cfg.init::<B>(&device);

        let optimizer = AdamConfig::new()
            .with_weight_decay(Some(WeightDecayConfig::new(config.weight_decay as f32)))
            .init();

        let graph = RouteGraph::new(&config);
        let ant_state = AntState::new(&config);
        let pheromone_config = PheromoneConfig::from_config(&config);

        let lr = config.learning_rate;
        Self {
            scorer,
            optimizer,
            graph,
            ant_state,
            config,
            pheromone_config,
            delta_stats: RunningDeltaStats::new(),
            step: 0,
            lr,
            device,
            total_steps: 0,
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
        // vocab_size from config — validated against actual tensor shapes below.
        let vocab_size = cfg.total_vocab_size();

        // Validate batch token count matches declared B*L.
        if batch.tokens.len() != b * l {
            return Err(ErmError::ShapeMismatch {
                expected: format!("batch.tokens.len() == B*L = {}*{} = {}", b, l, b * l),
                got: format!("{}", batch.tokens.len()),
            });
        }

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
        let mut last_follower_temp: f32 = 1.0;
        let mut last_leader_temp: f32 = 1.0;

        // Iterate from heavy noise (t=big_t) down to fine (t=1).
        for t in (1..=big_t).rev() {
            // 1. Corrupt clean tokens at noise level t → z_t.
            let corruption = corrupt(&x_i32, t, cfg, rng)?;
            let z_t_u32: Vec<u32> = corruption.y_t.iter().map(|&v| v as u32).collect();

            // Build corruption mask: 1.0 for positions changed by corrupt(), 0.0 for clean.
            let mask_vec: Vec<f32> = corruption.y_t.iter()
                .zip(x_i32.iter())
                .map(|(y, x)| if y != x { 1.0f32 } else { 0.0f32 })
                .collect();
            let num_corrupted: usize = mask_vec.iter().filter(|&&v| v > 0.5).count();
            let mask_tensor = {
                let mask_data = TensorData::new(mask_vec, [b * l]);
                Tensor::<B, 1>::from_data(mask_data, &self.device)
            };

            // 2. forward_with_hidden(z_t) → logits [B,L,V], hidden [B,L,d].
            let z_t_tensor = tokens_to_tensor::<B>(&z_t_u32, b, l, &self.device)?;
            let (logits_tensor, unc_tensor, hidden_tensor) =
                self.scorer.forward_with_hidden(z_t_tensor);

            // 3. Diffusion loss: γ(t) * CE(x | z_t), masked to corrupted positions only.
            let x_tensor = tokens_to_tensor::<B>(&batch.tokens, b, l, &self.device)?;
            let gamma = cfg.gamma(t);
            let step_loss = diffusion_ce_loss::<B>(
                logits_tensor.clone(),
                x_tensor,
                mask_tensor,
                gamma,
                num_corrupted,
            );
            accumulated_loss = Some(match accumulated_loss {
                None => step_loss,
                Some(acc) => acc + step_loss,
            });

            // 4. CPU colony step.
            // Transfer what we need to CPU and derive shapes from actual tensors.
            let logits_cpu = tensor_to_vec(logits_tensor)?;
            let uncertainty_cpu = tensor2d_to_vec(unc_tensor)?;
            let hidden_cpu = tensor_to_vec(hidden_tensor)?;

            // Derive V from actual logits buffer: logits_cpu.len() == B*L*V.
            let bl = b * l;
            if bl == 0 || logits_cpu.len() % bl != 0 {
                return Err(ErmError::ShapeMismatch {
                    expected: format!("logits.len() divisible by B*L={}*{}={}", b, l, bl),
                    got: format!("{}", logits_cpu.len()),
                });
            }
            let actual_v = logits_cpu.len() / bl;
            if actual_v != vocab_size {
                eprintln!(
                    "[diffusion_step] WARNING: runtime V={actual_v} != config V={vocab_size}; using runtime V"
                );
            }
            let v_rt = actual_v; // runtime vocab size for slicing

            // Derive d from actual hidden buffer: hidden_cpu.len() == B*L*d.
            if hidden_cpu.len() % bl != 0 {
                return Err(ErmError::ShapeMismatch {
                    expected: format!("hidden.len() divisible by B*L={}*{}={}", b, l, bl),
                    got: format!("{}", hidden_cpu.len()),
                });
            }
            let d = hidden_cpu.len() / bl;

            // Scale edit budget: coarse (more edits) at high t, fine (fewer) at low t.
            let edit_scale = t as f32 / big_t as f32;
            let effective_max_edits =
                ((cfg.max_edits() as f32 * edit_scale.max(0.1)).ceil() as usize).max(1);

            // Route aggregation using hidden states.
            let (_, edge_weights) = self.graph.route_aggregate(
                &hidden_cpu,
                d,
                cfg.route_epsilon,
                cfg.route_lambda,
                cfg.route_mu,
            )?;

            let num_followers = cfg.num_followers();
            let first_follower_id = cfg.num_leaders();
            let num_leaders = cfg.num_leaders();

            // Temperature scheduling: decay follower temp over training,
            // scale leader temp with uncertainty.
            let follower_cfg = if self.total_steps > 0 {
                let ft = follower_temperature_schedule(self.step, self.total_steps);
                last_follower_temp = ft;
                FollowerConfig::from_config(cfg).with_temperature(ft)
            } else {
                FollowerConfig::from_config(cfg)
            };
            // Compute mean uncertainty for leader temperature.
            let mean_unc: f32 = if !uncertainty_cpu.is_empty() {
                uncertainty_cpu.iter().sum::<f32>() / uncertainty_cpu.len() as f32
            } else {
                0.5
            };
            let leader_cfg = if self.total_steps > 0 {
                let lt = leader_temperature_schedule(mean_unc);
                last_leader_temp = lt;
                LeaderConfig::from_config(cfg).with_temperature(lt)
            } else {
                LeaderConfig::from_config(cfg)
            };

            let mut y_new_batch = Vec::with_capacity(b * l);
            let mut all_proposals = Vec::new();
            let mut all_edge_proposals = Vec::new();

            for batch_idx in 0..b {
                let logit_start = batch_idx * l * v_rt;
                let logit_end = logit_start + l * v_rt;
                if logit_end > logits_cpu.len() {
                    return Err(ErmError::ShapeMismatch {
                        expected: format!("logits[{logit_start}..{logit_end}]"),
                        got: format!("logits.len()={}", logits_cpu.len()),
                    });
                }
                let seq_logits = &logits_cpu[logit_start..logit_end];
                let unc_start = batch_idx * l;
                let unc_end = unc_start + l;
                if unc_end > uncertainty_cpu.len() {
                    return Err(ErmError::ShapeMismatch {
                        expected: format!("uncertainty[{unc_start}..{unc_end}]"),
                        got: format!("uncertainty.len()={}", uncertainty_cpu.len()),
                    });
                }
                let seq_uncertainty = &uncertainty_cpu[unc_start..unc_end];
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
                    v_rt,
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
                    v_rt,
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

            // Derive v_rt_new from actual logits_new tensor.
            let actual_v_new = if bl > 0 { logits_new_cpu.len() / bl } else { v_rt };

            let mut ant_deltas = vec![0.0_f32; cfg.num_ants];
            let mut position_deltas = vec![0.0_f32; b * l];
            for batch_idx in 0..b {
                let lo_start = batch_idx * l * v_rt;
                let lo_end = (batch_idx + 1) * l * v_rt;
                let ln_start = batch_idx * l * actual_v_new;
                let ln_end = (batch_idx + 1) * l * actual_v_new;
                if lo_end > logits_cpu.len() || ln_end > logits_new_cpu.len() {
                    return Err(ErmError::ShapeMismatch {
                        expected: format!("logits slice [{lo_start}..{lo_end}] and new [{ln_start}..{ln_end}]"),
                        got: format!("logits.len()={} new.len()={}", logits_cpu.len(), logits_new_cpu.len()),
                    });
                }
                let logits_b = &logits_cpu[lo_start..lo_end];
                let logits_new_b = &logits_new_cpu[ln_start..ln_end];
                let z_t_b = &z_t_u32[batch_idx * l..(batch_idx + 1) * l];
                let y_new_b = &y_new_batch[batch_idx * l..(batch_idx + 1) * l];

                let batch_deltas = compute_ant_deltas(
                    &all_proposals,
                    z_t_b,
                    y_new_b,
                    logits_b,
                    logits_new_b,
                    v_rt,
                    cfg.num_ants,
                )?;
                for (acc, &d) in ant_deltas.iter_mut().zip(batch_deltas.iter()) {
                    *acc += d;
                }

                // Per-position deltas for this batch element.
                let pos_deltas_b = compute_position_deltas(
                    z_t_b,
                    y_new_b,
                    logits_b,
                    logits_new_b,
                    v_rt,
                )?;
                let pd_start = batch_idx * l;
                position_deltas[pd_start..pd_start + l].copy_from_slice(&pos_deltas_b);
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
            let pstats = update_pheromones_with_position_credit(
                &mut self.graph,
                &traces,
                &ant_deltas,
                &self.pheromone_config,
                Some(&mut self.delta_stats),
                &hidden_cpu,
                d,
                &position_deltas,
            )?;

            // Pheromone rescaling (MuonClip analog): prevent φ from growing
            // large enough to collapse softmax into hard argmax.
            // Threshold = 80% of phi_max to give headroom.
            let rescale_threshold = self.pheromone_config.phi_max * 0.8;
            pheromone_rescale(&mut self.graph, rescale_threshold);

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
            lr: self.lr,
            follower_temp: last_follower_temp,
            leader_temp: last_leader_temp,
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

/// Compute `γ(t) * CE(x | z_t)` on GPU, averaged only over corrupted positions.
///
/// `logits`: `[B, L, V]`, `targets`: `[B, L]` int tensor of clean ids.
/// `mask`: flat `[B*L]` float tensor — 1.0 at corrupted positions, 0.0 elsewhere.
/// `num_corrupted`: pre-computed count of 1.0 entries in `mask` (avoids tensor→scalar round-trip).
/// Returns a scalar `[1]` tensor.
fn diffusion_ce_loss<B: AutodiffBackend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
    mask: Tensor<B, 1>,
    gamma: f32,
    num_corrupted: usize,
) -> Tensor<B, 1> {
    if num_corrupted == 0 {
        return Tensor::zeros([1], &logits.device());
    }
    // Derive B, L, V from the actual tensor shape — never trust caller dims.
    let [b_rt, l_rt, v_rt] = logits.dims();
    let bl = b_rt * l_rt;
    // [B,L,V] → [B*L, V]
    let logits_flat = logits.reshape([bl, v_rt]);
    // [B,L] → [B*L]
    let targets_flat = targets.reshape([bl]);
    // log-softmax for numerical stability
    let log_sm = burn::tensor::activation::log_softmax(logits_flat, 1);
    // gather log-prob at each target position
    let targets_idx = targets_flat.unsqueeze_dim::<2>(1); // [B*L, 1]
    let log_prob = log_sm.gather(1, targets_idx).reshape([bl]); // [B*L]
    // Sum NLL only at corrupted positions, normalize by count.
    let masked_nll = (-log_prob * mask).sum() * (1.0 / num_corrupted as f32);
    masked_nll * gamma
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

        // Forward pass — derive V from actual tensor shape.
        let y_tensor = tokens_to_tensor::<B>(&y, b, l, device)?;
        let (logits_tensor, _unc, _hidden) = scorer.forward_with_hidden(y_tensor);
        let logits_cpu = tensor_to_vec(logits_tensor)?;
        let v_rt = if l > 0 { logits_cpu.len() / l } else { 0 };

        // Fill top `to_fill` masked positions greedily.
        let mut candidates: Vec<(usize, u32, f32)> = masked
            .iter()
            .filter_map(|&pos| {
                let start = pos * v_rt;
                let end = start + v_rt;
                if end > logits_cpu.len() || v_rt == 0 {
                    return None;
                }
                let pos_logits = &logits_cpu[start..end];
                let best_tok = pos_logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, &v)| (i, v))
                    .unwrap_or((0, 0.0));
                Some((pos, best_tok.0 as u32, best_tok.1))
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
    let v_rt = if l > 0 { logits_cpu.len() / l } else { 0 };
    for pos in prompt_len..l {
        if y[pos] == mask_id {
            let start = pos * v_rt;
            let end = start + v_rt;
            if end > logits_cpu.len() || v_rt == 0 {
                continue;
            }
            let pos_logits = &logits_cpu[start..end];
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
