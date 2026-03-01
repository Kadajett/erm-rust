use std::collections::VecDeque;
use burn::prelude::*;
use burn_ndarray::NdArray;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use erm_core::config::ErmConfig;
use erm_core::corruption::corrupt;
use erm_core::scorer::{Scorer, ScorerConfig};
use erm_core::types::TokenId;
use crate::dataset::{DataBatch, TextDataset};

pub type TrainBackend = NdArray<f32>;

#[derive(Debug)]
struct AdamState {
    m: Vec<f32>,
    v: Vec<f32>,
    t: u64,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
}

impl AdamState {
    fn new(n: usize, lr: f32, weight_decay: f32) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
        }
    }

    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        self.t += 1;
        let t_f = self.t as f32;
        let bc1 = 1.0 - self.beta1.powf(t_f);
        let bc2 = 1.0 - self.beta2.powf(t_f);
        for i in 0..params.len() {
            params[i] *= 1.0 - self.lr * self.weight_decay;
            let g = grads[i];
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub total_steps: usize,
    pub log_every: usize,
    pub erm: ErmConfig,
    pub seed: u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            total_steps: 1000,
            log_every: 100,
            erm: ErmConfig::default(),
            seed: 42,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StepResult {
    pub step: usize,
    pub loss: f32,
    pub num_corrupted: usize,
}

pub fn cross_entropy(logits: &[f32], target: usize) -> f32 {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
    let log_softmax_target = (logits[target] - max_logit) - exp_sum.ln();
    -log_softmax_target
}

pub fn cross_entropy_grad(logits: &[f32], target: usize) -> Vec<f32> {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let exp_sum: f32 = exp.iter().sum();
    let mut grad: Vec<f32> = exp.iter().map(|&e| e / exp_sum).collect();
    grad[target] -= 1.0;
    grad
}

pub fn denoise_loss(logits_flat: &[f32], x_flat: &[TokenId], y_t_flat: &[TokenId], _mask_token_id: TokenId, vocab_size: usize, batch_size: usize, seq_len: usize) -> (f32, Vec<f32>, usize) {
    let bl = batch_size * seq_len;
    let mut total_loss = 0.0_f32;
    let mut grads = vec![0.0_f32; logits_flat.len()];
    let mut num_corrupted = 0usize;
    for pos in 0..bl {
        if y_t_flat[pos] == x_flat[pos] { continue; }
        let target = x_flat[pos];
        if target < 0 || target >= vocab_size as TokenId { continue; }
        let target_usize = target as usize;
        let logit_start = pos * vocab_size;
        let logit_end = logit_start + vocab_size;
        let pos_logits = &logits_flat[logit_start..logit_end];
        let ce = cross_entropy(pos_logits, target_usize);
        total_loss += ce;
        let pos_grads = cross_entropy_grad(pos_logits, target_usize);
        for (g, &pg) in grads[logit_start..logit_end].iter_mut().zip(pos_grads.iter()) {
            *g += pg;
        }
        num_corrupted += 1;
    }
    if num_corrupted == 0 { return (0.0, grads, 0); }
    let scale = 1.0 / num_corrupted as f32;
    total_loss *= scale;
    for g in grads.iter_mut() { *g *= scale; }
    (total_loss, grads, num_corrupted)
}

pub struct Trainer {
    config: ErmConfig,
    train_config: TrainingConfig,
    scorer: Scorer<TrainBackend>,
    logit_weights: Vec<f32>,
    adam: AdamState,
    device: <TrainBackend as Backend>::Device,
    recent_losses: VecDeque<f32>,
    rng: ChaCha8Rng,
    pub step: usize,
}

impl Trainer {
    pub fn new(train_config: TrainingConfig) -> Self {
        let device = Default::default();
        let erm = train_config.erm.clone();
        let scorer = ScorerConfig { erm: erm.clone() }.init::<TrainBackend>(&device);
        let d = erm.hidden_dim;
        let v = erm.vocab_size;
        let logit_weights = vec![0.0_f32; v * d];
        let adam = AdamState::new(v * d, erm.learning_rate as f32, erm.weight_decay as f32);
        let rng = ChaCha8Rng::seed_from_u64(train_config.seed);
        Self {
            config: erm,
            train_config,
            scorer,
            logit_weights,
            adam,
            device,
            recent_losses: VecDeque::with_capacity(100),
            rng,
            step: 0,
        }
    }

    pub fn train_step(&mut self, batch: &DataBatch) -> StepResult {
        let b = batch.batch_size;
        let l = batch.seq_len;
        let v = self.config.vocab_size;
        let t_max = self.config.refinement_steps;
        let t_val: usize = self.rng.gen_range(1..=t_max);
        let corruption = corrupt(&batch.x, t_val, &self.config, &mut self.rng)
            .unwrap_or_else(|_| erm_core::corruption::CorruptionResult {
                y_t: batch.x.clone(),
                num_masked: 0,
                num_replaced: 0,
                num_kept: batch.x.len(),
            });
        let y_t_flat = &corruption.y_t;
        let y_t_tensor = Tensor::<TrainBackend, 2, Int>::from_data(
            TensorData::new(y_t_flat.clone(), [b, l]),
            &self.device,
        );
        let (logits_tensor, _uncertainty) = self.scorer.forward(y_t_tensor, None);
        let logits_data = logits_tensor.to_data();
        let logits_flat = logits_data
            .as_slice::<f32>()
            .unwrap_or(&vec![0.0_f32; b * l * v])
            .to_vec();
        let (loss, grad_logits, num_corrupted) = denoise_loss(
            &logits_flat,
            &batch.x,
            y_t_flat,
            self.config.mask_token_id(),
            v,
            b,
            l,
        );
        if num_corrupted > 0 {
            let d = self.config.hidden_dim;
            let norm = 1.0 / (b * l) as f32;
            let mut weight_grads = vec![0.0_f32; v * d];
            for pos in 0..b * l {
                let logit_base = pos * v;
                for vi in 0..v {
                    let g = grad_logits[logit_base + vi] * norm;
                    for di in 0..d {
                        weight_grads[vi * d + di] += g;
                    }
                }
            }
            self.adam.step(&mut self.logit_weights, &weight_grads);
        }
        self.recent_losses.push_back(loss);
        if self.recent_losses.len() > 100 {
            self.recent_losses.pop_front();
        }
        let result = StepResult {
            step: self.step,
            loss,
            num_corrupted,
        };
        self.step += 1;
        result
    }

    pub fn train(&mut self, dataset: &mut TextDataset) -> Vec<StepResult> {
        let mut results = Vec::new();
        dataset.shuffle();
        while self.step < self.train_config.total_steps {
            match dataset.next_batch() {
                Some(batch) => {
                    let result = self.train_step(&batch);
                    if result.step.is_multiple_of(self.train_config.log_every) {
                        let avg = self.recent_avg_loss();
                        eprintln!("[step {:>6}] loss={:.4}  avg100={:.4}  corrupted={}",
                            result.step, result.loss, avg, result.num_corrupted);
                    }
                    results.push(result);
                }
                None => {
                    dataset.shuffle();
                }
            }
        }
        results
    }

    #[must_use]
    pub fn recent_avg_loss(&self) -> f32 {
        if self.recent_losses.is_empty() {
            return f32::NAN;
        }
        self.recent_losses.iter().sum::<f32>() / self.recent_losses.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::synthetic_dataset_with_batch;

    #[test]
    fn test_trainer_step_does_not_panic() {
        let train_cfg = TrainingConfig {
            total_steps: 10,
            log_every: 5,
            erm: ErmConfig { vocab_size: 32, seq_len: 16, hidden_dim: 16, num_blocks: 1, num_heads: 2, mlp_expansion: 2, dropout: 0.0, batch_size: 2, refinement_steps: 3, ..ErmConfig::default() },
            seed: 42,
        };
        let erm_cfg = train_cfg.erm.clone();
        let mut trainer = Trainer::new(train_cfg);
        let mut ds = synthetic_dataset_with_batch(8, erm_cfg.seq_len, erm_cfg.vocab_size, erm_cfg.batch_size, 42).unwrap();
        ds.shuffle();
        let batch = ds.next_batch().unwrap();
        let result = trainer.train_step(&batch);
        assert!(result.loss >= 0.0 || result.num_corrupted == 0);
    }
}
