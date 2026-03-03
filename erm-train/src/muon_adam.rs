//! MuonAdam optimizer: Newton-Schulz for eligible 2D weights, SGD+Nesterov for everything else.
//!
//! Combines Muon (orthogonalized momentum for hidden-layer weight matrices) with
//! SGD+Nesterov momentum (for biases, embeddings, output heads). Plugs directly
//! into Burn's `OptimizerAdaptor` with a single `SimpleOptimizer` implementation.
//!
//! # Parameter classification
//!
//! - **Muon path** (D=2, min_dim >= 16, aspect_ratio <= 8): Newton-Schulz orthogonalized
//!   momentum. Preserves gradient matrix structure, amplifies suppressed low-rank directions.
//! - **SGD path** (everything else): classical momentum with optional Nesterov.
//!
//! # Memory
//!
//! 1× parameters (single velocity buffer per param), vs AdamW's 2× (first + second moments).

use burn::module::AutodiffModule;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::grad_clipping::GradientClippingConfig;
use burn::optim::SimpleOptimizer;
use burn::prelude::*;
use burn::record::Record;
use burn::tensor::backend::AutodiffBackend;

use erm_core::config::ErmConfig;

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for the MuonAdam optimizer.
#[derive(Debug, Clone)]
pub struct MuonAdamConfig {
    /// Newton-Schulz coefficient a (default: 3.4445).
    pub ns_a: f32,
    /// Newton-Schulz coefficient b (default: -4.7750).
    pub ns_b: f32,
    /// Newton-Schulz coefficient c (default: 2.0315).
    pub ns_c: f32,
    /// Newton-Schulz iterations (default: 5).
    pub ns_steps: usize,
    /// Small constant for numerical stability (default: 1e-7).
    pub epsilon: f32,
    /// Momentum coefficient (default: 0.95).
    pub beta: f32,
    /// Momentum dampening (default: 0.0).
    pub dampening: f32,
    /// Use Nesterov momentum (default: true).
    pub nesterov: bool,
    /// Decoupled weight decay coefficient (default: 0.01).
    pub weight_decay: f32,
    /// Minimum dimension for Muon eligibility (default: 16).
    pub min_muon_dim: usize,
    /// Maximum aspect ratio for Muon eligibility (default: 8).
    pub max_aspect_ratio: usize,
    /// Gradient clipping norm (0 = disabled). Handled by OptimizerAdaptor.
    pub grad_clip_norm: f32,
}

impl Default for MuonAdamConfig {
    fn default() -> Self {
        Self {
            ns_a: 3.4445,
            ns_b: -4.7750,
            ns_c: 2.0315,
            ns_steps: 5,
            epsilon: 1e-7,
            beta: 0.95,
            dampening: 0.0,
            nesterov: true,
            weight_decay: 0.01,
            min_muon_dim: 16,
            max_aspect_ratio: 8,
            grad_clip_norm: 1.0,
        }
    }
}

impl MuonAdamConfig {
    /// Build from ERM configuration.
    pub fn from_erm(config: &ErmConfig) -> Self {
        Self {
            weight_decay: config.weight_decay as f32,
            grad_clip_norm: config.grad_clip_norm as f32,
            ..Self::default()
        }
    }

    /// Create the optimizer wrapped in Burn's `OptimizerAdaptor`.
    pub fn init<B: AutodiffBackend, M: AutodiffModule<B>>(
        &self,
    ) -> OptimizerAdaptor<MuonAdam, M, B> {
        let optim = MuonAdam {
            ns_a: self.ns_a,
            ns_b: self.ns_b,
            ns_c: self.ns_c,
            ns_steps: self.ns_steps,
            epsilon: self.epsilon,
            beta: self.beta,
            dampening: self.dampening,
            nesterov: self.nesterov,
            weight_decay: self.weight_decay,
            min_muon_dim: self.min_muon_dim,
            max_aspect_ratio: self.max_aspect_ratio,
        };
        let adaptor = OptimizerAdaptor::from(optim);
        if self.grad_clip_norm > 0.0 {
            adaptor.with_grad_clipping(
                GradientClippingConfig::Norm(self.grad_clip_norm).init(),
            )
        } else {
            adaptor
        }
    }
}

// ── Optimizer ─────────────────────────────────────────────────────────────────

/// Combined Muon+SGD optimizer.
///
/// For each parameter tensor:
/// - 2D weight matrices with min_dim >= 16 and aspect_ratio <= 8: Newton-Schulz
///   orthogonalized momentum (Muon path). Preserves matrix structure, amplifies
///   suppressed low-rank gradient directions.
/// - Everything else (biases, embeddings, output heads): SGD with optional Nesterov
///   momentum.
#[derive(Debug, Clone)]
pub struct MuonAdam {
    ns_a: f32,
    ns_b: f32,
    ns_c: f32,
    ns_steps: usize,
    epsilon: f32,
    beta: f32,
    dampening: f32,
    nesterov: bool,
    weight_decay: f32,
    min_muon_dim: usize,
    max_aspect_ratio: usize,
}

// ── State ─────────────────────────────────────────────────────────────────────

/// Per-parameter optimizer state: single momentum buffer (1× params memory).
#[derive(Record, Clone)]
pub struct MuonAdamState<B: Backend, const D: usize> {
    pub velocity: Tensor<B, D>,
}

// ── SimpleOptimizer impl ──────────────────────────────────────────────────────

impl<B: Backend> SimpleOptimizer<B> for MuonAdam {
    type State<const D: usize> = MuonAdamState<B, D>;

    fn step<const D: usize>(
        &self,
        lr: burn::optim::LearningRate,
        tensor: Tensor<B, D>,
        grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        // 1. Momentum: v = beta * v_prev + (1 - dampening) * grad
        //    First step: v = grad (matching PyTorch SGD convention)
        let velocity = if let Some(ref st) = state {
            st.velocity
                .clone()
                .mul_scalar(self.beta)
                .add(grad.clone().mul_scalar(1.0 - self.dampening))
        } else {
            grad.clone()
        };

        // 2. Compute update direction (with optional Nesterov lookahead)
        let update_dir = if self.nesterov {
            grad.add(velocity.clone().mul_scalar(self.beta))
        } else {
            velocity.clone()
        };

        // 3. Check Muon eligibility: D == 2, adequate size, reasonable aspect ratio
        let (update, adjusted_lr) = if D == 2 {
            // Safe: D == 2 so shape has exactly 2 elements.
            // Use slice coercion to avoid const-array bounds issues in dead-code
            // monomorphizations (D=1).
            let shape = tensor.shape();
            let dims: &[usize] = &shape.dims;
            let dim0 = dims[0];
            let dim1 = dims[1];
            let min_dim = dim0.min(dim1);
            let max_dim = dim0.max(dim1);
            let aspect = if min_dim > 0 { max_dim / min_dim } else { usize::MAX };

            if min_dim >= self.min_muon_dim && aspect <= self.max_aspect_ratio {
                // Muon path: Newton-Schulz orthogonalization
                let orth = self.newton_schulz(update_dir);
                // Adjust LR: scale by sqrt(max(1, rows/cols))
                let ratio = (dim0 as f64 / dim1.max(1) as f64).max(1.0).sqrt();
                (orth, lr * ratio)
            } else {
                // SGD path for ineligible 2D (embeddings, output heads)
                (update_dir, lr)
            }
        } else {
            // SGD path for non-2D (biases)
            (update_dir, lr)
        };

        // 4. Decoupled weight decay: w *= (1 - lr * wd)
        let tensor = if self.weight_decay > 0.0 {
            let decay = 1.0 - lr * self.weight_decay as f64;
            tensor.mul_scalar(decay)
        } else {
            tensor
        };

        // 5. Parameter update: w -= adjusted_lr * update
        let delta = update.mul_scalar(adjusted_lr);
        let new_state = MuonAdamState { velocity };

        (tensor - delta, Some(new_state))
    }

    fn to_device<const D: usize>(
        state: Self::State<D>,
        device: &B::Device,
    ) -> Self::State<D> {
        MuonAdamState {
            velocity: state.velocity.to_device(device),
        }
    }
}

// ── Newton-Schulz ─────────────────────────────────────────────────────────────

impl MuonAdam {
    /// Newton-Schulz orthogonalization via quintic iteration.
    ///
    /// Approximates the zero-power (sign function) of a matrix, producing a
    /// near-orthogonal result that preserves gradient direction structure.
    ///
    /// Algorithm:
    /// 1. If tall (rows > cols): transpose for better convergence
    /// 2. Normalize by Frobenius norm
    /// 3. 5× quintic iteration: A = X@Xᵀ, B = b*A + c*A², X = a*X + B@X
    /// 4. Restore original orientation
    ///
    /// Only called when D == 2; uses slice indexing to compile safely for all D.
    pub(crate) fn newton_schulz<B: Backend, const D: usize>(
        &self,
        g: Tensor<B, D>,
    ) -> Tensor<B, D> {
        let shape = g.shape();
        let dims: &[usize] = &shape.dims;
        let dim0 = dims[0];
        let dim1 = dims[1];

        // Transpose tall matrices (rows > cols) for better NS convergence
        let (mut x, transposed) = if dim0 > dim1 {
            (g.swap_dims(0, 1), true)
        } else {
            (g, false)
        };

        // Normalize by Frobenius norm
        let norm = x
            .clone()
            .powf_scalar(2.0)
            .sum()
            .sqrt()
            .clamp_min(self.epsilon)
            .unsqueeze();
        x = x.div(norm);

        // Quintic Newton-Schulz iterations:
        //   A = X @ Xᵀ
        //   B = b*A + c*A²
        //   X = a*X + B @ X
        for _ in 0..self.ns_steps {
            let x_t = x.clone().swap_dims(0, 1);
            let a = x.clone().matmul(x_t);
            let a_sq = a.clone().matmul(a.clone());
            let b_mat = a.mul_scalar(self.ns_b).add(a_sq.mul_scalar(self.ns_c));
            x = x.clone().mul_scalar(self.ns_a).add(b_mat.matmul(x));
        }

        // Restore original orientation
        if transposed {
            x.swap_dims(0, 1)
        } else {
            x
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn default_optim() -> MuonAdam {
        MuonAdam {
            ns_a: 3.4445,
            ns_b: -4.7750,
            ns_c: 2.0315,
            ns_steps: 5,
            epsilon: 1e-7,
            beta: 0.95,
            dampening: 0.0,
            nesterov: true,
            weight_decay: 0.01,
            min_muon_dim: 16,
            max_aspect_ratio: 8,
        }
    }

    fn scalar<const D: usize>(t: Tensor<B, D>) -> f32 {
        t.into_data().as_slice::<f32>().unwrap()[0]
    }

    #[test]
    fn test_ns_produces_near_orthogonal_output() {
        let device: <B as Backend>::Device = Default::default();
        let optim = default_optim();
        let g: Tensor<B, 2> =
            Tensor::random([32, 32], Distribution::Normal(0.0, 1.0), &device);
        let o = optim.newton_schulz(g);

        // Check OOᵀ ≈ I
        let oot = o.clone().matmul(o.clone().transpose());
        let eye: Tensor<B, 2> = Tensor::eye(32, &device);
        let diff = scalar((oot - eye).powf_scalar(2.0).sum().sqrt());
        // 5 NS iterations on a 32×32 random matrix yields diff ~2.0;
        // a non-orthogonalized random matrix would be much larger (~5+).
        assert!(
            diff < 3.5,
            "OOᵀ should be near identity, Frobenius diff = {diff}"
        );
    }

    #[test]
    fn test_ns_rectangular_matrix() {
        let device: <B as Backend>::Device = Default::default();
        let optim = default_optim();

        // Wide matrix (rows < cols)
        let g: Tensor<B, 2> =
            Tensor::random([32, 64], Distribution::Normal(0.0, 1.0), &device);
        let o = optim.newton_schulz(g);
        let oot = o.clone().matmul(o.clone().transpose());
        let eye: Tensor<B, 2> = Tensor::eye(32, &device);
        let diff = scalar((oot - eye).powf_scalar(2.0).sum().sqrt());
        assert!(diff < 3.5, "wide OOᵀ near identity, diff = {diff}");

        // Tall matrix (rows > cols) — NS transposes internally
        let g2: Tensor<B, 2> =
            Tensor::random([64, 32], Distribution::Normal(0.0, 1.0), &device);
        let o2 = optim.newton_schulz(g2);
        let o2t_o2 = o2.clone().transpose().matmul(o2.clone());
        let eye2: Tensor<B, 2> = Tensor::eye(32, &device);
        let diff2 = scalar((o2t_o2 - eye2).powf_scalar(2.0).sum().sqrt());
        assert!(diff2 < 3.5, "tall OᵀO near identity, diff = {diff2}");
    }

    #[test]
    fn test_2d_eligible_uses_muon_path() {
        let device: <B as Backend>::Device = Default::default();
        let optim = MuonAdam {
            weight_decay: 0.0,
            ..default_optim()
        };
        let tensor: Tensor<B, 2> = Tensor::ones([32, 32], &device);
        let grad: Tensor<B, 2> =
            Tensor::random([32, 32], Distribution::Normal(0.0, 1.0), &device);

        // MuonAdam step
        let (updated_muon, _) =
            optim.step(0.01, tensor.clone(), grad.clone(), None);

        // Naive SGD: tensor - lr * (1 + beta) * grad  (Nesterov first step)
        let nesterov_scale = 1.0 + optim.beta;
        let naive = tensor - grad.mul_scalar(0.01f64 * nesterov_scale as f64);

        // They should differ because Muon orthogonalizes the update
        let diff = scalar(
            (updated_muon - naive)
                .powf_scalar(2.0)
                .sum()
                .sqrt(),
        );
        assert!(
            diff > 1e-6,
            "Muon update should differ from naive SGD, diff = {diff}"
        );
    }

    #[test]
    fn test_1d_bias_no_panic() {
        let device: <B as Backend>::Device = Default::default();
        let optim = default_optim();
        let tensor: Tensor<B, 1> = Tensor::ones([64], &device);
        let grad: Tensor<B, 1> =
            Tensor::random([64], Distribution::Normal(0.0, 0.1), &device);

        let (updated, state) = optim.step(0.001, tensor, grad, None);
        assert!(state.is_some(), "state should be returned");
        let val = scalar(updated.sum());
        assert!(val.is_finite(), "1D update should be finite, got {val}");
    }

    #[test]
    fn test_high_aspect_ratio_uses_sgd_path() {
        let device: <B as Backend>::Device = Default::default();
        let optim = MuonAdam {
            weight_decay: 0.0,
            ..default_optim()
        };

        // Aspect ratio 512:1, way above max_aspect_ratio=8
        let tensor: Tensor<B, 2> = Tensor::ones([512, 1], &device);
        let grad: Tensor<B, 2> =
            Tensor::random([512, 1], Distribution::Normal(0.0, 1.0), &device);

        // Should use SGD path (no Newton-Schulz) — verify update is finite
        let (updated, state) = optim.step(0.01, tensor.clone(), grad.clone(), None);
        assert!(state.is_some());
        let val = scalar(updated.clone().sum());
        assert!(val.is_finite(), "high-aspect update should be finite");

        // SGD path: tensor - lr * nesterov_update, no NS
        // Nesterov first step: update = grad + beta * grad = (1+beta)*grad
        let nesterov_scale = 1.0 + optim.beta;
        let expected = tensor - grad.mul_scalar(0.01f64 * nesterov_scale as f64);
        let diff = scalar(
            (updated - expected)
                .powf_scalar(2.0)
                .sum()
                .sqrt(),
        );
        assert!(
            diff < 1e-4,
            "high-aspect should use SGD path (match naive), diff = {diff}"
        );
    }

    #[test]
    fn test_small_dim_uses_sgd_path() {
        let device: <B as Backend>::Device = Default::default();
        let optim = MuonAdam {
            weight_decay: 0.0,
            ..default_optim()
        };

        // 8x8 is below min_muon_dim=16
        let tensor: Tensor<B, 2> = Tensor::ones([8, 8], &device);
        let grad: Tensor<B, 2> =
            Tensor::random([8, 8], Distribution::Normal(0.0, 1.0), &device);

        let (updated, _) = optim.step(0.01, tensor.clone(), grad.clone(), None);

        // SGD path expected
        let nesterov_scale = 1.0 + optim.beta;
        let expected = tensor - grad.mul_scalar(0.01f64 * nesterov_scale as f64);
        let diff = scalar(
            (updated - expected)
                .powf_scalar(2.0)
                .sum()
                .sqrt(),
        );
        assert!(
            diff < 1e-4,
            "small-dim should use SGD path, diff = {diff}"
        );
    }

    #[test]
    fn test_weight_decay_reduces_norm() {
        let device: <B as Backend>::Device = Default::default();
        let optim = MuonAdam {
            weight_decay: 0.1,
            ..default_optim()
        };

        let tensor: Tensor<B, 1> = Tensor::ones([64], &device).mul_scalar(10.0);
        let grad: Tensor<B, 1> = Tensor::zeros([64], &device);

        let norm_before = scalar(
            tensor
                .clone()
                .powf_scalar(2.0)
                .sum()
                .sqrt(),
        );

        let (updated, _) = optim.step(0.01, tensor, grad, None);

        let norm_after = scalar(
            updated
                .powf_scalar(2.0)
                .sum()
                .sqrt(),
        );

        assert!(
            norm_after < norm_before,
            "weight decay should reduce norm: {norm_after} < {norm_before}"
        );
    }

    #[test]
    fn test_zero_gradient_no_nan() {
        let device: <B as Backend>::Device = Default::default();
        let optim = default_optim();

        // Test with 1D
        let t1: Tensor<B, 1> = Tensor::ones([32], &device);
        let g1: Tensor<B, 1> = Tensor::zeros([32], &device);
        let (u1, _) = optim.step(0.001, t1, g1, None);
        let v1 = scalar(u1.sum());
        assert!(!v1.is_nan(), "zero 1D grad should not produce NaN");

        // Test with 2D eligible
        let t2: Tensor<B, 2> = Tensor::ones([32, 32], &device);
        let g2: Tensor<B, 2> = Tensor::zeros([32, 32], &device);
        let (u2, _) = optim.step(0.001, t2, g2, None);
        let v2 = scalar(u2.sum());
        assert!(!v2.is_nan(), "zero 2D grad should not produce NaN");
    }

    #[test]
    fn test_momentum_accumulates() {
        let device: <B as Backend>::Device = Default::default();
        let optim = MuonAdam {
            nesterov: false,
            weight_decay: 0.0,
            min_muon_dim: 999, // force SGD path for all
            ..default_optim()
        };

        let tensor: Tensor<B, 1> = Tensor::ones([16], &device);
        let grad: Tensor<B, 1> = Tensor::ones([16], &device);

        // Step 1: v = grad (first step)
        let (_, state1) = optim.step(0.01, tensor.clone(), grad.clone(), None);
        let v1_norm = scalar(
            state1
                .as_ref()
                .unwrap()
                .velocity
                .clone()
                .powf_scalar(2.0)
                .sum()
                .sqrt(),
        );

        // Step 2: v = beta * v1 + grad
        let (_, state2) =
            optim.step(0.01, tensor.clone(), grad.clone(), state1);
        let v2_norm = scalar(
            state2
                .as_ref()
                .unwrap()
                .velocity
                .clone()
                .powf_scalar(2.0)
                .sum()
                .sqrt(),
        );

        assert!(
            v2_norm > v1_norm,
            "momentum should accumulate: v2={v2_norm} > v1={v1_norm}"
        );
    }

    #[test]
    fn test_config_from_erm() {
        let erm_cfg = ErmConfig {
            weight_decay: 0.05,
            grad_clip_norm: 2.0,
            ..ErmConfig::default()
        };
        let muon_cfg = MuonAdamConfig::from_erm(&erm_cfg);
        assert!((muon_cfg.weight_decay - 0.05).abs() < 1e-6);
        assert!((muon_cfg.grad_clip_norm - 2.0).abs() < 1e-6);
        assert_eq!(muon_cfg.ns_steps, 5);
        assert!(muon_cfg.nesterov);
    }
}
