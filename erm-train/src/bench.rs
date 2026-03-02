//! Benchmark harness for the Emergent Route Model.
//!
//! Measures throughput for key operations:
//! - Scorer forward pass (tokens/sec)
//! - Single refinement step (tokens/sec)
//! - Multi-step refinement (steps/sec, tokens/sec)
//! - Peak memory estimates
//!
//! # Usage
//!
//! ```text
//! erm bench --batch-size 8 --seq-len 128 --steps 6 --iters 10
//! ```

use std::time::Instant;

use erm_core::config::{ErmConfig, PheromoneConfig};
use erm_core::graph::RouteGraph;
use erm_core::refinement::{multi_step_refine, MultiStepConfig};
use erm_core::scorer::Scorer;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Result of a single benchmark measurement.
#[derive(Debug, Clone)]
pub struct BenchResult {
    /// Name of the benchmark.
    pub name: String,
    /// Total elapsed time in seconds.
    pub elapsed_secs: f64,
    /// Tokens processed.
    pub tokens: usize,
    /// Tokens per second.
    pub tokens_per_sec: f64,
    /// Refinement steps executed (if applicable).
    pub steps_executed: usize,
    /// Steps per second (if applicable).
    pub steps_per_sec: f64,
    /// Peak memory estimate in bytes.
    pub peak_memory: usize,
}

/// Run all benchmarks and print formatted results.
///
/// # Arguments
///
/// - `batch_size`: batch dimension `B`.
/// - `seq_len`: sequence length `L`.
/// - `steps`: number of refinement steps `T`.
/// - `iters`: number of iterations for each benchmark.
pub fn run_benchmarks(batch_size: usize, seq_len: usize, steps: usize, iters: usize) {
    println!("=== ERM Benchmarks ===");
    println!("Config: batch_size={batch_size}, seq_len={seq_len}, steps={steps}, iters={iters}");
    println!();

    let cfg = ErmConfig {
        batch_size,
        seq_len,
        vocab_size: 256,
        hidden_dim: 64,
        num_blocks: 2,
        num_heads: 2,
        mlp_expansion: 4,
        num_ants: 32,
        topk: 4,
        pmax: 4,
        emax: 4,
        refinement_steps: steps,
        max_edits_per_step: 0.15,
        leader_fraction: 0.10,
        dropout: 0.0,
        ..ErmConfig::default()
    };

    let results = vec![
        bench_scorer_forward(&cfg, iters),
        bench_single_refinement_step(&cfg, iters),
        bench_multi_step_refinement(&cfg, iters),
    ];

    println!("{:-<72}", "");
    println!(
        "{:<30} {:>10} {:>12} {:>12}",
        "Benchmark", "Time (s)", "Tokens/s", "Steps/s"
    );
    println!("{:-<72}", "");

    for r in &results {
        println!(
            "{:<30} {:>10.4} {:>12.0} {:>12.1}",
            r.name, r.elapsed_secs, r.tokens_per_sec, r.steps_per_sec
        );
    }

    println!("{:-<72}", "");
    println!();

    // Memory estimates.
    println!("Memory estimates:");
    for r in &results {
        println!(
            "  {}: ~{} bytes ({:.1} KB)",
            r.name,
            r.peak_memory,
            r.peak_memory as f64 / 1024.0
        );
    }
}

/// Benchmark scorer forward pass throughput.
fn bench_scorer_forward(cfg: &ErmConfig, iters: usize) -> BenchResult {
    let scorer = Scorer::new(cfg, cfg.vocab_size, 42);
    let y_t: Vec<u32> = (0..cfg.seq_len as u32).cycle().take(cfg.seq_len).collect();

    // Warm up.
    let _ = scorer.forward(&y_t, 1);

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = scorer.forward(&y_t, 1);
    }
    let elapsed = t0.elapsed().as_secs_f64();

    let total_tokens = iters * cfg.seq_len;
    let logits_mem = cfg.seq_len * cfg.vocab_size * std::mem::size_of::<f32>();
    let unc_mem = cfg.seq_len * std::mem::size_of::<f32>();

    BenchResult {
        name: "scorer_forward".to_string(),
        elapsed_secs: elapsed,
        tokens: total_tokens,
        tokens_per_sec: total_tokens as f64 / elapsed,
        steps_executed: 0,
        steps_per_sec: 0.0,
        peak_memory: logits_mem + unc_mem,
    }
}

/// Benchmark a single refinement step.
fn bench_single_refinement_step(cfg: &ErmConfig, iters: usize) -> BenchResult {
    let scorer = Scorer::new(cfg, cfg.vocab_size, 42);
    let mut graph = RouteGraph::new(cfg);
    let pconfig = PheromoneConfig::from_config(cfg);
    let editable = vec![true; cfg.seq_len];
    let y_t: Vec<u32> = (0..cfg.seq_len as u32).cycle().take(cfg.seq_len).collect();

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Warm up.
    let _ = erm_core::refinement::refine_step_with_pheromones(
        &y_t, &scorer, &mut graph, 0, cfg, &pconfig, &editable, &mut rng,
    );

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = erm_core::refinement::refine_step_with_pheromones(
            &y_t, &scorer, &mut graph, 0, cfg, &pconfig, &editable, &mut rng,
        );
    }
    let elapsed = t0.elapsed().as_secs_f64();

    let total_tokens = iters * cfg.seq_len;
    let logits_mem = cfg.seq_len * cfg.vocab_size * std::mem::size_of::<f32>() * 2;
    let graph_mem = graph.total_elements()
        * (std::mem::size_of::<i32>()
            + std::mem::size_of::<f32>() * 3
            + std::mem::size_of::<i32>());

    BenchResult {
        name: "single_refine_step".to_string(),
        elapsed_secs: elapsed,
        tokens: total_tokens,
        tokens_per_sec: total_tokens as f64 / elapsed,
        steps_executed: iters,
        steps_per_sec: iters as f64 / elapsed,
        peak_memory: logits_mem + graph_mem,
    }
}

/// Benchmark multi-step refinement (full T-step pipeline).
fn bench_multi_step_refinement(cfg: &ErmConfig, iters: usize) -> BenchResult {
    let scorer = Scorer::new(cfg, cfg.vocab_size, 42);
    let pconfig = PheromoneConfig::from_config(cfg);
    let multi_cfg = MultiStepConfig::from_config(cfg);
    let editable = vec![true; cfg.seq_len];
    let y_init: Vec<u32> = (0..cfg.seq_len as u32).cycle().take(cfg.seq_len).collect();

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut total_steps_executed = 0_usize;
    let mut peak_mem = 0_usize;

    // Warm up.
    {
        let mut graph = RouteGraph::new(cfg);
        let _ = multi_step_refine(
            &y_init, &scorer, &mut graph, 0, cfg, &pconfig, &multi_cfg, &editable, &mut rng,
        );
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        let mut graph = RouteGraph::new(cfg);
        if let Ok(result) = multi_step_refine(
            &y_init, &scorer, &mut graph, 0, cfg, &pconfig, &multi_cfg, &editable, &mut rng,
        ) {
            total_steps_executed += result.steps_executed;
            if result.peak_memory_estimate > peak_mem {
                peak_mem = result.peak_memory_estimate;
            }
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();

    let total_tokens = iters * cfg.seq_len;

    BenchResult {
        name: "multi_step_refine".to_string(),
        elapsed_secs: elapsed,
        tokens: total_tokens,
        tokens_per_sec: total_tokens as f64 / elapsed,
        steps_executed: total_steps_executed,
        steps_per_sec: total_steps_executed as f64 / elapsed,
        peak_memory: peak_mem,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bench_config() -> ErmConfig {
        ErmConfig {
            batch_size: 1,
            seq_len: 8,
            vocab_size: 16,
            hidden_dim: 8,
            num_blocks: 1,
            num_heads: 2,
            mlp_expansion: 2,
            num_ants: 8,
            topk: 4,
            pmax: 4,
            emax: 4,
            refinement_steps: 2,
            max_edits_per_step: 0.15,
            leader_fraction: 0.10,
            dropout: 0.0,
            ..ErmConfig::default()
        }
    }

    #[test]
    fn test_bench_scorer_forward_finite() {
        let cfg = bench_config();
        let result = bench_scorer_forward(&cfg, 3);
        assert!(
            result.tokens_per_sec.is_finite(),
            "tokens/sec must be finite"
        );
        assert!(result.tokens_per_sec > 0.0, "tokens/sec must be positive");
        assert!(result.elapsed_secs > 0.0, "elapsed must be positive");
        assert!(result.peak_memory > 0, "peak_memory must be positive");
    }

    #[test]
    fn test_bench_single_step_finite() {
        let cfg = bench_config();
        let result = bench_single_refinement_step(&cfg, 3);
        assert!(
            result.tokens_per_sec.is_finite(),
            "tokens/sec must be finite"
        );
        assert!(result.steps_per_sec.is_finite(), "steps/sec must be finite");
        assert!(result.elapsed_secs > 0.0);
    }

    #[test]
    fn test_bench_multi_step_finite() {
        let cfg = bench_config();
        let result = bench_multi_step_refinement(&cfg, 2);
        assert!(
            result.tokens_per_sec.is_finite(),
            "tokens/sec must be finite"
        );
        assert!(result.elapsed_secs > 0.0);
        assert!(
            result.steps_executed > 0,
            "should have executed at least 1 step"
        );
    }

    #[test]
    fn test_bench_results_are_reasonable() {
        let cfg = bench_config();
        let results = vec![
            bench_scorer_forward(&cfg, 2),
            bench_single_refinement_step(&cfg, 2),
            bench_multi_step_refinement(&cfg, 2),
        ];

        for r in &results {
            assert!(
                r.elapsed_secs.is_finite(),
                "{}: elapsed is not finite",
                r.name
            );
            assert!(
                r.tokens_per_sec.is_finite(),
                "{}: tokens/sec not finite",
                r.name
            );
            assert!(!r.elapsed_secs.is_nan(), "{}: elapsed is NaN", r.name);
            assert!(!r.tokens_per_sec.is_nan(), "{}: tokens/sec is NaN", r.name);
        }
    }
}
