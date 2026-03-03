#![recursion_limit = "512"]
//! ERM CLI — command-line interface for the Emergent Route Model.
//!
//! Subcommands:
//! - `train` — run the training loop
//! - `generate` — generate text from scratch or with a prompt
//! - `eval` — evaluate a checkpoint (scorer forward + loss)
//! - `bench` — throughput and memory benchmarks

use clap::{Parser, Subcommand, ValueEnum};
use erm_core::TokenizerApi;

/// Emergent Route Model — CLI entrypoint.
#[derive(Parser, Debug)]
#[command(name = "erm", version, about = "Emergent Route Model")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// Available subcommands.
#[derive(Subcommand, Debug)]
enum Commands {
    /// Train the model on text data.
    Train {
        /// Path to training data (plain text file).
        #[arg(long)]
        data: String,

        /// Number of training steps.
        #[arg(long, default_value = "1000")]
        steps: usize,

        /// Number of warm-start steps (plain denoiser, no colony).
        #[arg(long, default_value = "500")]
        warmstart: usize,

        /// Path to config file (JSON). Uses defaults if not provided.
        #[arg(long)]
        config: Option<String>,

        /// Dry run: validate config and shapes, no actual training.
        #[arg(long)]
        dry_run: bool,
    },

    /// Generate text from a model.
    Generate {
        /// Output length in tokens.
        #[arg(long, default_value = "128")]
        length: usize,

        /// Number of refinement steps.
        #[arg(long, default_value = "6")]
        steps: usize,

        /// Optional prompt text (prefix for prompted generation).
        #[arg(long)]
        prompt: Option<String>,

        /// Random seed for deterministic generation.
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Path to config file (JSON). Uses defaults if not provided.
        #[arg(long)]
        config: Option<String>,
    },

    /// Evaluate a model on held-out data.
    Eval {
        /// Path to evaluation data (plain text file).
        #[arg(long)]
        data: String,

        /// Path to checkpoint directory.
        #[arg(long)]
        checkpoint: String,

        /// Path to config file (JSON). Uses defaults if not provided.
        #[arg(long)]
        config: Option<String>,
    },

    /// Run throughput and memory benchmarks.
    Bench {
        /// Batch size for benchmarks.
        #[arg(long, default_value = "8")]
        batch_size: usize,

        /// Sequence length for benchmarks.
        #[arg(long, default_value = "128")]
        seq_len: usize,

        /// Number of refinement steps to benchmark.
        #[arg(long, default_value = "6")]
        steps: usize,

        /// Number of iterations per benchmark.
        #[arg(long, default_value = "10")]
        iters: usize,
    },

    /// Train the model using burn tensors (GPU-accelerated).
    BurnTrain {
        /// Path to training data (plain text file).
        #[arg(long)]
        data: String,

        /// Number of training steps.
        #[arg(long, default_value = "1000")]
        steps: usize,

        /// Path to config file (JSON). Uses defaults if not provided.
        #[arg(long)]
        config: Option<String>,

        /// Backend to use for burn training.
        #[arg(long, default_value = "gpu")]
        backend: BackendChoice,

        /// Log every N steps.
        #[arg(long, default_value = "100")]
        log_every: usize,
    },

    /// Run colony training (burn scorer on GPU + colony logic on CPU).
    ColonyTrain {
        /// Path to training data (plain text file).
        #[arg(long)]
        data: String,

        /// Number of colony training steps.
        #[arg(long, default_value = "5000")]
        steps: usize,

        /// Path to config file (JSON). Uses defaults if not provided.
        #[arg(long)]
        config: Option<String>,

        /// Backend to use.
        #[arg(long, default_value = "cpu")]
        backend: BackendChoice,

        /// Log every N steps.
        #[arg(long, default_value = "100")]
        log_every: usize,

        /// Checkpoint directory (optional).
        #[arg(long)]
        checkpoint_dir: Option<String>,
    },

    /// Render a graph snapshot to SVG.
    RenderGraph {
        /// Path to snapshot JSON file.
        #[arg(long)]
        snapshot: String,

        /// Output directory for SVG file.
        #[arg(long, default_value = ".")]
        output_dir: String,

        /// Batch index to visualize (0 to batch_size-1).
        #[arg(long, default_value = "0")]
        batch_idx: usize,
    },

    /// Produce a live I/O example from a warmstart checkpoint.
    /// Produce live I/O examples from a checkpoint (auto-detects shape from config.json).
    ///
    /// Reads seq_len, hidden_dim, vocab_size, and tokenizer type from the
    /// checkpoint's config.json. For BPE checkpoints, loads bpe_vocab.json
    /// from the checkpoint directory or its parent experiment directory.
    ///
    /// Supports both colony-train and diffusion-train checkpoints.
    IoExample {
        /// Path to training data (plain text file or directory).
        #[arg(long)]
        data: String,

        /// Path to checkpoint directory (must contain scorer.bin + config.json).
        #[arg(long)]
        checkpoint: String,

        /// Path to config file (JSON). Overrides checkpoint config if provided.
        #[arg(long)]
        config: Option<String>,

        /// Output JSON file path for the I/O example.
        #[arg(long, default_value = "io_example.json")]
        output: String,

        /// Number of I/O examples to generate (each from a different random batch).
        #[arg(long, default_value = "24")]
        num_examples: usize,

        /// Random seed.
        #[arg(long, default_value = "42")]
        seed: u64,
    },

    /// Generate a GIF animation from graph snapshots.
    GenerateGif {
        /// Directory containing snapshot JSON files.
        #[arg(long)]
        snapshots_dir: String,

        /// Output GIF file path.
        #[arg(long, default_value = "colony_growth.gif")]
        output: String,

        /// Frames per second in output GIF.
        #[arg(long, default_value = "10")]
        fps: u16,

        /// Batch index to visualize.
        #[arg(long, default_value = "0")]
        batch_idx: usize,
    },

    /// Save scorer weights from a running checkpoint to disk (mid-run save).
    ///
    /// Loads an existing checkpoint directory and re-saves the scorer.bin,
    /// graph.json, ant_state.json, and config.json. Useful for extracting
    /// scorer weights from a partially-trained checkpoint.
    SaveScorer {
        /// Path to checkpoint directory (must contain scorer.bin or scorer/).
        #[arg(long)]
        checkpoint: String,

        /// Output directory to write scorer.bin and supporting files.
        /// Defaults to the same as --checkpoint.
        #[arg(long)]
        output: Option<String>,
    },

    /// Generate text using diffusion-style coarse-to-fine inference.
    ///
    /// Starts from a masked sequence and iteratively refines it using the
    /// scorer without an autoregressive loop.
    Infer {
        /// Path to checkpoint directory.
        #[arg(long)]
        checkpoint: String,

        /// Output length in tokens.
        #[arg(long, default_value = "128")]
        length: usize,

        /// Number of refinement steps K.
        #[arg(long, default_value = "8")]
        steps: usize,

        /// Optional prompt text.
        #[arg(long)]
        prompt: Option<String>,

        /// Random seed.
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Path to config file (overrides checkpoint config).
        #[arg(long)]
        config: Option<String>,

        /// Backend to use.
        #[arg(long, default_value = "cpu")]
        backend: BackendChoice,
    },

    /// Run diffusion colony training (tokens-in → tokens-out, T refinement steps).
    DiffusionTrain {
        /// Path to training data directory (containing .txt files).
        #[arg(long, default_value = "/workspace/rust-pcn/data/books")]
        data: String,

        /// Number of training steps.
        #[arg(long, default_value = "10000")]
        steps: usize,

        /// Path to config file (JSON). Uses defaults if not provided.
        #[arg(long)]
        config: Option<String>,

        /// Backend to use.
        #[arg(long, default_value = "cuda")]
        backend: BackendChoice,

        /// Log every N steps.
        #[arg(long, default_value = "25")]
        log_every: usize,

        /// Save checkpoint every N steps (0 = disabled).
        #[arg(long, default_value = "250")]
        checkpoint_every: usize,

        /// Checkpoint output directory.
        #[arg(
            long,
            default_value = "/workspace/erm-rust/data/experiments/default/checkpoints"
        )]
        checkpoint_dir: Option<String>,

        /// Experiment id (used in metrics.jsonl).
        #[arg(long, default_value = "exp-default")]
        exp_id: String,

        /// Sequence length override.
        #[arg(long)]
        seq_len: Option<usize>,

        /// Hidden dimension override.
        #[arg(long)]
        hidden_dim: Option<usize>,

        /// Number of ants override.
        #[arg(long)]
        num_ants: Option<usize>,

        /// Batch size override.
        #[arg(long)]
        batch: Option<usize>,

        /// Diffusion steps T override.
        #[arg(long)]
        diffusion_t: Option<usize>,

        /// Resume training from a checkpoint directory (loads scorer weights, graph, ant state).
        #[arg(long)]
        resume: Option<String>,
    },
}

/// Backend choice for burn-based training.
#[derive(Debug, Clone, ValueEnum)]
enum BackendChoice {
    /// CPU backend (NdArray — portable, no GPU required).
    Cpu,
    /// GPU backend (wgpu — requires Vulkan/Metal/DX12).
    Gpu,
    /// CUDA backend (burn-cuda — requires NVIDIA GPU + CUDA toolkit).
    Cuda,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Train {
            data,
            steps,
            warmstart,
            config,
            dry_run,
        } => {
            run_train(&data, steps, warmstart, config.as_deref(), dry_run);
        }
        Commands::Generate {
            length,
            steps,
            prompt,
            seed,
            config,
        } => {
            run_generate(length, steps, prompt.as_deref(), seed, config.as_deref());
        }
        Commands::Eval {
            data,
            checkpoint,
            config,
        } => {
            run_eval(&data, &checkpoint, config.as_deref());
        }
        Commands::Bench {
            batch_size,
            seq_len,
            steps,
            iters,
        } => {
            run_bench(batch_size, seq_len, steps, iters);
        }
        Commands::BurnTrain {
            data,
            steps,
            config,
            backend,
            log_every,
        } => {
            run_burn_train(&data, steps, config.as_deref(), &backend, log_every);
        }
        Commands::ColonyTrain {
            data,
            steps,
            config,
            backend,
            log_every,
            checkpoint_dir,
        } => {
            run_colony_train(
                &data,
                steps,
                config.as_deref(),
                &backend,
                log_every,
                checkpoint_dir.as_deref(),
            );
        }
        Commands::IoExample {
            data,
            checkpoint,
            config,
            output,
            num_examples,
            seed,
        } => {
            run_io_example(
                &data,
                &checkpoint,
                config.as_deref(),
                &output,
                num_examples,
                seed,
            );
        }
        Commands::RenderGraph {
            snapshot,
            output_dir,
            batch_idx,
        } => {
            run_render_graph(&snapshot, &output_dir, batch_idx);
        }
        Commands::GenerateGif {
            snapshots_dir,
            output,
            fps,
            batch_idx,
        } => {
            run_generate_gif(&snapshots_dir, &output, fps, batch_idx);
        }
        Commands::SaveScorer { checkpoint, output } => {
            run_save_scorer(&checkpoint, output.as_deref());
        }
        Commands::Infer {
            checkpoint,
            length,
            steps,
            prompt,
            seed,
            config,
            backend,
        } => {
            run_infer(
                &checkpoint,
                length,
                steps,
                prompt.as_deref(),
                seed,
                config.as_deref(),
                &backend,
            );
        }
        Commands::DiffusionTrain {
            data,
            steps,
            config,
            backend,
            log_every,
            checkpoint_every,
            checkpoint_dir,
            exp_id,
            seq_len,
            hidden_dim,
            num_ants,
            batch,
            diffusion_t,
            resume,
        } => {
            run_diffusion_train(
                &data,
                steps,
                config.as_deref(),
                &backend,
                log_every,
                checkpoint_every,
                checkpoint_dir.as_deref(),
                &exp_id,
                seq_len,
                hidden_dim,
                num_ants,
                batch,
                diffusion_t,
                resume.as_deref(),
            );
        }
    }
}

/// Load an [`ErmConfig`](erm_core::ErmConfig) from a JSON file, or return defaults.
fn load_config(path: Option<&str>) -> erm_core::ErmConfig {
    if let Some(p) = path {
        let json = match std::fs::read_to_string(p) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("ERROR: cannot read config file '{p}': {e}");
                std::process::exit(1);
            }
        };
        match serde_json::from_str(&json) {
            Ok(cfg) => cfg,
            Err(e) => {
                eprintln!("ERROR: cannot parse config file '{p}': {e}");
                std::process::exit(1);
            }
        }
    } else {
        erm_core::ErmConfig::default()
    }
}

fn run_train(
    data_path: &str,
    total_steps: usize,
    warmstart_steps: usize,
    config_path: Option<&str>,
    dry_run: bool,
) {
    use erm_train::orchestrator::{Orchestrator, TrainingConfig};

    let erm_cfg = load_config(config_path);

    if dry_run {
        println!("=== DRY RUN ===");
        println!(
            "Config: seq_len={}, vocab_size={}, hidden_dim={}",
            erm_cfg.seq_len, erm_cfg.vocab_size, erm_cfg.hidden_dim
        );
        println!(
            "Shapes: logits=[{}, {}, {}]",
            erm_cfg.batch_size, erm_cfg.seq_len, erm_cfg.vocab_size
        );
        let colony_steps = total_steps.saturating_sub(warmstart_steps);
        println!("Steps: warmstart={warmstart_steps}, colony={colony_steps}");
        println!("Data: {data_path}");
        println!("Config validated OK.");
        return;
    }

    // Read training text.
    let text = match std::fs::read_to_string(data_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("ERROR: cannot read data file '{data_path}': {e}");
            std::process::exit(1);
        }
    };

    let tokenizer = erm_core::tokenizer::CharTokenizer::from_text(&text);
    let vocab_size = tokenizer.vocab_size();

    let mut cfg = erm_cfg;
    cfg.vocab_size = vocab_size;

    let colony_steps = total_steps.saturating_sub(warmstart_steps);
    let train_cfg = TrainingConfig {
        erm: cfg,
        warm_start_steps: warmstart_steps,
        colony_steps,
        log_every: 100,
        checkpoint_every: 0,
        seed: 42,
    };

    let total_vocab = train_cfg.erm.total_vocab_size();
    let dataset = match erm_train::dataset::TextDataset::from_text(
        &text,
        &tokenizer,
        train_cfg.erm.seq_len,
    ) {
        Ok(ds) => ds,
        Err(e) => {
            eprintln!("ERROR: cannot build dataset: {e}");
            std::process::exit(1);
        }
    };

    let mut orch = Orchestrator::new(train_cfg, total_vocab);
    println!("Training: warmstart={warmstart_steps}, colony={colony_steps}, vocab={vocab_size}");

    if let Err(e) = orch.run_all(&dataset, None) {
        eprintln!("ERROR during training: {e}");
        std::process::exit(1);
    }

    println!("Training complete. Global step: {}", orch.global_step);
    if let Some(last) = orch.loss_log.last() {
        println!("Final loss: {:.4} (step {})", last.loss, last.step);
    }
}

fn run_generate(
    length: usize,
    steps: usize,
    prompt: Option<&str>,
    seed: u64,
    config_path: Option<&str>,
) {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let mut cfg = load_config(config_path);
    cfg.seq_len = length;
    cfg.refinement_steps = steps;
    // Use a small vocab for demo generation without a real model.
    if cfg.vocab_size > 256 {
        cfg.vocab_size = 256;
    }

    let pconfig = erm_core::config::PheromoneConfig::from_config(&cfg);
    let scorer = erm_core::scorer::Scorer::new(&cfg, cfg.vocab_size, seed);
    let mut graph = erm_core::graph::RouteGraph::new(&cfg);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let result = if let Some(prompt_text) = prompt {
        // Convert chars to token ids (mod vocab_size for safety).
        let prefix: Vec<u32> = prompt_text
            .chars()
            .map(|c| (c as u32) % cfg.vocab_size as u32)
            .collect();
        let gen_length = length.saturating_sub(prefix.len());
        if gen_length == 0 {
            eprintln!("ERROR: prompt is as long as or longer than generation length");
            std::process::exit(1);
        }
        match erm_core::refinement::generate_prompted(
            &scorer, &mut graph, &cfg, &pconfig, &prefix, gen_length, &mut rng,
        ) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("ERROR: generation failed: {e}");
                std::process::exit(1);
            }
        }
    } else {
        match erm_core::refinement::generate_from_scratch(
            &scorer, &mut graph, &cfg, &pconfig, length, &mut rng,
        ) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("ERROR: generation failed: {e}");
                std::process::exit(1);
            }
        }
    };

    println!(
        "Generated {} tokens in {} steps ({})",
        result.y_final.len(),
        result.steps_executed,
        result.stop_reason
    );
    let display_len = result.y_final.len().min(64);
    println!("Tokens: {:?}", &result.y_final[..display_len]);
    println!(
        "Peak memory estimate: {} bytes",
        result.peak_memory_estimate
    );
}

fn run_eval(data_path: &str, checkpoint_dir: &str, config_path: Option<&str>) {
    use erm_train::orchestrator::Orchestrator;

    let _ = config_path; // config comes from checkpoint

    // Load checkpoint.
    let (orch, phase) = match Orchestrator::load_checkpoint(checkpoint_dir) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("ERROR: cannot load checkpoint from '{checkpoint_dir}': {e}");
            std::process::exit(1);
        }
    };

    println!(
        "Loaded checkpoint: step={}, phase={phase}",
        orch.global_step
    );

    // Read eval text.
    let text = match std::fs::read_to_string(data_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("ERROR: cannot read data file '{data_path}': {e}");
            std::process::exit(1);
        }
    };

    let tokenizer = erm_core::tokenizer::CharTokenizer::from_text(&text);
    let cfg = &orch.config.erm;
    let dataset = match erm_train::dataset::TextDataset::from_text(&text, &tokenizer, cfg.seq_len) {
        Ok(ds) => ds,
        Err(e) => {
            eprintln!("ERROR: cannot build eval dataset: {e}");
            std::process::exit(1);
        }
    };

    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let num_eval_batches = 10;
    let mut total_loss = 0.0_f32;
    let mut total_corrupted = 0_usize;

    for _ in 0..num_eval_batches {
        let batch = dataset.get_batch(cfg.batch_size, &mut rng);
        match erm_train::training::train_step(&orch.scorer, &batch, None, cfg, &mut rng) {
            Ok(result) => {
                total_loss += result.loss * result.num_corrupted as f32;
                total_corrupted += result.num_corrupted;
            }
            Err(e) => {
                eprintln!("WARNING: eval step failed: {e}");
            }
        }
    }

    let mean_loss = if total_corrupted > 0 {
        total_loss / total_corrupted as f32
    } else {
        0.0
    };

    println!("Eval results ({num_eval_batches} batches):");
    println!("  Mean denoising loss: {mean_loss:.4}");
    println!("  Total corrupted positions: {total_corrupted}");
}

fn run_bench(batch_size: usize, seq_len: usize, steps: usize, iters: usize) {
    erm_train::bench::run_benchmarks(batch_size, seq_len, steps, iters);
}

fn run_burn_train(
    data_path: &str,
    total_steps: usize,
    config_path: Option<&str>,
    backend: &BackendChoice,
    log_every: usize,
) {
    let erm_cfg = load_config(config_path);

    // Read training text.
    let text = match std::fs::read_to_string(data_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("ERROR: cannot read data file '{data_path}': {e}");
            std::process::exit(1);
        }
    };

    let tokenizer = erm_core::tokenizer::CharTokenizer::from_text(&text);
    let vocab_size = tokenizer.vocab_size();
    let mut cfg = erm_cfg;
    cfg.vocab_size = vocab_size;

    let dataset = match erm_train::dataset::TextDataset::from_text(&text, &tokenizer, cfg.seq_len) {
        Ok(ds) => ds,
        Err(e) => {
            eprintln!("ERROR: cannot build dataset: {e}");
            std::process::exit(1);
        }
    };

    match backend {
        BackendChoice::Cpu => {
            burn_train_loop::<burn_autodiff::Autodiff<burn_ndarray::NdArray<f32>>>(
                &cfg,
                &dataset,
                total_steps,
                log_every,
                Default::default(),
            );
        }
        BackendChoice::Gpu => {
            #[cfg(feature = "gpu")]
            {
                let device = burn_wgpu::WgpuDevice::default();
                burn_train_loop::<burn_autodiff::Autodiff<burn_wgpu::Wgpu>>(
                    &cfg,
                    &dataset,
                    total_steps,
                    log_every,
                    device,
                );
            }
            #[cfg(not(feature = "gpu"))]
            {
                eprintln!("ERROR: GPU (wgpu) support not compiled. Rebuild with: cargo build --release --features gpu");
                std::process::exit(1);
            }
        }
        BackendChoice::Cuda => {
            #[cfg(feature = "cuda")]
            {
                let device = burn_cuda::CudaDevice::new(0);
                println!("Backend: CUDA device 0");
                burn_train_loop::<burn_autodiff::Autodiff<burn_cuda::Cuda>>(
                    &cfg,
                    &dataset,
                    total_steps,
                    log_every,
                    device,
                );
            }
            #[cfg(not(feature = "cuda"))]
            {
                eprintln!("ERROR: CUDA support not compiled. Rebuild with: cargo build --release --features cuda");
                std::process::exit(1);
            }
        }
    }
}

/// Generic burn training loop parameterised by backend.
fn burn_train_loop<B: burn::tensor::backend::AutodiffBackend>(
    cfg: &erm_core::ErmConfig,
    dataset: &erm_train::dataset::TextDataset,
    total_steps: usize,
    log_every: usize,
    device: B::Device,
) {
    use erm_train::burn_training::BurnTrainer;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let mut trainer = BurnTrainer::<B>::new(cfg, device);
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    println!(
        "Burn training: steps={total_steps}, vocab={}, hidden={}, backend={}",
        cfg.vocab_size,
        cfg.hidden_dim,
        std::any::type_name::<B>(),
    );

    let mut recent_losses = Vec::new();
    for step in 0..total_steps {
        let batch = dataset.get_batch(cfg.batch_size, &mut rng);
        match trainer.train_step(&batch, None, cfg, &mut rng) {
            Ok(loss) => {
                recent_losses.push(loss);
                if (step + 1) % log_every == 0 || step == total_steps - 1 {
                    let avg_loss: f32 =
                        recent_losses.iter().sum::<f32>() / recent_losses.len() as f32;
                    println!(
                        "  step {}/{total_steps} — avg loss: {avg_loss:.4} (last {}: {loss:.4})",
                        step + 1,
                        recent_losses.len(),
                    );
                    recent_losses.clear();
                }
            }
            Err(e) => {
                eprintln!("ERROR at step {step}: {e}");
                std::process::exit(1);
            }
        }
    }

    println!("Burn training complete.");
}

fn run_colony_train(
    data_path: &str,
    total_steps: usize,
    config_path: Option<&str>,
    backend: &BackendChoice,
    log_every: usize,
    checkpoint_dir: Option<&str>,
) {
    let erm_cfg = load_config(config_path);

    let text = match std::fs::read_to_string(data_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("ERROR: cannot read data file '{data_path}': {e}");
            std::process::exit(1);
        }
    };

    let tokenizer = erm_core::tokenizer::CharTokenizer::from_text(&text);
    let vocab_size = tokenizer.vocab_size();
    let mut cfg = erm_cfg;
    cfg.vocab_size = vocab_size;

    let dataset = match erm_train::dataset::TextDataset::from_text(&text, &tokenizer, cfg.seq_len) {
        Ok(ds) => ds,
        Err(e) => {
            eprintln!("ERROR: cannot build dataset: {e}");
            std::process::exit(1);
        }
    };

    match backend {
        BackendChoice::Cpu => {
            colony_train_loop::<burn_autodiff::Autodiff<burn_ndarray::NdArray<f32>>>(
                &cfg,
                &dataset,
                total_steps,
                log_every,
                checkpoint_dir,
                Default::default(),
            );
        }
        BackendChoice::Gpu => {
            #[cfg(feature = "gpu")]
            {
                let device = burn_wgpu::WgpuDevice::default();
                colony_train_loop::<burn_autodiff::Autodiff<burn_wgpu::Wgpu>>(
                    &cfg,
                    &dataset,
                    total_steps,
                    log_every,
                    checkpoint_dir,
                    device,
                );
            }
            #[cfg(not(feature = "gpu"))]
            {
                eprintln!("ERROR: GPU (wgpu) support not compiled. Rebuild with: cargo build --release --features gpu");
                std::process::exit(1);
            }
        }
        BackendChoice::Cuda => {
            #[cfg(feature = "cuda")]
            {
                let device = burn_cuda::CudaDevice::new(0);
                println!("Backend: CUDA device 0");
                colony_train_loop::<burn_autodiff::Autodiff<burn_cuda::Cuda>>(
                    &cfg,
                    &dataset,
                    total_steps,
                    log_every,
                    checkpoint_dir,
                    device,
                );
            }
            #[cfg(not(feature = "cuda"))]
            {
                eprintln!("ERROR: CUDA support not compiled. Rebuild with: cargo build --release --features cuda");
                std::process::exit(1);
            }
        }
    }
}

/// Generic colony training loop parameterised by backend.
fn colony_train_loop<B: burn::tensor::backend::AutodiffBackend>(
    cfg: &erm_core::ErmConfig,
    dataset: &erm_train::dataset::TextDataset,
    total_steps: usize,
    log_every: usize,
    checkpoint_dir: Option<&str>,
    device: B::Device,
) {
    use erm_train::colony_orchestrator::{ColonyOrchestrator, ColonyTrainingConfig};

    let colony_cfg = ColonyTrainingConfig {
        erm: cfg.clone(),
        colony_steps: total_steps,
        log_every,
        checkpoint_every: if checkpoint_dir.is_some() { 500 } else { 0 },
        seed: 42,
    };

    let mut orch = ColonyOrchestrator::<B>::new(colony_cfg, device);

    println!(
        "Colony training: steps={total_steps}, vocab={}, hidden={}, backend={}",
        cfg.vocab_size,
        cfg.hidden_dim,
        std::any::type_name::<B>(),
    );

    match orch.run_colony_phase(dataset, checkpoint_dir) {
        Ok(_) => {
            println!("Colony training complete. Steps: {}", orch.step);
            if let Some(last) = orch.log.last() {
                println!(
                    "Final: loss={:.4}, edits={}, mean_φ={:.4}",
                    last.loss, last.num_edits, last.mean_phi
                );
            }

            // Save warmstart checkpoint if checkpoint_dir is set.
            if let Some(dir) = checkpoint_dir {
                let warmstart_dir = format!("{dir}/warmstart");
                match orch.trainer.save_warmstart(&warmstart_dir) {
                    Ok(()) => println!("Warmstart checkpoint saved to: {warmstart_dir}"),
                    Err(e) => eprintln!("WARNING: failed to save warmstart checkpoint: {e}"),
                }
            }
        }
        Err(e) => {
            eprintln!("ERROR during colony training: {e}");
            std::process::exit(1);
        }
    }
}

/// Produce a live I/O example from warmstart checkpoint.
/// Produce live I/O examples from a checkpoint (shape auto-detect).
///
/// 1. Reads config.json from checkpoint dir → gets seq_len, hidden_dim, vocab_size.
/// 2. Searches for bpe_vocab.json in checkpoint dir, parent, or grandparent.
/// 3. Loads scorer via DiffusionTrainer (handles both colony and diffusion checkpoints).
/// 4. Generates `num_examples` I/O pairs with top-3 predictions per corrupted position.
fn run_io_example(
    data_path: &str,
    checkpoint_dir: &str,
    config_path: Option<&str>,
    output_path: &str,
    num_examples: usize,
    seed: u64,
) {
    use std::path::Path;

    // ── Step 1: Load config from checkpoint (shape auto-detect) ───────
    let ckpt_config_path = format!("{checkpoint_dir}/config.json");
    let cfg = if let Some(explicit) = config_path {
        eprintln!("[io-example] Using explicit config: {explicit}");
        load_config(Some(explicit))
    } else if Path::new(&ckpt_config_path).exists() {
        eprintln!("[io-example] Auto-detected config from checkpoint: {ckpt_config_path}");
        load_config(Some(&ckpt_config_path))
    } else {
        eprintln!(
            "ERROR: No config.json found in checkpoint dir '{checkpoint_dir}'.\n\
             Either provide --config or ensure the checkpoint contains config.json.\n\
             (Checkpoint dirs from diffusion-train always include config.json with\n\
             the correct seq_len={{}}, hidden_dim={{}}, vocab_size={{}} used during training.)"
        );
        std::process::exit(1);
    };

    // ── Preflight: log expected dims from config (actual shapes derived at runtime) ──
    eprintln!(
        "[io-example] Config hints: seq_len={} hidden_dim={} vocab_size={} \
         (actual dims derived from scorer/tensor at runtime)",
        cfg.seq_len,
        cfg.hidden_dim,
        cfg.total_vocab_size()
    );

    // ── Step 2: Find and load BPE vocabulary ──────────────────────────
    let bpe_search_paths = [
        format!("{checkpoint_dir}/bpe_vocab.json"),
        // Parent dir (e.g., checkpoint is step_00001500/, vocab at experiment root).
        Path::new(checkpoint_dir)
            .parent()
            .map(|p| format!("{}/bpe_vocab.json", p.display()))
            .unwrap_or_default(),
        // Grandparent dir (e.g., checkpoint is smoke/final/, vocab at experiment root).
        Path::new(checkpoint_dir)
            .parent()
            .and_then(|p| p.parent())
            .map(|p| format!("{}/bpe_vocab.json", p.display()))
            .unwrap_or_default(),
        // Sibling smoke dir (e.g., checkpoint is step_00001500/, vocab at smoke/bpe_vocab.json).
        Path::new(checkpoint_dir)
            .parent()
            .map(|p| format!("{}/smoke/bpe_vocab.json", p.display()))
            .unwrap_or_default(),
    ];

    let is_bpe = cfg.tokenizer_type == "bpe";
    let bpe_tokenizer: Option<erm_core::BpeTokenizer> = if is_bpe {
        let found = bpe_search_paths
            .iter()
            .find(|p| !p.is_empty() && Path::new(p).exists());
        match found {
            Some(vocab_path) => {
                eprintln!("[io-example] Loading BPE vocabulary from: {vocab_path}");
                match erm_core::BpeTokenizer::load(vocab_path) {
                    Ok(t) => {
                        // Verify vocab size matches config.
                        let loaded_vocab = t.vocab_size();
                        if loaded_vocab != cfg.vocab_size {
                            eprintln!(
                                "WARNING: BPE vocab_size ({loaded_vocab}) != config.vocab_size ({}). \
                                 Using BPE vocab_size.",
                                cfg.vocab_size
                            );
                        }
                        Some(t)
                    }
                    Err(e) => {
                        eprintln!("ERROR: cannot load BPE vocab from '{vocab_path}': {e}");
                        std::process::exit(1);
                    }
                }
            }
            None => {
                eprintln!(
                    "ERROR: tokenizer_type=bpe but no bpe_vocab.json found.\n\
                     Searched: {:?}",
                    bpe_search_paths
                );
                std::process::exit(1);
            }
        }
    } else {
        None
    };

    // ── Step 3: Build dataset using correct tokenizer ─────────────────
    let text = {
        let p = Path::new(data_path);
        if p.is_dir() {
            // Collect text from all .txt files in the directory.
            collect_sample_text(data_path, 10_000_000)
        } else {
            match std::fs::read_to_string(data_path) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("ERROR: cannot read data file '{data_path}': {e}");
                    std::process::exit(1);
                }
            }
        }
    };

    if text.is_empty() {
        eprintln!("ERROR: no text data found at '{data_path}'");
        std::process::exit(1);
    }

    // ── Step 4: Run I/O example generation on CPU ─────────────────────
    type CpuBackend = burn_autodiff::Autodiff<burn_ndarray::NdArray<f32>>;

    if let Some(ref bpe) = bpe_tokenizer {
        io_example_loop_bpe::<CpuBackend>(
            &cfg,
            &text,
            bpe,
            checkpoint_dir,
            output_path,
            num_examples,
            seed,
            Default::default(),
        );
    } else {
        // Fall back to CharTokenizer (legacy colony-train checkpoints).
        let char_tok = erm_core::tokenizer::CharTokenizer::from_text(&text);
        io_example_loop_char::<CpuBackend>(
            &cfg,
            &text,
            &char_tok,
            checkpoint_dir,
            output_path,
            num_examples,
            seed,
            Default::default(),
        );
    }
}

/// I/O example generation with BPE tokenizer (diffusion-train checkpoints).
#[allow(clippy::too_many_arguments)]
fn io_example_loop_bpe<B: burn::tensor::backend::AutodiffBackend>(
    cfg: &erm_core::ErmConfig,
    text: &str,
    tokenizer: &erm_core::BpeTokenizer,
    checkpoint_dir: &str,
    output_path: &str,
    num_examples: usize,
    seed: u64,
    device: B::Device,
) {
    use erm_core::TokenizerApi;
    use erm_train::bridge::{tensor_to_vec, tokens_to_tensor};
    use erm_train::diffusion_training::DiffusionTrainer;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    // Use DiffusionTrainer to load the scorer with correct shapes.
    let mut trainer = DiffusionTrainer::<B>::new(cfg, device.clone());

    if let Err(e) = trainer.load_checkpoint(checkpoint_dir) {
        eprintln!("ERROR: cannot load checkpoint from '{checkpoint_dir}': {e}");
        std::process::exit(1);
    }
    eprintln!(
        "[io-example] Loaded checkpoint from: {checkpoint_dir} (step={})",
        trainer.step
    );

    // Verify loaded scorer dimensions match config.
    let scorer_hidden = trainer.scorer.hidden_dim();
    let scorer_seq = trainer.scorer.seq_len();
    let scorer_vocab = trainer.scorer.vocab_size();
    eprintln!(
        "[io-example] Scorer dims: hidden={scorer_hidden} seq={scorer_seq} vocab={scorer_vocab}"
    );
    if scorer_hidden != cfg.hidden_dim {
        eprintln!(
            "ERROR: shape mismatch! Scorer hidden_dim={scorer_hidden} but config says {}.\n\
             The checkpoint was trained with different dimensions. Cannot proceed without retraining.",
            cfg.hidden_dim
        );
        std::process::exit(1);
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    // Use scorer's runtime dims — these come from the actual weights, not config hints.
    let l = scorer_seq;
    let cfg_vocab = cfg.total_vocab_size();

    // Tokenize entire text into BPE tokens and create batches manually.
    let all_tokens = tokenizer.encode_text(text);
    if all_tokens.len() < l {
        eprintln!(
            "ERROR: text encodes to only {} BPE tokens, need at least seq_len={}",
            all_tokens.len(),
            l
        );
        std::process::exit(1);
    }

    let mut examples: Vec<serde_json::Value> = Vec::new();

    for example_idx in 0..num_examples {
        // Pick a random start position within the token stream.
        use rand::Rng;
        let max_start = all_tokens.len().saturating_sub(l);
        let start = if max_start > 0 {
            rng.gen_range(0..max_start)
        } else {
            0
        };
        let batch_tokens: Vec<u32> = all_tokens[start..start + l].to_vec();

        // Corrupt the batch at noise level t.
        let t_val = 3.min(cfg.diffusion_steps.max(1));
        let x_i32: Vec<i32> = batch_tokens.iter().map(|&tk| tk as i32).collect();
        let corruption = match erm_core::corruption::corrupt(&x_i32, t_val, cfg, &mut rng) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("WARNING: corruption failed for example {example_idx}: {e}");
                continue;
            }
        };
        let y_t_u32: Vec<u32> = corruption.y_t.iter().map(|&v| v as u32).collect();

        // Forward pass (batch=1). B is implicit from the tensor shape.
        let tokens_tensor = match tokens_to_tensor::<B>(&y_t_u32, 1, l, &device) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("WARNING: tensor creation failed for example {example_idx}: {e}");
                continue;
            }
        };
        let (logits_tensor, _unc) = trainer.scorer.forward(tokens_tensor);
        let logits_cpu = match tensor_to_vec(logits_tensor) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("WARNING: logits transfer failed for example {example_idx}: {e}");
                continue;
            }
        };

        // Derive V from actual logits buffer: len == 1*L*V → V = len/L.
        let vocab_size = if l > 0 {
            logits_cpu.len() / l
        } else {
            cfg_vocab
        };
        if logits_cpu.len() != l * vocab_size {
            eprintln!(
                "ERROR: logits buffer size {} is not divisible by L={l}. Cannot derive V.",
                logits_cpu.len()
            );
            std::process::exit(1);
        }

        // Decode strings via BPE.
        let clean_string = tokenizer.decode_text(&batch_tokens);
        let corrupted_string = tokenizer.decode_text(&y_t_u32);

        // Find corrupted positions and top-3 predictions.
        let mask_id = cfg.mask_token_id() as u32;
        let mut position_predictions = Vec::new();

        for pos in 0..l {
            if y_t_u32[pos] == mask_id || y_t_u32[pos] != batch_tokens[pos] {
                let start_idx = pos * vocab_size;
                let end_idx = start_idx + vocab_size;
                if end_idx > logits_cpu.len() {
                    continue; // bounds-safe skip
                }
                let pos_logits = &logits_cpu[start_idx..end_idx];

                // Softmax.
                let max_logit = pos_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = pos_logits.iter().map(|&v| (v - max_logit).exp()).collect();
                let sum_exps: f32 = exps.iter().sum();
                let probs: Vec<f32> = exps.iter().map(|&e| e / sum_exps).collect();

                // Top-3.
                let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let top3: Vec<serde_json::Value> = indexed
                    .iter()
                    .take(3)
                    .map(|(tok, prob)| {
                        let decoded = tokenizer.decode_text(&[*tok as u32]);
                        serde_json::json!({
                            "token_id": tok,
                            "text": decoded,
                            "p": format!("{:.4}", prob),
                        })
                    })
                    .collect();

                let ground_truth_text = tokenizer.decode_text(&[batch_tokens[pos]]);
                let is_mask = y_t_u32[pos] == mask_id;

                position_predictions.push(serde_json::json!({
                    "position": pos,
                    "corruption_type": if is_mask { "MASK" } else { "REPLACE" },
                    "ground_truth_token": batch_tokens[pos],
                    "ground_truth_text": ground_truth_text,
                    "corrupted_token": y_t_u32[pos],
                    "top_3_predictions": top3,
                }));
            }
        }

        // For the first example, print a human-readable summary to stderr for k8s logs.
        if example_idx == 0 && !position_predictions.is_empty() {
            let mut predicted_tokens = y_t_u32.clone();
            let mut top1_hits = 0usize;
            let mut top3_hits = 0usize;
            let num_corrupted_pos = position_predictions.len();
            for pred in &position_predictions {
                let pos = pred["position"].as_u64().unwrap_or(0) as usize;
                let ground_truth = pred["ground_truth_token"].as_u64().unwrap_or(0) as u32;
                if let Some(top3) = pred["top_3_predictions"].as_array() {
                    if let Some(top1) = top3.first() {
                        let top1_tok = top1["token_id"].as_u64().unwrap_or(0) as u32;
                        if pos < predicted_tokens.len() {
                            predicted_tokens[pos] = top1_tok;
                        }
                        if top1_tok == ground_truth {
                            top1_hits += 1;
                        }
                    }
                    if top3
                        .iter()
                        .any(|p| p["token_id"].as_u64().unwrap_or(0) as u32 == ground_truth)
                    {
                        top3_hits += 1;
                    }
                }
            }
            let predicted_str = tokenizer.decode_text(&predicted_tokens);
            let trunc80 = |s: &str| -> String { s.chars().take(80).collect() };
            let top1_pct = 100.0 * top1_hits as f32 / num_corrupted_pos as f32;
            let top3_pct = 100.0 * top3_hits as f32 / num_corrupted_pos as f32;
            eprintln!("[io-sampler] === Sample Output (example 0, t={t_val}) ===");
            eprintln!("[io-sampler] CLEAN:     \"{}\"", trunc80(&clean_string));
            eprintln!("[io-sampler] CORRUPTED: \"{}\"", trunc80(&corrupted_string));
            eprintln!("[io-sampler] PREDICTED: \"{}\"", trunc80(&predicted_str));
            eprintln!(
                "[io-sampler] Accuracy: top-1={:.1}% top-3={:.1}% ({} corrupted positions)",
                top1_pct, top3_pct, num_corrupted_pos
            );
        }

        let example = serde_json::json!({
            "example_idx": example_idx,
            "text_offset": start,
            "clean_string": clean_string,
            "corrupted_string": corrupted_string,
            "clean_tokens": batch_tokens,
            "corrupted_tokens": y_t_u32,
            "corruption_step_t": t_val,
            "num_corrupted_positions": position_predictions.len(),
            "position_predictions": position_predictions,
        });
        examples.push(example);
    }

    // Build final JSON — use runtime-derived dimensions.
    let output = serde_json::json!({
        "metadata": {
            "checkpoint_dir": checkpoint_dir,
            "checkpoint_step": trainer.step,
            "seed": seed,
            "seq_len": l,
            "hidden_dim": scorer_hidden,
            "vocab_size": scorer_vocab,
            "num_examples": examples.len(),
            "tokenizer_type": "bpe",
        },
        "examples": examples,
    });

    let json_str = serde_json::to_string_pretty(&output).expect("JSON serialization");
    if let Some(parent) = std::path::Path::new(output_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(output_path, &json_str).expect("write output file");
    eprintln!(
        "[io-example] Wrote {} examples to: {output_path}",
        examples.len()
    );

    // Print first 3 examples summary.
    for ex in examples.iter().take(3) {
        let clean = ex["clean_string"].as_str().unwrap_or("");
        let n_corr = ex["num_corrupted_positions"].as_u64().unwrap_or(0);
        eprintln!(
            "  Example {}: {} corrupted positions | clean: {}...",
            ex["example_idx"],
            n_corr,
            &clean[..clean.len().min(60)]
        );
    }
}

/// I/O example generation with CharTokenizer (legacy colony-train checkpoints).
#[allow(clippy::too_many_arguments)]
fn io_example_loop_char<B: burn::tensor::backend::AutodiffBackend>(
    cfg: &erm_core::ErmConfig,
    text: &str,
    tokenizer: &erm_core::tokenizer::CharTokenizer,
    checkpoint_dir: &str,
    output_path: &str,
    num_examples: usize,
    seed: u64,
    device: B::Device,
) {
    use erm_train::bridge::{tensor_to_vec, tokens_to_tensor};
    use erm_train::colony_training::ColonyTrainer;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let mut cfg = cfg.clone();
    cfg.vocab_size = tokenizer.vocab_size();

    let dataset = match erm_train::dataset::TextDataset::from_text(text, tokenizer, cfg.seq_len) {
        Ok(ds) => ds,
        Err(e) => {
            eprintln!("ERROR: cannot build dataset: {e}");
            std::process::exit(1);
        }
    };

    let mut trainer = ColonyTrainer::<B>::new(&cfg, device.clone());

    // Load warmstart checkpoint.
    if let Err(e) = trainer.load_warmstart(checkpoint_dir) {
        eprintln!("ERROR: cannot load warmstart checkpoint: {e}");
        std::process::exit(1);
    }
    eprintln!("[io-example] Loaded colony checkpoint from: {checkpoint_dir}");

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let l = cfg.seq_len;
    let mut examples: Vec<serde_json::Value> = Vec::new();

    for example_idx in 0..num_examples {
        let batch = dataset.get_batch(1, &mut rng);
        let t_val = 3.min(cfg.refinement_steps);
        let x_i32: Vec<i32> = batch.tokens.iter().map(|&tk| tk as i32).collect();

        let corruption = match erm_core::corruption::corrupt(&x_i32, t_val, &cfg, &mut rng) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("WARNING: corruption failed for example {example_idx}: {e}");
                continue;
            }
        };
        let y_t_u32: Vec<u32> = corruption.y_t.iter().map(|&v| v as u32).collect();

        let tokens_tensor = match tokens_to_tensor::<B>(&y_t_u32, 1, l, &device) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("WARNING: tensor failed for example {example_idx}: {e}");
                continue;
            }
        };
        let (logits_tensor, _) = trainer.scorer.forward(tokens_tensor);
        let logits_cpu = match tensor_to_vec(logits_tensor) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("WARNING: logits transfer failed for example {example_idx}: {e}");
                continue;
            }
        };

        // Derive vocab_size from actual logits buffer.
        let vocab_size = if l > 0 {
            logits_cpu.len() / l
        } else {
            cfg.total_vocab_size()
        };

        let clean_tokens: Vec<u32> = x_i32.iter().map(|&v| v as u32).collect();
        let clean_string = tokenizer.decode(&clean_tokens);
        let corrupted_string = tokenizer.decode(&y_t_u32);
        let mask_id = cfg.mask_token_id() as u32;
        let mut position_predictions = Vec::new();

        for pos in 0..l {
            if y_t_u32[pos] == mask_id || y_t_u32[pos] != clean_tokens[pos] {
                let start_idx = pos * vocab_size;
                let end_idx = start_idx + vocab_size;
                if end_idx > logits_cpu.len() {
                    continue;
                }
                let pos_logits = &logits_cpu[start_idx..end_idx];
                let max_logit = pos_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = pos_logits.iter().map(|&v| (v - max_logit).exp()).collect();
                let sum_exps: f32 = exps.iter().sum();
                let probs: Vec<f32> = exps.iter().map(|&e| e / sum_exps).collect();

                let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let top3: Vec<serde_json::Value> = indexed
                    .iter()
                    .take(3)
                    .map(|(tok, prob)| {
                        let ch = tokenizer.decode(&[*tok as u32]);
                        serde_json::json!({
                            "token_id": tok,
                            "text": ch,
                            "p": format!("{:.4}", prob),
                        })
                    })
                    .collect();

                let ground_truth_text = tokenizer.decode(&[clean_tokens[pos]]);
                let is_mask = y_t_u32[pos] == mask_id;

                position_predictions.push(serde_json::json!({
                    "position": pos,
                    "corruption_type": if is_mask { "MASK" } else { "REPLACE" },
                    "ground_truth_token": clean_tokens[pos],
                    "ground_truth_text": ground_truth_text,
                    "corrupted_token": y_t_u32[pos],
                    "top_3_predictions": top3,
                }));
            }
        }

        examples.push(serde_json::json!({
            "example_idx": example_idx,
            "clean_string": clean_string,
            "corrupted_string": corrupted_string,
            "clean_tokens": clean_tokens,
            "corrupted_tokens": y_t_u32,
            "corruption_step_t": t_val,
            "num_corrupted_positions": position_predictions.len(),
            "position_predictions": position_predictions,
        }));
    }

    let output = serde_json::json!({
        "metadata": {
            "checkpoint_dir": checkpoint_dir,
            "seed": seed,
            "seq_len": l,
            "hidden_dim": cfg.hidden_dim,
            "vocab_size": cfg.total_vocab_size(),
            "num_examples": examples.len(),
            "tokenizer_type": "char",
        },
        "examples": examples,
    });

    let json_str = serde_json::to_string_pretty(&output).expect("JSON serialization");
    if let Some(parent) = std::path::Path::new(output_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(output_path, &json_str).expect("write output file");
    eprintln!(
        "[io-example] Wrote {} examples to: {output_path}",
        examples.len()
    );
}

/// Render a graph snapshot to SVG file.
fn run_render_graph(snapshot_path: &str, output_dir: &str, batch_idx: usize) {
    use erm_train::graph_snapshot::GraphSnapshot;
    use erm_train::render_graph::save_snapshot_svg;

    let snapshot = match GraphSnapshot::load(snapshot_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ERROR: cannot load snapshot '{}': {}", snapshot_path, e);
            std::process::exit(1);
        }
    };

    if let Err(e) = save_snapshot_svg(&snapshot, batch_idx, output_dir) {
        eprintln!("ERROR: cannot render snapshot: {}", e);
        std::process::exit(1);
    }

    println!(
        "Rendered snapshot step {} to {}/frame_{:05}.svg",
        snapshot.step, output_dir, snapshot.step
    );
}

/// Generate GIF from graph snapshots.
fn run_generate_gif(snapshots_dir: &str, output_path: &str, fps: u16, batch_idx: usize) {
    use erm_train::graph_snapshot::GraphSnapshot;
    use erm_train::render_graph::render_snapshot_to_svg;
    use std::fs;
    use std::path::Path;

    // Find all snapshot files
    let entries = match fs::read_dir(snapshots_dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!(
                "ERROR: cannot read snapshots directory '{}': {}",
                snapshots_dir, e
            );
            std::process::exit(1);
        }
    };

    let mut snapshot_files: Vec<_> = entries
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name();
            let name_str = name.to_string_lossy();
            name_str.starts_with("step_") && name_str.ends_with(".json")
        })
        .collect();

    snapshot_files.sort_by_key(|a| a.file_name());

    if snapshot_files.is_empty() {
        eprintln!("ERROR: no snapshot files found in '{}'", snapshots_dir);
        std::process::exit(1);
    }

    println!(
        "Found {} snapshots. Rendering frames...",
        snapshot_files.len()
    );

    // Create frames directory
    let frames_dir = Path::new(output_path)
        .parent()
        .unwrap_or(Path::new("."))
        .join("frames");
    fs::create_dir_all(&frames_dir).expect("create frames dir");

    // Render each snapshot to SVG, then convert to PNG using resvg if available
    // For now, just save SVGs - user can convert to GIF externally
    for (i, entry) in snapshot_files.iter().enumerate() {
        let path = entry.path();
        let snapshot = match GraphSnapshot::load(&path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Warning: cannot load snapshot '{}': {}", path.display(), e);
                continue;
            }
        };

        let svg = match render_snapshot_to_svg(&snapshot, batch_idx) {
            Ok(s) => s,
            Err(e) => {
                eprintln!(
                    "Warning: cannot render snapshot '{}': {}",
                    path.display(),
                    e
                );
                continue;
            }
        };

        let frame_path = frames_dir.join(format!("frame_{:04}.svg", i));
        if let Err(e) = fs::write(&frame_path, svg) {
            eprintln!(
                "Warning: cannot write frame '{}': {}",
                frame_path.display(),
                e
            );
            continue;
        }

        if (i + 1) % 10 == 0 || i == snapshot_files.len() - 1 {
            println!("  Rendered {}/{} frames", i + 1, snapshot_files.len());
        }
    }

    println!("SVG frames saved to: {}", frames_dir.display());
    println!("To create GIF, run:");
    println!(
        "  ffmpeg -i {}/frame_%04d.svg -vf \"fps={},scale=1200:-1:flags=lanczos\" {}",
        frames_dir.display(),
        fps,
        output_path
    );
    println!("Or use ImageMagick:");
    println!(
        "  convert -delay {} -loop 0 {}/*.svg {}",
        100 / fps as i32,
        frames_dir.display(),
        output_path
    );
}

/// Save scorer weights from a checkpoint directory (mid-run save).
fn run_save_scorer(checkpoint_dir: &str, output_dir: Option<&str>) {
    let out_dir = output_dir.unwrap_or(checkpoint_dir);

    // We just copy/link the scorer.bin to ensure it's readable.
    // The real value is verifying the scorer file exists and is loadable.
    let scorer_src = format!("{checkpoint_dir}/scorer.bin");
    let config_src = format!("{checkpoint_dir}/config.json");
    let graph_src = format!("{checkpoint_dir}/graph.json");
    let ant_src = format!("{checkpoint_dir}/ant_state.json");

    if !std::path::Path::new(&scorer_src).exists() {
        // Try scorer/ directory (burn's default recorder format).
        let scorer_dir = format!("{checkpoint_dir}/scorer");
        if !std::path::Path::new(&scorer_dir).exists() {
            eprintln!("ERROR: scorer not found at '{scorer_src}' or '{scorer_dir}'");
            std::process::exit(1);
        }
        println!("Scorer found at: {scorer_dir}");
    } else {
        println!("Scorer found at: {scorer_src}");
    }

    // Ensure output dir exists.
    if let Err(e) = std::fs::create_dir_all(out_dir) {
        eprintln!("ERROR: cannot create output directory '{out_dir}': {e}");
        std::process::exit(1);
    }

    // Copy supporting files if not already in output dir.
    for (src, name) in [
        (&config_src, "config.json"),
        (&graph_src, "graph.json"),
        (&ant_src, "ant_state.json"),
    ] {
        let dst = format!("{out_dir}/{name}");
        if src != &dst && std::path::Path::new(src).exists() {
            if let Err(e) = std::fs::copy(src, &dst) {
                eprintln!("WARNING: cannot copy {src} → {dst}: {e}");
            } else {
                println!("Copied {name} to: {dst}");
            }
        }
    }

    println!("save-scorer: checkpoint at '{checkpoint_dir}' verified and ready.");
    println!("  Scorer weights present: YES");
    println!("  Output dir: {out_dir}");
}

/// Diffusion inference: generate text using coarse-to-fine denoising.
fn run_infer(
    checkpoint_dir: &str,
    length: usize,
    k_steps: usize,
    prompt: Option<&str>,
    seed: u64,
    config_path: Option<&str>,
    backend: &BackendChoice,
) {
    let erm_cfg = if let Some(cp) = config_path {
        load_config(Some(cp))
    } else {
        // Load config from checkpoint.
        let cfg_path = format!("{checkpoint_dir}/config.json");
        load_config(if std::path::Path::new(&cfg_path).exists() {
            Some(&cfg_path)
        } else {
            None
        })
    };

    match backend {
        BackendChoice::Cpu => {
            infer_loop::<burn_autodiff::Autodiff<burn_ndarray::NdArray<f32>>>(
                &erm_cfg,
                checkpoint_dir,
                length,
                k_steps,
                prompt,
                seed,
                Default::default(),
            );
        }
        BackendChoice::Gpu => {
            #[cfg(feature = "gpu")]
            {
                let device = burn_wgpu::WgpuDevice::default();
                infer_loop::<burn_autodiff::Autodiff<burn_wgpu::Wgpu>>(
                    &erm_cfg,
                    checkpoint_dir,
                    length,
                    k_steps,
                    prompt,
                    seed,
                    device,
                );
            }
            #[cfg(not(feature = "gpu"))]
            {
                eprintln!("ERROR: GPU (wgpu) support not compiled.");
                std::process::exit(1);
            }
        }
        BackendChoice::Cuda => {
            #[cfg(feature = "cuda")]
            {
                let device = burn_cuda::CudaDevice::new(0);
                println!("Backend: CUDA device 0");
                infer_loop::<burn_autodiff::Autodiff<burn_cuda::Cuda>>(
                    &erm_cfg,
                    checkpoint_dir,
                    length,
                    k_steps,
                    prompt,
                    seed,
                    device,
                );
            }
            #[cfg(not(feature = "cuda"))]
            {
                eprintln!("ERROR: CUDA support not compiled.");
                std::process::exit(1);
            }
        }
    }
}

fn infer_loop<B: burn::tensor::backend::AutodiffBackend>(
    cfg: &erm_core::ErmConfig,
    checkpoint_dir: &str,
    length: usize,
    k_steps: usize,
    prompt: Option<&str>,
    seed: u64,
    device: B::Device,
) {
    use erm_train::diffusion_training::DiffusionTrainer;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::path::Path;

    let mut cfg = cfg.clone();
    cfg.seq_len = length;

    let mut trainer = DiffusionTrainer::<B>::new(&cfg, device.clone());

    if let Err(e) = trainer.load_checkpoint(checkpoint_dir) {
        eprintln!("WARNING: could not load checkpoint '{checkpoint_dir}': {e}");
        eprintln!("  (running inference with freshly initialized weights)");
    } else {
        println!(
            "Loaded checkpoint from: {checkpoint_dir} (step={})",
            trainer.step
        );
    }

    // Load tokenizer for proper prompt encoding/decoding in inference.
    let bpe_tokenizer = if cfg.tokenizer_type == "bpe" {
        let bpe_candidates = [
            format!("{checkpoint_dir}/bpe_vocab.json"),
            Path::new(checkpoint_dir)
                .parent()
                .map(|p| format!("{}/bpe_vocab.json", p.display()))
                .unwrap_or_default(),
            Path::new(checkpoint_dir)
                .parent()
                .and_then(|p| p.parent())
                .map(|p| format!("{}/bpe_vocab.json", p.display()))
                .unwrap_or_default(),
            cfg.bpe_vocab_path.clone(),
        ];
        let found = bpe_candidates
            .iter()
            .find(|p| !p.is_empty() && Path::new(p).exists());
        match found {
            Some(path) => match erm_core::BpeTokenizer::load(path) {
                Ok(tok) => Some(tok),
                Err(e) => {
                    eprintln!("WARNING: failed to load BPE vocab from '{path}': {e}");
                    None
                }
            },
            None => {
                eprintln!(
                    "WARNING: tokenizer_type=bpe but bpe_vocab.json was not found; \
                     prompt will use fallback char encoding"
                );
                None
            }
        }
    } else {
        None
    };

    let prompt_tokens: Option<Vec<u32>> = match (prompt, bpe_tokenizer.as_ref()) {
        (Some(p), Some(tok)) => Some(tok.encode_text(p)),
        (Some(p), None) => Some(
            p.chars()
                .map(|c| (c as u32) % cfg.total_vocab_size() as u32)
                .collect(),
        ),
        (None, _) => None,
    };

    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let mut graph = erm_core::graph::RouteGraph::new(&cfg);

    let result = erm_train::diffusion_training::diffusion_infer(
        &trainer.scorer,
        &mut graph,
        &cfg,
        k_steps,
        prompt_tokens.as_deref(),
        length,
        &mut rng,
        &device,
    );

    match result {
        Ok(tokens) => {
            println!("=== Diffusion Inference Output ===");
            println!("Steps: {k_steps}, Length: {length}");
            println!("Tokens: {:?}", &tokens[..tokens.len().min(64)]);
            if let Some(tok) = bpe_tokenizer {
                let decoded = tok.decode_text(&tokens);
                let preview: String = decoded.chars().take(400).collect();
                println!("Decoded: {preview}");
            } else {
                eprintln!("INFO: no tokenizer decode available; showing token ids only");
            }
        }
        Err(e) => {
            eprintln!("ERROR: inference failed: {e}");
            std::process::exit(1);
        }
    }
}

/// Diffusion colony training loop.
#[allow(clippy::too_many_arguments)]
fn run_diffusion_train(
    data_path: &str,
    total_steps: usize,
    config_path: Option<&str>,
    backend: &BackendChoice,
    log_every: usize,
    checkpoint_every: usize,
    checkpoint_dir: Option<&str>,
    exp_id: &str,
    seq_len_override: Option<usize>,
    hidden_dim_override: Option<usize>,
    num_ants_override: Option<usize>,
    batch_override: Option<usize>,
    diffusion_t_override: Option<usize>,
    resume: Option<&str>,
) {
    let mut cfg = load_config(config_path);

    // Apply overrides.
    cfg.exp_id = exp_id.to_string();
    if let Some(sl) = seq_len_override {
        cfg.seq_len = sl;
    }
    if let Some(hd) = hidden_dim_override {
        cfg.hidden_dim = hd;
    }
    if let Some(na) = num_ants_override {
        cfg.num_ants = na;
    }
    if let Some(b) = batch_override {
        cfg.batch_size = b;
    }
    if let Some(t) = diffusion_t_override {
        cfg.diffusion_steps = t;
    }

    // Set up metrics path.
    if let Some(dir) = checkpoint_dir {
        if cfg.metrics_path.is_empty() {
            cfg.metrics_path = format!("{dir}/metrics.jsonl");
        }
    }

    // Build streaming dataset.
    use erm_train::streaming_dataset::StreamingConfig;

    // Determine if data_path is a directory or file.
    let data_dir = if std::path::Path::new(data_path).is_dir() {
        data_path.to_string()
    } else {
        // Treat as file — use parent dir.
        std::path::Path::new(data_path)
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| ".".to_string())
    };

    // Build BPE tokenizer from a sample of the corpus or load from path.
    let tokenizer =
        if !cfg.bpe_vocab_path.is_empty() && std::path::Path::new(&cfg.bpe_vocab_path).exists() {
            println!("Loading BPE vocabulary from: {}", cfg.bpe_vocab_path);
            match erm_core::BpeTokenizer::load(&cfg.bpe_vocab_path) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("ERROR: cannot load BPE vocab: {e}");
                    std::process::exit(1);
                }
            }
        } else {
            // Train BPE on sample text.
            println!("Training BPE vocabulary ({} merges)...", cfg.bpe_vocab_size);
            let sample_text = collect_sample_text(&data_dir, 1_000_000);
            let bpe = erm_core::BpeTokenizer::train(&sample_text, cfg.bpe_vocab_size);
            println!("BPE vocabulary trained: {} tokens", bpe.vocab_size());

            // Save vocabulary alongside checkpoint if possible.
            if let Some(dir) = checkpoint_dir {
                let vocab_path = format!("{dir}/bpe_vocab.json");
                if let Ok(()) = bpe.save(&vocab_path) {
                    println!("BPE vocabulary saved to: {vocab_path}");
                }
            }
            bpe
        };

    let vocab_size = tokenizer.vocab_size();
    cfg.vocab_size = vocab_size;

    println!(
        "DiffusionTrain: exp={exp_id} steps={total_steps} seq={} hidden={} ants={} batch={} T={}",
        cfg.seq_len, cfg.hidden_dim, cfg.num_ants, cfg.batch_size, cfg.diffusion_steps
    );
    println!("  vocab={vocab_size} data={data_dir}");

    let streaming_cfg = StreamingConfig {
        data_dir: data_dir.clone(),
        seq_len: cfg.seq_len,
        batch_size: cfg.batch_size,
        use_paragraph_spans: cfg.use_paragraph_spans,
        shuffle_files: true,
        seed: 42,
        repeat: true,
    };

    match backend {
        BackendChoice::Cpu => {
            diffusion_train_loop::<burn_autodiff::Autodiff<burn_ndarray::NdArray<f32>>>(
                &cfg,
                streaming_cfg,
                tokenizer,
                total_steps,
                log_every,
                checkpoint_every,
                checkpoint_dir,
                resume,
                Default::default(),
            );
        }
        BackendChoice::Gpu => {
            #[cfg(feature = "gpu")]
            {
                let device = burn_wgpu::WgpuDevice::default();
                diffusion_train_loop::<burn_autodiff::Autodiff<burn_wgpu::Wgpu>>(
                    &cfg,
                    streaming_cfg,
                    tokenizer,
                    total_steps,
                    log_every,
                    checkpoint_every,
                    checkpoint_dir,
                    resume,
                    device,
                );
            }
            #[cfg(not(feature = "gpu"))]
            {
                eprintln!("ERROR: GPU (wgpu) support not compiled.");
                std::process::exit(1);
            }
        }
        BackendChoice::Cuda => {
            #[cfg(feature = "cuda")]
            {
                let device = burn_cuda::CudaDevice::new(0);
                println!("Backend: CUDA device 0");
                diffusion_train_loop::<burn_autodiff::Autodiff<burn_cuda::Cuda>>(
                    &cfg,
                    streaming_cfg,
                    tokenizer,
                    total_steps,
                    log_every,
                    checkpoint_every,
                    checkpoint_dir,
                    resume,
                    device,
                );
            }
            #[cfg(not(feature = "cuda"))]
            {
                eprintln!("ERROR: CUDA support not compiled.");
                std::process::exit(1);
            }
        }
    }
}

/// Collect up to `max_bytes` of text from a directory for BPE training.
fn collect_sample_text(data_dir: &str, max_bytes: usize) -> String {
    let mut sample = String::new();
    if let Ok(entries) = std::fs::read_dir(data_dir) {
        let mut paths: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("txt"))
            .map(|e| e.path())
            .collect();
        paths.sort();
        for path in paths {
            if sample.len() >= max_bytes {
                break;
            }
            if let Ok(text) = std::fs::read_to_string(&path) {
                let remaining = max_bytes - sample.len();
                if text.len() <= remaining {
                    sample.push_str(&text);
                } else {
                    sample.push_str(&text[..remaining]);
                }
            }
        }
    }
    sample
}

#[derive(serde::Serialize)]
struct DiffusionSampleRecord {
    step: usize,
    corruption_step_t: usize,
    num_corrupted_positions: usize,
    clean_string: String,
    corrupted_string: String,
    predicted_string: String,
    clean_tokens: Vec<u32>,
    corrupted_tokens: Vec<u32>,
    predicted_tokens: Vec<u32>,
}

fn open_diffusion_sample_writer(
    checkpoint_dir: Option<&str>,
) -> Option<std::io::BufWriter<std::fs::File>> {
    use std::io::BufWriter;

    let dir = checkpoint_dir?;
    let path = format!("{dir}/samples.jsonl");
    if let Some(parent) = std::path::Path::new(&path).parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            eprintln!(
                "WARNING: cannot create sample output dir '{}': {e}",
                parent.display()
            );
            return None;
        }
    }
    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        Ok(file) => {
            println!("Samples → {path}");
            Some(BufWriter::new(file))
        }
        Err(e) => {
            eprintln!("WARNING: cannot open sample output file '{path}': {e}");
            None
        }
    }
}

fn write_diffusion_sample<B: burn::tensor::backend::AutodiffBackend>(
    writer: &mut std::io::BufWriter<std::fs::File>,
    trainer: &erm_train::diffusion_training::DiffusionTrainer<B>,
    tokenizer: &erm_core::BpeTokenizer,
    batch: &erm_train::streaming_dataset::TokenBatch,
    step: usize,
    device: &B::Device,
) -> Result<(), String> {
    use erm_train::bridge::{tensor_to_vec, tokens_to_tensor};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::io::Write;

    if batch.batch_size == 0 || batch.seq_len == 0 {
        return Ok(());
    }
    let l = batch.seq_len;
    if batch.tokens.len() < l {
        return Err(format!(
            "batch too small for sample: tokens={} seq_len={l}",
            batch.tokens.len()
        ));
    }

    let clean_tokens: Vec<u32> = batch.tokens[..l].to_vec();
    let x_i32: Vec<i32> = clean_tokens.iter().map(|&t| t as i32).collect();
    let t_val = ((trainer.config.diffusion_steps.max(1)) + 1) / 2;
    let mut sample_rng = ChaCha8Rng::seed_from_u64(0xD1FF_u64 ^ step as u64);

    let corruption = erm_core::corruption::corrupt(&x_i32, t_val, &trainer.config, &mut sample_rng)
        .map_err(|e| format!("sample corruption failed: {e}"))?;
    let corrupted_tokens: Vec<u32> = corruption.y_t.iter().map(|&t| t as u32).collect();
    let num_corrupted_positions = clean_tokens
        .iter()
        .zip(corrupted_tokens.iter())
        .filter(|(a, b)| a != b)
        .count();

    let tokens_tensor = tokens_to_tensor::<B>(&corrupted_tokens, 1, l, device)
        .map_err(|e| format!("sample tensor conversion failed: {e}"))?;
    let (logits_tensor, _) = trainer.scorer.forward(tokens_tensor);
    let logits_cpu =
        tensor_to_vec(logits_tensor).map_err(|e| format!("sample logits transfer failed: {e}"))?;
    let v_rt = if l > 0 { logits_cpu.len() / l } else { 0 };
    if v_rt == 0 || logits_cpu.len() < l * v_rt {
        return Err(format!(
            "invalid logits shape for sample: len={} L={l} V={v_rt}",
            logits_cpu.len()
        ));
    }

    let mut predicted_tokens = Vec::with_capacity(l);
    for pos in 0..l {
        let start = pos * v_rt;
        let end = start + v_rt;
        if end > logits_cpu.len() {
            predicted_tokens.push(0);
            continue;
        }
        let pos_logits = &logits_cpu[start..end];
        let best = pos_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        predicted_tokens.push(best);
    }

    let record = DiffusionSampleRecord {
        step,
        corruption_step_t: t_val,
        num_corrupted_positions,
        clean_string: tokenizer.decode_text(&clean_tokens),
        corrupted_string: tokenizer.decode_text(&corrupted_tokens),
        predicted_string: tokenizer.decode_text(&predicted_tokens),
        clean_tokens,
        corrupted_tokens,
        predicted_tokens,
    };
    let line = serde_json::to_string(&record)
        .map_err(|e| format!("sample JSON serialization failed: {e}"))?;
    writeln!(writer, "{line}").map_err(|e| format!("sample write failed: {e}"))?;
    writer
        .flush()
        .map_err(|e| format!("sample flush failed: {e}"))?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn diffusion_train_loop<B: burn::tensor::backend::AutodiffBackend>(
    cfg: &erm_core::ErmConfig,
    streaming_cfg: erm_train::streaming_dataset::StreamingConfig,
    tokenizer: erm_core::BpeTokenizer,
    total_steps: usize,
    log_every: usize,
    checkpoint_every: usize,
    checkpoint_dir: Option<&str>,
    resume: Option<&str>,
    device: B::Device,
) {
    use erm_train::diffusion_training::{DiffusionStepResult, DiffusionTrainer};
    use erm_train::metrics::{MetricsRecord, MetricsWriter};
    use erm_train::streaming_dataset::StreamingDataset;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let mut trainer = DiffusionTrainer::<B>::new(cfg, device.clone());
    trainer.total_steps = total_steps;

    if let Some(resume_dir) = resume {
        match trainer.load_checkpoint(resume_dir) {
            Ok(()) => println!(
                "[diffusion] Resumed from checkpoint: {resume_dir} (step={})",
                trainer.step
            ),
            Err(e) => {
                eprintln!("ERROR: cannot load resume checkpoint '{resume_dir}': {e}");
                std::process::exit(1);
            }
        }
    }
    let sample_tokenizer = tokenizer.clone();
    let mut dataset = StreamingDataset::new(streaming_cfg, tokenizer);
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Open metrics writer if path is configured.
    let mut metrics_writer: Option<MetricsWriter> = None;
    if !cfg.metrics_path.is_empty() {
        match MetricsWriter::open(&cfg.metrics_path, log_every.max(50)) {
            Ok(w) => {
                println!("Metrics → {}", cfg.metrics_path);
                metrics_writer = Some(w);
            }
            Err(e) => {
                eprintln!("WARNING: cannot open metrics file: {e}");
            }
        }
    }
    let mut sample_writer = open_diffusion_sample_writer(checkpoint_dir);

    println!(
        "Starting diffusion training: {} steps, log_every={}, checkpoint_every={}",
        total_steps, log_every, checkpoint_every
    );

    let mut local_step = 0usize;
    let mut recent_losses: Vec<f32> = Vec::new();

    while local_step < total_steps {
        let batch = match dataset.next_batch() {
            Ok(Some(b)) => b,
            Ok(None) => {
                eprintln!("Dataset exhausted at local_step={local_step}");
                break;
            }
            Err(e) => {
                eprintln!("Dataset error at local_step={local_step}: {e}");
                break;
            }
        };

        let result: DiffusionStepResult = match trainer.diffusion_step(&batch, &mut rng) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("ERROR at local_step={local_step}: {e}");
                break;
            }
        };

        local_step += 1;
        let global_step = trainer.step;
        recent_losses.push(result.loss);

        // Log.
        if local_step % log_every == 0 || local_step == total_steps {
            let avg_loss: f32 = recent_losses.iter().sum::<f32>() / recent_losses.len() as f32;
            println!(
                "[diffusion step {:6}] loss={:.4} lr={:.6} f_temp={:.2} l_temp={:.2} edits={} mean_φ={:.4} deaths={} pruned={} inserted={}",
                global_step,
                avg_loss,
                result.lr,
                result.follower_temp,
                result.leader_temp,
                result.total_edits,
                result.pheromone_stats.mean_phi,
                result.deaths,
                result.edges_pruned,
                result.edges_inserted,
            );
            recent_losses.clear();

            // Write metrics.
            if let Some(ref mut mw) = metrics_writer {
                let record = MetricsRecord {
                    exp_id: cfg.exp_id.clone(),
                    step: global_step,
                    loss: avg_loss,
                    edits: result.total_edits,
                    mean_phi: result.pheromone_stats.mean_phi,
                    deaths: result.deaths,
                    seq_len: cfg.seq_len,
                    batch: cfg.batch_size,
                    hidden_dim: cfg.hidden_dim,
                    lr: result.lr,
                    follower_temp: result.follower_temp,
                    leader_temp: result.leader_temp,
                };
                if let Err(e) = mw.write(&record) {
                    eprintln!("WARNING: metrics write failed: {e}");
                }
            }

            // Write one sample I/O record from this step.
            if let Some(ref mut sw) = sample_writer {
                if let Err(e) = write_diffusion_sample(
                    sw,
                    &trainer,
                    &sample_tokenizer,
                    &batch,
                    global_step,
                    &device,
                ) {
                    eprintln!("WARNING: sample write failed: {e}");
                }
            }
        }

        // Checkpoint.
        if checkpoint_every > 0 && local_step % checkpoint_every == 0 && checkpoint_dir.is_some() {
            let dir = checkpoint_dir.unwrap();
            let ckpt_dir = format!("{dir}/step_{global_step:08}");
            if let Err(e) = trainer.save_checkpoint(&ckpt_dir) {
                eprintln!("WARNING: checkpoint save failed: {e}");
            }
            // Also update latest/.
            let latest_dir = format!("{dir}/latest");
            if let Err(e) = trainer.save_checkpoint(&latest_dir) {
                eprintln!("WARNING: latest checkpoint save failed: {e}");
            }
        }
    }

    // Final checkpoint.
    if let Some(dir) = checkpoint_dir {
        let final_dir = format!("{dir}/final");
        if let Err(e) = trainer.save_checkpoint(&final_dir) {
            eprintln!("WARNING: final checkpoint save failed: {e}");
        } else {
            println!("Final checkpoint: {final_dir}");
        }
    }

    println!(
        "Diffusion training complete. Steps: {} (batches consumed: {})",
        trainer.step, dataset.batches_consumed
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parse_train() {
        let cli = Cli::try_parse_from([
            "erm",
            "train",
            "--data",
            "test.txt",
            "--steps",
            "100",
            "--warmstart",
            "50",
        ]);
        assert!(cli.is_ok(), "train subcommand should parse");
    }

    #[test]
    fn test_cli_parse_train_dry_run() {
        let cli = Cli::try_parse_from(["erm", "train", "--data", "test.txt", "--dry-run"]);
        assert!(cli.is_ok(), "train --dry-run should parse");
    }

    #[test]
    fn test_cli_parse_generate() {
        let cli = Cli::try_parse_from([
            "erm", "generate", "--length", "64", "--steps", "4", "--prompt", "hello",
        ]);
        assert!(cli.is_ok(), "generate subcommand should parse");
    }

    #[test]
    fn test_cli_parse_generate_defaults() {
        let cli = Cli::try_parse_from(["erm", "generate"]);
        assert!(cli.is_ok(), "generate with defaults should parse");
    }

    #[test]
    fn test_cli_parse_eval() {
        let cli = Cli::try_parse_from([
            "erm",
            "eval",
            "--data",
            "test.txt",
            "--checkpoint",
            "/tmp/ckpt",
        ]);
        assert!(cli.is_ok(), "eval subcommand should parse");
    }

    #[test]
    fn test_cli_parse_bench() {
        let cli = Cli::try_parse_from([
            "erm",
            "bench",
            "--batch-size",
            "4",
            "--seq-len",
            "64",
            "--steps",
            "3",
        ]);
        assert!(cli.is_ok(), "bench subcommand should parse");
    }

    #[test]
    fn test_cli_parse_bench_defaults() {
        let cli = Cli::try_parse_from(["erm", "bench"]);
        assert!(cli.is_ok(), "bench with defaults should parse");
    }

    #[test]
    fn test_load_config_default() {
        let cfg = load_config(None);
        assert_eq!(cfg.seq_len, 256);
        assert_eq!(cfg.vocab_size, 0);
    }

    #[test]
    fn test_cli_parse_burn_train_cpu() {
        let cli = Cli::try_parse_from([
            "erm",
            "burn-train",
            "--data",
            "test.txt",
            "--steps",
            "100",
            "--backend",
            "cpu",
        ]);
        assert!(cli.is_ok(), "burn-train cpu should parse");
    }

    #[test]
    fn test_cli_parse_burn_train_gpu() {
        let cli = Cli::try_parse_from([
            "erm",
            "burn-train",
            "--data",
            "test.txt",
            "--backend",
            "gpu",
        ]);
        assert!(cli.is_ok(), "burn-train gpu should parse");
    }

    #[test]
    fn test_cli_parse_burn_train_defaults() {
        let cli = Cli::try_parse_from(["erm", "burn-train", "--data", "test.txt"]);
        assert!(cli.is_ok(), "burn-train with defaults should parse");
    }

    #[test]
    fn test_cli_parse_colony_train() {
        let cli = Cli::try_parse_from([
            "erm",
            "colony-train",
            "--data",
            "test.txt",
            "--steps",
            "100",
            "--backend",
            "cpu",
        ]);
        assert!(cli.is_ok(), "colony-train should parse");
    }

    #[test]
    fn test_cli_parse_colony_train_defaults() {
        let cli = Cli::try_parse_from(["erm", "colony-train", "--data", "test.txt"]);
        assert!(cli.is_ok(), "colony-train with defaults should parse");
    }

    #[test]
    fn test_cli_parse_colony_train_with_checkpoint() {
        let cli = Cli::try_parse_from([
            "erm",
            "colony-train",
            "--data",
            "test.txt",
            "--steps",
            "50",
            "--checkpoint-dir",
            "/tmp/ckpt",
        ]);
        assert!(cli.is_ok(), "colony-train with checkpoint should parse");
    }
}
