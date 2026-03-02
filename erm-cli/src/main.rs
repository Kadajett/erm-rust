#![recursion_limit = "512"]
//! ERM CLI — command-line interface for the Emergent Route Model.
//!
//! Subcommands:
//! - `train` — run the training loop
//! - `generate` — generate text from scratch or with a prompt
//! - `eval` — evaluate a checkpoint (scorer forward + loss)
//! - `bench` — throughput and memory benchmarks

use clap::{Parser, Subcommand, ValueEnum};

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
}

/// Backend choice for burn-based training.
#[derive(Debug, Clone, ValueEnum)]
enum BackendChoice {
    /// CPU backend (NdArray — portable, no GPU required).
    Cpu,
    /// GPU backend (wgpu — requires Vulkan/Metal/DX12).
    Gpu,
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
                eprintln!("ERROR: GPU support not compiled. Rebuild with: cargo build --release --features gpu");
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
                eprintln!("ERROR: GPU support not compiled. Rebuild with: cargo build --release --features gpu");
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
        }
        Err(e) => {
            eprintln!("ERROR during colony training: {e}");
            std::process::exit(1);
        }
    }
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
        assert_eq!(cfg.seq_len, 128);
        assert_eq!(cfg.vocab_size, 16_384);
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
