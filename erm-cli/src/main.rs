//! ERM CLI — command-line interface for the Emergent Route Model.
//!
//! # Subcommands
//!
//! | Subcommand | Description |
//! |------------|-------------|
//! | `train`    | Run warm-start + colony training from text or directory |
//! | `eval`     | Evaluate a checkpoint: denoising accuracy and generation metrics |
//! | `generate` | Generate token sequences from a checkpoint using the scorer |
//! | `info`     | Print model architecture info for a checkpoint |

use std::process;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use erm_core::config::ErmConfig;
use erm_core::scorer::Scorer;
use erm_core::tokenizer::CharTokenizer;
use erm_train::comparison::{save_comparison_report, ComparisonMetrics};
use erm_train::dataset::TextDataset;
use erm_train::eval::{evaluate_denoising, evaluate_generation};
use erm_train::orchestrator::{Orchestrator, TrainingConfig};

#[derive(Parser, Debug)]
#[command(
    name = "erm",
    version,
    about = "Emergent Route Model: training, evaluation, and generation"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Train the ERM model (warm-start denoiser, then colony phase).
    Train {
        /// Path to a single `.txt` file or a directory of `.txt` files.
        #[arg(long, short = 'd')]
        data: String,

        /// Output directory for checkpoints and loss log.
        #[arg(long, short = 'o', default_value = "./erm-checkpoints")]
        output: String,

        /// Path to a JSON training config file. Uses defaults if omitted.
        #[arg(long, short = 'c')]
        config: Option<String>,

        /// Number of warm-start steps (overrides config if set).
        #[arg(long)]
        warm_steps: Option<usize>,

        /// Number of colony steps (overrides config if set).
        #[arg(long)]
        colony_steps: Option<usize>,

        /// RNG seed.
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Dry run: validate config and exit without training.
        #[arg(long)]
        dry_run: bool,
    },

    /// Evaluate a checkpoint on a text dataset.
    Eval {
        /// Path to a checkpoint directory (produced by `train`).
        #[arg(long, short = 'k')]
        checkpoint: String,

        /// Path to evaluation `.txt` file or directory.
        #[arg(long, short = 'd')]
        data: String,

        /// Number of evaluation batches.
        #[arg(long, default_value = "10")]
        num_batches: usize,

        /// Save a comparison report JSON to this path (optional).
        #[arg(long)]
        report: Option<String>,

        /// RNG seed.
        #[arg(long, default_value = "0")]
        seed: u64,
    },

    /// Generate token sequences from a checkpoint.
    Generate {
        /// Path to a checkpoint directory.
        #[arg(long, short = 'k')]
        checkpoint: String,

        /// Number of sequences to generate.
        #[arg(long, default_value = "4")]
        num_sequences: usize,

        /// Sequence length in tokens (uses checkpoint config if not set).
        #[arg(long)]
        length: Option<usize>,

        /// RNG seed.
        #[arg(long, default_value = "42")]
        seed: u64,
    },

    /// Print model architecture info for a config or checkpoint.
    Info {
        /// Path to a JSON config file or checkpoint directory.
        /// If omitted, prints default config info.
        #[arg(long)]
        source: Option<String>,
    },
}

fn main() {
    let cli = Cli::parse();
    if let Err(e) = run(cli) {
        eprintln!("error: {e:#}");
        process::exit(1);
    }
}

fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Train {
            data,
            output,
            config,
            warm_steps,
            colony_steps,
            seed,
            dry_run,
        } => cmd_train(
            &data,
            &output,
            config.as_deref(),
            warm_steps,
            colony_steps,
            seed,
            dry_run,
        ),
        Commands::Eval {
            checkpoint,
            data,
            num_batches,
            report,
            seed,
        } => cmd_eval(&checkpoint, &data, num_batches, report.as_deref(), seed),
        Commands::Generate {
            checkpoint,
            num_sequences,
            length,
            seed,
        } => cmd_generate(&checkpoint, num_sequences, length, seed),
        Commands::Info { source } => cmd_info(source.as_deref()),
    }
}

// ── train ───────────────────────────────────────────────────────────────────

fn cmd_train(
    data: &str,
    output: &str,
    config_path: Option<&str>,
    warm_steps: Option<usize>,
    colony_steps: Option<usize>,
    seed: u64,
    dry_run: bool,
) -> Result<()> {
    let mut train_cfg = if let Some(path) = config_path {
        let json =
            std::fs::read_to_string(path).with_context(|| format!("reading config {path}"))?;
        serde_json::from_str::<TrainingConfig>(&json)
            .with_context(|| format!("parsing config {path}"))?
    } else {
        TrainingConfig {
            seed,
            ..TrainingConfig::default()
        }
    };

    if let Some(w) = warm_steps {
        train_cfg.warm_start_steps = w;
    }
    if let Some(c) = colony_steps {
        train_cfg.colony_steps = c;
    }
    train_cfg.seed = seed;

    println!("ERM Training");
    println!("  data:         {data}");
    println!("  output:       {output}");
    println!("  warm_steps:   {}", train_cfg.warm_start_steps);
    println!("  colony_steps: {}", train_cfg.colony_steps);
    println!("  seed:         {}", train_cfg.seed);

    let corpus = load_corpus(data)?;
    let tokenizer = CharTokenizer::from_text(&corpus);
    let vocab = tokenizer.vocab_size();
    println!("  vocab_size:   {vocab}");

    train_cfg.erm.vocab_size = vocab;
    let seq_len = train_cfg.erm.seq_len;

    if dry_run {
        println!(
            "Dry run: config valid. Scorer params: {}",
            Scorer::new(&train_cfg.erm, vocab, seed).num_parameters()
        );
        return Ok(());
    }

    let dataset =
        TextDataset::from_text(&corpus, &tokenizer, seq_len).with_context(|| "building dataset")?;
    println!("  sequences:    {}", dataset.len());

    let mut orch = Orchestrator::new(train_cfg, vocab);
    orch.run_all(&dataset, Some(output))
        .map_err(|e| anyhow::anyhow!("training failed: {e}"))?;

    orch.save_checkpoint(output, "done")
        .map_err(|e| anyhow::anyhow!("checkpoint save failed: {e}"))?;

    let log_path = format!("{output}/loss_log.json");
    let log_json = serde_json::to_string_pretty(&orch.loss_log).context("serialising loss log")?;
    std::fs::write(&log_path, log_json)
        .with_context(|| format!("writing loss log to {log_path}"))?;

    println!("Training complete. Checkpoint saved to: {output}");
    println!("Loss log: {log_path}");
    Ok(())
}

// ── eval ────────────────────────────────────────────────────────────────────

fn cmd_eval(
    checkpoint: &str,
    data: &str,
    num_batches: usize,
    report_path: Option<&str>,
    seed: u64,
) -> Result<()> {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    println!("ERM Evaluation");
    println!("  checkpoint: {checkpoint}");
    println!("  data:       {data}");
    println!("  batches:    {num_batches}");

    let (orch, phase) = Orchestrator::load_checkpoint(checkpoint)
        .map_err(|e| anyhow::anyhow!("loading checkpoint: {e}"))?;
    println!("  loaded phase: {phase}, step: {}", orch.global_step);

    let corpus = load_corpus(data)?;
    let tokenizer = CharTokenizer::from_text(&corpus);
    let seq_len = orch.config.erm.seq_len;
    let dataset = TextDataset::from_text(&corpus, &tokenizer, seq_len)
        .with_context(|| "building eval dataset")?;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let denoise_metrics = evaluate_denoising(
        &orch.scorer,
        &dataset,
        &orch.config.erm,
        num_batches,
        None,
        &mut rng,
    )
    .map_err(|e| anyhow::anyhow!("denoising eval: {e}"))?;

    println!("\nDenoising evaluation:");
    println!(
        "  masked_token_accuracy: {:?}",
        denoise_metrics.masked_token_accuracy
    );
    println!("  avg_loss:              {:?}", denoise_metrics.avg_loss);
    println!("  num_positions:         {}", denoise_metrics.num_positions);

    let gen_metrics = evaluate_generation(&orch.scorer, &orch.config.erm, num_batches, &mut rng)
        .map_err(|e| anyhow::anyhow!("generation eval: {e}"))?;

    println!("\nGeneration evaluation:");
    println!("  token_entropy:      {:?}", gen_metrics.token_entropy);
    println!("  unique_token_ratio: {:?}", gen_metrics.unique_token_ratio);
    println!("  num_positions:      {}", gen_metrics.num_positions);

    if let Some(path) = report_path {
        let entries = vec![
            ComparisonMetrics::new("denoising", denoise_metrics),
            ComparisonMetrics::new("generation", gen_metrics),
        ];
        save_comparison_report(path, &entries)
            .map_err(|e| anyhow::anyhow!("saving report: {e}"))?;
        println!("\nComparison report saved to: {path}");
    }

    Ok(())
}

// ── generate ────────────────────────────────────────────────────────────────

fn cmd_generate(
    checkpoint: &str,
    num_sequences: usize,
    length: Option<usize>,
    seed: u64,
) -> Result<()> {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    println!("ERM Generate");
    println!("  checkpoint:     {checkpoint}");
    println!("  num_sequences:  {num_sequences}");

    let (orch, phase) = Orchestrator::load_checkpoint(checkpoint)
        .map_err(|e| anyhow::anyhow!("loading checkpoint: {e}"))?;
    println!("  loaded phase: {phase}");

    let seq_len = length.unwrap_or(orch.config.erm.seq_len);
    let cfg = &orch.config.erm;
    let v = orch.scorer.vocab_size;
    let mask_id = cfg.mask_token_id() as u32;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let _ = &mut rng;

    println!("  seq_len:        {seq_len}");
    println!("  vocab_size:     {v}\n");

    let fully_masked: Vec<u32> = vec![mask_id; num_sequences * seq_len];
    let output = orch
        .scorer
        .forward(&fully_masked, num_sequences)
        .map_err(|e| anyhow::anyhow!("scorer forward: {e}"))?;

    for b in 0..num_sequences {
        let mut tokens: Vec<u32> = Vec::with_capacity(seq_len);
        for pos in 0..seq_len {
            let logit_start = (b * seq_len + pos) * v;
            let logit_slice = &output.logits[logit_start..logit_start + v];
            let pred = argmax(logit_slice) as u32;
            tokens.push(pred);
        }
        println!("sequence {b}: {:?}", &tokens[..seq_len.min(32)]);
        if seq_len > 32 {
            println!("  ... ({} total tokens)", seq_len);
        }
    }

    Ok(())
}

// ── info ────────────────────────────────────────────────────────────────────

fn cmd_info(source: Option<&str>) -> Result<()> {
    let cfg = match source {
        None => {
            println!("(no source specified — showing default config)");
            ErmConfig::default()
        }
        Some(path) => {
            let config_path = format!("{path}/config.json");
            if std::path::Path::new(&config_path).exists() {
                let json = std::fs::read_to_string(&config_path)
                    .with_context(|| format!("reading {config_path}"))?;
                let train_cfg: TrainingConfig =
                    serde_json::from_str(&json).with_context(|| "parsing checkpoint config")?;
                println!("Loaded from checkpoint: {path}");
                if let Ok(meta_json) = std::fs::read_to_string(format!("{path}/step.json")) {
                    println!("Checkpoint meta: {meta_json}");
                }
                train_cfg.erm
            } else {
                let json =
                    std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?;
                serde_json::from_str::<ErmConfig>(&json)
                    .with_context(|| "parsing ErmConfig JSON")?
            }
        }
    };

    let vocab = cfg.total_vocab_size();
    let scorer = Scorer::new(&cfg, vocab, 0);
    let n_params = scorer.num_parameters();

    println!("\nERM Model Architecture");
    println!(
        "  vocab_size:      {} ({} with MASK)",
        cfg.vocab_size, vocab
    );
    println!("  seq_len:         {}", cfg.seq_len);
    println!("  hidden_dim:      {}", cfg.hidden_dim);
    println!("  num_blocks:      {}", cfg.num_blocks);
    println!("  num_heads:       {}", cfg.num_heads);
    println!("  mlp_expansion:   {}", cfg.mlp_expansion);
    println!("  num_ants:        {}", cfg.num_ants);
    println!("  refinement_steps:{}", cfg.refinement_steps);
    println!("  batch_size:      {}", cfg.batch_size);
    println!("  learning_rate:   {}", cfg.learning_rate);
    println!("\n  Total parameters: {n_params}");

    Ok(())
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn load_corpus(path: &str) -> Result<String> {
    let p = std::path::Path::new(path);
    if p.is_dir() {
        let mut files: Vec<String> = Vec::new();
        collect_txt_recursive(path, &mut files)?;
        files.sort();
        if files.is_empty() {
            anyhow::bail!("no .txt files found in directory: {path}");
        }
        let mut corpus = String::new();
        for f in &files {
            let content = std::fs::read_to_string(f).with_context(|| format!("reading {f}"))?;
            if !corpus.is_empty() {
                corpus.push('\n');
            }
            corpus.push_str(&content);
        }
        Ok(corpus)
    } else {
        std::fs::read_to_string(path).with_context(|| format!("reading {path}"))
    }
}

fn collect_txt_recursive(dir: &str, out: &mut Vec<String>) -> Result<()> {
    for entry in std::fs::read_dir(dir).with_context(|| format!("reading directory {dir}"))? {
        let entry = entry.context("directory entry")?;
        let path = entry.path();
        if path.is_dir() {
            let sub = path.to_str().context("non-UTF-8 path")?.to_string();
            collect_txt_recursive(&sub, out)?;
        } else if path.extension().and_then(|e| e.to_str()) == Some("txt") {
            out.push(path.to_str().context("non-UTF-8 path")?.to_string());
        }
    }
    Ok(())
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
            if v > bv {
                (i, v)
            } else {
                (bi, bv)
            }
        })
        .0
}
