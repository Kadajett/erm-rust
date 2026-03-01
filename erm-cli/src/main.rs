//! ERM CLI — command-line interface for the Emergent Route Model.
//!
//! Subcommands:
//! - `train` — run the training loop
//! - `eval` — evaluate a checkpoint
//! - `generate` — generate text from a checkpoint
//!
//! Currently a skeleton — full implementation in Phase 7.

use clap::{Parser, Subcommand};

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
    /// Train the model.
    Train {
        /// Path to config file (TOML or JSON).
        #[arg(long)]
        config: Option<String>,

        /// Path to training data.
        #[arg(long)]
        data: Option<String>,

        /// Output directory for checkpoints.
        #[arg(long)]
        output: Option<String>,

        /// Dry run: validate config and shapes, no GPU.
        #[arg(long)]
        dry_run: bool,
    },

    /// Evaluate a checkpoint.
    Eval {
        /// Path to checkpoint directory.
        #[arg(long)]
        checkpoint: String,

        /// Path to evaluation data.
        #[arg(long)]
        data: String,
    },

    /// Generate text.
    Generate {
        /// Path to checkpoint directory.
        #[arg(long)]
        checkpoint: String,

        /// Optional prompt text.
        #[arg(long)]
        prompt: Option<String>,

        /// Generate unconditionally (no prompt).
        #[arg(long)]
        unconditional: bool,

        /// Output length in tokens.
        #[arg(long, default_value = "128")]
        length: usize,

        /// Number of refinement steps.
        #[arg(long, default_value = "6")]
        steps: usize,

        /// Random seed.
        #[arg(long, default_value = "42")]
        seed: u64,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Train {
            config,
            data,
            output,
            dry_run,
        } => {
            println!("Training not yet implemented.");
            println!("  config={config:?} data={data:?} output={output:?} dry_run={dry_run}");

            if dry_run {
                // Validate config and shapes.
                let cfg = erm_core::ErmConfig::default();
                println!("Config validated: {cfg:?}");
                println!(
                    "Shapes OK: logits=[{}, {}, {}]",
                    cfg.batch_size, cfg.seq_len, cfg.vocab_size
                );
            }
        }
        Commands::Eval { checkpoint, data } => {
            println!("Eval not yet implemented.");
            println!("  checkpoint={checkpoint} data={data}");
        }
        Commands::Generate {
            checkpoint,
            prompt,
            unconditional,
            length,
            steps,
            seed,
        } => {
            println!("Generate not yet implemented.");
            println!("  checkpoint={checkpoint} prompt={prompt:?}");
            println!("  unconditional={unconditional} length={length} steps={steps} seed={seed}");
        }
    }
}
