//! # erm-train
//!
//! Training loop, data pipeline, evaluation, orchestration, and benchmarks
//! for the Emergent Route Model.
//!
//! ## Modules
//!
//! - [`dataset`] — text loading, tokenisation, epoch tracking, directory scanning
//! - [`image_dataset`] — grayscale image (PGM P5) dataset for pixel-level sequences
//! - [`training`] — single training step (forward + loss)
//! - [`orchestrator`] — two-phase training (warm-start → colony), checkpoints
//! - [`eval`] — evaluation: denoising accuracy, entropy, unique-token ratio
//! - [`comparison`] — comparison harness for multi-run experiments; JSON report
//! - [`bench`] — throughput and memory benchmarking harness

pub mod bench;
pub mod burn_training;
pub mod comparison;
pub mod dataset;
pub mod eval;
pub mod image_dataset;
pub mod orchestrator;
pub mod training;
