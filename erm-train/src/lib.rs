//! # erm-train
//!
//! Training loop, data pipeline, and optimization for the Emergent Route Model.
//!
//! ## Phase 1 (Milestone 1) — Baseline Discrete Denoiser
//!
//! - [`dataset`] — text loading, tokenization, chunking, and batch iteration
//! - [`training`] — training step with corruption, denoising loss computation

pub mod dataset;
pub mod training;
