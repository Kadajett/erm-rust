//! # erm-core
//!
//! Core library for the **Emergent Route Model** (ERM) — a non-autoregressive
//! text generator that learns via stigmergic exploration and exploitation.
//!
//! This crate provides:
//!
//! - [`config`] — full hyperparameter configuration with serde support
//! - [`types`] — core type aliases (TokenId, EdgeType, etc.)
//! - [`error`] — error types for the entire ERM pipeline
//! - [`graph`] — dense-neighbor route graph storage and mutation
//! - [`corruption`] — discrete corruption schedule (mask/replace/keep)
//! - [`scorer`] — neural scorer network (transformer encoder + output heads)
//! - [`ants`] — ant colony structures, edit proposals, and conflict-free merge
//! - [`topk`] — top-k extraction from logits
//! - [`merge`] — conflict-free merge of edit proposals with per-ant Δ
//! - [`refinement`] — one-step refinement pipeline

pub mod ants;
pub mod bpe_tokenizer;
pub mod burn_scorer;
pub mod config;
pub mod corruption;
pub mod error;
pub mod graph;
pub mod merge;
pub mod pheromone;
pub mod refinement;
pub mod scorer;
pub mod tokenizer;
pub mod topk;
pub mod types;

// Re-export the most commonly used items at crate root.
pub use bpe_tokenizer::{BpeTokenizer, TokenizerApi};
pub use config::ErmConfig;
pub use error::{ErmError, ErmResult};
