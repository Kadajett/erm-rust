//! Error types for the ERM core library.
//!
//! All public functions in `erm-core` return `Result<_, ErmError>` instead of
//! panicking. Variants are added as the codebase grows.

use thiserror::Error;

/// Errors that can occur within the ERM core library.
#[derive(Debug, Error)]
pub enum ErmError {
    /// A tensor dimension did not match the expected shape.
    #[error("shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch {
        /// Human-readable description of the expected shape.
        expected: String,
        /// Human-readable description of the actual shape.
        got: String,
    },

    /// An index was out of the valid range.
    #[error("index out of bounds: {0}")]
    IndexOutOfBounds(String),

    /// Configuration value is invalid.
    #[error("invalid config: {0}")]
    InvalidConfig(String),

    /// Graph operation failed (e.g., edge insertion into a full slot).
    #[error("graph error: {0}")]
    GraphError(String),

    /// Serialization / deserialization failed.
    #[error("serde error: {0}")]
    SerdeError(#[from] serde_json::Error),

    /// I/O error.
    #[error("io error: {0}")]
    IoError(#[from] std::io::Error),

    /// A feature or code path is not yet implemented.
    #[error("not implemented: {0}")]
    NotImplemented(String),
}

/// Convenience alias used throughout the crate.
pub type ErmResult<T> = Result<T, ErmError>;
