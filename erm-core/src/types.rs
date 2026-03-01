//! Core type aliases used across the ERM codebase.
//!
//! These aliases keep tensor-heavy code readable and make it easy to change
//! underlying representations later.

/// A token identifier. Values in `[0, V)` are real tokens; `V` is the MASK sentinel.
pub type TokenId = i32;

/// Batch index.
pub type BatchIdx = usize;

/// Sequence position index.
pub type PosIdx = usize;

/// Edge slot index within the dense-neighbor array (range `[0, Emax)`).
pub type EdgeSlot = usize;

/// Ant identifier within a colony.
pub type AntIdx = usize;

/// Refinement step counter (1 = lightest corruption, T = heaviest).
pub type StepT = usize;

/// Edge type discriminant.
///
/// - `0` = local (short-range positional)
/// - `1` = long-range
/// - `2` = concept
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum EdgeType {
    /// Short-range positional edge.
    Local = 0,
    /// Long-range dependency edge.
    LongRange = 1,
    /// Concept/cluster edge.
    Concept = 2,
}

impl EdgeType {
    /// Convert from a raw `u8`.
    ///
    /// # Errors
    ///
    /// Returns `None` for values outside `[0, 2]`.
    #[must_use]
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Local),
            1 => Some(Self::LongRange),
            2 => Some(Self::Concept),
            _ => None,
        }
    }
}
