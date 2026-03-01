//! Character-level tokenizer for the Emergent Route Model.
//!
//! Maps individual characters to `u32` token ids. Reserves:
//! - id `0` for **PAD** (padding)
//! - id `1` for **MASK** (corruption sentinel)
//!
//! All printable characters encountered during vocabulary building are assigned
//! sequential ids starting from `2`.
//!
//! # Example
//!
//! ```
//! use erm_core::tokenizer::CharTokenizer;
//!
//! let tokenizer = CharTokenizer::from_text("hello");
//! let ids = tokenizer.encode("hello");
//! let decoded = tokenizer.decode(&ids);
//! assert_eq!(decoded, "hello");
//! ```

use std::collections::HashMap;

use crate::error::{ErmError, ErmResult};

/// Reserved token id for padding.
pub const PAD_ID: u32 = 0;
/// Reserved token id for the MASK sentinel.
pub const MASK_ID: u32 = 1;
/// First id available for real characters.
const FIRST_CHAR_ID: u32 = 2;

/// A simple character-level tokenizer.
///
/// Builds a bidirectional mapping between `char` and `u32` ids.
/// PAD (0) and MASK (1) are always reserved.
#[derive(Debug, Clone)]
pub struct CharTokenizer {
    /// Character → id mapping.
    char_to_id: HashMap<char, u32>,
    /// Id → character mapping (only for real chars, not PAD/MASK).
    id_to_char: HashMap<u32, char>,
    /// Total vocabulary size (PAD + MASK + unique chars).
    vocab_size: usize,
}

impl CharTokenizer {
    /// Build a tokenizer from a corpus of text.
    ///
    /// Scans all characters in `text` and assigns each unique character a
    /// sequential id starting from [`FIRST_CHAR_ID`] (2).
    ///
    /// # Arguments
    ///
    /// - `text`: the full training corpus text.
    #[must_use]
    pub fn from_text(text: &str) -> Self {
        let mut char_to_id = HashMap::new();
        let mut id_to_char = HashMap::new();
        let mut next_id = FIRST_CHAR_ID;

        // Collect unique chars in order of first appearance for determinism.
        for ch in text.chars() {
            use std::collections::hash_map::Entry;
            if let Entry::Vacant(e) = char_to_id.entry(ch) {
                e.insert(next_id);
                id_to_char.insert(next_id, ch);
                next_id += 1;
            }
        }

        let vocab_size = next_id as usize; // PAD=0, MASK=1, then chars

        Self {
            char_to_id,
            id_to_char,
            vocab_size,
        }
    }

    /// Encode a string into a sequence of token ids.
    ///
    /// Unknown characters (not seen during construction) are silently skipped.
    ///
    /// # Arguments
    ///
    /// - `text`: the string to encode.
    ///
    /// # Returns
    ///
    /// A `Vec<u32>` of token ids.
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.chars()
            .filter_map(|ch| self.char_to_id.get(&ch).copied())
            .collect()
    }

    /// Decode a sequence of token ids back to a string.
    ///
    /// PAD and MASK tokens are skipped. Unknown ids are replaced with the
    /// Unicode replacement character `'�'`.
    ///
    /// # Arguments
    ///
    /// - `ids`: the token id sequence.
    ///
    /// # Returns
    ///
    /// The decoded string.
    #[must_use]
    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|&id| {
                if id == PAD_ID || id == MASK_ID {
                    None
                } else {
                    Some(self.id_to_char.get(&id).copied().unwrap_or('\u{FFFD}'))
                }
            })
            .collect()
    }

    /// Total vocabulary size including PAD and MASK.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Look up the id for a character.
    ///
    /// # Errors
    ///
    /// Returns [`ErmError::IndexOutOfBounds`] if the character is unknown.
    pub fn char_id(&self, ch: char) -> ErmResult<u32> {
        self.char_to_id
            .get(&ch)
            .copied()
            .ok_or_else(|| ErmError::IndexOutOfBounds(format!("unknown character: {ch:?}")))
    }

    /// The MASK token id (always 1).
    #[must_use]
    pub fn mask_id(&self) -> u32 {
        MASK_ID
    }

    /// The PAD token id (always 0).
    #[must_use]
    pub fn pad_id(&self) -> u32 {
        PAD_ID
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        let tokenizer = CharTokenizer::from_text("hello world");
        let ids = tokenizer.encode("hello world");
        let decoded = tokenizer.decode(&ids);
        assert_eq!(decoded, "hello world");
    }

    #[test]
    fn test_encode_decode_roundtrip_diverse() {
        let corpus = "abcdefghijklmnopqrstuvwxyz 0123456789!@#$%";
        let tokenizer = CharTokenizer::from_text(corpus);
        let text = "hello 123!";
        let ids = tokenizer.encode(text);
        let decoded = tokenizer.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_mask_token_exists() {
        let tokenizer = CharTokenizer::from_text("abc");
        assert_eq!(tokenizer.mask_id(), MASK_ID);
        assert_eq!(MASK_ID, 1);
    }

    #[test]
    fn test_pad_token_exists() {
        let tokenizer = CharTokenizer::from_text("abc");
        assert_eq!(tokenizer.pad_id(), PAD_ID);
        assert_eq!(PAD_ID, 0);
    }

    #[test]
    fn test_vocab_size_correct() {
        // "abc" has 3 unique chars → vocab = PAD + MASK + 3 = 5
        let tokenizer = CharTokenizer::from_text("abc");
        assert_eq!(tokenizer.vocab_size(), 5);
    }

    #[test]
    fn test_vocab_size_with_duplicates() {
        // "aabbcc" still has 3 unique chars → vocab = 5
        let tokenizer = CharTokenizer::from_text("aabbcc");
        assert_eq!(tokenizer.vocab_size(), 5);
    }

    #[test]
    fn test_reserved_ids() {
        let tokenizer = CharTokenizer::from_text("xyz");
        // No character should map to 0 or 1.
        for id in tokenizer.encode("xyz") {
            assert!(id >= FIRST_CHAR_ID, "char mapped to reserved id {id}");
        }
    }

    #[test]
    fn test_unknown_chars_skipped() {
        let tokenizer = CharTokenizer::from_text("abc");
        let ids = tokenizer.encode("axbycz");
        // Only a, b, c should produce ids; x, y, z are unknown.
        assert_eq!(ids.len(), 3);
        let decoded = tokenizer.decode(&ids);
        assert_eq!(decoded, "abc");
    }

    #[test]
    fn test_decode_skips_pad_and_mask() {
        let tokenizer = CharTokenizer::from_text("ab");
        let ids = vec![PAD_ID, 2, MASK_ID, 3]; // PAD, 'a', MASK, 'b'
        let decoded = tokenizer.decode(&ids);
        assert_eq!(decoded, "ab");
    }

    #[test]
    fn test_empty_text() {
        let tokenizer = CharTokenizer::from_text("");
        assert_eq!(tokenizer.vocab_size(), 2); // just PAD + MASK
        assert!(tokenizer.encode("hello").is_empty());
    }

    #[test]
    fn test_deterministic_ids() {
        let t1 = CharTokenizer::from_text("hello");
        let t2 = CharTokenizer::from_text("hello");
        assert_eq!(t1.encode("hello"), t2.encode("hello"));
    }
}
