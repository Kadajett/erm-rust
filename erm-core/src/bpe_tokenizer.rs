//! BPE subword tokenizer for the Emergent Route Model.
//!
//! Provides a [`BpeTokenizer`] that wraps a byte-pair encoding vocabulary for
//! subword tokenization, and a [`TokenizerApi`] trait that both [`BpeTokenizer`]
//! and the legacy [`crate::tokenizer::CharTokenizer`] implement so callers can
//! be generic over tokenizer type.
//!
//! # Vocabulary layout
//!
//! - `id 0`: **PAD**
//! - `id 1`: **MASK** (corruption sentinel)
//! - `id 2`: **UNK** (unknown subword)
//! - `ids 3..`: real subword tokens in UTF-8 byte-pair order
//!
//! The BPE vocabulary is built by a simple iterative merge of the most frequent
//! adjacent byte pairs in the training corpus. For production use with large
//! corpora the vocabulary should be pre-built and serialized; this implementation
//! includes a lightweight in-process trainer suitable for corpora up to ~100 MB.
//!
//! # Stub modalities
//!
//! The `TokenizerApi` trait provides stubs for code and image modalities. These
//! return `Err(ErmError::NotImplemented)` and will be wired to real encoders in
//! future milestones.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::{ErmError, ErmResult};

// ── Reserved token ids ────────────────────────────────────────────────────────

/// Reserved id for padding.
pub const PAD_ID: u32 = 0;
/// Reserved id for the MASK sentinel (diffusion corruption).
pub const MASK_ID: u32 = 1;
/// Reserved id for unknown subwords.
pub const UNK_ID: u32 = 2;
/// First id available for real subword merges.
pub const FIRST_SUBWORD_ID: u32 = 3;

// ── TokenizerApi trait ────────────────────────────────────────────────────────

/// Common interface for all ERM tokenizers.
///
/// Implementing this trait allows training and inference code to be generic
/// over the tokenizer in use.
pub trait TokenizerApi: Send + Sync {
    /// Encode a UTF-8 text string into a sequence of token ids.
    fn encode_text(&self, text: &str) -> Vec<u32>;

    /// Decode a sequence of token ids back to a UTF-8 string.
    ///
    /// PAD and MASK ids are silently skipped.
    fn decode_text(&self, ids: &[u32]) -> String;

    /// Total vocabulary size (including PAD, MASK, UNK, and all real tokens).
    fn vocab_size(&self) -> usize;

    /// The MASK token id.
    fn mask_id(&self) -> u32;

    /// The PAD token id.
    fn pad_id(&self) -> u32;

    /// Encode source code (stub — returns `ErmError::NotImplemented`).
    fn encode_code(&self, _src: &str) -> ErmResult<Vec<u32>> {
        Err(ErmError::NotImplemented(
            "code tokenization not yet implemented".to_string(),
        ))
    }

    /// Encode an image as a flat patch token sequence (stub — returns `ErmError::NotImplemented`).
    fn encode_image(&self, _bytes: &[u8]) -> ErmResult<Vec<u32>> {
        Err(ErmError::NotImplemented(
            "image tokenization not yet implemented".to_string(),
        ))
    }
}

// ── BPE vocabulary entry ──────────────────────────────────────────────────────

/// A single entry in the BPE vocabulary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabEntry {
    /// The UTF-8 text this token represents (empty for PAD/MASK/UNK).
    pub text: String,
    /// Token id.
    pub id: u32,
}

// ── BpeTokenizer ──────────────────────────────────────────────────────────────

/// BPE subword tokenizer.
///
/// Encodes text as UTF-8 byte pairs merged iteratively. The vocabulary can be
/// built from a training corpus via [`BpeTokenizer::train`] or loaded from a
/// serialized JSON file via [`BpeTokenizer::from_json`].
///
/// # Thread safety
///
/// All methods take `&self` (immutable reference) and are safe to call from
/// multiple threads. Use `Arc<BpeTokenizer>` in concurrent contexts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BpeTokenizer {
    /// The ordered merge rules: (left, right) → merged token id.
    ///
    /// Applied left-to-right, first match wins.
    merges: Vec<(String, String)>,
    /// String → token id forward map.
    vocab: HashMap<String, u32>,
    /// Token id → string backward map.
    id_to_token: HashMap<u32, String>,
    /// Total vocabulary size.
    vocab_size: usize,
}

impl BpeTokenizer {
    /// Detect dominant whitespace marker style in the loaded vocabulary.
    ///
    /// - `Prefix`: tokens like `Ġword` / `▁word` (GPT/SentencePiece-like).
    /// - `Suffix`: tokens like `wordĠ` (legacy ERM trainer style).
    /// - `None`: no strong marker usage in vocab.
    fn marker_style(&self) -> MarkerStyle {
        let mut prefix = 0usize;
        let mut suffix = 0usize;
        for tok in self.vocab.keys() {
            if tok.starts_with('Ġ') || tok.starts_with('▁') {
                prefix += 1;
            }
            if tok.ends_with('Ġ') || tok.ends_with('▁') {
                suffix += 1;
            }
        }
        if prefix > suffix && prefix > 0 {
            MarkerStyle::Prefix
        } else if suffix > 0 {
            MarkerStyle::Suffix
        } else {
            MarkerStyle::None
        }
    }

    /// Longest token length in Unicode scalar values.
    fn max_token_len_chars(&self) -> usize {
        self.vocab
            .keys()
            .map(|k| k.chars().count())
            .max()
            .unwrap_or(1)
    }

    /// Encode text for vocabularies that use prefix whitespace markers (`Ġ`/`▁`).
    ///
    /// Input whitespace is preserved as boundary information via:
    /// - prefix marker on the next lexical token
    /// - standalone whitespace marker tokens for additional spaces in a run
    fn encode_text_prefix_markers(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut current_word = String::new();
        let mut pending_ws = 0usize;

        for ch in text.chars() {
            if ch.is_whitespace() {
                if !current_word.is_empty() {
                    ids.extend(self.encode_prefix_word(&current_word, pending_ws));
                    current_word.clear();
                    pending_ws = 0;
                }
                pending_ws += 1;
            } else {
                current_word.push(ch);
            }
        }

        if !current_word.is_empty() {
            ids.extend(self.encode_prefix_word(&current_word, pending_ws));
        }

        ids
    }

    /// Encode one lexical word in prefix-marker mode.
    ///
    /// `leading_ws` is the number of whitespace chars before the word.
    fn encode_prefix_word(&self, word: &str, leading_ws: usize) -> Vec<u32> {
        if word.is_empty() {
            return Vec::new();
        }

        let mut ids = Vec::new();
        let whitespace_id = self
            .vocab
            .get("Ġ")
            .copied()
            .or_else(|| self.vocab.get("▁").copied())
            .or_else(|| self.vocab.get(" ").copied());

        // Preserve extra whitespace (>1) using standalone marker/space tokens.
        if leading_ws > 1 {
            for _ in 0..(leading_ws - 1) {
                if let Some(ws_id) = whitespace_id {
                    ids.push(ws_id);
                }
            }
        }

        let chars: Vec<char> = word.chars().collect();
        let max_len = self.max_token_len_chars().max(1);
        let mut i = 0usize;
        let mut needs_prefix = leading_ws > 0;

        while i < chars.len() {
            let mut matched: Option<(u32, usize)> = None;
            let max_span = (chars.len() - i).min(max_len);

            for span in (1..=max_span).rev() {
                let chunk: String = chars[i..i + span].iter().collect();
                if needs_prefix {
                    for candidate in [format!("Ġ{chunk}"), format!("▁{chunk}"), chunk.clone()] {
                        if let Some(&id) = self.vocab.get(&candidate) {
                            matched = Some((id, span));
                            break;
                        }
                    }
                } else if let Some(&id) = self.vocab.get(&chunk) {
                    matched = Some((id, span));
                }

                if matched.is_some() {
                    break;
                }
            }

            if let Some((id, span)) = matched {
                ids.push(id);
                i += span;
            } else {
                let ch = chars[i].to_string();
                if needs_prefix {
                    let mut found = false;
                    for candidate in [format!("Ġ{ch}"), format!("▁{ch}"), ch.clone()] {
                        if let Some(&id) = self.vocab.get(&candidate) {
                            ids.push(id);
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        ids.push(UNK_ID);
                    }
                } else {
                    ids.push(self.vocab.get(&ch).copied().unwrap_or(UNK_ID));
                }
                i += 1;
            }
            needs_prefix = false;
        }

        ids
    }

    /// Greedy raw-text tokenizer for vocabularies without explicit marker style.
    ///
    /// This path preserves literal whitespace when space/newline tokens exist.
    fn encode_text_raw_greedy(&self, text: &str) -> Vec<u32> {
        let chars: Vec<char> = text.chars().collect();
        let max_len = self.max_token_len_chars().max(1);
        let mut ids = Vec::new();
        let mut i = 0usize;

        while i < chars.len() {
            let mut matched: Option<(u32, usize)> = None;
            let max_span = (chars.len() - i).min(max_len);
            for span in (1..=max_span).rev() {
                let chunk: String = chars[i..i + span].iter().collect();
                if let Some(&id) = self.vocab.get(&chunk) {
                    matched = Some((id, span));
                    break;
                }
            }
            if let Some((id, span)) = matched {
                ids.push(id);
                i += span;
            } else {
                ids.push(
                    self.vocab
                        .get(&chars[i].to_string())
                        .copied()
                        .unwrap_or(UNK_ID),
                );
                i += 1;
            }
        }

        ids
    }

    /// Build a BPE tokenizer from a training corpus.
    ///
    /// Runs `num_merges` rounds of byte-pair merging on the corpus. Each round
    /// finds the most frequent adjacent pair of tokens and merges them into a
    /// new token.
    ///
    /// # Arguments
    ///
    /// - `corpus`: the full training text.
    /// - `num_merges`: number of merge operations (≈ vocabulary size beyond base).
    ///
    /// # Returns
    ///
    /// A trained [`BpeTokenizer`] ready for use.
    pub fn train(corpus: &str, num_merges: usize) -> Self {
        // Start with character-level vocabulary (UTF-8 chars as base units).
        // Add special tokens first.
        let mut vocab: HashMap<String, u32> = HashMap::new();
        let mut id_to_token: HashMap<u32, String> = HashMap::new();

        // Reserve special tokens.
        for (id, name) in [(PAD_ID, "<pad>"), (MASK_ID, "<mask>"), (UNK_ID, "<unk>")] {
            vocab.insert(name.to_string(), id);
            id_to_token.insert(id, name.to_string());
        }

        let mut next_id: u32 = FIRST_SUBWORD_ID;

        // Collect all unique characters from corpus as base vocabulary.
        let mut chars: Vec<char> = corpus
            .chars()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        chars.sort_by_key(|c| *c as u32);

        for ch in &chars {
            let s = ch.to_string();
            if !vocab.contains_key(&s) {
                vocab.insert(s.clone(), next_id);
                id_to_token.insert(next_id, s);
                next_id += 1;
            }
        }

        // Tokenize corpus into initial character sequences (for merge counting).
        // Work on word-segmented corpus: split by whitespace, add sentinel.
        let words: Vec<Vec<String>> = corpus
            .split_whitespace()
            .map(|w| {
                let mut chars: Vec<String> = w.chars().map(|c| c.to_string()).collect();
                // Add end-of-word sentinel.
                if let Some(last) = chars.last_mut() {
                    *last = format!("{last}Ġ");
                }
                chars
            })
            .collect();

        // Make sure sentinel chars are in vocab.
        for word in &words {
            for tok in word {
                if !vocab.contains_key(tok) {
                    vocab.insert(tok.clone(), next_id);
                    id_to_token.insert(next_id, tok.clone());
                    next_id += 1;
                }
            }
        }

        let mut merges: Vec<(String, String)> = Vec::new();
        let mut current_words = words;

        for _ in 0..num_merges {
            // Count pairs.
            let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
            for word in &current_words {
                for window in word.windows(2) {
                    let pair = (window[0].clone(), window[1].clone());
                    *pair_counts.entry(pair).or_default() += 1;
                }
            }

            if pair_counts.is_empty() {
                break;
            }

            // Find most frequent pair.
            let best_pair = pair_counts
                .iter()
                .max_by_key(|(_, &count)| count)
                .map(|(pair, _)| pair.clone())
                .unwrap();

            let merged = format!("{}{}", best_pair.0, best_pair.1);

            // Register merged token.
            if !vocab.contains_key(&merged) {
                vocab.insert(merged.clone(), next_id);
                id_to_token.insert(next_id, merged.clone());
                next_id += 1;
            }
            merges.push(best_pair.clone());

            // Apply merge to all words.
            current_words = current_words
                .into_iter()
                .map(|word| apply_merge(&word, &best_pair, &merged))
                .collect();
        }

        let vocab_size = next_id as usize;

        Self {
            merges,
            vocab,
            id_to_token,
            vocab_size,
        }
    }

    /// Encode a single word (already split by whitespace) using the merge rules.
    fn encode_word(&self, word: &str) -> Vec<u32> {
        if word.is_empty() {
            return vec![];
        }

        // Start with character-level tokens + end-of-word sentinel on last char.
        let chars: Vec<char> = word.chars().collect();
        let mut tokens: Vec<String> = chars
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                if i == chars.len() - 1 {
                    format!("{c}Ġ")
                } else {
                    c.to_string()
                }
            })
            .collect();

        // Apply merge rules.
        for (left, right) in &self.merges {
            let merged = format!("{left}{right}");
            tokens = apply_merge(&tokens, &(left.clone(), right.clone()), &merged);
        }

        // Map to ids.
        tokens
            .iter()
            .map(|t| self.vocab.get(t).copied().unwrap_or(UNK_ID))
            .collect()
    }

    /// Save the tokenizer vocabulary and merge rules to a JSON string.
    ///
    /// # Errors
    ///
    /// Returns `ErmError::InvalidConfig` if serialization fails.
    pub fn to_json(&self) -> ErmResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| ErmError::InvalidConfig(format!("BPE serialization failed: {e}")))
    }

    /// Load a tokenizer from a JSON string produced by [`to_json`](Self::to_json).
    ///
    /// # Errors
    ///
    /// Returns `ErmError::InvalidConfig` if parsing fails.
    pub fn from_json(json: &str) -> ErmResult<Self> {
        serde_json::from_str(json)
            .map_err(|e| ErmError::InvalidConfig(format!("BPE deserialization failed: {e}")))
    }

    /// Save to a file path.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn save(&self, path: &str) -> ErmResult<()> {
        let json = self.to_json()?;
        std::fs::write(path, json)
            .map_err(|e| ErmError::InvalidConfig(format!("cannot write BPE vocab to {path}: {e}")))
    }

    /// Load from a file path.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn load(path: &str) -> ErmResult<Self> {
        let json = std::fs::read_to_string(path).map_err(|e| {
            ErmError::InvalidConfig(format!("cannot read BPE vocab from {path}: {e}"))
        })?;
        Self::from_json(&json)
    }
}

impl TokenizerApi for BpeTokenizer {
    fn encode_text(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        match self.marker_style() {
            MarkerStyle::Prefix => self.encode_text_prefix_markers(text),
            MarkerStyle::Suffix => {
                // Legacy suffix-marker mode from the in-repo trainer.
                for word in text.split_whitespace() {
                    ids.extend(self.encode_word(word));
                }
                ids
            }
            MarkerStyle::None => self.encode_text_raw_greedy(text),
        }
    }

    fn decode_text(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for &id in ids {
            if id == PAD_ID || id == MASK_ID {
                continue;
            }
            if let Some(tok) = self.id_to_token.get(&id) {
                if tok == "<unk>" {
                    out.push('?');
                    continue;
                }
                if tok == "<pad>" || tok == "<mask>" {
                    continue;
                }
                // Replace both common whitespace markers with a literal space.
                let text = tok.replace(['Ġ', '▁'], " ");
                out.push_str(&text);
            } else {
                out.push('?');
            }
        }
        out
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn mask_id(&self) -> u32 {
        MASK_ID
    }

    fn pad_id(&self) -> u32 {
        PAD_ID
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MarkerStyle {
    Prefix,
    Suffix,
    None,
}

// ── CharTokenizer adapter ─────────────────────────────────────────────────────

impl TokenizerApi for crate::tokenizer::CharTokenizer {
    fn encode_text(&self, text: &str) -> Vec<u32> {
        self.encode(text)
    }

    fn decode_text(&self, ids: &[u32]) -> String {
        self.decode(ids)
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }

    fn mask_id(&self) -> u32 {
        self.mask_id()
    }

    fn pad_id(&self) -> u32 {
        self.pad_id()
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Apply a single BPE merge rule to a sequence of string tokens.
fn apply_merge(tokens: &[String], pair: &(String, String), merged: &str) -> Vec<String> {
    let mut result: Vec<String> = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() {
        if i + 1 < tokens.len() && tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
            result.push(merged.to_string());
            i += 2;
        } else {
            result.push(tokens[i].clone());
            i += 1;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_corpus() -> &'static str {
        "the quick brown fox jumps over the lazy dog the fox"
    }

    #[test]
    fn test_bpe_train_basic() {
        let bpe = BpeTokenizer::train(small_corpus(), 20);
        assert!(bpe.vocab_size() > 3); // at least PAD + MASK + UNK + some chars
    }

    #[test]
    fn test_bpe_encode_nonempty() {
        let bpe = BpeTokenizer::train(small_corpus(), 20);
        let ids = bpe.encode_text("the fox");
        assert!(!ids.is_empty());
        for &id in &ids {
            assert!(id < bpe.vocab_size() as u32, "id {id} out of range");
        }
    }

    #[test]
    fn test_bpe_decode_round_trip_approx() {
        let bpe = BpeTokenizer::train(small_corpus(), 20);
        let ids = bpe.encode_text("the fox");
        let decoded = bpe.decode_text(&ids);
        // After decode, "the fox" → " the  fox" (with spaces from sentinel removal).
        assert!(decoded.contains("the"));
        assert!(decoded.contains("fox"));
    }

    #[test]
    fn test_bpe_no_reserved_ids_in_encode() {
        let bpe = BpeTokenizer::train(small_corpus(), 10);
        let ids = bpe.encode_text("quick brown");
        // encode_text should never emit PAD or MASK.
        for &id in &ids {
            assert_ne!(id, PAD_ID, "PAD should not appear in encode output");
            assert_ne!(id, MASK_ID, "MASK should not appear in encode output");
        }
    }

    #[test]
    fn test_bpe_serde_round_trip() {
        let bpe = BpeTokenizer::train(small_corpus(), 15);
        let json = bpe.to_json().unwrap();
        let bpe2 = BpeTokenizer::from_json(&json).unwrap();

        // Vocabularies should match.
        assert_eq!(bpe.vocab_size(), bpe2.vocab_size());
        let ids1 = bpe.encode_text("the fox");
        let ids2 = bpe2.encode_text("the fox");
        assert_eq!(ids1, ids2);
    }

    #[test]
    fn test_char_tokenizer_api() {
        let ct = crate::tokenizer::CharTokenizer::from_text("hello world");
        let ids = ct.encode_text("hello world");
        assert!(!ids.is_empty());
        let decoded = ct.decode_text(&ids);
        assert_eq!(decoded, "hello world");
    }

    #[test]
    fn test_bpe_vocab_size_constant() {
        let bpe = BpeTokenizer::train("abcde abcde", 5);
        // vocab_size should not change after construction.
        assert_eq!(bpe.vocab_size(), TokenizerApi::vocab_size(&bpe));
    }
}
