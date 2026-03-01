//! Text dataset for training the ERM discrete denoiser.
//!
//! Loads a text file, tokenizes it using a [`CharTokenizer`], and chunks the
//! token stream into overlapping fixed-length sequences.
//!
//! # Chunking strategy
//!
//! Given token stream of length `N` and sequence length `L`, produces chunks:
//! - `[0..L)`, `[L/2..L/2+L)`, `[L..2L)`, `[3L/2..3L/2+L)`, ...
//!
//! Stride = `L / 2` (50% overlap) for better coverage of token boundaries.

use rand::seq::SliceRandom;
use rand::Rng;

use erm_core::error::{ErmError, ErmResult};
use erm_core::tokenizer::CharTokenizer;

/// A batch of token sequences for training.
#[derive(Debug, Clone)]
pub struct DataBatch {
    /// Flat token ids. Shape: `[batch_size * seq_len]`.
    /// Values are `u32` (from the tokenizer).
    pub tokens: Vec<u32>,
    /// Batch size `B`.
    pub batch_size: usize,
    /// Sequence length `L`.
    pub seq_len: usize,
}

/// A text dataset chunked into fixed-length token sequences.
#[derive(Debug, Clone)]
pub struct TextDataset {
    /// All sequences, each of length `seq_len`.
    sequences: Vec<Vec<u32>>,
    /// Sequence length.
    seq_len: usize,
    /// Current read position for sequential iteration.
    cursor: usize,
}

impl TextDataset {
    /// Build a dataset from a text file.
    ///
    /// Reads the file, tokenizes with the provided tokenizer, and chunks
    /// into sequences of length `seq_len` with stride `seq_len / 2`.
    ///
    /// # Arguments
    ///
    /// - `path`: path to the text file.
    /// - `tokenizer`: character-level tokenizer.
    /// - `seq_len`: target sequence length `L`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or produces too few tokens.
    pub fn from_file(path: &str, tokenizer: &CharTokenizer, seq_len: usize) -> ErmResult<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| ErmError::InvalidConfig(format!("cannot read file {path}: {e}")))?;

        Self::from_text(&text, tokenizer, seq_len)
    }

    /// Build a dataset from a string (useful for testing).
    ///
    /// # Arguments
    ///
    /// - `text`: the corpus text.
    /// - `tokenizer`: character-level tokenizer.
    /// - `seq_len`: target sequence length `L`.
    ///
    /// # Errors
    ///
    /// Returns an error if the text produces fewer tokens than `seq_len`.
    pub fn from_text(text: &str, tokenizer: &CharTokenizer, seq_len: usize) -> ErmResult<Self> {
        let all_tokens = tokenizer.encode(text);

        if all_tokens.len() < seq_len {
            return Err(ErmError::InvalidConfig(format!(
                "text has {} tokens, need at least {seq_len}",
                all_tokens.len()
            )));
        }

        let stride = seq_len / 2;
        let stride = if stride == 0 { 1 } else { stride };
        let mut sequences = Vec::new();

        let mut start = 0;
        while start + seq_len <= all_tokens.len() {
            sequences.push(all_tokens[start..start + seq_len].to_vec());
            start += stride;
        }

        // If there are leftover tokens at the end, take the last `seq_len` tokens.
        if start < all_tokens.len() && all_tokens.len() >= seq_len {
            let last_start = all_tokens.len() - seq_len;
            if sequences.is_empty() || last_start != start - stride {
                sequences.push(all_tokens[last_start..].to_vec());
            }
        }

        Ok(Self {
            sequences,
            seq_len,
            cursor: 0,
        })
    }

    /// Number of sequences in the dataset.
    #[must_use]
    pub fn len(&self) -> usize {
        self.sequences.len()
    }

    /// Whether the dataset is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    /// Get a random batch of sequences.
    ///
    /// Samples `batch_size` sequences uniformly at random with replacement.
    ///
    /// # Arguments
    ///
    /// - `batch_size`: number of sequences in the batch.
    /// - `rng`: random number generator.
    ///
    /// # Returns
    ///
    /// A [`DataBatch`] with flat `[B * L]` token ids.
    pub fn get_batch<R: Rng>(&self, batch_size: usize, rng: &mut R) -> DataBatch {
        let mut tokens = Vec::with_capacity(batch_size * self.seq_len);

        for _ in 0..batch_size {
            let idx = rng.gen_range(0..self.sequences.len());
            tokens.extend_from_slice(&self.sequences[idx]);
        }

        DataBatch {
            tokens,
            batch_size,
            seq_len: self.seq_len,
        }
    }

    /// Shuffle the internal sequence order (for epoch-based iteration).
    pub fn shuffle<R: Rng>(&mut self, rng: &mut R) {
        self.sequences.shuffle(rng);
        self.cursor = 0;
    }

    /// Get the next sequential batch, returning `None` if exhausted.
    ///
    /// Call [`shuffle`](Self::shuffle) to reset for a new epoch.
    pub fn next_batch(&mut self, batch_size: usize) -> Option<DataBatch> {
        if self.cursor + batch_size > self.sequences.len() {
            return None;
        }

        let mut tokens = Vec::with_capacity(batch_size * self.seq_len);
        for i in 0..batch_size {
            tokens.extend_from_slice(&self.sequences[self.cursor + i]);
        }
        self.cursor += batch_size;

        Some(DataBatch {
            tokens,
            batch_size,
            seq_len: self.seq_len,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn make_dataset() -> TextDataset {
        let text = "abcdefghijklmnopqrstuvwxyz ".repeat(100);
        let tokenizer = CharTokenizer::from_text(&text);
        TextDataset::from_text(&text, &tokenizer, 16).unwrap()
    }

    #[test]
    fn test_dataset_not_empty() {
        let ds = make_dataset();
        assert!(!ds.is_empty());
        assert!(!ds.is_empty());
    }

    #[test]
    fn test_batch_correct_shape() {
        let ds = make_dataset();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let batch = ds.get_batch(4, &mut rng);

        assert_eq!(batch.batch_size, 4);
        assert_eq!(batch.seq_len, 16);
        assert_eq!(batch.tokens.len(), 4 * 16);
    }

    #[test]
    fn test_sequences_correct_length() {
        let ds = make_dataset();
        for seq in &ds.sequences {
            assert_eq!(seq.len(), 16, "sequence length mismatch");
        }
    }

    #[test]
    fn test_batch_tokens_in_vocab_range() {
        let text = "hello world! ";
        let tokenizer = CharTokenizer::from_text(text);
        let big_text = text.repeat(50);
        let ds = TextDataset::from_text(&big_text, &tokenizer, 8).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let batch = ds.get_batch(2, &mut rng);

        let vocab = tokenizer.vocab_size() as u32;
        for &tok in &batch.tokens {
            assert!(tok < vocab, "token {tok} out of vocab range [0, {vocab})");
        }
    }

    #[test]
    fn test_too_short_text_errors() {
        let tokenizer = CharTokenizer::from_text("ab");
        let result = TextDataset::from_text("ab", &tokenizer, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_sequential_iteration() {
        let ds = make_dataset();
        let mut ds = ds;
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        ds.shuffle(&mut rng);

        let mut count = 0;
        while let Some(batch) = ds.next_batch(4) {
            assert_eq!(batch.tokens.len(), 4 * 16);
            count += 1;
        }
        assert!(count > 0);
    }

    #[test]
    fn test_overlap_produces_more_sequences() {
        // With stride L/2, we should get roughly 2x more sequences than stride L.
        let text = "a".repeat(200);
        let tokenizer = CharTokenizer::from_text(&text);
        let ds = TextDataset::from_text(&text, &tokenizer, 16).unwrap();
        // 200 tokens, stride 8 → roughly (200-16)/8 + 1 = 24 sequences
        assert!(
            ds.len() >= 20,
            "expected >= 20 sequences with overlap, got {}",
            ds.len()
        );
    }
}
