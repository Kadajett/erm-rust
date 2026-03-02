//! Text dataset for training the ERM discrete denoiser.
//!
//! Loads a text file, tokenizes it using a [`CharTokenizer`], and chunks the
//! token stream into overlapping fixed-length sequences.
//!
//! # Chunking strategy
//!
//! Given token stream of length `N` and sequence length `L`, produces chunks:
//! - `[0..L)`, `[L/2..L/2+L)`, `[L..2L)`, ...
//!
//! Stride = `L / 2` (50% overlap) for better coverage of token boundaries.
//!
//! # Directory loading
//!
//! [`TextDataset::from_directory`] recursively scans a directory for `*.txt`
//! files and concatenates their content before tokenisation. Files are sorted
//! by path for determinism.

use rand::seq::SliceRandom;
use rand::Rng;

use erm_core::error::{ErmError, ErmResult};
use erm_core::tokenizer::CharTokenizer;

/// A batch of token sequences for training.
#[derive(Debug, Clone)]
pub struct DataBatch {
    /// Flat token ids. Shape: `[batch_size * seq_len]`.
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
    /// Epoch counter — incremented on each call to [`shuffle`](Self::shuffle).
    epoch: usize,
}

impl TextDataset {
    /// Build a dataset from a single text file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or produces too few tokens.
    pub fn from_file(path: &str, tokenizer: &CharTokenizer, seq_len: usize) -> ErmResult<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| ErmError::InvalidConfig(format!("cannot read file {path}: {e}")))?;
        Self::from_text(&text, tokenizer, seq_len)
    }

    /// Build a dataset by recursively scanning a directory for `*.txt` files.
    ///
    /// Files are sorted by path for determinism before concatenation.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be read, no `.txt` files are
    /// found, or the combined corpus is too short.
    pub fn from_directory(dir: &str, tokenizer: &CharTokenizer, seq_len: usize) -> ErmResult<Self> {
        let mut paths: Vec<String> = Vec::new();
        collect_txt_paths(dir, &mut paths)?;
        paths.sort();

        if paths.is_empty() {
            return Err(ErmError::InvalidConfig(format!(
                "no .txt files found in directory: {dir}"
            )));
        }

        let mut combined = String::new();
        for path in &paths {
            match std::fs::read_to_string(path) {
                Ok(content) => {
                    if !combined.is_empty() {
                        combined.push('\n');
                    }
                    combined.push_str(&content);
                }
                Err(e) => {
                    eprintln!("warning: cannot read {path}: {e}");
                }
            }
        }

        Self::from_text(&combined, tokenizer, seq_len)
    }

    /// Build a dataset from a string (useful for testing).
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

        let stride = (seq_len / 2).max(1);
        let mut sequences = Vec::new();

        let mut start = 0;
        while start + seq_len <= all_tokens.len() {
            sequences.push(all_tokens[start..start + seq_len].to_vec());
            start += stride;
        }

        // Capture leftover tail.
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
            epoch: 0,
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

    /// Current epoch number. Starts at `0`; incremented by each [`shuffle`](Self::shuffle).
    #[must_use]
    pub fn epoch(&self) -> usize {
        self.epoch
    }

    /// Sample a random batch with replacement.
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

    /// Shuffle the sequence order and increment the epoch counter.
    ///
    /// Resets the sequential iteration cursor to 0.
    pub fn shuffle<R: Rng>(&mut self, rng: &mut R) {
        self.sequences.shuffle(rng);
        self.cursor = 0;
        self.epoch += 1;
    }

    /// Return the next sequential batch, or `None` if the epoch is exhausted.
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

// ── Directory helpers ──────────────────────────────────────────────────────

/// Recursively collect all `*.txt` file paths under `dir`.
fn collect_txt_paths(dir: &str, out: &mut Vec<String>) -> ErmResult<()> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| ErmError::InvalidConfig(format!("cannot read directory {dir}: {e}")))?;

    for entry in entries {
        let entry =
            entry.map_err(|e| ErmError::InvalidConfig(format!("directory entry error: {e}")))?;
        let path = entry.path();
        if path.is_dir() {
            let sub = path
                .to_str()
                .ok_or_else(|| ErmError::InvalidConfig("non-UTF-8 path".to_string()))?
                .to_string();
            collect_txt_paths(&sub, out)?;
        } else if path.extension().and_then(|e| e.to_str()) == Some("txt") {
            let s = path
                .to_str()
                .ok_or_else(|| ErmError::InvalidConfig("non-UTF-8 path".to_string()))?
                .to_string();
            out.push(s);
        }
    }
    Ok(())
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
        let text = "a".repeat(200);
        let tokenizer = CharTokenizer::from_text(&text);
        let ds = TextDataset::from_text(&text, &tokenizer, 16).unwrap();
        assert!(
            ds.len() >= 20,
            "expected >= 20 sequences with overlap, got {}",
            ds.len()
        );
    }

    #[test]
    fn test_epoch_starts_at_zero() {
        let ds = make_dataset();
        assert_eq!(ds.epoch(), 0);
    }

    #[test]
    fn test_shuffle_increments_epoch() {
        let mut ds = make_dataset();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        assert_eq!(ds.epoch(), 0);
        ds.shuffle(&mut rng);
        assert_eq!(ds.epoch(), 1);
        ds.shuffle(&mut rng);
        assert_eq!(ds.epoch(), 2);
    }

    #[test]
    fn test_from_directory() {
        let dir = "/tmp/erm_dataset_dir_test";
        let _ = std::fs::create_dir_all(dir);
        let corpus = "the quick brown fox jumps over the lazy dog. ".repeat(20);
        std::fs::write(format!("{dir}/a.txt"), &corpus).unwrap();
        std::fs::write(format!("{dir}/b.txt"), &corpus).unwrap();
        std::fs::write(format!("{dir}/ignore.md"), "ignored").unwrap();

        let tokenizer = CharTokenizer::from_text(&corpus);
        let ds = TextDataset::from_directory(dir, &tokenizer, 16).unwrap();
        assert!(!ds.is_empty(), "dataset from directory should not be empty");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_from_directory_missing() {
        let tokenizer = CharTokenizer::from_text("abc");
        let result = TextDataset::from_directory("/nonexistent/path/xyz", &tokenizer, 8);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_directory_no_txt_files() {
        let dir = "/tmp/erm_no_txt_test";
        let _ = std::fs::create_dir_all(dir);
        std::fs::write(format!("{dir}/data.csv"), "1,2,3").unwrap();
        let tokenizer = CharTokenizer::from_text("abc");
        let result = TextDataset::from_directory(dir, &tokenizer, 8);
        assert!(result.is_err());
        let _ = std::fs::remove_dir_all(dir);
    }
}
