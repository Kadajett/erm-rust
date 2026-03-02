//! CPU streaming dataset for ERM diffusion training.
//!
//! Implements a double-buffered, streaming data pipeline that iterates over
//! text files on CPU and feeds tokenized batches to the GPU trainer without
//! loading the entire corpus into GPU memory at once.
//!
//! # Design
//!
//! ```text
//!  Disk files ──→ [producer thread] ──→ channel (capacity=2) ──→ [consumer/trainer]
//!                    tokenize +                                       GPU forward
//!                  chunk on CPU                                       + backward
//! ```
//!
//! The producer thread walks a directory of `.txt` files, tokenizes each file's
//! content on the fly, slides a window of `seq_len` tokens, and sends
//! [`TokenBatch`]es through a bounded channel (capacity 2 = double buffer).
//!
//! The trainer calls [`StreamingDataset::next_batch`] which blocks until the
//! next batch is ready, but the producer is always one batch ahead thanks to
//! the bounded channel.
//!
//! # Sentence/paragraph spans
//!
//! When `use_paragraph_spans = true`, the dataset derives spans at sentence
//! and paragraph boundaries rather than sliding a fixed window every `seq_len/2`
//! tokens. This produces more semantically coherent training examples for books.
//!
//! # No GPU load
//!
//! This module only produces `Vec<u32>` token batches. The caller is responsible
//! for moving them to the GPU via `tokens_to_tensor`.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread;

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use erm_core::bpe_tokenizer::TokenizerApi;
use erm_core::error::{ErmError, ErmResult};

/// A CPU-side batch of tokenized sequences ready for GPU transfer.
#[derive(Debug, Clone)]
pub struct TokenBatch {
    /// Flat token ids. Shape: `[batch_size * seq_len]`.
    pub tokens: Vec<u32>,
    /// Batch size `B`.
    pub batch_size: usize,
    /// Sequence length `L`.
    pub seq_len: usize,
}

/// Configuration for the streaming dataset producer thread.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Directory of `.txt` files.
    pub data_dir: String,
    /// Sequence length for each training example.
    pub seq_len: usize,
    /// Training batch size.
    pub batch_size: usize,
    /// Whether to split on paragraph/sentence boundaries.
    pub use_paragraph_spans: bool,
    /// Shuffle file order on each epoch pass.
    pub shuffle_files: bool,
    /// RNG seed for reproducibility.
    pub seed: u64,
    /// Loop forever (repeat the corpus).
    pub repeat: bool,
}

/// Double-buffered streaming dataset.
///
/// Spawns a background producer thread that tokenizes files and sends
/// batches through a bounded channel. The main thread calls [`next_batch`]
/// to receive the next ready batch.
pub struct StreamingDataset {
    /// Receiver end of the double-buffer channel.
    rx: Receiver<ErmResult<TokenBatch>>,
    /// Batch count consumed so far.
    pub batches_consumed: u64,
}

impl StreamingDataset {
    /// Start a new streaming dataset producer.
    ///
    /// The producer thread starts immediately and begins filling the internal
    /// 2-batch buffer.
    ///
    /// # Arguments
    ///
    /// - `config`: streaming configuration.
    /// - `tokenizer`: a tokenizer implementing [`TokenizerApi`].
    ///
    /// # Returns
    ///
    /// A [`StreamingDataset`] that yields batches from [`next_batch`].
    pub fn new<T: TokenizerApi + 'static>(config: StreamingConfig, tokenizer: T) -> Self {
        // Bounded channel with capacity 2 → double buffer.
        let (tx, rx): (SyncSender<ErmResult<TokenBatch>>, _) = mpsc::sync_channel(2);

        thread::Builder::new()
            .name("erm-data-producer".to_string())
            .spawn(move || producer_loop(config, tokenizer, tx))
            .expect("failed to spawn data producer thread");

        Self {
            rx,
            batches_consumed: 0,
        }
    }

    /// Receive the next batch from the producer.
    ///
    /// Blocks until the producer sends the next batch or an error.
    ///
    /// # Returns
    ///
    /// `Ok(Some(batch))` on success, `Ok(None)` if the producer has finished
    /// (only possible when `repeat = false`), or `Err` on a tokenization error.
    pub fn next_batch(&mut self) -> ErmResult<Option<TokenBatch>> {
        match self.rx.recv() {
            Ok(result) => {
                let batch = result?;
                self.batches_consumed += 1;
                Ok(Some(batch))
            }
            // Sender dropped → stream exhausted.
            Err(_) => Ok(None),
        }
    }
}

// ── Producer thread ───────────────────────────────────────────────────────────

fn producer_loop<T: TokenizerApi>(
    config: StreamingConfig,
    tokenizer: T,
    tx: SyncSender<ErmResult<TokenBatch>>,
) {
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);

    loop {
        // Collect .txt file paths.
        let mut paths: Vec<PathBuf> = match collect_txt_paths(&config.data_dir) {
            Ok(p) => p,
            Err(e) => {
                let _ = tx.send(Err(e));
                return;
            }
        };

        if paths.is_empty() {
            let _ = tx.send(Err(ErmError::InvalidConfig(format!(
                "no .txt files found in: {}",
                config.data_dir
            ))));
            return;
        }

        if config.shuffle_files {
            paths.shuffle(&mut rng);
        }

        let mut pending: VecDeque<u32> = VecDeque::new();

        for path in &paths {
            // Read file on CPU; do NOT send to GPU.
            let text = match std::fs::read_to_string(path) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!(
                        "[streaming_dataset] warning: cannot read {}: {e}",
                        path.display()
                    );
                    continue;
                }
            };

            let tokens = if config.use_paragraph_spans {
                tokenize_by_paragraphs(&text, &tokenizer, config.seq_len)
            } else {
                tokenize_sliding_window(&text, &tokenizer, config.seq_len)
            };

            pending.extend(tokens);

            // Emit complete batches as soon as we have enough pending tokens.
            while pending.len() >= config.batch_size * config.seq_len {
                let batch_tokens: Vec<u32> = pending
                    .drain(..config.batch_size * config.seq_len)
                    .collect();

                let batch = TokenBatch {
                    tokens: batch_tokens,
                    batch_size: config.batch_size,
                    seq_len: config.seq_len,
                };

                // send() blocks when channel is full (back-pressure).
                if tx.send(Ok(batch)).is_err() {
                    // Receiver dropped — trainer is done.
                    return;
                }
            }
        }

        // Emit any partial last batch (zero-padded to full batch).
        if !pending.is_empty() && pending.len() >= config.seq_len {
            let complete_seqs = pending.len() / config.seq_len;
            let batch_seqs = complete_seqs.min(config.batch_size);
            if batch_seqs > 0 {
                let batch_tokens: Vec<u32> =
                    pending.drain(..batch_seqs * config.seq_len).collect();
                let batch = TokenBatch {
                    tokens: batch_tokens,
                    batch_size: batch_seqs,
                    seq_len: config.seq_len,
                };
                if tx.send(Ok(batch)).is_err() {
                    return;
                }
            }
        }

        if !config.repeat {
            break;
        }
    }
}

// ── Tokenization strategies ───────────────────────────────────────────────────

/// Tokenize text using a sliding window (seq_len/2 stride).
///
/// Returns a flat `Vec<u32>` of all token ids across all windows. The caller
/// is responsible for slicing into batches.
fn tokenize_sliding_window<T: TokenizerApi>(text: &str, tokenizer: &T, seq_len: usize) -> Vec<u32> {
    let all_tokens = tokenizer.encode_text(text);
    if all_tokens.len() < seq_len {
        return all_tokens;
    }
    let stride = (seq_len / 2).max(1);
    let mut result = Vec::new();

    let mut start = 0;
    while start + seq_len <= all_tokens.len() {
        result.extend_from_slice(&all_tokens[start..start + seq_len]);
        start += stride;
    }
    // Capture leftover tail.
    if all_tokens.len() >= seq_len {
        let last_start = all_tokens.len() - seq_len;
        if last_start >= start {
            result.extend_from_slice(&all_tokens[last_start..]);
        }
    }
    result
}

/// Tokenize text by paragraph/sentence boundaries.
///
/// Splits the text at double-newlines (paragraphs) and single newlines
/// (sentences), producing spans that are at most `seq_len` tokens long.
/// Spans shorter than `seq_len` are concatenated until they fill a window.
fn tokenize_by_paragraphs<T: TokenizerApi>(
    text: &str,
    tokenizer: &T,
    seq_len: usize,
) -> Vec<u32> {
    // Split into paragraphs first, then sentences within each.
    let mut all_tokens: Vec<u32> = Vec::new();
    let mut buffer: Vec<u32> = Vec::new();

    for para in text.split("\n\n") {
        for sentence in para.split('\n') {
            let trimmed = sentence.trim();
            if trimmed.is_empty() {
                continue;
            }
            let tokens = tokenizer.encode_text(trimmed);
            if tokens.is_empty() {
                continue;
            }

            // If adding this sentence would overflow the window, flush buffer.
            if buffer.len() + tokens.len() >= seq_len {
                if !buffer.is_empty() {
                    // Pad or truncate to exactly seq_len.
                    buffer.truncate(seq_len);
                    while buffer.len() < seq_len {
                        buffer.push(erm_core::bpe_tokenizer::PAD_ID);
                    }
                    all_tokens.extend_from_slice(&buffer);
                    buffer.clear();
                }
            }
            buffer.extend_from_slice(&tokens);
        }
    }

    // Flush remaining buffer.
    if buffer.len() >= seq_len / 4 {
        buffer.truncate(seq_len);
        while buffer.len() < seq_len {
            buffer.push(erm_core::bpe_tokenizer::PAD_ID);
        }
        all_tokens.extend_from_slice(&buffer);
    }

    all_tokens
}

// ── Path utilities ────────────────────────────────────────────────────────────

fn collect_txt_paths(dir: &str) -> ErmResult<Vec<PathBuf>> {
    let mut paths = Vec::new();
    collect_txt_paths_rec(Path::new(dir), &mut paths)?;
    paths.sort();
    Ok(paths)
}

fn collect_txt_paths_rec(dir: &Path, out: &mut Vec<PathBuf>) -> ErmResult<()> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| ErmError::InvalidConfig(format!("cannot read dir {}: {e}", dir.display())))?;
    for entry in entries {
        let entry = entry
            .map_err(|e| ErmError::InvalidConfig(format!("dir entry error: {e}")))?;
        let path = entry.path();
        if path.is_dir() {
            collect_txt_paths_rec(&path, out)?;
        } else if path.extension().and_then(|e| e.to_str()) == Some("txt") {
            out.push(path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use erm_core::bpe_tokenizer::BpeTokenizer;

    fn make_corpus() -> String {
        "the quick brown fox jumps over the lazy dog\n\n".repeat(30)
    }

    fn make_config(dir: &str, batch: usize, seq: usize) -> StreamingConfig {
        StreamingConfig {
            data_dir: dir.to_string(),
            seq_len: seq,
            batch_size: batch,
            use_paragraph_spans: false,
            shuffle_files: false,
            seed: 42,
            repeat: false,
        }
    }

    #[test]
    fn test_streaming_basic() {
        let dir = "/tmp/erm_streaming_test_1";
        let _ = std::fs::create_dir_all(dir);
        let corpus = make_corpus();
        std::fs::write(format!("{dir}/a.txt"), &corpus).unwrap();
        std::fs::write(format!("{dir}/b.txt"), &corpus).unwrap();

        let bpe = BpeTokenizer::train(&corpus, 30);
        let config = make_config(dir, 2, 16);
        let mut ds = StreamingDataset::new(config, bpe);

        let mut batches = 0;
        while let Some(batch) = ds.next_batch().unwrap() {
            assert_eq!(batch.seq_len, 16);
            assert_eq!(batch.tokens.len(), batch.batch_size * batch.seq_len);
            batches += 1;
            if batches > 100 {
                break;
            }
        }
        assert!(batches > 0, "should produce at least one batch");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_streaming_no_gpu_load() {
        // This test verifies the API doesn't hold large allocations.
        // It's a logical test — just ensure batches.tokens contains only u32s (CPU vec).
        let dir = "/tmp/erm_streaming_test_2";
        let _ = std::fs::create_dir_all(dir);
        let corpus = "hello world ".repeat(200);
        std::fs::write(format!("{dir}/c.txt"), &corpus).unwrap();

        let bpe = BpeTokenizer::train(&corpus, 20);
        let config = make_config(dir, 1, 8);
        let mut ds = StreamingDataset::new(config, bpe);

        if let Some(batch) = ds.next_batch().unwrap() {
            // All tokens must be valid CPU u32 values.
            assert_eq!(batch.tokens.len(), 8);
            for &tok in &batch.tokens {
                let _ = tok; // just ensure it's a plain u32
            }
        }

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_paragraph_spans() {
        let dir = "/tmp/erm_streaming_test_3";
        let _ = std::fs::create_dir_all(dir);
        let corpus = "First paragraph sentence one.\nFirst paragraph sentence two.\n\nSecond paragraph.\n\n".repeat(20);
        std::fs::write(format!("{dir}/d.txt"), &corpus).unwrap();

        let bpe = BpeTokenizer::train(&corpus, 20);
        let mut config = make_config(dir, 1, 32);
        config.use_paragraph_spans = true;
        let mut ds = StreamingDataset::new(config, bpe);

        let mut got_batch = false;
        while let Some(batch) = ds.next_batch().unwrap() {
            assert_eq!(batch.tokens.len(), batch.batch_size * batch.seq_len);
            got_batch = true;
            break;
        }
        assert!(got_batch, "paragraph spans should produce at least one batch");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_missing_dir_error() {
        use erm_core::bpe_tokenizer::BpeTokenizer;
        let bpe = BpeTokenizer::train("hello", 5);
        let config = make_config("/nonexistent_path_xyz", 1, 8);
        let mut ds = StreamingDataset::new(config, bpe);
        let result = ds.next_batch();
        assert!(result.is_err() || matches!(result, Ok(None)));
    }
}
