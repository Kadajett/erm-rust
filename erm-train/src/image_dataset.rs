//! Image dataset for training ERM on grayscale pixel data.
//!
//! Treats each pixel (0–255) as a token id in a flat sequence. Supports
//! PGM P5 (binary grayscale) format parsed without any external crates.
//!
//! # Token encoding
//!
//! Pixel value `p ∈ [0, 255]` maps directly to token id `p + 2` to reserve
//! `0` (PAD) and `1` (MASK) as per [`erm_core::tokenizer`] convention.
//!
//! # Chunking strategy
//!
//! Same 50% overlap strategy as [`crate::dataset::TextDataset`]:
//! stride = `seq_len / 2`.

use rand::seq::SliceRandom;
use rand::Rng;

use erm_core::error::{ErmError, ErmResult};

use crate::dataset::DataBatch;

/// Token id offset so that 0=PAD and 1=MASK are preserved.
const PIXEL_OFFSET: u32 = 2;

/// An image dataset backed by PGM P5 (binary grayscale) files.
///
/// Each image is flattened row-major and tokenised as pixel values `[0, 255]`
/// offset by [`PIXEL_OFFSET`] to avoid collisions with PAD/MASK.
#[derive(Debug, Clone)]
pub struct ImageDataset {
    /// All fixed-length pixel sequences.
    sequences: Vec<Vec<u32>>,
    /// Sequence length `L`.
    seq_len: usize,
    /// Sequential iteration cursor.
    cursor: usize,
    /// Epoch counter (increments on each full pass / shuffle).
    epoch: usize,
}

impl ImageDataset {
    /// Build a dataset from a single PGM P5 file.
    ///
    /// # Arguments
    ///
    /// - `path`: path to a `.pgm` file in binary P5 format.
    /// - `seq_len`: fixed sequence length `L`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read, is not valid P5 PGM, or
    /// produces fewer pixels than `seq_len`.
    pub fn from_file(path: &str, seq_len: usize) -> ErmResult<Self> {
        let pixels = load_pgm_p5(path)?;
        let sequences = chunk_pixels(&pixels, seq_len)?;
        Ok(Self {
            sequences,
            seq_len,
            cursor: 0,
            epoch: 0,
        })
    }

    /// Build a dataset by recursively scanning a directory for `*.pgm` files.
    ///
    /// Files are sorted by path for determinism before processing.
    ///
    /// # Arguments
    ///
    /// - `dir`: directory path to scan recursively.
    /// - `seq_len`: fixed sequence length `L`.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be read, or if no valid PGM
    /// files produce at least `seq_len` pixels.
    pub fn from_directory(dir: &str, seq_len: usize) -> ErmResult<Self> {
        let mut paths = Vec::new();
        collect_pgm_paths(dir, &mut paths)?;
        paths.sort();

        if paths.is_empty() {
            return Err(ErmError::InvalidConfig(format!(
                "no .pgm files found in directory: {dir}"
            )));
        }

        let mut all_sequences: Vec<Vec<u32>> = Vec::new();
        for path in &paths {
            match load_pgm_p5(path) {
                Ok(pixels) => match chunk_pixels(&pixels, seq_len) {
                    Ok(seqs) => all_sequences.extend(seqs),
                    Err(_) => {
                        // Image too small — skip silently.
                    }
                },
                Err(_) => {
                    // Not a valid PGM — skip silently.
                }
            }
        }

        if all_sequences.is_empty() {
            return Err(ErmError::InvalidConfig(format!(
                "no valid PGM images with >= {seq_len} pixels found in: {dir}"
            )));
        }

        Ok(Self {
            sequences: all_sequences,
            seq_len,
            cursor: 0,
            epoch: 0,
        })
    }

    /// Build a dataset from raw pixel bytes (useful for testing).
    ///
    /// # Arguments
    ///
    /// - `pixels`: grayscale pixel values `[0, 255]`.
    /// - `seq_len`: fixed sequence length `L`.
    ///
    /// # Errors
    ///
    /// Returns an error if `pixels.len() < seq_len`.
    pub fn from_pixels(pixels: &[u8], seq_len: usize) -> ErmResult<Self> {
        let sequences = chunk_pixels(pixels, seq_len)?;
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

    /// Current epoch number.
    #[must_use]
    pub fn epoch(&self) -> usize {
        self.epoch
    }

    /// Sequence length `L`.
    #[must_use]
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Vocabulary size for pixel tokens: 256 pixels + PAD + MASK = 258.
    #[must_use]
    pub fn vocab_size() -> usize {
        256 + PIXEL_OFFSET as usize
    }

    /// Sample a random batch with replacement.
    ///
    /// # Arguments
    ///
    /// - `batch_size`: number of sequences per batch.
    /// - `rng`: random number generator.
    ///
    /// # Returns
    ///
    /// A [`DataBatch`] with flat `[B * L]` pixel token ids.
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
    ///
    /// Call [`shuffle`](Self::shuffle) to begin a new epoch.
    ///
    /// # Arguments
    ///
    /// - `batch_size`: number of sequences per batch.
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

// ── PGM P5 parser ──────────────────────────────────────────────────────────

/// Parse a PGM P5 (binary grayscale) file and return pixel bytes.
///
/// # PGM P5 format
///
/// ```text
/// P5\n
/// <width> <height>\n
/// <maxval>\n
/// <binary pixel data: width*height bytes>
/// ```
///
/// Comments (`# ...`) may appear between header tokens.
///
/// # Errors
///
/// Returns [`ErmError::InvalidConfig`] for malformed or unsupported files.
fn load_pgm_p5(path: &str) -> ErmResult<Vec<u8>> {
    let data = std::fs::read(path)
        .map_err(|e| ErmError::InvalidConfig(format!("cannot read file {path}: {e}")))?;

    parse_pgm_p5_bytes(&data)
        .map_err(|e| ErmError::InvalidConfig(format!("invalid PGM file {path}: {e}")))
}

/// Parse PGM P5 from raw bytes.
///
/// # Errors
///
/// Returns a string error for malformed data.
fn parse_pgm_p5_bytes(data: &[u8]) -> Result<Vec<u8>, String> {
    let mut pos = 0usize;

    // Helper: skip whitespace and comments.
    let skip_ws_comments = |pos: &mut usize| {
        loop {
            // Skip whitespace.
            while *pos < data.len()
                && (data[*pos] == b' '
                    || data[*pos] == b'\t'
                    || data[*pos] == b'\n'
                    || data[*pos] == b'\r')
            {
                *pos += 1;
            }
            // Skip comment lines.
            if *pos < data.len() && data[*pos] == b'#' {
                while *pos < data.len() && data[*pos] != b'\n' {
                    *pos += 1;
                }
            } else {
                break;
            }
        }
    };

    // Helper: read ASCII token until whitespace.
    let read_token = |pos: &mut usize| -> Result<String, String> {
        let start = *pos;
        while *pos < data.len()
            && data[*pos] != b' '
            && data[*pos] != b'\t'
            && data[*pos] != b'\n'
            && data[*pos] != b'\r'
        {
            *pos += 1;
        }
        if *pos == start {
            return Err("unexpected end of file".to_string());
        }
        std::str::from_utf8(&data[start..*pos])
            .map(|s| s.to_string())
            .map_err(|e| format!("utf8 error: {e}"))
    };

    // 1. Magic number
    skip_ws_comments(&mut pos);
    let magic = read_token(&mut pos)?;
    if magic != "P5" {
        return Err(format!("expected P5, got {magic:?}"));
    }

    // 2. Width
    skip_ws_comments(&mut pos);
    let width_str = read_token(&mut pos)?;
    let width: usize = width_str
        .parse()
        .map_err(|e| format!("bad width {width_str:?}: {e}"))?;

    // 3. Height
    skip_ws_comments(&mut pos);
    let height_str = read_token(&mut pos)?;
    let height: usize = height_str
        .parse()
        .map_err(|e| format!("bad height {height_str:?}: {e}"))?;

    // 4. Max value
    skip_ws_comments(&mut pos);
    let maxval_str = read_token(&mut pos)?;
    let maxval: usize = maxval_str
        .parse()
        .map_err(|e| format!("bad maxval {maxval_str:?}: {e}"))?;
    if maxval > 255 {
        return Err(format!("maxval {maxval} > 255 (16-bit PGM not supported)"));
    }

    // 5. Exactly one whitespace byte before binary data.
    if pos >= data.len() {
        return Err("file truncated before pixel data".to_string());
    }
    pos += 1; // consume the single whitespace separator

    // 6. Binary pixel data: width * height bytes.
    let n_pixels = width * height;
    if pos + n_pixels > data.len() {
        return Err(format!(
            "expected {n_pixels} pixel bytes, only {} available",
            data.len() - pos
        ));
    }

    Ok(data[pos..pos + n_pixels].to_vec())
}

/// Write a minimal PGM P5 file (for tests).
///
/// # Errors
///
/// Returns an error if the file cannot be written.
pub fn write_pgm_p5(path: &str, width: usize, height: usize, pixels: &[u8]) -> ErmResult<()> {
    if pixels.len() != width * height {
        return Err(ErmError::ShapeMismatch {
            expected: format!("{}", width * height),
            got: format!("{}", pixels.len()),
        });
    }
    let header = format!("P5\n{width} {height}\n255\n");
    let mut buf = header.into_bytes();
    buf.extend_from_slice(pixels);
    std::fs::write(path, buf)
        .map_err(|e| ErmError::InvalidConfig(format!("cannot write pgm {path}: {e}")))
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Chunk a pixel slice into overlapping fixed-length token sequences.
///
/// Token id = pixel + [`PIXEL_OFFSET`].
/// Stride = `seq_len / 2` (50% overlap).
///
/// # Errors
///
/// Returns an error if `pixels.len() < seq_len`.
fn chunk_pixels(pixels: &[u8], seq_len: usize) -> ErmResult<Vec<Vec<u32>>> {
    if pixels.len() < seq_len {
        return Err(ErmError::InvalidConfig(format!(
            "image has {} pixels, need at least {seq_len}",
            pixels.len()
        )));
    }

    let stride = (seq_len / 2).max(1);
    let mut sequences = Vec::new();
    let mut start = 0;

    while start + seq_len <= pixels.len() {
        let seq: Vec<u32> = pixels[start..start + seq_len]
            .iter()
            .map(|&p| p as u32 + PIXEL_OFFSET)
            .collect();
        sequences.push(seq);
        start += stride;
    }

    // Capture any leftover tail.
    if start < pixels.len() && pixels.len() >= seq_len {
        let last_start = pixels.len() - seq_len;
        if sequences.is_empty() || last_start != start - stride {
            let seq: Vec<u32> = pixels[last_start..]
                .iter()
                .map(|&p| p as u32 + PIXEL_OFFSET)
                .collect();
            sequences.push(seq);
        }
    }

    Ok(sequences)
}

/// Recursively collect all `*.pgm` file paths under `dir`.
fn collect_pgm_paths(dir: &str, out: &mut Vec<String>) -> ErmResult<()> {
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
            collect_pgm_paths(&sub, out)?;
        } else if path.extension().and_then(|e| e.to_str()) == Some("pgm") {
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

    // ── PGM parser tests ──────────────────────────────────────────────────

    #[test]
    fn test_pgm_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_roundtrip.pgm");
        let path_str = path.to_str().unwrap();
        let pixels: Vec<u8> = (0u8..64).collect();
        write_pgm_p5(path_str, 8, 8, &pixels).unwrap();
        let loaded = load_pgm_p5(path_str).unwrap();
        assert_eq!(loaded, pixels);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_pgm_comment_skip() {
        // PGM with comment lines in header.
        let data = b"P5\n# a comment\n4 2\n# another\n255\n\x00\x01\x02\x03\x04\x05\x06\x07";
        let pixels = parse_pgm_p5_bytes(data).unwrap();
        assert_eq!(pixels, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_pgm_wrong_magic() {
        let data = b"P2\n4 2\n255\n";
        assert!(parse_pgm_p5_bytes(data).is_err());
    }

    #[test]
    fn test_pgm_truncated() {
        let data = b"P5\n4 4\n255\n\x00\x01\x02"; // only 3 bytes for 16 pixels
        assert!(parse_pgm_p5_bytes(data).is_err());
    }

    // ── Dataset construction ──────────────────────────────────────────────

    #[test]
    fn test_from_pixels_basic() {
        let pixels: Vec<u8> = (0..64).map(|i| (i * 4) as u8).collect();
        let ds = ImageDataset::from_pixels(&pixels, 16).unwrap();
        assert!(!ds.is_empty());
        // 64 pixels, stride 8 → chunks at 0, 8, 16, 24, 32, 40, 48 → 7 full + maybe tail
        assert!(ds.len() >= 7);
        assert_eq!(ds.seq_len(), 16);
        assert_eq!(ds.epoch(), 0);
    }

    #[test]
    fn test_from_pixels_too_short() {
        let pixels: Vec<u8> = vec![1, 2, 3];
        assert!(ImageDataset::from_pixels(&pixels, 16).is_err());
    }

    #[test]
    fn test_pixel_offset_applied() {
        let pixels = vec![0u8, 128, 255];
        let ds = ImageDataset::from_pixels(&pixels, 3).unwrap();
        // Single sequence: [2, 130, 257]
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let batch = ds.get_batch(1, &mut rng);
        assert_eq!(batch.tokens[0], 2); // 0 + 2
        assert_eq!(batch.tokens[1], 130); // 128 + 2
        assert_eq!(batch.tokens[2], 257); // 255 + 2
    }

    #[test]
    fn test_vocab_size() {
        assert_eq!(ImageDataset::vocab_size(), 258);
    }

    // ── get_batch ─────────────────────────────────────────────────────────

    #[test]
    fn test_get_batch_shape() {
        let pixels: Vec<u8> = (0..128).collect();
        let ds = ImageDataset::from_pixels(&pixels, 16).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let batch = ds.get_batch(4, &mut rng);
        assert_eq!(batch.batch_size, 4);
        assert_eq!(batch.seq_len, 16);
        assert_eq!(batch.tokens.len(), 4 * 16);
    }

    #[test]
    fn test_get_batch_token_range() {
        let pixels: Vec<u8> = (0..128).collect();
        let ds = ImageDataset::from_pixels(&pixels, 16).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let batch = ds.get_batch(8, &mut rng);
        for &tok in &batch.tokens {
            assert!(tok >= 2, "token {tok} below PIXEL_OFFSET");
            assert!(tok <= 257, "token {tok} above max pixel token");
        }
    }

    // ── next_batch / shuffle ──────────────────────────────────────────────

    #[test]
    fn test_next_batch_sequential() {
        let pixels: Vec<u8> = (0..128u8).collect();
        let mut ds = ImageDataset::from_pixels(&pixels, 16).unwrap();
        let mut count = 0;
        while let Some(batch) = ds.next_batch(2) {
            assert_eq!(batch.tokens.len(), 2 * 16);
            count += 1;
        }
        assert!(count > 0, "should have produced at least one batch");
    }

    #[test]
    fn test_next_batch_exhausted_returns_none() {
        let pixels: Vec<u8> = (0..32u8).collect();
        let mut ds = ImageDataset::from_pixels(&pixels, 16).unwrap();
        // Exhaust all sequences.
        while ds.next_batch(1).is_some() {}
        assert!(ds.next_batch(1).is_none());
    }

    #[test]
    fn test_shuffle_increments_epoch() {
        let pixels: Vec<u8> = (0..128u8).collect();
        let mut ds = ImageDataset::from_pixels(&pixels, 16).unwrap();
        assert_eq!(ds.epoch(), 0);
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        ds.shuffle(&mut rng);
        assert_eq!(ds.epoch(), 1);
        ds.shuffle(&mut rng);
        assert_eq!(ds.epoch(), 2);
    }

    #[test]
    fn test_shuffle_resets_cursor() {
        let pixels: Vec<u8> = (0..128u8).collect();
        let mut ds = ImageDataset::from_pixels(&pixels, 16).unwrap();
        // Exhaust.
        while ds.next_batch(1).is_some() {}
        assert!(ds.next_batch(1).is_none());
        // Shuffle → cursor reset.
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        ds.shuffle(&mut rng);
        assert!(ds.next_batch(1).is_some());
    }

    // ── from_directory ────────────────────────────────────────────────────

    #[test]
    fn test_from_directory() {
        let dir = tempdir_with_pgms();
        let ds = ImageDataset::from_directory(&dir, 16).unwrap();
        assert!(!ds.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_from_directory_missing() {
        assert!(ImageDataset::from_directory("/nonexistent/path/xyz", 16).is_err());
    }

    #[test]
    fn test_from_directory_no_pgms() {
        let dir = std::env::temp_dir()
            .join("erm_no_pgms")
            .to_str()
            .unwrap()
            .to_string();
        let _ = std::fs::create_dir_all(&dir);
        // Put a non-pgm file.
        let _ = std::fs::write(format!("{dir}/file.txt"), b"hello");
        let result = ImageDataset::from_directory(&dir, 16);
        assert!(result.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    /// Create a temp directory with a few PGM files for testing.
    fn tempdir_with_pgms() -> String {
        let dir = std::env::temp_dir()
            .join("erm_pgm_test")
            .to_str()
            .unwrap()
            .to_string();
        let _ = std::fs::create_dir_all(&dir);
        let pixels: Vec<u8> = (0u8..=255u8).collect();
        write_pgm_p5(&format!("{dir}/a.pgm"), 16, 16, &pixels).unwrap();
        write_pgm_p5(&format!("{dir}/b.pgm"), 16, 16, &pixels).unwrap();
        dir
    }
}
