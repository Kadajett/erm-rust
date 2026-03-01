use std::path::Path;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use erm_core::error::{ErmError, ErmResult};
use erm_core::tokenizer::CharTokenizer;
use erm_core::types::TokenId;

#[derive(Debug, Clone)]
pub struct DataBatch {
    pub x: Vec<TokenId>,
    pub y_t: Vec<TokenId>,
    pub editable: Vec<bool>,
    pub t: Vec<i32>,
    pub batch_size: usize,
    pub seq_len: usize,
}

impl DataBatch {
    #[must_use]
    pub fn x_row(&self, b: usize) -> &[TokenId] {
        let start = b * self.seq_len;
        &self.x[start..start + self.seq_len]
    }
}

#[derive(Debug, Clone)]
pub struct DatasetConfig {
    pub seq_len: usize,
    pub stride: usize,
    pub batch_size: usize,
    pub seed: u64,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            seq_len: 128,
            stride: 128,
            batch_size: 8,
            seed: 42,
        }
    }
}

#[derive(Debug)]
pub struct TextDataset {
    sequences: Vec<Vec<TokenId>>,
    config: DatasetConfig,
    rng: ChaCha8Rng,
    cursor: usize,
    shuffled_indices: Vec<usize>,
}

impl TextDataset {
    pub fn from_ids(ids: Vec<TokenId>, config: DatasetConfig) -> ErmResult<Self> {
        if config.seq_len == 0 {
            return Err(ErmError::InvalidConfig("seq_len must be > 0".into()));
        }
        let mut sequences = Vec::new();
        let mut start = 0;
        while start < ids.len() {
            let end = (start + config.seq_len).min(ids.len());
            let mut seq: Vec<TokenId> = ids[start..end].to_vec();
            while seq.len() < config.seq_len {
                seq.push(0); // PAD
            }
            sequences.push(seq);
            start += config.stride;
        }
        let n = sequences.len();
        let rng = ChaCha8Rng::seed_from_u64(config.seed);
        let shuffled_indices: Vec<usize> = (0..n).collect();
        Ok(Self {
            sequences,
            config,
            rng,
            cursor: 0,
            shuffled_indices,
        })
    }

    pub fn from_sequences(sequences: Vec<Vec<TokenId>>, config: DatasetConfig) -> ErmResult<Self> {
        for (i, seq) in sequences.iter().enumerate() {
            if seq.len() != config.seq_len {
                return Err(ErmError::ShapeMismatch {
                    expected: format!("seq_len={}", config.seq_len),
                    got: format!("sequence[{i}] has length {}", seq.len()),
                });
            }
        }
        let n = sequences.len();
        let rng = ChaCha8Rng::seed_from_u64(config.seed);
        let shuffled_indices: Vec<usize> = (0..n).collect();
        Ok(Self {
            sequences,
            config,
            rng,
            cursor: 0,
            shuffled_indices,
        })
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.sequences.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    pub fn shuffle(&mut self) {
        self.shuffled_indices.shuffle(&mut self.rng);
        self.cursor = 0;
    }

    #[must_use]
    pub fn next_batch(&mut self) -> Option<DataBatch> {
        let b = self.config.batch_size;
        if self.cursor + b > self.shuffled_indices.len() {
            return None;
        }
        let indices = &self.shuffled_indices[self.cursor..self.cursor + b];
        let l = self.config.seq_len;
        let mut x = Vec::with_capacity(b * l);
        for &idx in indices {
            x.extend_from_slice(&self.sequences[idx]);
        }
        self.cursor += b;
        Some(DataBatch {
            y_t: x.clone(),
            editable: vec![true; b * l],
            t: vec![1; b],
            x,
            batch_size: b,
            seq_len: l,
        })
    }
}

pub fn synthetic_dataset_with_batch(num_sequences: usize, seq_len: usize, vocab_size: usize, batch_size: usize, seed: u64) -> ErmResult<TextDataset> {
    use rand::Rng;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let sequences: Vec<Vec<TokenId>> = (0..num_sequences)
        .map(|_| (0..seq_len).map(|_| rng.gen_range(0..vocab_size as TokenId)).collect())
        .collect();
    let config = DatasetConfig { seq_len, stride: seq_len, batch_size, seed };
    TextDataset::from_sequences(sequences, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_from_ids_basic() {
        let ids: Vec<TokenId> = (0..32).collect();
        let cfg = DatasetConfig { seq_len: 8, stride: 8, batch_size: 4, seed: 42 };
        let ds = TextDataset::from_ids(ids, cfg).unwrap();
        assert_eq!(ds.len(), 4);
    }
}
