//! Metrics logging for ERM training experiments.
//!
//! Writes structured JSONL records to a file every N steps. Each record
//! contains the experiment id, step, loss, and colony statistics.
//!
//! # Record format
//!
//! ```json
//! {"exp_id":"exp-a","step":50,"loss":3.42,"edits":47,"mean_phi":0.12,
//!  "deaths":18,"seq_len":512,"batch":2,"hidden_dim":192}
//! ```
//!
//! Records are appended (not overwritten) so a crashed run can be resumed and
//! the existing records remain intact.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};

use serde::{Deserialize, Serialize};

use erm_core::error::{ErmError, ErmResult};

/// A single metrics record emitted every `log_every` steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsRecord {
    /// Experiment identifier (e.g., "exp-a").
    pub exp_id: String,
    /// Training step number.
    pub step: usize,
    /// Denoising loss for this step.
    pub loss: f32,
    /// Number of colony edits applied.
    pub edits: usize,
    /// Mean pheromone level across all route graph edges.
    pub mean_phi: f32,
    /// Number of ant deaths (and respawns) this step.
    pub deaths: usize,
    /// Sequence length used in this experiment.
    pub seq_len: usize,
    /// Batch size used in this experiment.
    pub batch: usize,
    /// Hidden dimension of the scorer network.
    pub hidden_dim: usize,
}

/// JSONL metrics writer.
///
/// Appends one record per write call. Thread-safety is not provided; call from
/// the training thread only.
pub struct MetricsWriter {
    writer: BufWriter<File>,
    /// Log every N steps.
    pub log_every: usize,
}

impl MetricsWriter {
    /// Open a JSONL metrics file for appending.
    ///
    /// Creates the file (and any missing parent directories) if it doesn't exist.
    ///
    /// # Arguments
    ///
    /// - `path`: path to the JSONL file.
    /// - `log_every`: emit a record every this many steps.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened for appending.
    pub fn open(path: &str, log_every: usize) -> ErmResult<Self> {
        // Create parent directories.
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                ErmError::IoError(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("cannot create metrics dir: {e}"),
                ))
            })?;
        }

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| {
                ErmError::IoError(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("cannot open metrics file {path}: {e}"),
                ))
            })?;

        Ok(Self {
            writer: BufWriter::new(file),
            log_every,
        })
    }

    /// Write a record if `step` is a multiple of `log_every`.
    ///
    /// Returns `true` if a record was written, `false` if skipped.
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails.
    pub fn maybe_write(&mut self, step: usize, record: &MetricsRecord) -> ErmResult<bool> {
        if step % self.log_every != 0 {
            return Ok(false);
        }
        self.write(record)?;
        Ok(true)
    }

    /// Unconditionally write a record.
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails.
    pub fn write(&mut self, record: &MetricsRecord) -> ErmResult<()> {
        let line = serde_json::to_string(record).map_err(|e| {
            ErmError::InvalidConfig(format!("metrics serialization failed: {e}"))
        })?;
        writeln!(self.writer, "{line}").map_err(|e| {
            ErmError::IoError(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("metrics write failed: {e}"),
            ))
        })?;
        self.writer.flush().map_err(|e| {
            ErmError::IoError(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("metrics flush failed: {e}"),
            ))
        })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_record(step: usize) -> MetricsRecord {
        MetricsRecord {
            exp_id: "exp-test".to_string(),
            step,
            loss: 3.14,
            edits: 42,
            mean_phi: 0.5,
            deaths: 3,
            seq_len: 512,
            batch: 2,
            hidden_dim: 192,
        }
    }

    #[test]
    fn test_write_and_read() {
        let path = "/tmp/erm_metrics_test.jsonl";
        let _ = std::fs::remove_file(path);

        let mut writer = MetricsWriter::open(path, 50).unwrap();
        writer.write(&test_record(50)).unwrap();
        writer.write(&test_record(100)).unwrap();

        let content = std::fs::read_to_string(path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2);

        let rec: MetricsRecord = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(rec.step, 50);
        assert_eq!(rec.exp_id, "exp-test");
        assert!((rec.loss - 3.14).abs() < 1e-3);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_maybe_write_skips() {
        let path = "/tmp/erm_metrics_maybe_test.jsonl";
        let _ = std::fs::remove_file(path);

        let mut writer = MetricsWriter::open(path, 50).unwrap();
        let wrote = writer.maybe_write(49, &test_record(49)).unwrap();
        assert!(!wrote);
        let wrote = writer.maybe_write(50, &test_record(50)).unwrap();
        assert!(wrote);

        let content = std::fs::read_to_string(path).unwrap();
        assert_eq!(content.lines().count(), 1);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_append_mode() {
        let path = "/tmp/erm_metrics_append_test.jsonl";
        let _ = std::fs::remove_file(path);

        {
            let mut w = MetricsWriter::open(path, 1).unwrap();
            w.write(&test_record(1)).unwrap();
        }
        {
            let mut w = MetricsWriter::open(path, 1).unwrap();
            w.write(&test_record(2)).unwrap();
        }

        let content = std::fs::read_to_string(path).unwrap();
        assert_eq!(content.lines().count(), 2, "should have 2 records after 2 opens");
        let _ = std::fs::remove_file(path);
    }
}
