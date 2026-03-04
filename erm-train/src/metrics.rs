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
    /// Maximum pheromone level across all route graph edges.
    #[serde(default)]
    pub max_phi: f32,
    /// Mean taint level across all route graph edges.
    #[serde(default)]
    pub mean_taint: f32,
    /// Number of edges with taint > 0.
    #[serde(default)]
    pub tainted_count: usize,
    /// Number of ant deaths (and respawns) this step.
    pub deaths: usize,
    /// Number of active (non-empty) edges.
    #[serde(default)]
    pub active_edges: usize,
    /// Number of active leader-introduced edges.
    #[serde(default)]
    pub leader_edges: usize,
    /// Fraction of active edges introduced by leaders.
    #[serde(default)]
    pub leader_edge_fraction: f32,
    /// Mean edge age across active edges.
    #[serde(default)]
    pub mean_age: f32,
    /// Maximum edge age across active edges.
    #[serde(default)]
    pub max_age: u32,
    /// Fraction of active edges clamped at `phi_max`.
    #[serde(default)]
    pub phi_clamped_fraction: f32,
    /// Fraction of active edges clamped at `taint_max`.
    #[serde(default)]
    pub taint_clamped_fraction: f32,
    /// Mean per-destination edge-weight entropy.
    #[serde(default)]
    pub edge_weight_entropy_mean: f32,
    /// Mean per-destination top-1 edge share.
    #[serde(default)]
    pub top1_edge_share_mean: f32,
    /// Fraction of previous-step leader edges still present.
    #[serde(default)]
    pub leader_edge_survival_rate: f32,
    /// Number of edges pruned during this step.
    #[serde(default)]
    pub edges_pruned: usize,
    /// Number of edges inserted during this step.
    #[serde(default)]
    pub edges_inserted: usize,
    /// Sequence length used in this experiment.
    pub seq_len: usize,
    /// Batch size used in this experiment.
    pub batch: usize,
    /// Hidden dimension of the scorer network.
    pub hidden_dim: usize,
    /// Current learning rate.
    pub lr: f64,
    /// Current follower temperature.
    pub follower_temp: f32,
    /// Current leader temperature.
    pub leader_temp: f32,
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
                ErmError::IoError(std::io::Error::other(format!(
                    "cannot create metrics dir: {e}"
                )))
            })?;
        }

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| {
                ErmError::IoError(std::io::Error::other(format!(
                    "cannot open metrics file {path}: {e}"
                )))
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
        if !step.is_multiple_of(self.log_every) {
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
        let line = serde_json::to_string(record)
            .map_err(|e| ErmError::InvalidConfig(format!("metrics serialization failed: {e}")))?;
        writeln!(self.writer, "{line}").map_err(|e| {
            ErmError::IoError(std::io::Error::other(format!("metrics write failed: {e}")))
        })?;
        self.writer.flush().map_err(|e| {
            ErmError::IoError(std::io::Error::other(format!("metrics flush failed: {e}")))
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
            max_phi: 0.9,
            mean_taint: 0.1,
            tainted_count: 3,
            deaths: 3,
            active_edges: 64,
            leader_edges: 12,
            leader_edge_fraction: 0.1875,
            mean_age: 4.0,
            max_age: 10,
            phi_clamped_fraction: 0.02,
            taint_clamped_fraction: 0.01,
            edge_weight_entropy_mean: 0.55,
            top1_edge_share_mean: 0.73,
            leader_edge_survival_rate: 0.88,
            edges_pruned: 5,
            edges_inserted: 7,
            seq_len: 512,
            batch: 2,
            hidden_dim: 192,
            lr: 0.0005,
            follower_temp: 1.0,
            leader_temp: 1.0,
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
        assert_eq!(rec.active_edges, 64);
        assert!((rec.leader_edge_fraction - 0.1875).abs() < 1e-6);
        assert_eq!(rec.edges_pruned, 5);

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
        assert_eq!(
            content.lines().count(),
            2,
            "should have 2 records after 2 opens"
        );
        let _ = std::fs::remove_file(path);
    }
}
