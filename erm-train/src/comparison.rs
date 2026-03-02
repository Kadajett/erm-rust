//! Comparison harness for ERM experiments.
//!
//! Aggregates evaluation metrics from multiple runs and produces a structured
//! JSON comparison report.

use serde::{Deserialize, Serialize};

use erm_core::error::{ErmError, ErmResult};

use crate::eval::EvalMetrics;

/// A labelled evaluation result for comparison across runs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ComparisonMetrics {
    /// Human-readable label for this run/configuration.
    pub label: String,
    /// Core evaluation metrics.
    pub metrics: EvalMetrics,
    /// Delta in masked-token accuracy relative to the first entry.
    pub accuracy_delta: Option<f32>,
    /// Delta in token entropy relative to the first entry.
    pub entropy_delta: Option<f32>,
    /// Delta in average loss relative to the first entry.
    pub loss_delta: Option<f32>,
}

impl ComparisonMetrics {
    /// Create a new entry with no delta fields set (acts as baseline).
    #[must_use]
    pub fn new(label: &str, metrics: EvalMetrics) -> Self {
        Self {
            label: label.to_string(),
            metrics,
            accuracy_delta: None,
            entropy_delta: None,
            loss_delta: None,
        }
    }

    /// Create an entry with deltas computed relative to `baseline`.
    #[must_use]
    pub fn with_baseline(label: &str, metrics: EvalMetrics, baseline: &EvalMetrics) -> Self {
        let accuracy_delta = match (
            metrics.masked_token_accuracy,
            baseline.masked_token_accuracy,
        ) {
            (Some(m), Some(b)) => Some(m - b),
            _ => None,
        };
        let entropy_delta = match (metrics.token_entropy, baseline.token_entropy) {
            (Some(m), Some(b)) => Some(m - b),
            _ => None,
        };
        let loss_delta = match (metrics.avg_loss, baseline.avg_loss) {
            (Some(m), Some(b)) => Some(m - b),
            _ => None,
        };
        Self {
            label: label.to_string(),
            metrics,
            accuracy_delta,
            entropy_delta,
            loss_delta,
        }
    }
}

/// A full comparison report containing multiple labelled entries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonReport {
    /// Timestamp (ISO-8601 string, best-effort).
    pub timestamp: String,
    /// All comparison entries.
    pub entries: Vec<ComparisonMetrics>,
    /// Summary: label of the entry with the highest masked token accuracy.
    pub best_accuracy_label: Option<String>,
    /// Summary: label of the entry with the lowest average loss.
    pub best_loss_label: Option<String>,
}

impl ComparisonReport {
    /// Build a report from a list of labelled metrics.
    ///
    /// Computes deltas relative to the first entry (treated as baseline).
    /// Identifies the best-accuracy and lowest-loss entries.
    #[must_use]
    pub fn build(entries_raw: &[ComparisonMetrics]) -> Self {
        if entries_raw.is_empty() {
            return Self {
                timestamp: timestamp_now(),
                entries: Vec::new(),
                best_accuracy_label: None,
                best_loss_label: None,
            };
        }

        let baseline_metrics = &entries_raw[0].metrics;
        let mut entries: Vec<ComparisonMetrics> = Vec::with_capacity(entries_raw.len());
        entries.push(entries_raw[0].clone());
        for e in &entries_raw[1..] {
            entries.push(ComparisonMetrics::with_baseline(
                &e.label,
                e.metrics.clone(),
                baseline_metrics,
            ));
        }

        let best_accuracy_label = entries
            .iter()
            .filter_map(|e| {
                e.metrics
                    .masked_token_accuracy
                    .map(|a| (e.label.clone(), a))
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(label, _)| label);

        let best_loss_label = entries
            .iter()
            .filter_map(|e| e.metrics.avg_loss.map(|l| (e.label.clone(), l)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(label, _)| label);

        Self {
            timestamp: timestamp_now(),
            entries,
            best_accuracy_label,
            best_loss_label,
        }
    }
}

/// Serialise a list of [`ComparisonMetrics`] to a JSON file.
///
/// # Errors
///
/// Returns an error if serialisation fails or the file cannot be written.
pub fn save_comparison_report(path: &str, entries: &[ComparisonMetrics]) -> ErmResult<()> {
    let report = ComparisonReport::build(entries);
    let json = serde_json::to_string_pretty(&report)?;
    std::fs::write(path, json).map_err(|e| {
        ErmError::InvalidConfig(format!("cannot write comparison report to {path}: {e}"))
    })
}

/// Load a comparison report from a JSON file.
///
/// # Errors
///
/// Returns an error if the file cannot be read or deserialised.
pub fn load_comparison_report(path: &str) -> ErmResult<ComparisonReport> {
    let json = std::fs::read_to_string(path)
        .map_err(|e| ErmError::InvalidConfig(format!("cannot read report {path}: {e}")))?;
    serde_json::from_str(&json).map_err(ErmError::SerdeError)
}

fn timestamp_now() -> String {
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => {
            let secs = d.as_secs();
            let s = secs % 60;
            let m = (secs / 60) % 60;
            let h = (secs / 3600) % 24;
            let days = secs / 86400;
            let year = 1970 + days / 365;
            let day_of_year = days % 365;
            let month = day_of_year / 30 + 1;
            let day = day_of_year % 30 + 1;
            format!("{year:04}-{month:02}-{day:02}T{h:02}:{m:02}:{s:02}Z")
        }
        Err(_) => "unknown".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_metrics(acc: f32, loss: f32, entropy: f32) -> EvalMetrics {
        EvalMetrics {
            masked_token_accuracy: Some(acc),
            token_entropy: Some(entropy),
            unique_token_ratio: Some(0.3),
            num_positions: 100,
            avg_loss: Some(loss),
        }
    }

    #[test]
    fn test_comparison_metrics_new() {
        let m = sample_metrics(0.5, 1.5, 2.0);
        let cm = ComparisonMetrics::new("baseline", m.clone());
        assert_eq!(cm.label, "baseline");
        assert_eq!(cm.metrics, m);
        assert!(cm.accuracy_delta.is_none());
    }

    #[test]
    fn test_comparison_metrics_with_baseline() {
        let base = sample_metrics(0.4, 2.0, 2.5);
        let exp = sample_metrics(0.6, 1.5, 3.0);
        let cm = ComparisonMetrics::with_baseline("exp", exp, &base);
        let acc_delta = cm.accuracy_delta.unwrap();
        let loss_delta = cm.loss_delta.unwrap();
        let ent_delta = cm.entropy_delta.unwrap();
        assert!((acc_delta - 0.2).abs() < 1e-5);
        assert!((loss_delta - (-0.5)).abs() < 1e-5);
        assert!((ent_delta - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_report_build_empty() {
        let report = ComparisonReport::build(&[]);
        assert!(report.entries.is_empty());
        assert!(report.best_accuracy_label.is_none());
    }

    #[test]
    fn test_report_identifies_best() {
        let entries = vec![
            ComparisonMetrics::new("low", sample_metrics(0.3, 2.5, 2.0)),
            ComparisonMetrics::new("mid", sample_metrics(0.5, 1.8, 3.0)),
            ComparisonMetrics::new("high", sample_metrics(0.7, 1.0, 3.5)),
        ];
        let report = ComparisonReport::build(&entries);
        assert_eq!(report.best_accuracy_label.as_deref(), Some("high"));
        assert_eq!(report.best_loss_label.as_deref(), Some("high"));
    }

    #[test]
    fn test_save_load_roundtrip() {
        let entries = vec![
            ComparisonMetrics::new("a", sample_metrics(0.4, 2.0, 2.5)),
            ComparisonMetrics::new("b", sample_metrics(0.6, 1.5, 3.1)),
        ];

        let path = format!(
            "/tmp/erm_comparison_{}.json",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );

        save_comparison_report(&path, &entries).unwrap();
        let report = load_comparison_report(&path).unwrap();

        assert_eq!(report.entries.len(), 2);
        assert_eq!(report.entries[0].label, "a");
        assert_eq!(report.entries[1].label, "b");

        let _ = std::fs::remove_file(&path);
    }
}
