//! Feature Extraction for GPU Processing
//!
//! Implements Section 4.3: Feature Extraction
//! Converts OIP defect classifications into GPU-friendly numerical features
//!
//! OPT-001: Integrated BatchProcessor for efficient bulk extraction

use crate::citl::{ErrorCodeClass, SuggestionApplicability};
use crate::perf::{BatchProcessor, PerfStats};
use anyhow::Result;
use chrono::{Datelike, Timelike};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Commit features optimized for GPU processing
///
/// NLP-014: Extended from 8 to 14 dimensions for CITL integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitFeatures {
    // Categorical (one-hot encoded for GPU)
    pub defect_category: u8, // 0-17 (18 categories from OIP)

    // Numerical (GPU-native f32)
    pub files_changed: f32,
    pub lines_added: f32,
    pub lines_deleted: f32,
    pub complexity_delta: f32, // Cyclomatic complexity change

    // Temporal
    pub timestamp: f64,  // Unix epoch
    pub hour_of_day: u8, // 0-23 (circadian patterns)
    pub day_of_week: u8, // 0-6

    // NLP-014: CITL features (6 new dims)
    /// Error code class: 0=type, 1=borrow, 2=name, 3=trait, 4=other
    #[serde(default)]
    pub error_code_class: u8,
    /// Whether a suggestion was provided: 0 or 1
    #[serde(default)]
    pub has_suggestion: u8,
    /// Suggestion applicability: 0=none, 1=machine, 2=maybe, 3=placeholder
    #[serde(default)]
    pub suggestion_applicability: u8,
    /// Count of clippy lints (0-255)
    #[serde(default)]
    pub clippy_lint_count: u8,
    /// Distance from function start (normalized span line delta)
    #[serde(default)]
    pub span_line_delta: f32,
    /// Diagnostic confidence from taxonomy mapping
    #[serde(default)]
    pub diagnostic_confidence: f32,
}

impl CommitFeatures {
    /// Convert to flat vector for GPU processing
    ///
    /// Fixed-size vector enables efficient GPU batching
    /// NLP-014: Extended to 14 dimensions
    pub fn to_vector(&self) -> Vec<f32> {
        vec![
            self.defect_category as f32,
            self.files_changed,
            self.lines_added,
            self.lines_deleted,
            self.complexity_delta,
            self.timestamp as f32,
            self.hour_of_day as f32,
            self.day_of_week as f32,
            // NLP-014: CITL features
            self.error_code_class as f32,
            self.has_suggestion as f32,
            self.suggestion_applicability as f32,
            self.clippy_lint_count as f32,
            self.span_line_delta,
            self.diagnostic_confidence,
        ]
    }

    /// Vector dimension count (for GPU buffer allocation)
    /// NLP-014: Extended from 8 to 14 dimensions
    pub const DIMENSION: usize = 14;
}

impl Default for CommitFeatures {
    fn default() -> Self {
        Self {
            defect_category: 0,
            files_changed: 0.0,
            lines_added: 0.0,
            lines_deleted: 0.0,
            complexity_delta: 0.0,
            timestamp: 0.0,
            hour_of_day: 0,
            day_of_week: 0,
            error_code_class: ErrorCodeClass::Other.as_u8(),
            has_suggestion: 0,
            suggestion_applicability: SuggestionApplicability::None.as_u8(),
            clippy_lint_count: 0,
            span_line_delta: 0.0,
            diagnostic_confidence: 0.0,
        }
    }
}

/// Extract features from OIP defect record
pub struct FeatureExtractor;

impl FeatureExtractor {
    pub fn new() -> Self {
        Self
    }

    /// Extract features from defect category and metadata
    ///
    /// Uses default values for CITL fields (backwards compatible)
    pub fn extract(
        &self,
        category: u8,
        files_changed: usize,
        lines_added: usize,
        lines_deleted: usize,
        timestamp: i64,
    ) -> Result<CommitFeatures> {
        self.extract_with_citl(
            category,
            files_changed,
            lines_added,
            lines_deleted,
            timestamp,
            ErrorCodeClass::Other,
            false,
            SuggestionApplicability::None,
            0,
            0.0,
            0.0,
        )
    }

    /// Extract features with CITL diagnostic information (NLP-014)
    #[allow(clippy::too_many_arguments)]
    pub fn extract_with_citl(
        &self,
        category: u8,
        files_changed: usize,
        lines_added: usize,
        lines_deleted: usize,
        timestamp: i64,
        error_code_class: ErrorCodeClass,
        has_suggestion: bool,
        suggestion_applicability: SuggestionApplicability,
        clippy_lint_count: u8,
        span_line_delta: f32,
        diagnostic_confidence: f32,
    ) -> Result<CommitFeatures> {
        // Convert timestamp to hour/day
        let datetime = chrono::DateTime::from_timestamp(timestamp, 0)
            .ok_or_else(|| anyhow::anyhow!("Invalid timestamp"))?;

        let hour_of_day = datetime.hour() as u8;
        let day_of_week = datetime.weekday().num_days_from_monday() as u8;

        Ok(CommitFeatures {
            defect_category: category,
            files_changed: files_changed as f32,
            lines_added: lines_added as f32,
            lines_deleted: lines_deleted as f32,
            complexity_delta: 0.0, // Will compute in future iteration
            timestamp: timestamp as f64,
            hour_of_day,
            day_of_week,
            // NLP-014: CITL features
            error_code_class: error_code_class.as_u8(),
            has_suggestion: u8::from(has_suggestion),
            suggestion_applicability: suggestion_applicability.as_u8(),
            clippy_lint_count,
            span_line_delta,
            diagnostic_confidence,
        })
    }
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Input data for batch feature extraction
#[derive(Debug, Clone)]
pub struct FeatureInput {
    pub category: u8,
    pub files_changed: usize,
    pub lines_added: usize,
    pub lines_deleted: usize,
    pub timestamp: i64,
}

/// Batch feature extractor with performance tracking
///
/// OPT-001: Uses BatchProcessor for efficient bulk extraction
pub struct BatchFeatureExtractor {
    extractor: FeatureExtractor,
    batch_processor: BatchProcessor<FeatureInput>,
    stats: PerfStats,
}

impl BatchFeatureExtractor {
    /// Create batch extractor with default batch size (1000)
    pub fn new() -> Self {
        Self::with_batch_size(1000)
    }

    /// Create batch extractor with custom batch size
    pub fn with_batch_size(batch_size: usize) -> Self {
        Self {
            extractor: FeatureExtractor::new(),
            batch_processor: BatchProcessor::new(batch_size),
            stats: PerfStats::new(),
        }
    }

    /// Add input to batch, returns extracted features if batch is full
    pub fn add(&mut self, input: FeatureInput) -> Option<Vec<CommitFeatures>> {
        self.batch_processor
            .add(input)
            .map(|batch| self.extract_batch(batch))
    }

    /// Flush remaining inputs and extract features
    pub fn flush(&mut self) -> Vec<CommitFeatures> {
        let batch = self.batch_processor.flush();
        if batch.is_empty() {
            Vec::new()
        } else {
            self.extract_batch(batch)
        }
    }

    /// Extract features from batch with performance tracking
    fn extract_batch(&mut self, inputs: Vec<FeatureInput>) -> Vec<CommitFeatures> {
        let start = Instant::now();

        let features: Vec<CommitFeatures> = inputs
            .into_iter()
            .filter_map(|input| {
                self.extractor
                    .extract(
                        input.category,
                        input.files_changed,
                        input.lines_added,
                        input.lines_deleted,
                        input.timestamp,
                    )
                    .ok()
            })
            .collect();

        let duration_ns = start.elapsed().as_nanos() as u64;
        self.stats.record(duration_ns);

        features
    }

    /// Extract all features at once (convenience method)
    pub fn extract_all(&mut self, inputs: Vec<FeatureInput>) -> Vec<CommitFeatures> {
        let start = Instant::now();

        let features: Vec<CommitFeatures> = inputs
            .into_iter()
            .filter_map(|input| {
                self.extractor
                    .extract(
                        input.category,
                        input.files_changed,
                        input.lines_added,
                        input.lines_deleted,
                        input.timestamp,
                    )
                    .ok()
            })
            .collect();

        let duration_ns = start.elapsed().as_nanos() as u64;
        self.stats.record(duration_ns);

        features
    }

    /// Get performance statistics
    pub fn stats(&self) -> &PerfStats {
        &self.stats
    }

    /// Get pending item count
    pub fn pending(&self) -> usize {
        self.batch_processor.len()
    }
}

impl Default for BatchFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_extractor_creation() {
        let _extractor = FeatureExtractor::new();
        // Extractor is zero-sized type, just verify it compiles
    }

    #[test]
    fn test_extract_basic_features() {
        let extractor = FeatureExtractor::new();

        // Category 2, 3 files, 100 lines added, 50 deleted
        let features = extractor
            .extract(
                2, 3, 100, 50, 1700000000, // 2023-11-14
            )
            .unwrap();

        assert_eq!(features.defect_category, 2);
        assert_eq!(features.files_changed, 3.0);
        assert_eq!(features.lines_added, 100.0);
        assert_eq!(features.lines_deleted, 50.0);
    }

    #[test]
    fn test_to_vector_dimension() {
        let features = CommitFeatures {
            defect_category: 1,
            files_changed: 2.0,
            lines_added: 10.0,
            lines_deleted: 5.0,
            complexity_delta: 0.0,
            timestamp: 1700000000.0,
            hour_of_day: 14,
            day_of_week: 2,
            // NLP-014: CITL fields
            error_code_class: 0,
            has_suggestion: 1,
            suggestion_applicability: 2,
            clippy_lint_count: 3,
            span_line_delta: 4.5,
            diagnostic_confidence: 0.95,
        };

        let vec = features.to_vector();
        assert_eq!(vec.len(), CommitFeatures::DIMENSION);
        assert_eq!(vec.len(), 14); // NLP-014: 14 dimensions
        assert_eq!(vec[0], 1.0); // category
        assert_eq!(vec[1], 2.0); // files
        assert_eq!(vec[2], 10.0); // lines added
                                  // NLP-014: CITL features
        assert_eq!(vec[8], 0.0); // error_code_class
        assert_eq!(vec[9], 1.0); // has_suggestion
        assert_eq!(vec[10], 2.0); // suggestion_applicability
        assert_eq!(vec[11], 3.0); // clippy_lint_count
        assert!((vec[12] - 4.5).abs() < 0.001); // span_line_delta
        assert!((vec[13] - 0.95).abs() < 0.001); // diagnostic_confidence
    }

    #[test]
    fn test_temporal_features() {
        let extractor = FeatureExtractor::new();

        // Known timestamp: 2023-11-14 14:30:00 UTC (Tuesday)
        let features = extractor.extract(0, 1, 1, 1, 1699971000).unwrap();

        assert_eq!(features.hour_of_day, 14);
        assert_eq!(features.day_of_week, 1); // Tuesday (0=Mon, 1=Tue)
    }

    #[test]
    fn test_invalid_timestamp() {
        let extractor = FeatureExtractor::new();

        // Out of range timestamp should error
        let result = extractor.extract(0, 1, 1, 1, i64::MAX);
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_extractor_creation() {
        let extractor = BatchFeatureExtractor::new();
        assert_eq!(extractor.pending(), 0);
    }

    #[test]
    fn test_batch_extractor_add() {
        let mut extractor = BatchFeatureExtractor::with_batch_size(3);

        let input1 = FeatureInput {
            category: 0,
            files_changed: 1,
            lines_added: 10,
            lines_deleted: 5,
            timestamp: 1700000000,
        };

        // First two shouldn't trigger extraction
        assert!(extractor.add(input1.clone()).is_none());
        assert!(extractor.add(input1.clone()).is_none());
        assert_eq!(extractor.pending(), 2);

        // Third should trigger
        let batch = extractor.add(input1);
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().len(), 3);
        assert_eq!(extractor.pending(), 0);
    }

    #[test]
    fn test_batch_extractor_flush() {
        let mut extractor = BatchFeatureExtractor::with_batch_size(10);

        let input = FeatureInput {
            category: 1,
            files_changed: 2,
            lines_added: 20,
            lines_deleted: 10,
            timestamp: 1700000000,
        };

        extractor.add(input.clone());
        extractor.add(input.clone());
        extractor.add(input);

        let remaining = extractor.flush();
        assert_eq!(remaining.len(), 3);
        assert_eq!(extractor.pending(), 0);
    }

    #[test]
    fn test_batch_extractor_extract_all() {
        let mut extractor = BatchFeatureExtractor::new();

        let inputs: Vec<FeatureInput> = (0..5)
            .map(|i| FeatureInput {
                category: i as u8,
                files_changed: i + 1,
                lines_added: (i + 1) * 10,
                lines_deleted: (i + 1) * 5,
                timestamp: 1700000000 + i as i64,
            })
            .collect();

        let features = extractor.extract_all(inputs);
        assert_eq!(features.len(), 5);
        assert_eq!(features[0].defect_category, 0);
        assert_eq!(features[4].defect_category, 4);
    }

    #[test]
    fn test_batch_extractor_stats() {
        let mut extractor = BatchFeatureExtractor::new();

        let inputs: Vec<FeatureInput> = (0..100)
            .map(|i| FeatureInput {
                category: (i % 10) as u8,
                files_changed: i + 1,
                lines_added: (i + 1) * 10,
                lines_deleted: (i + 1) * 5,
                timestamp: 1700000000 + i as i64,
            })
            .collect();

        extractor.extract_all(inputs);

        let stats = extractor.stats();
        assert_eq!(stats.operation_count, 1);
        assert!(stats.avg_ns() > 0);
    }
}
