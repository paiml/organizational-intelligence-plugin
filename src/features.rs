//! Feature Extraction for GPU Processing
//!
//! Implements Section 4.3: Feature Extraction
//! Converts OIP defect classifications into GPU-friendly numerical features

use anyhow::Result;
use chrono::{Datelike, Timelike};

/// Commit features optimized for GPU processing
#[derive(Debug, Clone)]
pub struct CommitFeatures {
    // Categorical (one-hot encoded for GPU)
    pub defect_category: u8, // 0-9 (10 categories from OIP)

    // Numerical (GPU-native f32)
    pub files_changed: f32,
    pub lines_added: f32,
    pub lines_deleted: f32,
    pub complexity_delta: f32, // Cyclomatic complexity change

    // Temporal
    pub timestamp: f64,  // Unix epoch
    pub hour_of_day: u8, // 0-23 (circadian patterns)
    pub day_of_week: u8, // 0-6
}

impl CommitFeatures {
    /// Convert to flat vector for GPU processing
    ///
    /// Fixed-size vector enables efficient GPU batching
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
        ]
    }

    /// Vector dimension count (for GPU buffer allocation)
    pub const DIMENSION: usize = 8;
}

/// Extract features from OIP defect record
pub struct FeatureExtractor;

impl FeatureExtractor {
    pub fn new() -> Self {
        Self
    }

    /// Extract features from defect category and metadata
    pub fn extract(
        &self,
        category: u8,
        files_changed: usize,
        lines_added: usize,
        lines_deleted: usize,
        timestamp: i64,
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
        })
    }
}

impl Default for FeatureExtractor {
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
        };

        let vec = features.to_vector();
        assert_eq!(vec.len(), CommitFeatures::DIMENSION);
        assert_eq!(vec[0], 1.0); // category
        assert_eq!(vec[1], 2.0); // files
        assert_eq!(vec[2], 10.0); // lines added
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
}
