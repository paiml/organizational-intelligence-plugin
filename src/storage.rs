//! Feature Storage with trueno-db
//!
//! Implements Section 4.4: Trueno-DB Storage Layer
//! GPU-first columnar storage using Arrow/Parquet
//!
//! UAT: Added JSON persistence for Phase 1

use anyhow::Result;
use std::path::Path;

use crate::features::CommitFeatures;

/// Feature storage using trueno-db
///
/// Phase 1: Basic in-memory storage with file persistence
/// Phase 2: Full trueno-db integration with GPU-resident data
pub struct FeatureStore {
    features: Vec<CommitFeatures>,
}

impl FeatureStore {
    /// Create new feature store
    pub fn new() -> Result<Self> {
        Ok(Self {
            features: Vec::new(),
        })
    }

    /// Insert single feature
    pub fn insert(&mut self, feature: CommitFeatures) -> Result<()> {
        self.features.push(feature);
        Ok(())
    }

    /// Bulk insert features (optimized for GPU batch processing)
    pub fn bulk_insert(&mut self, features: Vec<CommitFeatures>) -> Result<()> {
        self.features.extend(features);
        Ok(())
    }

    /// Query features by defect category
    pub fn query_by_category(&self, category: u8) -> Result<Vec<&CommitFeatures>> {
        Ok(self
            .features
            .iter()
            .filter(|f| f.defect_category == category)
            .collect())
    }

    /// Query features by time range (for sliding window correlation)
    ///
    /// Returns features where start_time <= timestamp < end_time
    /// Time is in Unix epoch seconds (f64)
    pub fn query_by_time_range(
        &self,
        start_time: f64,
        end_time: f64,
    ) -> Result<Vec<&CommitFeatures>> {
        Ok(self
            .features
            .iter()
            .filter(|f| f.timestamp >= start_time && f.timestamp < end_time)
            .collect())
    }

    /// Get all features (for compatibility)
    pub fn all_features(&self) -> &[CommitFeatures] {
        &self.features
    }

    /// Get all features as vectors (for GPU transfer)
    pub fn to_vectors(&self) -> Vec<Vec<f32>> {
        self.features.iter().map(|f| f.to_vector()).collect()
    }

    /// Get feature count
    pub fn len(&self) -> usize {
        self.features.len()
    }

    /// Check if store is empty
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    /// Save to file (JSON in Phase 1, Parquet in Phase 2)
    pub async fn save(&self, path: &Path) -> Result<()> {
        // Phase 1: JSON persistence
        let json = serde_json::to_string(&self.features)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load from file (JSON in Phase 1, Parquet in Phase 2)
    pub async fn load(path: &Path) -> Result<Self> {
        // Phase 1: JSON persistence
        if !path.exists() {
            return Self::new();
        }
        let json = std::fs::read_to_string(path)?;
        let features: Vec<CommitFeatures> = serde_json::from_str(&json)?;
        Ok(Self { features })
    }
}

impl Default for FeatureStore {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_feature(category: u8, files: usize) -> CommitFeatures {
        CommitFeatures {
            defect_category: category,
            files_changed: files as f32,
            lines_added: 100.0,
            lines_deleted: 50.0,
            complexity_delta: 0.0,
            timestamp: 1700000000.0,
            hour_of_day: 14,
            day_of_week: 2,
            ..Default::default()
        }
    }

    #[test]
    fn test_store_creation() {
        let store = FeatureStore::new();
        assert!(store.is_ok());
        assert!(store.unwrap().is_empty());
    }

    #[test]
    fn test_insert_single() {
        let mut store = FeatureStore::new().unwrap();
        let feature = make_test_feature(1, 3);

        store.insert(feature).unwrap();

        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
    }

    #[test]
    fn test_bulk_insert() {
        let mut store = FeatureStore::new().unwrap();

        let features = vec![
            make_test_feature(1, 2),
            make_test_feature(2, 3),
            make_test_feature(1, 1),
        ];

        store.bulk_insert(features).unwrap();

        assert_eq!(store.len(), 3);
    }

    #[test]
    fn test_query_by_category() {
        let mut store = FeatureStore::new().unwrap();

        store
            .bulk_insert(vec![
                make_test_feature(1, 2),
                make_test_feature(2, 3),
                make_test_feature(1, 1),
                make_test_feature(3, 5),
            ])
            .unwrap();

        let category1 = store.query_by_category(1).unwrap();
        assert_eq!(category1.len(), 2);

        let category2 = store.query_by_category(2).unwrap();
        assert_eq!(category2.len(), 1);

        let category9 = store.query_by_category(9).unwrap();
        assert_eq!(category9.len(), 0);
    }

    #[test]
    fn test_to_vectors() {
        let mut store = FeatureStore::new().unwrap();

        store
            .bulk_insert(vec![make_test_feature(1, 2), make_test_feature(2, 3)])
            .unwrap();

        let vectors = store.to_vectors();

        assert_eq!(vectors.len(), 2);
        assert_eq!(vectors[0].len(), CommitFeatures::DIMENSION);
        assert_eq!(vectors[0][0], 1.0); // category
        assert_eq!(vectors[0][1], 2.0); // files
        assert_eq!(vectors[1][0], 2.0); // category
        assert_eq!(vectors[1][1], 3.0); // files
    }

    #[test]
    fn test_query_by_time_range() {
        let mut store = FeatureStore::new().unwrap();

        // Insert features with different timestamps
        let mut f1 = make_test_feature(1, 2);
        f1.timestamp = 1000.0;
        let mut f2 = make_test_feature(2, 3);
        f2.timestamp = 2000.0;
        let mut f3 = make_test_feature(3, 4);
        f3.timestamp = 3000.0;
        let mut f4 = make_test_feature(4, 5);
        f4.timestamp = 4000.0;

        store.bulk_insert(vec![f1, f2, f3, f4]).unwrap();

        // Query range [2000, 4000) - should get f2 and f3
        let range_result = store.query_by_time_range(2000.0, 4000.0).unwrap();
        assert_eq!(range_result.len(), 2);
        assert_eq!(range_result[0].timestamp, 2000.0);
        assert_eq!(range_result[1].timestamp, 3000.0);

        // Query range [1000, 2500) - should get f1 and f2
        let range_result2 = store.query_by_time_range(1000.0, 2500.0).unwrap();
        assert_eq!(range_result2.len(), 2);

        // Query empty range
        let empty_range = store.query_by_time_range(5000.0, 6000.0).unwrap();
        assert_eq!(empty_range.len(), 0);
    }

    #[tokio::test]
    async fn test_save_load() {
        let store = FeatureStore::new().unwrap();

        // Phase 1: save/load are stubs, just verify they compile
        store.save(Path::new("test.parquet")).await.unwrap();
        let _loaded = FeatureStore::load(Path::new("test.parquet")).await.unwrap();
    }
}
