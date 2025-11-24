//! Feature Storage with trueno-db
//!
//! Implements Section 4.4: Trueno-DB Storage Layer
//! GPU-first columnar storage using Arrow/Parquet

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

    /// Save to file (Parquet in Phase 2)
    pub async fn save(&self, _path: &Path) -> Result<()> {
        // Phase 1: stub
        // Phase 2: write to Parquet via trueno-db
        Ok(())
    }

    /// Load from file (Parquet in Phase 2)
    pub async fn load(_path: &Path) -> Result<Self> {
        // Phase 1: stub
        // Phase 2: read from Parquet via trueno-db
        Self::new()
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

    #[tokio::test]
    async fn test_save_load() {
        let store = FeatureStore::new().unwrap();

        // Phase 1: save/load are stubs, just verify they compile
        store.save(Path::new("test.parquet")).await.unwrap();
        let _loaded = FeatureStore::load(Path::new("test.parquet")).await.unwrap();
    }
}
