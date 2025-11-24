//! GPU-Resident Feature Store
//!
//! Implements Section A.1 of the spec: GPU Memory Persistence Strategy
//! Phase 1: In-memory storage (trueno-db integration in Phase 2)

use anyhow::Result;

/// GPU-resident feature store
///
/// Phase 1: Simple in-memory storage
/// Phase 2: trueno-db for GPU-resident columnar storage
pub struct GPUHotStore {
    // Feature dimensions
    feature_count: usize,
    commit_count: usize,
}

impl GPUHotStore {
    /// Create new hot store
    pub fn new() -> Result<Self> {
        Ok(Self {
            feature_count: 0,
            commit_count: 0,
        })
    }

    /// Get feature dimensions
    pub fn dimensions(&self) -> (usize, usize) {
        (self.commit_count, self.feature_count)
    }
}

impl Default for GPUHotStore {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hot_store_creation() {
        let store = GPUHotStore::new();
        assert!(store.is_ok());
    }

    #[test]
    fn test_dimensions_initial() {
        let store = GPUHotStore::new().unwrap();
        assert_eq!(store.dimensions(), (0, 0));
    }
}
