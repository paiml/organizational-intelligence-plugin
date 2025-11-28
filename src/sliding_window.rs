//! Sliding Window Correlation for Concept Drift Detection
//!
//! Implements PHASE2-003: Time-windowed correlation matrices
//! Detects changing defect patterns over time (concept drift)

use crate::correlation::pearson_correlation;
use crate::features::CommitFeatures;
use crate::storage::FeatureStore;
use anyhow::Result;
use trueno::Vector;

/// Time window duration in seconds (6 months ≈ 15,768,000 seconds)
pub const SIX_MONTHS_SECONDS: f64 = 6.0 * 30.0 * 24.0 * 3600.0;

/// Time window for correlation analysis
#[derive(Debug, Clone)]
pub struct TimeWindow {
    pub start_time: f64, // Unix epoch
    pub end_time: f64,   // Unix epoch
}

impl TimeWindow {
    /// Create time window
    pub fn new(start_time: f64, end_time: f64) -> Self {
        Self {
            start_time,
            end_time,
        }
    }

    /// Create 6-month window starting at given time
    pub fn six_months_from(start_time: f64) -> Self {
        Self {
            start_time,
            end_time: start_time + SIX_MONTHS_SECONDS,
        }
    }

    /// Check if window contains timestamp
    pub fn contains(&self, timestamp: f64) -> bool {
        timestamp >= self.start_time && timestamp < self.end_time
    }

    /// Get window duration in seconds
    pub fn duration(&self) -> f64 {
        self.end_time - self.start_time
    }
}

/// Correlation matrix for a time window
#[derive(Debug, Clone)]
pub struct WindowedCorrelationMatrix {
    pub window: TimeWindow,
    pub matrix: Vec<Vec<f32>>, // n×n correlation matrix
    pub feature_count: usize,  // Number of features in window
}

/// Sliding window correlation analyzer
pub struct SlidingWindowAnalyzer {
    window_size: f64, // Window duration in seconds
    stride: f64,      // Window stride in seconds (for overlap)
}

impl SlidingWindowAnalyzer {
    /// Create analyzer with 6-month windows, 3-month stride (50% overlap)
    pub fn new_six_month() -> Self {
        Self {
            window_size: SIX_MONTHS_SECONDS,
            stride: SIX_MONTHS_SECONDS / 2.0,
        }
    }

    /// Create analyzer with custom window size and stride
    pub fn new(window_size: f64, stride: f64) -> Self {
        Self {
            window_size,
            stride,
        }
    }

    /// Generate time windows for given data range
    pub fn generate_windows(&self, start_time: f64, end_time: f64) -> Vec<TimeWindow> {
        let mut windows = Vec::new();
        let mut current_start = start_time;

        while current_start + self.window_size <= end_time {
            windows.push(TimeWindow::new(
                current_start,
                current_start + self.window_size,
            ));
            current_start += self.stride;
        }

        windows
    }

    /// Compute correlation matrix for features in a time window
    ///
    /// Returns correlation matrix between all feature dimensions
    pub fn compute_window_correlation(
        &self,
        store: &FeatureStore,
        window: &TimeWindow,
    ) -> Result<WindowedCorrelationMatrix> {
        // Query features in window
        let features = store.query_by_time_range(window.start_time, window.end_time)?;

        if features.is_empty() {
            anyhow::bail!(
                "No features in window [{}, {})",
                window.start_time,
                window.end_time
            );
        }

        // Convert to vectors (8 dimensions per feature)
        let vectors: Vec<Vec<f32>> = features.iter().map(|f| f.to_vector()).collect();
        let n_samples = vectors.len();
        let n_dims = CommitFeatures::DIMENSION;

        // Build dimension-wise arrays
        let mut dim_arrays: Vec<Vec<f32>> = vec![Vec::new(); n_dims];
        for v in &vectors {
            for (dim_idx, &value) in v.iter().enumerate() {
                dim_arrays[dim_idx].push(value);
            }
        }

        // Compute correlation matrix (n_dims × n_dims)
        let mut matrix = vec![vec![0.0; n_dims]; n_dims];
        for i in 0..n_dims {
            for j in 0..n_dims {
                if i == j {
                    matrix[i][j] = 1.0; // Self-correlation is always 1
                } else {
                    let vec_i = Vector::from_slice(&dim_arrays[i]);
                    let vec_j = Vector::from_slice(&dim_arrays[j]);
                    matrix[i][j] = pearson_correlation(&vec_i, &vec_j)?;
                }
            }
        }

        Ok(WindowedCorrelationMatrix {
            window: window.clone(),
            matrix,
            feature_count: n_samples,
        })
    }

    /// Compute correlation matrices for all windows
    pub fn compute_all_windows(
        &self,
        store: &FeatureStore,
    ) -> Result<Vec<WindowedCorrelationMatrix>> {
        // Find time range in data
        let all_features = store.all_features();
        if all_features.is_empty() {
            anyhow::bail!("No features in store");
        }

        let start_time = all_features
            .iter()
            .map(|f| f.timestamp)
            .fold(f64::INFINITY, f64::min);
        let end_time = all_features
            .iter()
            .map(|f| f.timestamp)
            .fold(f64::NEG_INFINITY, f64::max);

        // Generate windows
        let windows = self.generate_windows(start_time, end_time);

        // Compute correlation for each window
        let mut results = Vec::new();
        for window in windows {
            match self.compute_window_correlation(store, &window) {
                Ok(wcm) => results.push(wcm),
                Err(_) => continue, // Skip windows with no data
            }
        }

        Ok(results)
    }
}

/// Concept drift detection
#[derive(Debug, Clone)]
pub struct ConceptDrift {
    pub window1_idx: usize,
    pub window2_idx: usize,
    pub matrix_diff: f32,     // Frobenius norm of difference
    pub is_significant: bool, // Above threshold
}

/// Detect concept drift between consecutive windows
///
/// Uses Frobenius norm to measure matrix difference
pub fn detect_drift(
    matrices: &[WindowedCorrelationMatrix],
    threshold: f32,
) -> Result<Vec<ConceptDrift>> {
    if matrices.len() < 2 {
        return Ok(Vec::new());
    }

    let mut drifts = Vec::new();

    for i in 0..matrices.len() - 1 {
        let mat1 = &matrices[i].matrix;
        let mat2 = &matrices[i + 1].matrix;

        // Compute Frobenius norm: sqrt(sum of squared differences)
        let mut sum_sq_diff = 0.0;
        for (row1, row2) in mat1.iter().zip(mat2.iter()) {
            for (&val1, &val2) in row1.iter().zip(row2.iter()) {
                let diff = val1 - val2;
                sum_sq_diff += diff * diff;
            }
        }
        let frobenius_norm = sum_sq_diff.sqrt();

        drifts.push(ConceptDrift {
            window1_idx: i,
            window2_idx: i + 1,
            matrix_diff: frobenius_norm,
            is_significant: frobenius_norm > threshold,
        });
    }

    Ok(drifts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_window_creation() {
        let window = TimeWindow::new(1000.0, 2000.0);
        assert_eq!(window.duration(), 1000.0);
        assert!(window.contains(1500.0));
        assert!(!window.contains(2500.0));
    }

    #[test]
    fn test_six_month_window() {
        let window = TimeWindow::six_months_from(0.0);
        assert_eq!(window.duration(), SIX_MONTHS_SECONDS);
    }

    #[test]
    fn test_generate_windows() {
        let analyzer = SlidingWindowAnalyzer::new_six_month();
        let windows = analyzer.generate_windows(0.0, SIX_MONTHS_SECONDS * 3.0);

        // With 50% overlap: windows at 0, 0.5×6mo, 1×6mo, 1.5×6mo, 2×6mo
        assert_eq!(windows.len(), 5);
    }

    #[test]
    fn test_window_correlation_computation() {
        let mut store = FeatureStore::new().unwrap();

        // Create test features with different timestamps
        for i in 0..10 {
            let f = CommitFeatures {
                defect_category: 1,
                files_changed: (i + 1) as f32,
                lines_added: (i * 10) as f32,
                lines_deleted: (i * 5) as f32,
                complexity_delta: (i as f32) * 0.5,
                timestamp: (i * 1000) as f64,
                hour_of_day: 10,
                day_of_week: 1,
                ..Default::default()
            };
            store.insert(f).unwrap();
        }

        let analyzer = SlidingWindowAnalyzer::new(5000.0, 2500.0);
        let window = TimeWindow::new(0.0, 5000.0);

        let result = analyzer
            .compute_window_correlation(&store, &window)
            .unwrap();

        // Should have 8×8 correlation matrix
        assert_eq!(result.matrix.len(), CommitFeatures::DIMENSION);
        assert_eq!(result.matrix[0].len(), CommitFeatures::DIMENSION);

        // Diagonal should be 1.0
        for i in 0..CommitFeatures::DIMENSION {
            assert!((result.matrix[i][i] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_window_contains_boundaries() {
        let window = TimeWindow::new(1000.0, 2000.0);

        // Start boundary (inclusive)
        assert!(window.contains(1000.0));

        // End boundary (exclusive)
        assert!(!window.contains(2000.0));

        // Before window
        assert!(!window.contains(999.9));

        // After window
        assert!(!window.contains(2000.1));
    }

    #[test]
    fn test_empty_store_compute_all_windows() {
        let store = FeatureStore::new().unwrap();
        let analyzer = SlidingWindowAnalyzer::new_six_month();

        let result = analyzer.compute_all_windows(&store);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No features in store"));
    }

    #[test]
    fn test_window_with_no_features() {
        let mut store = FeatureStore::new().unwrap();

        // Add features outside the window
        let f = CommitFeatures {
            defect_category: 1,
            files_changed: 5.0,
            lines_added: 50.0,
            lines_deleted: 20.0,
            complexity_delta: 0.5,
            timestamp: 10000.0, // Way outside window
            hour_of_day: 10,
            day_of_week: 1,
            ..Default::default()
        };
        store.insert(f).unwrap();

        let analyzer = SlidingWindowAnalyzer::new(5000.0, 2500.0);
        let window = TimeWindow::new(0.0, 5000.0); // Features are at 10000.0

        let result = analyzer.compute_window_correlation(&store, &window);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No features in window"));
    }

    #[test]
    fn test_detect_drift_with_no_matrices() {
        let matrices = Vec::new();
        let drifts = detect_drift(&matrices, 0.5).unwrap();
        assert_eq!(drifts.len(), 0);
    }

    #[test]
    fn test_detect_drift_with_one_matrix() {
        let matrix = WindowedCorrelationMatrix {
            window: TimeWindow::new(0.0, 1000.0),
            matrix: vec![vec![1.0; 8]; 8],
            feature_count: 10,
        };

        let drifts = detect_drift(&[matrix], 0.5).unwrap();
        assert_eq!(drifts.len(), 0);
    }

    #[test]
    fn test_detect_drift_identical_matrices() {
        let matrix1 = WindowedCorrelationMatrix {
            window: TimeWindow::new(0.0, 1000.0),
            matrix: vec![vec![1.0; 8]; 8],
            feature_count: 10,
        };

        let matrix2 = WindowedCorrelationMatrix {
            window: TimeWindow::new(1000.0, 2000.0),
            matrix: vec![vec![1.0; 8]; 8],
            feature_count: 10,
        };

        let drifts = detect_drift(&[matrix1, matrix2], 0.5).unwrap();
        assert_eq!(drifts.len(), 1);
        assert!(!drifts[0].is_significant); // No difference
        assert_eq!(drifts[0].matrix_diff, 0.0);
    }

    #[test]
    fn test_detect_drift_different_matrices() {
        let mut matrix1_data = vec![vec![1.0; 8]; 8];
        matrix1_data[0][1] = 0.5; // Change one value

        let matrix1 = WindowedCorrelationMatrix {
            window: TimeWindow::new(0.0, 1000.0),
            matrix: matrix1_data,
            feature_count: 10,
        };

        let matrix2 = WindowedCorrelationMatrix {
            window: TimeWindow::new(1000.0, 2000.0),
            matrix: vec![vec![1.0; 8]; 8],
            feature_count: 10,
        };

        let drifts = detect_drift(&[matrix1, matrix2], 0.01).unwrap();
        assert_eq!(drifts.len(), 1);
        assert!(drifts[0].is_significant); // Difference above threshold
        assert!(drifts[0].matrix_diff > 0.0);
    }

    #[test]
    fn test_detect_drift_multiple_windows() {
        let mat1 = WindowedCorrelationMatrix {
            window: TimeWindow::new(0.0, 1000.0),
            matrix: vec![vec![1.0; 8]; 8],
            feature_count: 10,
        };

        let mat2 = WindowedCorrelationMatrix {
            window: TimeWindow::new(1000.0, 2000.0),
            matrix: vec![vec![0.9; 8]; 8],
            feature_count: 10,
        };

        let mat3 = WindowedCorrelationMatrix {
            window: TimeWindow::new(2000.0, 3000.0),
            matrix: vec![vec![0.8; 8]; 8],
            feature_count: 10,
        };

        let drifts = detect_drift(&[mat1, mat2, mat3], 0.1).unwrap();
        assert_eq!(drifts.len(), 2); // Two transitions
        assert_eq!(drifts[0].window1_idx, 0);
        assert_eq!(drifts[0].window2_idx, 1);
        assert_eq!(drifts[1].window1_idx, 1);
        assert_eq!(drifts[1].window2_idx, 2);
    }

    #[test]
    fn test_custom_analyzer_creation() {
        let analyzer = SlidingWindowAnalyzer::new(1000.0, 500.0);
        let windows = analyzer.generate_windows(0.0, 3000.0);

        // Windows: [0-1000], [500-1500], [1000-2000], [1500-2500], [2000-3000]
        assert_eq!(windows.len(), 5);
        assert_eq!(windows[0].start_time, 0.0);
        assert_eq!(windows[0].end_time, 1000.0);
        assert_eq!(windows[1].start_time, 500.0);
    }

    #[test]
    fn test_generate_windows_no_full_window_at_end() {
        let analyzer = SlidingWindowAnalyzer::new(1000.0, 500.0);
        let windows = analyzer.generate_windows(0.0, 1500.0);

        // Windows: [0-1000], [500-1500]
        // No [1000-2000] because end_time is only 1500
        assert_eq!(windows.len(), 2);
    }

    #[test]
    fn test_compute_all_windows_skips_empty_windows() {
        let mut store = FeatureStore::new().unwrap();

        // Add features only in first half of time range
        for i in 0..5 {
            let f = CommitFeatures {
                defect_category: 1,
                files_changed: (i + 1) as f32,
                lines_added: (i * 10) as f32,
                lines_deleted: (i * 5) as f32,
                complexity_delta: (i as f32) * 0.5,
                timestamp: (i * 1000) as f64, // 0-4000
                hour_of_day: 10,
                day_of_week: 1,
                ..Default::default()
            };
            store.insert(f).unwrap();
        }

        let analyzer = SlidingWindowAnalyzer::new(3000.0, 1500.0);
        let results = analyzer.compute_all_windows(&store).unwrap();

        // Only windows with data should be in results
        assert!(!results.is_empty());
        assert!(results.len() <= 3); // Won't have empty windows
    }

    #[test]
    fn test_concept_drift_structure() {
        let drift = ConceptDrift {
            window1_idx: 0,
            window2_idx: 1,
            matrix_diff: 0.75,
            is_significant: true,
        };

        assert_eq!(drift.window1_idx, 0);
        assert_eq!(drift.window2_idx, 1);
        assert_eq!(drift.matrix_diff, 0.75);
        assert!(drift.is_significant);
    }

    #[test]
    fn test_windowed_correlation_matrix_structure() {
        let wcm = WindowedCorrelationMatrix {
            window: TimeWindow::new(0.0, 1000.0),
            matrix: vec![vec![1.0; 8]; 8],
            feature_count: 42,
        };

        assert_eq!(wcm.window.start_time, 0.0);
        assert_eq!(wcm.window.end_time, 1000.0);
        assert_eq!(wcm.matrix.len(), 8);
        assert_eq!(wcm.feature_count, 42);
    }

    #[test]
    fn test_six_months_constant() {
        // 6 months * 30 days * 24 hours * 3600 seconds
        let expected = 6.0 * 30.0 * 24.0 * 3600.0;
        assert_eq!(SIX_MONTHS_SECONDS, expected);
        assert_eq!(SIX_MONTHS_SECONDS, 15_552_000.0);
    }
}
