//! Correlation Matrix Computation
//!
//! Implements Section 5.2: Correlation Matrix Computation
//! Phase 1: SIMD via trueno (GPU in Phase 2)
//!
//! OPT-001: Added LruCache for correlation result caching

use crate::perf::{LruCache, PerfStats};
use anyhow::Result;
use std::time::{Duration, Instant};
use trueno::{Matrix, Vector};

/// Correlation matrix result
#[derive(Debug)]
pub struct CorrelationMatrix {
    /// Number of categories
    pub categories: usize,

    /// Correlation values (categories × categories)
    pub values: Matrix<f32>,
}

/// Compute Pearson correlation between two vectors
///
/// r = cov(X,Y) / (std(X) * std(Y))
pub fn pearson_correlation(x: &Vector<f32>, y: &Vector<f32>) -> Result<f32> {
    if x.len() != y.len() {
        anyhow::bail!("Vectors must have same length");
    }

    let n = x.len() as f32;

    // Mean values (trueno SIMD-accelerated)
    let mean_x = x.sum()? / n;
    let mean_y = y.sum()? / n;

    // Compute covariance and variances
    let mut cov = 0.0_f32;
    let mut var_x = 0.0_f32;
    let mut var_y = 0.0_f32;

    let x_slice = x.as_slice();
    let y_slice = y.as_slice();

    for i in 0..x.len() {
        let dx = x_slice[i] - mean_x;
        let dy = y_slice[i] - mean_y;

        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    // Handle zero variance
    if var_x == 0.0 || var_y == 0.0 {
        return Ok(0.0);
    }

    // Pearson correlation: r = cov / sqrt(var_x * var_y)
    let r = cov / (var_x * var_y).sqrt();

    Ok(r)
}

/// Cached correlation computer with performance tracking
///
/// OPT-001: Caches correlation results to avoid redundant computation
pub struct CachedCorrelation {
    cache: LruCache<(usize, usize), f32>,
    stats: PerfStats,
    cache_hits: u64,
    cache_misses: u64,
}

impl CachedCorrelation {
    /// Create cached correlation with default capacity (1000 entries, 5 min TTL)
    pub fn new() -> Self {
        Self::with_capacity(1000)
    }

    /// Create cached correlation with custom capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            cache: LruCache::with_ttl(capacity, Duration::from_secs(300)),
            stats: PerfStats::new(),
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    /// Compute correlation with caching
    ///
    /// Cache key is based on vector indices (assumes stable vector ordering)
    pub fn correlate(
        &mut self,
        x: &Vector<f32>,
        y: &Vector<f32>,
        x_idx: usize,
        y_idx: usize,
    ) -> Result<f32> {
        let key = if x_idx <= y_idx {
            (x_idx, y_idx)
        } else {
            (y_idx, x_idx)
        };

        // Check cache
        if let Some(cached) = self.cache.get(&key) {
            self.cache_hits += 1;
            return Ok(cached);
        }

        self.cache_misses += 1;

        // Compute correlation
        let start = Instant::now();
        let result = pearson_correlation(x, y)?;
        let duration_ns = start.elapsed().as_nanos() as u64;
        self.stats.record(duration_ns);

        // Cache result
        self.cache.insert(key, result);

        Ok(result)
    }

    /// Compute correlation matrix for multiple vectors
    pub fn correlation_matrix(&mut self, vectors: &[Vector<f32>]) -> Result<Vec<Vec<f32>>> {
        let n = vectors.len();
        let mut matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            matrix[i][i] = 1.0; // Self-correlation
            for j in (i + 1)..n {
                let corr = self.correlate(&vectors[i], &vectors[j], i, j)?;
                matrix[i][j] = corr;
                matrix[j][i] = corr; // Symmetric
            }
        }

        Ok(matrix)
    }

    /// Get performance statistics
    pub fn stats(&self) -> &PerfStats {
        &self.stats
    }

    /// Get cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (u64, u64) {
        (self.cache_hits, self.cache_misses)
    }

    /// Clear cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.cache_hits = 0;
        self.cache_misses = 0;
    }
}

impl Default for CachedCorrelation {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pearson_perfect_positive() {
        let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);

        let r = pearson_correlation(&x, &y).unwrap();

        assert!((r - 1.0).abs() < 1e-6, "Expected r=1.0, got {}", r);
    }

    #[test]
    fn test_pearson_perfect_negative() {
        let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Vector::from_slice(&[5.0, 4.0, 3.0, 2.0, 1.0]);

        let r = pearson_correlation(&x, &y).unwrap();

        assert!((r + 1.0).abs() < 1e-6, "Expected r=-1.0, got {}", r);
    }

    #[test]
    fn test_pearson_zero_variance() {
        let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let y = Vector::from_slice(&[1.0, 1.0, 1.0, 1.0]);

        let r = pearson_correlation(&x, &y).unwrap();

        assert_eq!(r, 0.0, "Zero variance should return 0.0");
    }

    #[test]
    fn test_pearson_different_lengths() {
        let x = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let y = Vector::from_slice(&[1.0, 2.0]);

        let result = pearson_correlation(&x, &y);

        assert!(result.is_err(), "Should error on different lengths");
    }

    #[test]
    fn test_cached_correlation_creation() {
        let cached = CachedCorrelation::new();
        assert_eq!(cached.cache_hit_rate(), 0.0);
    }

    #[test]
    fn test_cached_correlation_compute() {
        let mut cached = CachedCorrelation::new();

        let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);

        let r = cached.correlate(&x, &y, 0, 1).unwrap();
        assert!((r - 1.0).abs() < 1e-6);

        let (hits, misses) = cached.cache_stats();
        assert_eq!(hits, 0);
        assert_eq!(misses, 1);
    }

    #[test]
    fn test_cached_correlation_cache_hit() {
        let mut cached = CachedCorrelation::new();

        let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);

        // First call - cache miss
        cached.correlate(&x, &y, 0, 1).unwrap();

        // Second call - cache hit
        let r = cached.correlate(&x, &y, 0, 1).unwrap();
        assert!((r - 1.0).abs() < 1e-6);

        let (hits, misses) = cached.cache_stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
        assert!((cached.cache_hit_rate() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_cached_correlation_symmetric_key() {
        let mut cached = CachedCorrelation::new();

        let x = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let y = Vector::from_slice(&[3.0, 2.0, 1.0]);

        // Call with (0, 1)
        cached.correlate(&x, &y, 0, 1).unwrap();

        // Call with (1, 0) - should hit cache due to symmetric key
        cached.correlate(&y, &x, 1, 0).unwrap();

        let (hits, _) = cached.cache_stats();
        assert_eq!(hits, 1);
    }

    #[test]
    fn test_cached_correlation_matrix() {
        let mut cached = CachedCorrelation::new();

        let vectors = vec![
            Vector::from_slice(&[1.0, 2.0, 3.0]),
            Vector::from_slice(&[2.0, 4.0, 6.0]),
            Vector::from_slice(&[3.0, 2.0, 1.0]),
        ];

        let matrix = cached.correlation_matrix(&vectors).unwrap();

        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix[0].len(), 3);

        // Diagonal should be 1.0
        assert!((matrix[0][0] - 1.0).abs() < 1e-6);
        assert!((matrix[1][1] - 1.0).abs() < 1e-6);
        assert!((matrix[2][2] - 1.0).abs() < 1e-6);

        // Should be symmetric
        assert!((matrix[0][1] - matrix[1][0]).abs() < 1e-6);
    }

    #[test]
    fn test_cached_correlation_clear() {
        let mut cached = CachedCorrelation::new();

        let x = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let y = Vector::from_slice(&[2.0, 4.0, 6.0]);

        cached.correlate(&x, &y, 0, 1).unwrap();
        cached.clear_cache();

        let (hits, misses) = cached.cache_stats();
        assert_eq!(hits, 0);
        assert_eq!(misses, 0);
    }
}
