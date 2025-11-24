//! Correlation Matrix Computation
//!
//! Implements Section 5.2: Correlation Matrix Computation
//! Phase 1: SIMD via trueno (GPU in Phase 2)

use anyhow::Result;
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
}
