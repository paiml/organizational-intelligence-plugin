//! Class Imbalance Handling for Defect Prediction
//!
//! Implements PHASE2-004: SMOTE, Focal Loss, and AUPRC
//! Addresses class imbalance in defect data (typically <1% defect rate)
//!
//! References:
//! - Chawla et al. (2002): SMOTE - Synthetic Minority Over-sampling Technique
//! - Lin et al. (2017): Focal Loss for Dense Object Detection

use crate::features::CommitFeatures;
use anyhow::Result;

/// SMOTE (Synthetic Minority Over-sampling Technique)
///
/// Generates synthetic samples for minority class by interpolating
/// between existing minority samples and their k-nearest neighbors.
pub struct Smote {
    k_neighbors: usize,
}

impl Smote {
    /// Create SMOTE with default k=5 neighbors
    pub fn new() -> Self {
        Self { k_neighbors: 5 }
    }

    /// Create SMOTE with custom k neighbors
    pub fn with_k(k_neighbors: usize) -> Self {
        Self { k_neighbors }
    }

    /// Oversample minority class to balance dataset
    ///
    /// # Arguments
    /// * `features` - All features (majority + minority)
    /// * `minority_category` - Category ID to oversample (0-9)
    /// * `target_ratio` - Target minority:majority ratio (e.g., 0.5 = 50%)
    ///
    /// # Returns
    /// Original features + synthetic minority samples
    pub fn oversample(
        &self,
        features: &[CommitFeatures],
        minority_category: u8,
        target_ratio: f32,
    ) -> Result<Vec<CommitFeatures>> {
        // Separate minority and majority
        let minority: Vec<&CommitFeatures> = features
            .iter()
            .filter(|f| f.defect_category == minority_category)
            .collect();

        let majority_count = features.len() - minority.len();

        if minority.is_empty() {
            anyhow::bail!("No samples found for category {}", minority_category);
        }

        // Calculate how many synthetic samples to generate
        let target_minority = (majority_count as f32 * target_ratio) as usize;
        let samples_needed = target_minority.saturating_sub(minority.len());

        if samples_needed == 0 {
            return Ok(features.to_vec());
        }

        // Convert minority to vectors for distance computation
        let minority_vecs: Vec<Vec<f32>> = minority.iter().map(|f| f.to_vector()).collect();

        // Generate synthetic samples
        let mut synthetic = Vec::with_capacity(samples_needed);
        let mut sample_idx = 0;

        while synthetic.len() < samples_needed {
            let base_idx = sample_idx % minority.len();
            let base = &minority_vecs[base_idx];

            // Find k nearest neighbors
            let neighbors = self.find_k_nearest(base, &minority_vecs, base_idx);

            // Pick random neighbor and interpolate
            let neighbor_idx = neighbors[sample_idx % neighbors.len()];
            let neighbor = &minority_vecs[neighbor_idx];

            // Generate synthetic sample via linear interpolation
            let synthetic_vec = self.interpolate(base, neighbor, sample_idx);
            let synthetic_feature = self.vector_to_features(&synthetic_vec, minority_category);

            synthetic.push(synthetic_feature);
            sample_idx += 1;
        }

        // Combine original + synthetic
        let mut result = features.to_vec();
        result.extend(synthetic);

        Ok(result)
    }

    /// Find k nearest neighbors using Euclidean distance
    fn find_k_nearest(&self, target: &[f32], all: &[Vec<f32>], exclude_idx: usize) -> Vec<usize> {
        let mut distances: Vec<(usize, f32)> = all
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != exclude_idx)
            .map(|(i, v)| (i, self.euclidean_distance(target, v)))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        distances
            .iter()
            .take(self.k_neighbors.min(distances.len()))
            .map(|(i, _)| *i)
            .collect()
    }

    /// Euclidean distance between two vectors
    fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Interpolate between two vectors (SMOTE core algorithm)
    fn interpolate(&self, base: &[f32], neighbor: &[f32], seed: usize) -> Vec<f32> {
        // Deterministic "random" factor based on seed
        let gap = ((seed * 17 + 42) % 100) as f32 / 100.0;

        base.iter()
            .zip(neighbor.iter())
            .map(|(b, n)| b + gap * (n - b))
            .collect()
    }

    /// Convert vector back to CommitFeatures
    ///
    /// NLP-014: Extended to support 14-dimensional feature vectors
    fn vector_to_features(&self, vec: &[f32], category: u8) -> CommitFeatures {
        CommitFeatures {
            defect_category: category,
            files_changed: vec[1].max(0.0),
            lines_added: vec[2].max(0.0),
            lines_deleted: vec[3].max(0.0),
            complexity_delta: vec[4],
            timestamp: vec[5] as f64,
            hour_of_day: (vec[6] as u8).min(23),
            day_of_week: (vec[7] as u8).min(6),
            // NLP-014: CITL features (synthesize from vector if available)
            error_code_class: if vec.len() > 8 { vec[8] as u8 } else { 4 },
            has_suggestion: if vec.len() > 9 { vec[9] as u8 } else { 0 },
            suggestion_applicability: if vec.len() > 10 { vec[10] as u8 } else { 0 },
            clippy_lint_count: if vec.len() > 11 { vec[11] as u8 } else { 0 },
            span_line_delta: if vec.len() > 12 { vec[12] } else { 0.0 },
            diagnostic_confidence: if vec.len() > 13 { vec[13] } else { 0.0 },
        }
    }
}

impl Default for Smote {
    fn default() -> Self {
        Self::new()
    }
}

/// Focal Loss for imbalanced classification
///
/// FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
///
/// Where:
/// - p_t = probability of correct class
/// - α_t = class weight (higher for minority)
/// - γ = focusing parameter (typically 2.0)
pub struct FocalLoss {
    gamma: f32,      // Focusing parameter
    alpha: Vec<f32>, // Class weights (per category)
}

impl FocalLoss {
    /// Create Focal Loss with default parameters
    pub fn new() -> Self {
        Self {
            gamma: 2.0,
            alpha: vec![1.0; 10], // Default: equal weights for 10 categories
        }
    }

    /// Create Focal Loss with custom gamma and class weights
    pub fn with_params(gamma: f32, alpha: Vec<f32>) -> Self {
        Self { gamma, alpha }
    }

    /// Compute class weights from sample distribution
    ///
    /// Inverse frequency weighting: α_i = N / (K * n_i)
    /// Where N = total samples, K = classes, n_i = samples in class i
    pub fn compute_weights(features: &[CommitFeatures]) -> Vec<f32> {
        let mut counts = [0usize; 10];
        for f in features {
            let idx = (f.defect_category as usize).min(9);
            counts[idx] += 1;
        }

        let total = features.len() as f32;
        let k = counts.iter().filter(|&&c| c > 0).count() as f32;

        counts
            .iter()
            .map(|&c| if c > 0 { total / (k * c as f32) } else { 0.0 })
            .collect()
    }

    /// Compute focal loss for a single prediction
    ///
    /// # Arguments
    /// * `prob` - Predicted probability for correct class (0-1)
    /// * `class` - True class label (0-9)
    pub fn loss(&self, prob: f32, class: u8) -> f32 {
        let p_t = prob.clamp(1e-7, 1.0 - 1e-7);
        let alpha_t = self.alpha.get(class as usize).copied().unwrap_or(1.0);

        -alpha_t * (1.0 - p_t).powf(self.gamma) * p_t.ln()
    }

    /// Compute batch focal loss
    pub fn batch_loss(&self, probs: &[f32], classes: &[u8]) -> f32 {
        probs
            .iter()
            .zip(classes.iter())
            .map(|(&p, &c)| self.loss(p, c))
            .sum::<f32>()
            / probs.len() as f32
    }
}

impl Default for FocalLoss {
    fn default() -> Self {
        Self::new()
    }
}

/// AUPRC (Area Under Precision-Recall Curve)
///
/// Better metric than AUROC for imbalanced datasets
/// Computes precision and recall at various thresholds
pub struct AuprcMetric;

impl AuprcMetric {
    /// Compute AUPRC from predictions and labels
    ///
    /// # Arguments
    /// * `predictions` - Predicted probabilities (0-1)
    /// * `labels` - True binary labels (0 or 1)
    ///
    /// # Returns
    /// AUPRC score (0-1, higher is better)
    pub fn compute(predictions: &[f32], labels: &[u8]) -> Result<f32> {
        if predictions.len() != labels.len() {
            anyhow::bail!("Predictions and labels must have same length");
        }

        if predictions.is_empty() {
            anyhow::bail!("Empty predictions");
        }

        // Sort by prediction score (descending)
        let mut pairs: Vec<(f32, u8)> = predictions
            .iter()
            .copied()
            .zip(labels.iter().copied())
            .collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let total_positives = labels.iter().filter(|&&l| l == 1).count() as f32;
        if total_positives == 0.0 {
            anyhow::bail!("No positive samples in labels");
        }

        // Compute precision-recall curve
        let mut true_positives = 0.0;
        let mut false_positives = 0.0;
        let mut auprc = 0.0;
        let mut prev_recall = 0.0;

        for (_, label) in &pairs {
            if *label == 1 {
                true_positives += 1.0;
            } else {
                false_positives += 1.0;
            }

            let precision = true_positives / (true_positives + false_positives);
            let recall = true_positives / total_positives;

            // Trapezoidal integration
            auprc += precision * (recall - prev_recall);
            prev_recall = recall;
        }

        Ok(auprc)
    }

    /// Compute precision at given recall threshold
    pub fn precision_at_recall(
        predictions: &[f32],
        labels: &[u8],
        target_recall: f32,
    ) -> Result<f32> {
        if predictions.len() != labels.len() {
            anyhow::bail!("Predictions and labels must have same length");
        }

        let mut pairs: Vec<(f32, u8)> = predictions
            .iter()
            .copied()
            .zip(labels.iter().copied())
            .collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let total_positives = labels.iter().filter(|&&l| l == 1).count() as f32;
        if total_positives == 0.0 {
            anyhow::bail!("No positive samples");
        }

        let mut true_positives = 0.0;
        let mut false_positives = 0.0;

        for (_, label) in &pairs {
            if *label == 1 {
                true_positives += 1.0;
            } else {
                false_positives += 1.0;
            }

            let recall = true_positives / total_positives;
            if recall >= target_recall {
                let precision = true_positives / (true_positives + false_positives);
                return Ok(precision);
            }
        }

        Ok(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_feature(category: u8, files: u32) -> CommitFeatures {
        CommitFeatures {
            defect_category: category,
            files_changed: files as f32,
            lines_added: 100.0,
            lines_deleted: 50.0,
            complexity_delta: 0.0,
            timestamp: 1700000000.0,
            hour_of_day: 10,
            day_of_week: 1,
            ..Default::default()
        }
    }

    #[test]
    fn test_smote_creation() {
        let smote = Smote::new();
        assert_eq!(smote.k_neighbors, 5);

        let smote_k3 = Smote::with_k(3);
        assert_eq!(smote_k3.k_neighbors, 3);
    }

    #[test]
    fn test_smote_oversample() {
        // Create imbalanced dataset: 90 majority (cat 0), 10 minority (cat 1)
        let mut features = Vec::new();
        for i in 0..90 {
            features.push(make_feature(0, i));
        }
        for i in 0..10 {
            features.push(make_feature(1, i + 100));
        }

        let smote = Smote::new();
        let balanced = smote.oversample(&features, 1, 0.5).unwrap();

        // Should have generated synthetic samples
        assert!(balanced.len() > features.len());

        // Count minority after oversampling
        let minority_count = balanced.iter().filter(|f| f.defect_category == 1).count();
        assert!(minority_count > 10);
    }

    #[test]
    fn test_smote_no_minority() {
        let features = vec![make_feature(0, 1), make_feature(0, 2)];
        let smote = Smote::new();
        let result = smote.oversample(&features, 1, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_focal_loss_creation() {
        let fl = FocalLoss::new();
        assert_eq!(fl.gamma, 2.0);
        assert_eq!(fl.alpha.len(), 10);
    }

    #[test]
    fn test_focal_loss_compute_weights() {
        let features = vec![
            make_feature(0, 1),
            make_feature(0, 2),
            make_feature(0, 3),
            make_feature(1, 4),
        ];

        let weights = FocalLoss::compute_weights(&features);

        // Minority class (1) should have higher weight
        assert!(weights[1] > weights[0]);
    }

    #[test]
    fn test_focal_loss_value() {
        let fl = FocalLoss::new();

        // High confidence correct prediction = low loss
        let loss_high = fl.loss(0.9, 0);

        // Low confidence = high loss
        let loss_low = fl.loss(0.3, 0);

        assert!(loss_high < loss_low);
    }

    #[test]
    fn test_auprc_perfect() {
        // Perfect predictions
        let predictions = vec![0.9, 0.8, 0.2, 0.1];
        let labels = vec![1, 1, 0, 0];

        let auprc = AuprcMetric::compute(&predictions, &labels).unwrap();
        assert!((auprc - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_auprc_range() {
        // Varying predictions on imbalanced data
        let predictions = vec![0.9, 0.7, 0.5, 0.3, 0.2, 0.15, 0.1, 0.05, 0.02, 0.01];
        let labels = vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0]; // 10% positive, ranked first

        let auprc = AuprcMetric::compute(&predictions, &labels).unwrap();
        // Perfect ranking (positive sample first) = AUPRC of 1.0
        assert!((auprc - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_precision_at_recall() {
        let predictions = vec![0.9, 0.7, 0.5, 0.3];
        let labels = vec![1, 1, 0, 0];

        let p_at_50 = AuprcMetric::precision_at_recall(&predictions, &labels, 0.5).unwrap();
        assert!((p_at_50 - 1.0).abs() < 0.01); // First prediction is positive

        let p_at_100 = AuprcMetric::precision_at_recall(&predictions, &labels, 1.0).unwrap();
        assert!((p_at_100 - 1.0).abs() < 0.01); // Both positives ranked first
    }

    #[test]
    fn test_smote_default() {
        let smote = Smote::default();
        assert_eq!(smote.k_neighbors, 5);
    }

    #[test]
    fn test_smote_no_samples_needed() {
        // Minority already balanced
        let features = vec![
            make_feature(0, 1),
            make_feature(0, 2),
            make_feature(1, 10),
            make_feature(1, 11),
        ];

        let smote = Smote::new();
        let result = smote.oversample(&features, 1, 0.5).unwrap();

        // Should return original features (no oversampling needed)
        assert_eq!(result.len(), features.len());
    }

    #[test]
    fn test_smote_vector_to_features_clamping() {
        let smote = Smote::new();

        // Test negative values are clamped to 0
        let vec = vec![0.0, -5.0, -10.0, -1.0, 0.5, 1700000000.0, 25.0, 8.0];
        let feature = smote.vector_to_features(&vec, 3);

        assert_eq!(feature.files_changed, 0.0); // Clamped from -5.0
        assert_eq!(feature.lines_added, 0.0); // Clamped from -10.0
        assert_eq!(feature.lines_deleted, 0.0); // Clamped from -1.0
        assert_eq!(feature.hour_of_day, 23); // Clamped from 25
        assert_eq!(feature.day_of_week, 6); // Clamped from 8
    }

    #[test]
    fn test_focal_loss_with_params() {
        let weights = vec![2.0, 1.5, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let fl = FocalLoss::with_params(3.0, weights.clone());

        assert_eq!(fl.gamma, 3.0);
        assert_eq!(fl.alpha, weights);
    }

    #[test]
    fn test_focal_loss_default() {
        let fl = FocalLoss::default();
        assert_eq!(fl.gamma, 2.0);
        assert_eq!(fl.alpha.len(), 10);
    }

    #[test]
    fn test_focal_loss_batch() {
        let fl = FocalLoss::new();
        let probs = vec![0.9, 0.8, 0.7, 0.6];
        let classes = vec![0, 0, 1, 1];

        let batch_loss = fl.batch_loss(&probs, &classes);
        assert!(batch_loss > 0.0);

        // Average of individual losses
        let expected =
            (fl.loss(0.9, 0) + fl.loss(0.8, 0) + fl.loss(0.7, 1) + fl.loss(0.6, 1)) / 4.0;
        assert!((batch_loss - expected).abs() < 0.001);
    }

    #[test]
    fn test_focal_loss_compute_weights_edge_cases() {
        // Single class
        let features = vec![make_feature(0, 1), make_feature(0, 2), make_feature(0, 3)];
        let weights = FocalLoss::compute_weights(&features);

        // All weight should go to class 0
        assert!(weights[0] > 0.0);
        assert_eq!(weights[1], 0.0); // No samples in class 1
    }

    #[test]
    fn test_auprc_length_mismatch() {
        let predictions = vec![0.9, 0.8, 0.7];
        let labels = vec![1, 0]; // Different length

        let result = AuprcMetric::compute(&predictions, &labels);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("same length"));
    }

    #[test]
    fn test_auprc_empty() {
        let predictions: Vec<f32> = vec![];
        let labels: Vec<u8> = vec![];

        let result = AuprcMetric::compute(&predictions, &labels);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Empty predictions"));
    }

    #[test]
    fn test_auprc_no_positives() {
        let predictions = vec![0.9, 0.8, 0.7];
        let labels = vec![0, 0, 0]; // All negatives

        let result = AuprcMetric::compute(&predictions, &labels);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No positive samples"));
    }

    #[test]
    fn test_precision_at_recall_length_mismatch() {
        let predictions = vec![0.9, 0.8];
        let labels = vec![1, 0, 0];

        let result = AuprcMetric::precision_at_recall(&predictions, &labels, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_precision_at_recall_no_positives() {
        let predictions = vec![0.9, 0.8, 0.7];
        let labels = vec![0, 0, 0];

        let result = AuprcMetric::precision_at_recall(&predictions, &labels, 0.5);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No positive samples"));
    }

    #[test]
    fn test_precision_at_recall_target_not_reached() {
        let predictions = vec![0.9, 0.7, 0.3];
        let labels = vec![0, 0, 1]; // Positive ranked last

        // Can't reach 100% recall without going through all samples
        let p = AuprcMetric::precision_at_recall(&predictions, &labels, 1.0).unwrap();
        assert!((p - 1.0 / 3.0).abs() < 0.01); // 1 positive out of 3 samples
    }

    #[test]
    fn test_smote_interpolate_deterministic() {
        let smote = Smote::new();
        let base = vec![1.0, 2.0, 3.0];
        let neighbor = vec![2.0, 4.0, 6.0];

        let synthetic1 = smote.interpolate(&base, &neighbor, 0);
        let synthetic2 = smote.interpolate(&base, &neighbor, 0);

        // Same seed produces same result
        assert_eq!(synthetic1, synthetic2);

        // Different seeds produce different results
        let synthetic3 = smote.interpolate(&base, &neighbor, 1);
        assert_ne!(synthetic1, synthetic3);
    }
}
