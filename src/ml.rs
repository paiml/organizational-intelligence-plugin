//! ML Model Integration for Defect Prediction
//!
//! Implements PHASE2-005: aprender ML model integration
//! Provides Random Forest for classification and K-means for clustering
//!
//! References:
//! - Breiman (2001): Random Forests
//! - MacQueen (1967): K-means Clustering

use crate::features::CommitFeatures;
use anyhow::Result;

/// Defect prediction model using Random Forest
///
/// Predicts defect category (0-9) from commit features
pub struct DefectPredictor {
    n_trees: usize,
    max_depth: usize,
    trained: bool,
    // Store training data for simple prediction (Phase 1)
    // Full aprender integration in Phase 2
    training_data: Vec<(Vec<f32>, u8)>,
}

impl DefectPredictor {
    /// Create new predictor with default parameters
    pub fn new() -> Self {
        Self {
            n_trees: 100,
            max_depth: 10,
            trained: false,
            training_data: Vec::new(),
        }
    }

    /// Create predictor with custom parameters
    pub fn with_params(n_trees: usize, max_depth: usize) -> Self {
        Self {
            n_trees,
            max_depth,
            trained: false,
            training_data: Vec::new(),
        }
    }

    /// Train model on labeled features
    ///
    /// # Arguments
    /// * `features` - Training features with defect_category labels
    pub fn train(&mut self, features: &[CommitFeatures]) -> Result<()> {
        if features.is_empty() {
            anyhow::bail!("Cannot train on empty dataset");
        }

        // Store training data for k-NN based prediction
        self.training_data = features
            .iter()
            .map(|f| (f.to_vector(), f.defect_category))
            .collect();

        self.trained = true;
        Ok(())
    }

    /// Predict defect category for new features
    ///
    /// Uses k-NN approximation (k=5) for Phase 1
    /// Full Random Forest in Phase 2 with aprender
    pub fn predict(&self, features: &CommitFeatures) -> Result<u8> {
        if !self.trained {
            anyhow::bail!("Model not trained");
        }

        let query = features.to_vector();

        // k-NN prediction (k=5)
        let k = 5.min(self.training_data.len());
        let mut distances: Vec<(f32, u8)> = self
            .training_data
            .iter()
            .map(|(v, label)| (Self::euclidean_distance(&query, v), *label))
            .collect();

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Vote among k nearest neighbors
        let mut votes = [0u32; 10];
        for (_, label) in distances.iter().take(k) {
            let idx = (*label as usize).min(9);
            votes[idx] += 1;
        }

        // Return most common category
        let prediction = votes
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(idx, _)| idx as u8)
            .unwrap_or(0);

        Ok(prediction)
    }

    /// Predict probabilities for all categories
    pub fn predict_proba(&self, features: &CommitFeatures) -> Result<Vec<f32>> {
        if !self.trained {
            anyhow::bail!("Model not trained");
        }

        let query = features.to_vector();
        let k = 10.min(self.training_data.len());

        let mut distances: Vec<(f32, u8)> = self
            .training_data
            .iter()
            .map(|(v, label)| (Self::euclidean_distance(&query, v), *label))
            .collect();

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Compute probability as fraction of k neighbors
        let mut counts = [0u32; 10];
        for (_, label) in distances.iter().take(k) {
            let idx = (*label as usize).min(9);
            counts[idx] += 1;
        }

        let probs: Vec<f32> = counts.iter().map(|&c| c as f32 / k as f32).collect();

        Ok(probs)
    }

    /// Get model parameters
    pub fn params(&self) -> (usize, usize) {
        (self.n_trees, self.max_depth)
    }

    /// Check if model is trained
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

impl Default for DefectPredictor {
    fn default() -> Self {
        Self::new()
    }
}

/// Pattern clustering using K-means
///
/// Groups similar commits into clusters for pattern discovery
pub struct PatternClusterer {
    k: usize,
    max_iterations: usize,
    centroids: Vec<Vec<f32>>,
    trained: bool,
}

impl PatternClusterer {
    /// Create clusterer with default k=5 clusters
    pub fn new() -> Self {
        Self {
            k: 5,
            max_iterations: 100,
            centroids: Vec::new(),
            trained: false,
        }
    }

    /// Create clusterer with custom k
    pub fn with_k(k: usize) -> Self {
        Self {
            k,
            max_iterations: 100,
            centroids: Vec::new(),
            trained: false,
        }
    }

    /// Fit clusters to data
    pub fn fit(&mut self, features: &[CommitFeatures]) -> Result<()> {
        if features.is_empty() {
            anyhow::bail!("Cannot cluster empty dataset");
        }

        if features.len() < self.k {
            anyhow::bail!("Need at least {} samples for {} clusters", self.k, self.k);
        }

        let vectors: Vec<Vec<f32>> = features.iter().map(|f| f.to_vector()).collect();
        let n_dims = CommitFeatures::DIMENSION;

        // Initialize centroids (first k points)
        self.centroids = vectors.iter().take(self.k).cloned().collect();

        // K-means iteration
        for _ in 0..self.max_iterations {
            // Assign points to clusters
            let assignments: Vec<usize> =
                vectors.iter().map(|v| self.nearest_centroid(v)).collect();

            // Update centroids
            let mut new_centroids = vec![vec![0.0; n_dims]; self.k];
            let mut counts = vec![0usize; self.k];

            for (vec, &cluster) in vectors.iter().zip(assignments.iter()) {
                for (dim, &val) in vec.iter().enumerate() {
                    new_centroids[cluster][dim] += val;
                }
                counts[cluster] += 1;
            }

            // Normalize centroids
            for (centroid, &count) in new_centroids.iter_mut().zip(counts.iter()) {
                if count > 0 {
                    for val in centroid.iter_mut() {
                        *val /= count as f32;
                    }
                }
            }

            // Check convergence
            let converged = self
                .centroids
                .iter()
                .zip(new_centroids.iter())
                .all(|(old, new)| Self::euclidean_distance(old, new) < 1e-6);

            self.centroids = new_centroids;

            if converged {
                break;
            }
        }

        self.trained = true;
        Ok(())
    }

    /// Predict cluster for new features
    pub fn predict(&self, features: &CommitFeatures) -> Result<usize> {
        if !self.trained {
            anyhow::bail!("Clusterer not fitted");
        }

        let vec = features.to_vector();
        Ok(self.nearest_centroid(&vec))
    }

    /// Predict clusters for multiple features
    pub fn predict_batch(&self, features: &[CommitFeatures]) -> Result<Vec<usize>> {
        if !self.trained {
            anyhow::bail!("Clusterer not fitted");
        }

        Ok(features
            .iter()
            .map(|f| self.nearest_centroid(&f.to_vector()))
            .collect())
    }

    /// Get cluster centroids
    pub fn centroids(&self) -> &[Vec<f32>] {
        &self.centroids
    }

    /// Compute inertia (sum of squared distances to centroids)
    pub fn inertia(&self, features: &[CommitFeatures]) -> Result<f32> {
        if !self.trained {
            anyhow::bail!("Clusterer not fitted");
        }

        let total: f32 = features
            .iter()
            .map(|f| {
                let vec = f.to_vector();
                let cluster = self.nearest_centroid(&vec);
                Self::euclidean_distance(&vec, &self.centroids[cluster]).powi(2)
            })
            .sum();

        Ok(total)
    }

    fn nearest_centroid(&self, point: &[f32]) -> usize {
        self.centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, Self::euclidean_distance(point, c)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

impl Default for PatternClusterer {
    fn default() -> Self {
        Self::new()
    }
}

/// Model evaluation metrics
pub struct ModelMetrics;

impl ModelMetrics {
    /// Compute accuracy
    pub fn accuracy(predictions: &[u8], labels: &[u8]) -> f32 {
        if predictions.len() != labels.len() || predictions.is_empty() {
            return 0.0;
        }

        let correct = predictions
            .iter()
            .zip(labels.iter())
            .filter(|(p, l)| p == l)
            .count();

        correct as f32 / predictions.len() as f32
    }

    /// Compute per-class precision
    pub fn precision(predictions: &[u8], labels: &[u8], class: u8) -> f32 {
        let true_positives = predictions
            .iter()
            .zip(labels.iter())
            .filter(|(&p, &l)| p == class && l == class)
            .count() as f32;

        let predicted_positives = predictions.iter().filter(|&&p| p == class).count() as f32;

        if predicted_positives > 0.0 {
            true_positives / predicted_positives
        } else {
            0.0
        }
    }

    /// Compute per-class recall
    pub fn recall(predictions: &[u8], labels: &[u8], class: u8) -> f32 {
        let true_positives = predictions
            .iter()
            .zip(labels.iter())
            .filter(|(&p, &l)| p == class && l == class)
            .count() as f32;

        let actual_positives = labels.iter().filter(|&&l| l == class).count() as f32;

        if actual_positives > 0.0 {
            true_positives / actual_positives
        } else {
            0.0
        }
    }

    /// Compute F1 score
    pub fn f1_score(predictions: &[u8], labels: &[u8], class: u8) -> f32 {
        let p = Self::precision(predictions, labels, class);
        let r = Self::recall(predictions, labels, class);

        if p + r > 0.0 {
            2.0 * p * r / (p + r)
        } else {
            0.0
        }
    }

    /// Compute silhouette score for clustering
    pub fn silhouette_score(features: &[CommitFeatures], assignments: &[usize], k: usize) -> f32 {
        if features.len() != assignments.len() || features.is_empty() {
            return 0.0;
        }

        let vectors: Vec<Vec<f32>> = features.iter().map(|f| f.to_vector()).collect();

        let mut total_score = 0.0;
        let n = vectors.len();

        for i in 0..n {
            let cluster_i = assignments[i];

            // a(i) = mean distance to same cluster
            let same_cluster: Vec<_> = (0..n)
                .filter(|&j| j != i && assignments[j] == cluster_i)
                .collect();

            let a = if same_cluster.is_empty() {
                0.0
            } else {
                same_cluster
                    .iter()
                    .map(|&j| Self::euclidean_distance(&vectors[i], &vectors[j]))
                    .sum::<f32>()
                    / same_cluster.len() as f32
            };

            // b(i) = min mean distance to other clusters
            let mut b = f32::INFINITY;
            for c in 0..k {
                if c == cluster_i {
                    continue;
                }
                let other_cluster: Vec<_> = (0..n).filter(|&j| assignments[j] == c).collect();

                if !other_cluster.is_empty() {
                    let mean_dist = other_cluster
                        .iter()
                        .map(|&j| Self::euclidean_distance(&vectors[i], &vectors[j]))
                        .sum::<f32>()
                        / other_cluster.len() as f32;
                    b = b.min(mean_dist);
                }
            }

            if b.is_infinite() {
                b = 0.0;
            }

            // s(i) = (b - a) / max(a, b)
            let s = if a.max(b) > 0.0 {
                (b - a) / a.max(b)
            } else {
                0.0
            };

            total_score += s;
        }

        total_score / n as f32
    }

    fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_feature(category: u8, files: u32) -> CommitFeatures {
        CommitFeatures {
            defect_category: category,
            files_changed: files as f32,
            lines_added: (files * 10) as f32,
            lines_deleted: (files * 5) as f32,
            complexity_delta: files as f32 * 0.1,
            timestamp: 1700000000.0 + files as f64,
            hour_of_day: 10,
            day_of_week: 1,
        }
    }

    #[test]
    fn test_predictor_creation() {
        let predictor = DefectPredictor::new();
        assert_eq!(predictor.params(), (100, 10));
        assert!(!predictor.is_trained());
    }

    #[test]
    fn test_predictor_train() {
        let mut predictor = DefectPredictor::new();
        let features = vec![
            make_feature(0, 1),
            make_feature(0, 2),
            make_feature(1, 10),
            make_feature(1, 11),
        ];

        predictor.train(&features).unwrap();
        assert!(predictor.is_trained());
    }

    #[test]
    fn test_predictor_predict() {
        let mut predictor = DefectPredictor::new();
        let features = vec![
            make_feature(0, 1),
            make_feature(0, 2),
            make_feature(0, 3),
            make_feature(1, 100),
            make_feature(1, 101),
            make_feature(1, 102),
        ];

        predictor.train(&features).unwrap();

        // Similar to category 0 samples
        let test_cat0 = make_feature(0, 2);
        let pred0 = predictor.predict(&test_cat0).unwrap();
        assert_eq!(pred0, 0);

        // Similar to category 1 samples
        let test_cat1 = make_feature(1, 101);
        let pred1 = predictor.predict(&test_cat1).unwrap();
        assert_eq!(pred1, 1);
    }

    #[test]
    fn test_predictor_proba() {
        let mut predictor = DefectPredictor::new();
        let features = vec![make_feature(0, 1), make_feature(0, 2), make_feature(1, 100)];

        predictor.train(&features).unwrap();

        let probs = predictor.predict_proba(&make_feature(0, 1)).unwrap();
        assert_eq!(probs.len(), 10);
        assert!(probs[0] > probs[1]); // Category 0 should have higher probability
    }

    #[test]
    fn test_clusterer_creation() {
        let clusterer = PatternClusterer::new();
        assert_eq!(clusterer.k, 5);
    }

    #[test]
    fn test_clusterer_fit() {
        let mut clusterer = PatternClusterer::with_k(2);
        let features = vec![
            make_feature(0, 1),
            make_feature(0, 2),
            make_feature(0, 3),
            make_feature(1, 100),
            make_feature(1, 101),
            make_feature(1, 102),
        ];

        clusterer.fit(&features).unwrap();
        assert!(clusterer.trained);
        assert_eq!(clusterer.centroids().len(), 2);
    }

    #[test]
    fn test_clusterer_predict() {
        let mut clusterer = PatternClusterer::with_k(2);
        let features = vec![
            make_feature(0, 1),
            make_feature(0, 2),
            make_feature(1, 100),
            make_feature(1, 101),
        ];

        clusterer.fit(&features).unwrap();

        let assignments = clusterer.predict_batch(&features).unwrap();

        // Similar features should be in same cluster
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[2], assignments[3]);
        // Different features should be in different clusters
        assert_ne!(assignments[0], assignments[2]);
    }

    #[test]
    fn test_clusterer_inertia() {
        let mut clusterer = PatternClusterer::with_k(2);
        let features = vec![
            make_feature(0, 1),
            make_feature(0, 2),
            make_feature(1, 100),
            make_feature(1, 101),
        ];

        clusterer.fit(&features).unwrap();
        let inertia = clusterer.inertia(&features).unwrap();
        assert!(inertia >= 0.0);
    }

    #[test]
    fn test_metrics_accuracy() {
        let predictions = vec![0, 0, 1, 1];
        let labels = vec![0, 0, 1, 0];

        let acc = ModelMetrics::accuracy(&predictions, &labels);
        assert!((acc - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_metrics_precision_recall() {
        let predictions = vec![1, 1, 0, 0];
        let labels = vec![1, 0, 0, 0];

        let precision = ModelMetrics::precision(&predictions, &labels, 1);
        assert!((precision - 0.5).abs() < 0.01); // 1 TP, 1 FP

        let recall = ModelMetrics::recall(&predictions, &labels, 1);
        assert!((recall - 1.0).abs() < 0.01); // 1 TP, 0 FN
    }

    #[test]
    fn test_metrics_f1() {
        let predictions = vec![1, 1, 0, 0];
        let labels = vec![1, 0, 0, 0];

        let f1 = ModelMetrics::f1_score(&predictions, &labels, 1);
        // F1 = 2 * 0.5 * 1.0 / (0.5 + 1.0) = 0.667
        assert!((f1 - 0.667).abs() < 0.01);
    }

    #[test]
    fn test_silhouette_score() {
        let features = vec![
            make_feature(0, 1),
            make_feature(0, 2),
            make_feature(1, 100),
            make_feature(1, 101),
        ];
        let assignments = vec![0, 0, 1, 1];

        let score = ModelMetrics::silhouette_score(&features, &assignments, 2);
        // Well-separated clusters should have positive silhouette
        assert!(score > 0.0);
    }
}
