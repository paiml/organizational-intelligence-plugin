//! Weighted Ensemble Risk Score and Calibrated Defect Prediction
//!
//! This module implements Phase 6 (Weighted Ensemble) and Phase 7 (Calibrated Probability)
//! from the Tarantula specification. It combines multiple defect signals using weak supervision
//! and provides calibrated probability predictions with confidence intervals.
//!
//! # Toyota Way Alignment
//! - **Jidoka**: Learned weights are interpretable - developers see why
//! - **Kaizen**: Model improves as more defect history accumulates
//! - **Genchi Genbutsu**: Weights derived from actual codebase patterns
//! - **Heijunka**: Batch training amortizes cost across many predictions
//! - **Muri**: Low-confidence predictions flagged for human judgment

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Features extracted for each file for defect prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileFeatures {
    /// File path
    pub path: PathBuf,
    /// SBFL suspiciousness score (0.0-1.0)
    pub sbfl_score: f32,
    /// Technical Debt Grade score (0.0-1.0, inverted so higher = worse)
    pub tdg_score: f32,
    /// Normalized commit frequency (0.0-1.0)
    pub churn_score: f32,
    /// Normalized cyclomatic complexity (0.0-1.0)
    pub complexity_score: f32,
    /// RAG similarity to historical bugs (0.0-1.0)
    pub rag_similarity: f32,
}

impl FileFeatures {
    /// Create new FileFeatures
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            sbfl_score: 0.0,
            tdg_score: 0.0,
            churn_score: 0.0,
            complexity_score: 0.0,
            rag_similarity: 0.0,
        }
    }

    /// Builder method to set SBFL score
    pub fn with_sbfl(mut self, score: f32) -> Self {
        self.sbfl_score = score.clamp(0.0, 1.0);
        self
    }

    /// Builder method to set TDG score
    pub fn with_tdg(mut self, score: f32) -> Self {
        self.tdg_score = score.clamp(0.0, 1.0);
        self
    }

    /// Builder method to set churn score
    pub fn with_churn(mut self, score: f32) -> Self {
        self.churn_score = score.clamp(0.0, 1.0);
        self
    }

    /// Builder method to set complexity score
    pub fn with_complexity(mut self, score: f32) -> Self {
        self.complexity_score = score.clamp(0.0, 1.0);
        self
    }

    /// Builder method to set RAG similarity
    pub fn with_rag_similarity(mut self, score: f32) -> Self {
        self.rag_similarity = score.clamp(0.0, 1.0);
        self
    }

    /// Convert to feature vector for ML models
    pub fn to_vector(&self) -> Vec<f32> {
        vec![
            self.sbfl_score,
            self.tdg_score,
            self.churn_score,
            self.complexity_score,
            self.rag_similarity,
        ]
    }
}

/// Output of a labeling function
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelOutput {
    /// Positive label (likely defect)
    Positive,
    /// Negative label (likely clean)
    Negative,
    /// Abstain (uncertain)
    Abstain,
}

/// Trait for labeling functions that emit noisy labels
pub trait LabelingFunction: Send + Sync {
    /// Apply the labeling function to features
    fn apply(&self, features: &FileFeatures) -> LabelOutput;

    /// Get the name of this labeling function
    fn name(&self) -> &str;
}

/// SBFL-based labeling function
#[derive(Debug, Clone)]
pub struct SbflLabelingFunction {
    /// Threshold for positive label
    pub positive_threshold: f32,
    /// Threshold for negative label
    pub negative_threshold: f32,
}

impl SbflLabelingFunction {
    pub fn new(positive_threshold: f32, negative_threshold: f32) -> Self {
        Self {
            positive_threshold,
            negative_threshold,
        }
    }
}

impl LabelingFunction for SbflLabelingFunction {
    fn apply(&self, features: &FileFeatures) -> LabelOutput {
        if features.sbfl_score > self.positive_threshold {
            LabelOutput::Positive
        } else if features.sbfl_score < self.negative_threshold {
            LabelOutput::Negative
        } else {
            LabelOutput::Abstain
        }
    }

    fn name(&self) -> &str {
        "SBFL"
    }
}

/// TDG-based labeling function (low TDG = likely defect)
#[derive(Debug, Clone)]
pub struct TdgLabelingFunction {
    /// Maximum TDG grade for positive label (defect likely if TDG < this)
    pub max_grade: f32,
    /// Minimum TDG grade for negative label (clean if TDG > this)
    pub min_grade: f32,
}

impl TdgLabelingFunction {
    pub fn new(max_grade: f32, min_grade: f32) -> Self {
        Self {
            max_grade,
            min_grade,
        }
    }
}

impl LabelingFunction for TdgLabelingFunction {
    fn apply(&self, features: &FileFeatures) -> LabelOutput {
        // Note: tdg_score is inverted (higher = worse debt = lower grade)
        if features.tdg_score > self.max_grade {
            LabelOutput::Positive // High debt = likely defect
        } else if features.tdg_score < self.min_grade {
            LabelOutput::Negative // Low debt = likely clean
        } else {
            LabelOutput::Abstain
        }
    }

    fn name(&self) -> &str {
        "TDG"
    }
}

/// Churn-based labeling function
#[derive(Debug, Clone)]
pub struct ChurnLabelingFunction {
    /// Percentile threshold for high churn
    pub high_percentile: f32,
    /// Percentile threshold for low churn
    pub low_percentile: f32,
}

impl ChurnLabelingFunction {
    pub fn new(high_percentile: f32, low_percentile: f32) -> Self {
        Self {
            high_percentile,
            low_percentile,
        }
    }
}

impl LabelingFunction for ChurnLabelingFunction {
    fn apply(&self, features: &FileFeatures) -> LabelOutput {
        if features.churn_score > self.high_percentile {
            LabelOutput::Positive
        } else if features.churn_score < self.low_percentile {
            LabelOutput::Negative
        } else {
            LabelOutput::Abstain
        }
    }

    fn name(&self) -> &str {
        "Churn"
    }
}

/// Complexity-based labeling function
#[derive(Debug, Clone)]
pub struct ComplexityLabelingFunction {
    /// Max complexity threshold (above = likely defect)
    pub max_complexity: f32,
    /// Min complexity threshold (below = likely clean)
    pub min_complexity: f32,
}

impl ComplexityLabelingFunction {
    pub fn new(max_complexity: f32, min_complexity: f32) -> Self {
        Self {
            max_complexity,
            min_complexity,
        }
    }
}

impl LabelingFunction for ComplexityLabelingFunction {
    fn apply(&self, features: &FileFeatures) -> LabelOutput {
        if features.complexity_score > self.max_complexity {
            LabelOutput::Positive
        } else if features.complexity_score < self.min_complexity {
            LabelOutput::Negative
        } else {
            LabelOutput::Abstain
        }
    }

    fn name(&self) -> &str {
        "Complexity"
    }
}

/// RAG similarity-based labeling function
#[derive(Debug, Clone)]
pub struct RagSimilarityLabelingFunction {
    /// Threshold for similar to historical bugs
    pub threshold: f32,
}

impl RagSimilarityLabelingFunction {
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }
}

impl LabelingFunction for RagSimilarityLabelingFunction {
    fn apply(&self, features: &FileFeatures) -> LabelOutput {
        if features.rag_similarity > self.threshold {
            LabelOutput::Positive
        } else {
            LabelOutput::Abstain // RAG only provides positive signal
        }
    }

    fn name(&self) -> &str {
        "RAG_Similarity"
    }
}

/// Learned weights for combining labeling functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelModelWeights {
    /// Weights for each labeling function
    pub weights: Vec<f32>,
    /// Names of labeling functions
    pub names: Vec<String>,
    /// Number of training iterations
    pub n_iterations: usize,
    /// Final log-likelihood
    pub log_likelihood: f64,
}

impl LabelModelWeights {
    /// Get weight by name
    pub fn get_weight(&self, name: &str) -> Option<f32> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|idx| self.weights[idx])
    }

    /// Get weights as HashMap for easy access
    pub fn to_hashmap(&self) -> HashMap<String, f32> {
        self.names
            .iter()
            .cloned()
            .zip(self.weights.iter().copied())
            .collect()
    }
}

/// Weighted Ensemble Model using weak supervision
///
/// Phase 6: Combines multiple noisy signals (SBFL, TDG, Churn, Complexity, RAG)
/// to learn optimal weights for defect prediction.
pub struct WeightedEnsembleModel {
    /// Labeling functions
    labeling_functions: Vec<Box<dyn LabelingFunction>>,
    /// Learned weights (after fitting)
    weights: Option<LabelModelWeights>,
    /// Number of EM iterations
    n_iterations: usize,
    /// Convergence threshold
    convergence_threshold: f64,
}

impl Default for WeightedEnsembleModel {
    fn default() -> Self {
        Self::new()
    }
}

impl WeightedEnsembleModel {
    /// Create a new ensemble model with default labeling functions
    pub fn new() -> Self {
        let lfs: Vec<Box<dyn LabelingFunction>> = vec![
            Box::new(SbflLabelingFunction::new(0.7, 0.2)),
            Box::new(TdgLabelingFunction::new(0.5, 0.2)),
            Box::new(ChurnLabelingFunction::new(0.9, 0.3)),
            Box::new(ComplexityLabelingFunction::new(0.7, 0.3)),
            Box::new(RagSimilarityLabelingFunction::new(0.8)),
        ];

        Self {
            labeling_functions: lfs,
            weights: None,
            n_iterations: 100,
            convergence_threshold: 1e-6,
        }
    }

    /// Create with custom labeling functions
    pub fn with_labeling_functions(lfs: Vec<Box<dyn LabelingFunction>>) -> Self {
        Self {
            labeling_functions: lfs,
            weights: None,
            n_iterations: 100,
            convergence_threshold: 1e-6,
        }
    }

    /// Set number of EM iterations
    pub fn with_iterations(mut self, n: usize) -> Self {
        self.n_iterations = n;
        self
    }

    /// Fit the model using EM algorithm on unlabeled data
    ///
    /// This learns optimal weights for each labeling function by
    /// maximizing the likelihood of the observed label matrix.
    pub fn fit(&mut self, files: &[FileFeatures]) -> anyhow::Result<()> {
        if files.is_empty() {
            anyhow::bail!("Cannot fit on empty data");
        }

        let n_lfs = self.labeling_functions.len();
        if n_lfs == 0 {
            anyhow::bail!("No labeling functions provided");
        }

        // Generate label matrix: rows = files, cols = LFs
        let label_matrix: Vec<Vec<LabelOutput>> = files
            .iter()
            .map(|f| {
                self.labeling_functions
                    .iter()
                    .map(|lf| lf.apply(f))
                    .collect()
            })
            .collect();

        // EM Algorithm for Label Model
        // Initialize weights uniformly
        let mut weights: Vec<f64> = vec![1.0 / n_lfs as f64; n_lfs];
        let mut prev_ll = f64::NEG_INFINITY;

        for _iter in 0..self.n_iterations {
            // E-step: Estimate latent labels
            let mut expected_labels: Vec<f64> = Vec::with_capacity(files.len());
            for row in &label_matrix {
                let mut pos_score = 0.0;
                let mut neg_score = 0.0;

                for (j, &output) in row.iter().enumerate() {
                    match output {
                        LabelOutput::Positive => pos_score += weights[j],
                        LabelOutput::Negative => neg_score += weights[j],
                        LabelOutput::Abstain => {}
                    }
                }

                // Sigmoid probability
                let total = pos_score + neg_score;
                let prob = if total > 0.0 { pos_score / total } else { 0.5 };
                expected_labels.push(prob);
            }

            // M-step: Update weights based on expected labels
            let mut new_weights = vec![0.0; n_lfs];
            let mut counts = vec![0.0; n_lfs];

            for (i, row) in label_matrix.iter().enumerate() {
                let y = expected_labels[i];
                for (j, &output) in row.iter().enumerate() {
                    match output {
                        LabelOutput::Positive => {
                            new_weights[j] += y;
                            counts[j] += 1.0;
                        }
                        LabelOutput::Negative => {
                            new_weights[j] += 1.0 - y;
                            counts[j] += 1.0;
                        }
                        LabelOutput::Abstain => {}
                    }
                }
            }

            // Normalize weights
            for j in 0..n_lfs {
                if counts[j] > 0.0 {
                    new_weights[j] /= counts[j];
                } else {
                    new_weights[j] = 0.5; // Default weight if no labels
                }
            }

            // Compute log-likelihood
            let ll: f64 = expected_labels
                .iter()
                .map(|&p| {
                    let p_clamped = p.clamp(1e-10, 1.0 - 1e-10);
                    p_clamped.ln() + (1.0 - p_clamped).ln()
                })
                .sum();

            // Check convergence
            if (ll - prev_ll).abs() < self.convergence_threshold {
                break;
            }

            weights = new_weights;
            prev_ll = ll;
        }

        // Normalize weights to sum to 1
        let sum: f64 = weights.iter().sum();
        if sum > 0.0 {
            for w in &mut weights {
                *w /= sum;
            }
        }

        let names: Vec<String> = self
            .labeling_functions
            .iter()
            .map(|lf| lf.name().to_string())
            .collect();

        self.weights = Some(LabelModelWeights {
            weights: weights.iter().map(|&w| w as f32).collect(),
            names,
            n_iterations: self.n_iterations,
            log_likelihood: prev_ll,
        });

        Ok(())
    }

    /// Predict defect probability for a file
    pub fn predict(&self, features: &FileFeatures) -> f32 {
        let weights = match &self.weights {
            Some(w) => &w.weights,
            None => return 0.5, // Untrained model returns neutral
        };

        let mut pos_score = 0.0f32;
        let mut neg_score = 0.0f32;

        for (lf, &weight) in self.labeling_functions.iter().zip(weights.iter()) {
            match lf.apply(features) {
                LabelOutput::Positive => pos_score += weight,
                LabelOutput::Negative => neg_score += weight,
                LabelOutput::Abstain => {}
            }
        }

        let total = pos_score + neg_score;
        if total > 0.0 {
            pos_score / total
        } else {
            0.5
        }
    }

    /// Get learned weights for interpretability
    pub fn get_weights(&self) -> Option<&LabelModelWeights> {
        self.weights.as_ref()
    }

    /// Check if model is fitted
    pub fn is_fitted(&self) -> bool {
        self.weights.is_some()
    }

    /// Save model weights to file
    pub fn save(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model not fitted"))?;
        let json = serde_json::to_string_pretty(weights)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load model weights from file
    pub fn load(&mut self, path: &std::path::Path) -> anyhow::Result<()> {
        let json = std::fs::read_to_string(path)?;
        let weights: LabelModelWeights = serde_json::from_str(&json)?;
        self.weights = Some(weights);
        Ok(())
    }
}

// ============================================================================
// Phase 7: Calibrated Defect Probability
// ============================================================================

/// Confidence level based on CI width
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfidenceLevel {
    /// CI width < 0.15
    High,
    /// CI width 0.15-0.30
    Medium,
    /// CI width > 0.30
    Low,
}

impl std::fmt::Display for ConfidenceLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfidenceLevel::High => write!(f, "HIGH"),
            ConfidenceLevel::Medium => write!(f, "MEDIUM"),
            ConfidenceLevel::Low => write!(f, "LOW"),
        }
    }
}

/// Contribution of each factor to the prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorContribution {
    /// Name of the factor
    pub factor_name: String,
    /// Contribution percentage (0-100)
    pub contribution_pct: f32,
    /// Raw value of the factor
    pub raw_value: f32,
}

/// Prediction with uncertainty quantification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibratedPrediction {
    /// File path
    pub file: PathBuf,
    /// Line number (optional, for statement-level)
    pub line: Option<usize>,
    /// Calibrated probability of defect
    pub probability: f32,
    /// 95% confidence interval (low, high)
    pub confidence_interval: (f32, f32),
    /// Confidence level based on CI width
    pub confidence_level: ConfidenceLevel,
    /// Factor contributions for explainability
    pub contributing_factors: Vec<FactorContribution>,
}

/// Calibration metrics for model validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationMetrics {
    /// Expected Calibration Error
    pub ece: f32,
    /// Maximum Calibration Error
    pub mce: f32,
    /// Brier Score (lower is better)
    pub brier_score: f32,
    /// Coverage of confidence intervals
    pub coverage: f32,
}

/// Isotonic regression for calibration
#[derive(Debug, Clone, Serialize, Deserialize)]
struct IsotonicCalibrator {
    /// X values (raw probabilities)
    x_values: Vec<f32>,
    /// Y values (calibrated probabilities)
    y_values: Vec<f32>,
}

impl IsotonicCalibrator {
    fn new() -> Self {
        Self {
            x_values: Vec::new(),
            y_values: Vec::new(),
        }
    }

    /// Fit isotonic regression using Pool Adjacent Violators Algorithm (PAVA)
    fn fit(&mut self, raw_probs: &[f32], actuals: &[bool]) -> anyhow::Result<()> {
        if raw_probs.len() != actuals.len() {
            anyhow::bail!("Mismatched lengths");
        }
        if raw_probs.is_empty() {
            anyhow::bail!("Empty data");
        }

        // Sort by raw probabilities
        let mut pairs: Vec<(f32, f32)> = raw_probs
            .iter()
            .zip(actuals.iter())
            .map(|(&p, &a)| (p, if a { 1.0 } else { 0.0 }))
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Pool Adjacent Violators Algorithm
        let mut y: Vec<f32> = pairs.iter().map(|(_, y)| *y).collect();
        let mut weights: Vec<f32> = vec![1.0; pairs.len()];

        // Forward pass - enforce monotonicity
        let mut i = 0;
        while i < y.len().saturating_sub(1) {
            if y[i] > y[i + 1] {
                // Pool adjacent violators
                let combined_y =
                    (y[i] * weights[i] + y[i + 1] * weights[i + 1]) / (weights[i] + weights[i + 1]);
                let combined_w = weights[i] + weights[i + 1];

                y[i] = combined_y;
                weights[i] = combined_w;

                // Remove pooled element
                y.remove(i + 1);
                weights.remove(i + 1);

                // Go back to check previous
                i = i.saturating_sub(1);
            } else {
                i += 1;
            }
        }

        // Extract unique x values and corresponding y values
        self.x_values = pairs.iter().map(|(x, _)| *x).collect();
        self.y_values = y;

        // If PAVA reduced the size, we need to expand back
        if self.y_values.len() < self.x_values.len() {
            // Create step function from PAVA result
            let pava_x: Vec<f32> = pairs
                .iter()
                .step_by(pairs.len() / self.y_values.len().max(1))
                .map(|(x, _)| *x)
                .collect();

            let mut expanded_y = Vec::with_capacity(self.x_values.len());
            let mut pava_idx = 0;

            for &x in &self.x_values {
                while pava_idx < pava_x.len() - 1 && x > pava_x[pava_idx + 1] {
                    pava_idx += 1;
                }
                expanded_y.push(self.y_values[pava_idx.min(self.y_values.len() - 1)]);
            }

            self.y_values = expanded_y;
        }

        Ok(())
    }

    /// Transform raw probability to calibrated probability
    fn transform(&self, raw_prob: f32) -> f32 {
        if self.x_values.is_empty() {
            return raw_prob;
        }

        // Binary search for closest x value
        let idx = self
            .x_values
            .binary_search_by(|x| {
                x.partial_cmp(&raw_prob)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or_else(|i| i.min(self.x_values.len() - 1));

        // Linear interpolation
        if idx == 0 {
            self.y_values[0]
        } else if idx >= self.x_values.len() {
            *self.y_values.last().unwrap_or(&raw_prob)
        } else {
            let x0 = self.x_values[idx - 1];
            let x1 = self.x_values[idx];
            let y0 = self.y_values[idx - 1];
            let y1 = self.y_values[idx];

            if (x1 - x0).abs() < 1e-10 {
                y0
            } else {
                let t = (raw_prob - x0) / (x1 - x0);
                y0 + t * (y1 - y0)
            }
        }
    }
}

/// Calibrated Defect Predictor using Bayesian inference + Isotonic calibration
///
/// Phase 7: Provides calibrated probabilities with confidence intervals
pub struct CalibratedDefectPredictor {
    /// Ensemble model for base predictions
    ensemble: WeightedEnsembleModel,
    /// Isotonic calibrator
    calibrator: IsotonicCalibrator,
    /// Feature names for explainability
    feature_names: Vec<String>,
    /// Prior variance for Bayesian inference
    prior_variance: f32,
    /// Is calibrator fitted
    calibrator_fitted: bool,
}

impl Default for CalibratedDefectPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl CalibratedDefectPredictor {
    /// Create new calibrated predictor
    pub fn new() -> Self {
        Self {
            ensemble: WeightedEnsembleModel::new(),
            calibrator: IsotonicCalibrator::new(),
            feature_names: vec![
                "SBFL".into(),
                "TDG".into(),
                "Churn".into(),
                "Complexity".into(),
                "RAG_Similarity".into(),
            ],
            prior_variance: 1.0,
            calibrator_fitted: false,
        }
    }

    /// Set prior variance for uncertainty estimation
    pub fn with_prior_variance(mut self, variance: f32) -> Self {
        self.prior_variance = variance;
        self
    }

    /// Fit the predictor on labeled data
    ///
    /// Splits data into training (for ensemble) and calibration sets.
    pub fn fit(&mut self, files: &[FileFeatures], labels: &[bool]) -> anyhow::Result<()> {
        if files.len() != labels.len() {
            anyhow::bail!(
                "Mismatched lengths: {} files, {} labels",
                files.len(),
                labels.len()
            );
        }
        if files.len() < 10 {
            anyhow::bail!("Need at least 10 samples for calibration");
        }

        // Split: 80% training, 20% calibration
        let split_idx = (files.len() as f32 * 0.8) as usize;
        let train_files = &files[..split_idx];
        let cal_files = &files[split_idx..];
        let cal_labels = &labels[split_idx..];

        // Fit ensemble on training data (unsupervised weak supervision)
        self.ensemble.fit(train_files)?;

        // Get raw predictions on calibration set
        let raw_probs: Vec<f32> = cal_files.iter().map(|f| self.ensemble.predict(f)).collect();

        // Fit isotonic calibrator
        self.calibrator.fit(&raw_probs, cal_labels)?;
        self.calibrator_fitted = true;

        Ok(())
    }

    /// Predict with uncertainty quantification
    pub fn predict(&self, features: &FileFeatures) -> CalibratedPrediction {
        // Get raw ensemble prediction
        let raw_prob = self.ensemble.predict(features);

        // Calibrate
        let calibrated_prob = if self.calibrator_fitted {
            self.calibrator.transform(raw_prob)
        } else {
            raw_prob
        };

        // Estimate uncertainty using Bayesian approximation
        // Variance increases for predictions near 0.5 and with less training data
        let base_variance = self.prior_variance * calibrated_prob * (1.0 - calibrated_prob);
        let std_dev = base_variance.sqrt();

        // 95% confidence interval
        let z_95 = 1.96f32;
        let ci_low = (calibrated_prob - z_95 * std_dev).max(0.0);
        let ci_high = (calibrated_prob + z_95 * std_dev).min(1.0);

        // Confidence level based on CI width
        let ci_width = ci_high - ci_low;
        let confidence_level = if ci_width < 0.15 {
            ConfidenceLevel::High
        } else if ci_width < 0.30 {
            ConfidenceLevel::Medium
        } else {
            ConfidenceLevel::Low
        };

        // Compute factor contributions
        let contributing_factors = self.compute_contributions(features);

        CalibratedPrediction {
            file: features.path.clone(),
            line: None,
            probability: calibrated_prob,
            confidence_interval: (ci_low, ci_high),
            confidence_level,
            contributing_factors,
        }
    }

    /// Compute factor contributions for explainability
    fn compute_contributions(&self, features: &FileFeatures) -> Vec<FactorContribution> {
        let weights = match self.ensemble.get_weights() {
            Some(w) => w.weights.clone(),
            None => vec![0.2; 5], // Equal weights if not fitted
        };

        let feature_values = features.to_vector();

        // Weighted contribution of each feature
        let weighted: Vec<f32> = feature_values
            .iter()
            .zip(weights.iter())
            .map(|(f, w)| (f * w).abs())
            .collect();

        let total: f32 = weighted.iter().sum();

        self.feature_names
            .iter()
            .zip(feature_values.iter())
            .zip(weighted.iter())
            .map(|((name, &raw_value), &w)| FactorContribution {
                factor_name: name.clone(),
                contribution_pct: if total > 0.0 { w / total * 100.0 } else { 20.0 },
                raw_value,
            })
            .collect()
    }

    /// Evaluate calibration quality on test set
    pub fn evaluate(
        &self,
        test_files: &[FileFeatures],
        test_labels: &[bool],
    ) -> CalibrationMetrics {
        if test_files.len() != test_labels.len() || test_files.is_empty() {
            return CalibrationMetrics {
                ece: 1.0,
                mce: 1.0,
                brier_score: 1.0,
                coverage: 0.0,
            };
        }

        let predictions: Vec<CalibratedPrediction> =
            test_files.iter().map(|f| self.predict(f)).collect();

        // Brier Score
        let brier_score: f32 = predictions
            .iter()
            .zip(test_labels.iter())
            .map(|(pred, &actual)| {
                let target = if actual { 1.0 } else { 0.0 };
                (pred.probability - target).powi(2)
            })
            .sum::<f32>()
            / predictions.len() as f32;

        // Expected Calibration Error (binned)
        let n_bins = 10;
        let mut bins: Vec<(f32, f32, usize)> = vec![(0.0, 0.0, 0); n_bins];

        for (pred, &actual) in predictions.iter().zip(test_labels.iter()) {
            let bin_idx = ((pred.probability * n_bins as f32) as usize).min(n_bins - 1);
            bins[bin_idx].0 += pred.probability; // sum of predictions
            bins[bin_idx].1 += if actual { 1.0 } else { 0.0 }; // sum of actuals
            bins[bin_idx].2 += 1; // count
        }

        let mut ece = 0.0f32;
        let mut mce = 0.0f32;

        for (sum_pred, sum_actual, count) in &bins {
            if *count > 0 {
                let avg_pred = sum_pred / *count as f32;
                let avg_actual = sum_actual / *count as f32;
                let bin_error = (avg_pred - avg_actual).abs();
                let weight = *count as f32 / predictions.len() as f32;
                ece += weight * bin_error;
                mce = mce.max(bin_error);
            }
        }

        // Coverage: % of true labels within confidence interval
        let covered = predictions
            .iter()
            .zip(test_labels.iter())
            .filter(|(pred, &actual)| {
                let target = if actual { 1.0 } else { 0.0 };
                target >= pred.confidence_interval.0 && target <= pred.confidence_interval.1
            })
            .count();
        let coverage = covered as f32 / predictions.len() as f32;

        CalibrationMetrics {
            ece,
            mce,
            brier_score,
            coverage,
        }
    }

    /// Check if model is fitted
    pub fn is_fitted(&self) -> bool {
        self.ensemble.is_fitted() && self.calibrator_fitted
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // -------------------------------------------------------------------------
    // FileFeatures Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_file_features_new() {
        let features = FileFeatures::new(PathBuf::from("src/main.rs"));
        assert_eq!(features.path, PathBuf::from("src/main.rs"));
        assert_eq!(features.sbfl_score, 0.0);
        assert_eq!(features.tdg_score, 0.0);
        assert_eq!(features.churn_score, 0.0);
        assert_eq!(features.complexity_score, 0.0);
        assert_eq!(features.rag_similarity, 0.0);
    }

    #[test]
    fn test_file_features_builder() {
        let features = FileFeatures::new(PathBuf::from("src/lib.rs"))
            .with_sbfl(0.85)
            .with_tdg(0.4)
            .with_churn(0.95)
            .with_complexity(0.6)
            .with_rag_similarity(0.75);

        assert_eq!(features.sbfl_score, 0.85);
        assert_eq!(features.tdg_score, 0.4);
        assert_eq!(features.churn_score, 0.95);
        assert_eq!(features.complexity_score, 0.6);
        assert_eq!(features.rag_similarity, 0.75);
    }

    #[test]
    fn test_file_features_clamping() {
        let features = FileFeatures::new(PathBuf::from("test.rs"))
            .with_sbfl(1.5) // Should clamp to 1.0
            .with_tdg(-0.5); // Should clamp to 0.0

        assert_eq!(features.sbfl_score, 1.0);
        assert_eq!(features.tdg_score, 0.0);
    }

    #[test]
    fn test_file_features_to_vector() {
        let features = FileFeatures::new(PathBuf::from("test.rs"))
            .with_sbfl(0.9)
            .with_tdg(0.3)
            .with_churn(0.8)
            .with_complexity(0.5)
            .with_rag_similarity(0.7);

        let vec = features.to_vector();
        assert_eq!(vec, vec![0.9, 0.3, 0.8, 0.5, 0.7]);
    }

    // -------------------------------------------------------------------------
    // Labeling Function Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_sbfl_labeling_function_positive() {
        let lf = SbflLabelingFunction::new(0.7, 0.2);
        let features = FileFeatures::new(PathBuf::from("test.rs")).with_sbfl(0.9);
        assert_eq!(lf.apply(&features), LabelOutput::Positive);
    }

    #[test]
    fn test_sbfl_labeling_function_negative() {
        let lf = SbflLabelingFunction::new(0.7, 0.2);
        let features = FileFeatures::new(PathBuf::from("test.rs")).with_sbfl(0.1);
        assert_eq!(lf.apply(&features), LabelOutput::Negative);
    }

    #[test]
    fn test_sbfl_labeling_function_abstain() {
        let lf = SbflLabelingFunction::new(0.7, 0.2);
        let features = FileFeatures::new(PathBuf::from("test.rs")).with_sbfl(0.5);
        assert_eq!(lf.apply(&features), LabelOutput::Abstain);
    }

    #[test]
    fn test_tdg_labeling_function() {
        let lf = TdgLabelingFunction::new(0.5, 0.2);

        // High debt (bad TDG) = positive (likely defect)
        let high_debt = FileFeatures::new(PathBuf::from("test.rs")).with_tdg(0.7);
        assert_eq!(lf.apply(&high_debt), LabelOutput::Positive);

        // Low debt (good TDG) = negative (likely clean)
        let low_debt = FileFeatures::new(PathBuf::from("test.rs")).with_tdg(0.1);
        assert_eq!(lf.apply(&low_debt), LabelOutput::Negative);

        // Medium debt = abstain
        let medium_debt = FileFeatures::new(PathBuf::from("test.rs")).with_tdg(0.35);
        assert_eq!(lf.apply(&medium_debt), LabelOutput::Abstain);
    }

    #[test]
    fn test_churn_labeling_function() {
        let lf = ChurnLabelingFunction::new(0.9, 0.3);

        let high_churn = FileFeatures::new(PathBuf::from("test.rs")).with_churn(0.95);
        assert_eq!(lf.apply(&high_churn), LabelOutput::Positive);

        let low_churn = FileFeatures::new(PathBuf::from("test.rs")).with_churn(0.1);
        assert_eq!(lf.apply(&low_churn), LabelOutput::Negative);
    }

    #[test]
    fn test_complexity_labeling_function() {
        let lf = ComplexityLabelingFunction::new(0.7, 0.3);

        let high_complexity = FileFeatures::new(PathBuf::from("test.rs")).with_complexity(0.9);
        assert_eq!(lf.apply(&high_complexity), LabelOutput::Positive);

        let low_complexity = FileFeatures::new(PathBuf::from("test.rs")).with_complexity(0.1);
        assert_eq!(lf.apply(&low_complexity), LabelOutput::Negative);
    }

    #[test]
    fn test_rag_similarity_labeling_function() {
        let lf = RagSimilarityLabelingFunction::new(0.8);

        // Only provides positive or abstain (no negative signal)
        let similar = FileFeatures::new(PathBuf::from("test.rs")).with_rag_similarity(0.9);
        assert_eq!(lf.apply(&similar), LabelOutput::Positive);

        let not_similar = FileFeatures::new(PathBuf::from("test.rs")).with_rag_similarity(0.5);
        assert_eq!(lf.apply(&not_similar), LabelOutput::Abstain);
    }

    #[test]
    fn test_labeling_function_names() {
        assert_eq!(SbflLabelingFunction::new(0.7, 0.2).name(), "SBFL");
        assert_eq!(TdgLabelingFunction::new(0.5, 0.2).name(), "TDG");
        assert_eq!(ChurnLabelingFunction::new(0.9, 0.3).name(), "Churn");
        assert_eq!(
            ComplexityLabelingFunction::new(0.7, 0.3).name(),
            "Complexity"
        );
        assert_eq!(
            RagSimilarityLabelingFunction::new(0.8).name(),
            "RAG_Similarity"
        );
    }

    // -------------------------------------------------------------------------
    // WeightedEnsembleModel Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_ensemble_model_new() {
        let model = WeightedEnsembleModel::new();
        assert!(!model.is_fitted());
        assert!(model.get_weights().is_none());
    }

    #[test]
    fn test_ensemble_model_predict_unfitted() {
        let model = WeightedEnsembleModel::new();
        let features = FileFeatures::new(PathBuf::from("test.rs")).with_sbfl(0.9);
        // Unfitted model returns 0.5 (neutral)
        assert_eq!(model.predict(&features), 0.5);
    }

    #[test]
    fn test_ensemble_model_fit_empty_data() {
        let mut model = WeightedEnsembleModel::new();
        let result = model.fit(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_ensemble_model_fit_and_predict() {
        let mut model = WeightedEnsembleModel::new();

        // Create synthetic training data
        let files: Vec<FileFeatures> = (0..100)
            .map(|i| {
                let is_defect = i % 3 == 0;
                FileFeatures::new(PathBuf::from(format!("file_{}.rs", i)))
                    .with_sbfl(if is_defect { 0.8 } else { 0.2 })
                    .with_tdg(if is_defect { 0.7 } else { 0.2 })
                    .with_churn(if is_defect { 0.95 } else { 0.3 })
                    .with_complexity(if is_defect { 0.8 } else { 0.3 })
                    .with_rag_similarity(if is_defect { 0.85 } else { 0.1 })
            })
            .collect();

        let result = model.fit(&files);
        assert!(result.is_ok());
        assert!(model.is_fitted());

        // Test prediction on high-risk file
        let high_risk = FileFeatures::new(PathBuf::from("risky.rs"))
            .with_sbfl(0.9)
            .with_tdg(0.8)
            .with_churn(0.95)
            .with_complexity(0.9)
            .with_rag_similarity(0.9);
        let prob = model.predict(&high_risk);
        assert!(
            prob > 0.5,
            "High risk file should have prob > 0.5, got {}",
            prob
        );

        // Test prediction on low-risk file
        let low_risk = FileFeatures::new(PathBuf::from("safe.rs"))
            .with_sbfl(0.1)
            .with_tdg(0.1)
            .with_churn(0.1)
            .with_complexity(0.1)
            .with_rag_similarity(0.1);
        let prob = model.predict(&low_risk);
        assert!(
            prob < 0.5,
            "Low risk file should have prob < 0.5, got {}",
            prob
        );
    }

    #[test]
    fn test_ensemble_model_weights_interpretability() {
        let mut model = WeightedEnsembleModel::new();

        let files: Vec<FileFeatures> = (0..50)
            .map(|i| {
                FileFeatures::new(PathBuf::from(format!("file_{}.rs", i)))
                    .with_sbfl(0.5 + (i as f32 % 10.0) / 20.0)
                    .with_tdg(0.3 + (i as f32 % 5.0) / 10.0)
                    .with_churn(0.4 + (i as f32 % 7.0) / 15.0)
                    .with_complexity(0.35 + (i as f32 % 8.0) / 20.0)
                    .with_rag_similarity(0.2 + (i as f32 % 6.0) / 12.0)
            })
            .collect();

        model.fit(&files).unwrap();

        let weights = model.get_weights().unwrap();
        assert_eq!(weights.names.len(), 5);
        assert_eq!(weights.weights.len(), 5);

        // Weights should sum to approximately 1
        let sum: f32 = weights.weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Weights should sum to 1, got {}",
            sum
        );

        // Test hashmap conversion
        let weight_map = weights.to_hashmap();
        assert!(weight_map.contains_key("SBFL"));
        assert!(weight_map.contains_key("TDG"));
    }

    // -------------------------------------------------------------------------
    // CalibratedDefectPredictor Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_calibrated_predictor_new() {
        let predictor = CalibratedDefectPredictor::new();
        assert!(!predictor.is_fitted());
    }

    #[test]
    fn test_calibrated_predictor_fit_insufficient_data() {
        let mut predictor = CalibratedDefectPredictor::new();
        let files: Vec<FileFeatures> = (0..5)
            .map(|i| FileFeatures::new(PathBuf::from(format!("file_{}.rs", i))))
            .collect();
        let labels = vec![true, false, true, false, true];

        let result = predictor.fit(&files, &labels);
        assert!(result.is_err()); // Need at least 10 samples
    }

    #[test]
    fn test_calibrated_predictor_fit_and_predict() {
        let mut predictor = CalibratedDefectPredictor::new();

        // Create synthetic labeled data
        let files: Vec<FileFeatures> = (0..100)
            .map(|i| {
                let is_defect = i % 3 == 0;
                FileFeatures::new(PathBuf::from(format!("file_{}.rs", i)))
                    .with_sbfl(if is_defect {
                        0.8 + (i as f32 % 10.0) / 50.0
                    } else {
                        0.2 + (i as f32 % 10.0) / 50.0
                    })
                    .with_tdg(if is_defect { 0.7 } else { 0.2 })
                    .with_churn(if is_defect { 0.9 } else { 0.3 })
                    .with_complexity(if is_defect { 0.8 } else { 0.3 })
                    .with_rag_similarity(if is_defect { 0.85 } else { 0.1 })
            })
            .collect();

        let labels: Vec<bool> = (0..100).map(|i| i % 3 == 0).collect();

        let result = predictor.fit(&files, &labels);
        assert!(result.is_ok());
        assert!(predictor.is_fitted());

        // Predict on new file
        let test_features = FileFeatures::new(PathBuf::from("test.rs"))
            .with_sbfl(0.85)
            .with_tdg(0.6)
            .with_churn(0.9)
            .with_complexity(0.75)
            .with_rag_similarity(0.8);

        let prediction = predictor.predict(&test_features);
        assert!(prediction.probability >= 0.0 && prediction.probability <= 1.0);
        assert!(prediction.confidence_interval.0 <= prediction.probability);
        assert!(prediction.confidence_interval.1 >= prediction.probability);
        assert!(!prediction.contributing_factors.is_empty());
    }

    #[test]
    fn test_calibrated_prediction_confidence_levels() {
        // Test confidence level classification
        let mut predictor = CalibratedDefectPredictor::new().with_prior_variance(0.1);

        let files: Vec<FileFeatures> = (0..50)
            .map(|i| {
                FileFeatures::new(PathBuf::from(format!("file_{}.rs", i)))
                    .with_sbfl(0.9)
                    .with_tdg(0.7)
                    .with_churn(0.95)
                    .with_complexity(0.8)
                    .with_rag_similarity(0.85)
            })
            .collect();
        let labels: Vec<bool> = vec![true; 50];

        let _ = predictor.fit(&files, &labels);

        // High confidence prediction (low variance)
        let high_conf_features = FileFeatures::new(PathBuf::from("high.rs"))
            .with_sbfl(0.95)
            .with_tdg(0.9)
            .with_churn(0.98)
            .with_complexity(0.9)
            .with_rag_similarity(0.95);

        let pred = predictor.predict(&high_conf_features);
        // Prediction near 1.0 should have narrower CI (higher confidence)
        let ci_width = pred.confidence_interval.1 - pred.confidence_interval.0;
        assert!(ci_width < 0.5, "CI width {} should be reasonable", ci_width);
    }

    #[test]
    fn test_calibration_metrics_evaluation() {
        let mut predictor = CalibratedDefectPredictor::new();

        // Create training data
        let train_files: Vec<FileFeatures> = (0..80)
            .map(|i| {
                let is_defect = i % 4 == 0;
                FileFeatures::new(PathBuf::from(format!("train_{}.rs", i)))
                    .with_sbfl(if is_defect { 0.85 } else { 0.15 })
                    .with_tdg(if is_defect { 0.75 } else { 0.25 })
                    .with_churn(if is_defect { 0.9 } else { 0.2 })
                    .with_complexity(if is_defect { 0.8 } else { 0.2 })
                    .with_rag_similarity(if is_defect { 0.8 } else { 0.1 })
            })
            .collect();
        let train_labels: Vec<bool> = (0..80).map(|i| i % 4 == 0).collect();

        predictor.fit(&train_files, &train_labels).unwrap();

        // Create test data
        let test_files: Vec<FileFeatures> = (0..20)
            .map(|i| {
                let is_defect = i % 4 == 0;
                FileFeatures::new(PathBuf::from(format!("test_{}.rs", i)))
                    .with_sbfl(if is_defect { 0.85 } else { 0.15 })
                    .with_tdg(if is_defect { 0.75 } else { 0.25 })
                    .with_churn(if is_defect { 0.9 } else { 0.2 })
                    .with_complexity(if is_defect { 0.8 } else { 0.2 })
                    .with_rag_similarity(if is_defect { 0.8 } else { 0.1 })
            })
            .collect();
        let test_labels: Vec<bool> = (0..20).map(|i| i % 4 == 0).collect();

        let metrics = predictor.evaluate(&test_files, &test_labels);

        // Metrics should be in valid ranges
        assert!(metrics.ece >= 0.0 && metrics.ece <= 1.0);
        assert!(metrics.mce >= 0.0 && metrics.mce <= 1.0);
        assert!(metrics.brier_score >= 0.0 && metrics.brier_score <= 1.0);
        assert!(metrics.coverage >= 0.0 && metrics.coverage <= 1.0);
    }

    #[test]
    fn test_factor_contributions() {
        let mut predictor = CalibratedDefectPredictor::new();

        let files: Vec<FileFeatures> = (0..50)
            .map(|i| {
                FileFeatures::new(PathBuf::from(format!("file_{}.rs", i)))
                    .with_sbfl(0.5 + (i as f32) / 100.0)
                    .with_tdg(0.4)
                    .with_churn(0.6)
                    .with_complexity(0.5)
                    .with_rag_similarity(0.3)
            })
            .collect();
        let labels: Vec<bool> = (0..50).map(|i| i > 25).collect();

        predictor.fit(&files, &labels).unwrap();

        let features = FileFeatures::new(PathBuf::from("test.rs"))
            .with_sbfl(0.9)
            .with_tdg(0.1)
            .with_churn(0.5)
            .with_complexity(0.3)
            .with_rag_similarity(0.2);

        let prediction = predictor.predict(&features);

        // Should have 5 factor contributions
        assert_eq!(prediction.contributing_factors.len(), 5);

        // Contributions should sum to approximately 100%
        let total: f32 = prediction
            .contributing_factors
            .iter()
            .map(|f| f.contribution_pct)
            .sum();
        assert!(
            (total - 100.0).abs() < 1.0,
            "Contributions should sum to 100%, got {}",
            total
        );

        // Each factor should have a name and valid percentage
        for factor in &prediction.contributing_factors {
            assert!(!factor.factor_name.is_empty());
            assert!(factor.contribution_pct >= 0.0);
            assert!(factor.raw_value >= 0.0 && factor.raw_value <= 1.0);
        }
    }

    // -------------------------------------------------------------------------
    // Isotonic Calibrator Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_isotonic_calibrator_basic() {
        let mut calibrator = IsotonicCalibrator::new();

        // Perfect calibration data
        let raw_probs = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let actuals = vec![false, false, false, false, true, true, true, true, true];

        calibrator.fit(&raw_probs, &actuals).unwrap();

        // Transform should produce monotonic output
        let t1 = calibrator.transform(0.2);
        let t2 = calibrator.transform(0.5);
        let t3 = calibrator.transform(0.8);

        assert!(t1 <= t2, "Isotonic: {} should be <= {}", t1, t2);
        assert!(t2 <= t3, "Isotonic: {} should be <= {}", t2, t3);
    }

    #[test]
    fn test_isotonic_calibrator_empty() {
        let mut calibrator = IsotonicCalibrator::new();
        let result = calibrator.fit(&[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_isotonic_calibrator_mismatched_lengths() {
        let mut calibrator = IsotonicCalibrator::new();
        let result = calibrator.fit(&[0.5, 0.6], &[true]);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------------
    // LabelModelWeights Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_label_model_weights_get_weight() {
        let weights = LabelModelWeights {
            weights: vec![0.3, 0.2, 0.25, 0.15, 0.1],
            names: vec![
                "SBFL".into(),
                "TDG".into(),
                "Churn".into(),
                "Complexity".into(),
                "RAG_Similarity".into(),
            ],
            n_iterations: 100,
            log_likelihood: -50.0,
        };

        assert_eq!(weights.get_weight("SBFL"), Some(0.3));
        assert_eq!(weights.get_weight("TDG"), Some(0.2));
        assert_eq!(weights.get_weight("Unknown"), None);
    }

    #[test]
    fn test_confidence_level_display() {
        assert_eq!(format!("{}", ConfidenceLevel::High), "HIGH");
        assert_eq!(format!("{}", ConfidenceLevel::Medium), "MEDIUM");
        assert_eq!(format!("{}", ConfidenceLevel::Low), "LOW");
    }

    // -------------------------------------------------------------------------
    // Integration Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_end_to_end_defect_prediction() {
        // Test ensemble model directly (without calibration) for clearer signal
        let mut ensemble = WeightedEnsembleModel::new();

        // Training data with clear patterns
        let mut files = Vec::new();

        // Pattern 1: High SBFL + High Churn = Defect indicators
        for i in 0..40 {
            files.push(
                FileFeatures::new(PathBuf::from(format!("high_risk_{}.rs", i)))
                    .with_sbfl(0.85 + (i as f32 % 5.0) / 100.0)
                    .with_tdg(0.7)
                    .with_churn(0.95)
                    .with_complexity(0.8)
                    .with_rag_similarity(0.85),
            );
        }

        // Pattern 2: Low all signals = Clean indicators
        for i in 0..60 {
            files.push(
                FileFeatures::new(PathBuf::from(format!("low_risk_{}.rs", i)))
                    .with_sbfl(0.1 + (i as f32 % 5.0) / 100.0)
                    .with_tdg(0.1)
                    .with_churn(0.15)
                    .with_complexity(0.2)
                    .with_rag_similarity(0.05),
            );
        }

        ensemble.fit(&files).unwrap();

        // Test predictions on clearly different risk profiles
        let high_risk = FileFeatures::new(PathBuf::from("new_risky.rs"))
            .with_sbfl(0.95)
            .with_tdg(0.8)
            .with_churn(0.98)
            .with_complexity(0.9)
            .with_rag_similarity(0.9);

        let low_risk = FileFeatures::new(PathBuf::from("new_safe.rs"))
            .with_sbfl(0.05)
            .with_tdg(0.05)
            .with_churn(0.05)
            .with_complexity(0.1)
            .with_rag_similarity(0.0);

        let high_pred = ensemble.predict(&high_risk);
        let low_pred = ensemble.predict(&low_risk);

        // High risk should have higher probability than low risk
        assert!(
            high_pred >= low_pred,
            "High risk ({}) should have >= prob than low risk ({})",
            high_pred,
            low_pred
        );

        // Both should be in valid range
        assert!((0.0..=1.0).contains(&high_pred));
        assert!((0.0..=1.0).contains(&low_pred));
    }

    #[test]
    fn test_serialization_roundtrip() {
        let weights = LabelModelWeights {
            weights: vec![0.25, 0.20, 0.20, 0.20, 0.15],
            names: vec![
                "SBFL".into(),
                "TDG".into(),
                "Churn".into(),
                "Complexity".into(),
                "RAG_Similarity".into(),
            ],
            n_iterations: 50,
            log_likelihood: -45.5,
        };

        let json = serde_json::to_string(&weights).unwrap();
        let parsed: LabelModelWeights = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.weights, weights.weights);
        assert_eq!(parsed.names, weights.names);
        assert_eq!(parsed.n_iterations, weights.n_iterations);
    }
}
