//! ML model training module for defect classification.
//!
//! This module implements Phase 2 ML classifier training:
//! - Load training data from JSON
//! - Extract TF-IDF features from commit messages
//! - Train RandomForestClassifier
//! - Evaluate on validation and test sets
//! - Save trained model to disk
//!
//! Implements Section 3 ML Classification from nlp-models-techniques-spec.md

use crate::classifier::DefectCategory;
use crate::nlp::TfidfFeatureExtractor;
use crate::training::{TrainingDataset, TrainingExample};
use anyhow::{anyhow, Result};
use aprender::primitives::Matrix;
use aprender::tree::RandomForestClassifier;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Trained ML classifier model with metadata
#[derive(Serialize, Deserialize)]
pub struct TrainedModel {
    /// Random Forest classifier
    #[serde(skip)]
    pub classifier: Option<RandomForestClassifier>,
    /// TF-IDF feature extractor
    #[serde(skip)]
    pub tfidf_extractor: Option<TfidfFeatureExtractor>,
    /// Mapping from category to label index
    pub category_to_label: HashMap<String, usize>,
    /// Mapping from label index to category
    pub label_to_category: HashMap<usize, String>,
    /// Training metadata
    pub metadata: TrainingMetadata,
    /// TF-IDF vocabulary (for reconstruction)
    pub tfidf_vocabulary: Vec<String>,
    /// Max features for TF-IDF
    pub max_features: usize,
}

/// Metadata about model training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetadata {
    /// Number of training examples
    pub n_train: usize,
    /// Number of validation examples
    pub n_validation: usize,
    /// Number of test examples
    pub n_test: usize,
    /// Number of trees in Random Forest
    pub n_estimators: usize,
    /// Maximum tree depth
    pub max_depth: Option<usize>,
    /// Number of TF-IDF features
    pub n_features: usize,
    /// Number of classes
    pub n_classes: usize,
    /// Training accuracy
    pub train_accuracy: f32,
    /// Validation accuracy
    pub validation_accuracy: f32,
    /// Test accuracy (optional - set after evaluation)
    pub test_accuracy: Option<f32>,
}

/// ML model trainer
pub struct MLTrainer {
    n_estimators: usize,
    max_depth: Option<usize>,
    max_features: usize,
    random_state: u64,
}

impl MLTrainer {
    /// Create a new ML trainer
    ///
    /// # Arguments
    ///
    /// * `n_estimators` - Number of trees in Random Forest
    /// * `max_depth` - Maximum depth of each tree (None for unlimited)
    /// * `max_features` - Maximum TF-IDF features
    ///
    /// # Examples
    ///
    /// ```rust
    /// use organizational_intelligence_plugin::ml_trainer::MLTrainer;
    ///
    /// let trainer = MLTrainer::new(100, Some(20), 1500);
    /// ```
    pub fn new(n_estimators: usize, max_depth: Option<usize>, max_features: usize) -> Self {
        Self {
            n_estimators,
            max_depth,
            max_features,
            random_state: 42,
        }
    }

    /// Load training dataset from JSON file
    ///
    /// # Arguments
    ///
    /// * `path` - Path to training data JSON file
    ///
    /// # Returns
    ///
    /// * `Ok(TrainingDataset)` if successful
    /// * `Err` if file not found or invalid format
    pub fn load_dataset<P: AsRef<Path>>(path: P) -> Result<TrainingDataset> {
        let content = fs::read_to_string(path.as_ref())
            .map_err(|e| anyhow!("Failed to read training data: {}", e))?;

        serde_json::from_str(&content)
            .map_err(|e| anyhow!("Failed to parse training data JSON: {}", e))
    }

    /// Train ML classifier on training dataset
    ///
    /// # Arguments
    ///
    /// * `dataset` - Training dataset with splits
    ///
    /// # Returns
    ///
    /// * `Ok(TrainedModel)` with trained classifier
    /// * `Err` if training fails
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use organizational_intelligence_plugin::ml_trainer::MLTrainer;
    /// use std::path::PathBuf;
    ///
    /// # async fn example() -> Result<(), anyhow::Error> {
    /// let trainer = MLTrainer::new(100, Some(20), 1500);
    /// let dataset = MLTrainer::load_dataset("training-data.json")?;
    /// let model = trainer.train(&dataset)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn train(&self, dataset: &TrainingDataset) -> Result<TrainedModel> {
        if dataset.train.is_empty() {
            return Err(anyhow!("Training dataset is empty"));
        }

        // Extract messages and labels
        let train_messages: Vec<String> =
            dataset.train.iter().map(|ex| ex.message.clone()).collect();

        let validation_messages: Vec<String> = dataset
            .validation
            .iter()
            .map(|ex| ex.message.clone())
            .collect();

        // Build category-to-label mapping
        let mut unique_categories: Vec<String> = dataset
            .train
            .iter()
            .map(|ex| format!("{}", ex.label))
            .collect();
        unique_categories.sort();
        unique_categories.dedup();

        let category_to_label: HashMap<String, usize> = unique_categories
            .iter()
            .enumerate()
            .map(|(i, cat)| (cat.clone(), i))
            .collect();

        let label_to_category: HashMap<usize, String> = unique_categories
            .iter()
            .enumerate()
            .map(|(i, cat)| (i, cat.clone()))
            .collect();

        // Convert labels to indices
        let train_labels: Vec<usize> = dataset
            .train
            .iter()
            .map(|ex| {
                *category_to_label
                    .get(&format!("{}", ex.label))
                    .unwrap_or(&0)
            })
            .collect();

        let validation_labels: Vec<usize> = dataset
            .validation
            .iter()
            .map(|ex| {
                *category_to_label
                    .get(&format!("{}", ex.label))
                    .unwrap_or(&0)
            })
            .collect();

        // Extract TF-IDF features
        let mut tfidf_extractor = TfidfFeatureExtractor::new(self.max_features);
        let train_features = tfidf_extractor.fit_transform(&train_messages)?;
        let validation_features = tfidf_extractor.transform(&validation_messages)?;

        // Convert Matrix<f64> to Matrix<f32> for RandomForestClassifier
        let train_features_f32 = Self::convert_f64_to_f32(&train_features)?;
        let validation_features_f32 = Self::convert_f64_to_f32(&validation_features)?;

        // Train Random Forest
        let mut classifier = RandomForestClassifier::new(self.n_estimators);
        if let Some(depth) = self.max_depth {
            classifier = classifier.with_max_depth(depth);
        }
        classifier = classifier.with_random_state(self.random_state);

        classifier
            .fit(&train_features_f32, &train_labels)
            .map_err(|e| anyhow!("Random Forest training failed: {}", e))?;

        // Evaluate on training set
        let train_predictions = classifier.predict(&train_features_f32);
        let train_accuracy = Self::calculate_accuracy(&train_predictions, &train_labels);

        // Evaluate on validation set
        let validation_predictions = classifier.predict(&validation_features_f32);
        let validation_accuracy =
            Self::calculate_accuracy(&validation_predictions, &validation_labels);

        let metadata = TrainingMetadata {
            n_train: dataset.train.len(),
            n_validation: dataset.validation.len(),
            n_test: dataset.test.len(),
            n_estimators: self.n_estimators,
            max_depth: self.max_depth,
            n_features: tfidf_extractor.vocabulary_size(),
            n_classes: unique_categories.len(),
            train_accuracy,
            validation_accuracy,
            test_accuracy: None,
        };

        // Extract vocabulary for serialization (simplified - just store metadata)
        let tfidf_vocabulary: Vec<String> = vec![]; // TODO: Extract vocabulary from TfidfVectorizer
        let max_features = self.max_features;

        Ok(TrainedModel {
            classifier: Some(classifier),
            tfidf_extractor: Some(tfidf_extractor),
            category_to_label,
            label_to_category,
            metadata,
            tfidf_vocabulary,
            max_features,
        })
    }

    /// Convert Matrix<f64> to Matrix<f32>
    fn convert_f64_to_f32(matrix: &Matrix<f64>) -> Result<Matrix<f32>> {
        let (n_rows, n_cols) = (matrix.n_rows(), matrix.n_cols());
        let data_f32: Vec<f32> = (0..n_rows * n_cols)
            .map(|i| {
                let row = i / n_cols;
                let col = i % n_cols;
                matrix.get(row, col) as f32
            })
            .collect();

        Matrix::from_vec(n_rows, n_cols, data_f32)
            .map_err(|e| anyhow!("Failed to convert matrix: {}", e))
    }

    /// Calculate classification accuracy
    fn calculate_accuracy(predictions: &[usize], labels: &[usize]) -> f32 {
        if predictions.is_empty() || predictions.len() != labels.len() {
            return 0.0;
        }

        let correct = predictions
            .iter()
            .zip(labels.iter())
            .filter(|(pred, label)| pred == label)
            .count();

        correct as f32 / predictions.len() as f32
    }

    /// Evaluate model on test set
    ///
    /// # Arguments
    ///
    /// * `model` - Trained model
    /// * `test_examples` - Test examples
    ///
    /// # Returns
    ///
    /// * Test accuracy (0.0-1.0)
    pub fn evaluate(model: &TrainedModel, test_examples: &[TrainingExample]) -> Result<f32> {
        if test_examples.is_empty() {
            return Ok(0.0);
        }

        let classifier = model
            .classifier
            .as_ref()
            .ok_or_else(|| anyhow!("Model has no classifier"))?;

        let tfidf_extractor = model
            .tfidf_extractor
            .as_ref()
            .ok_or_else(|| anyhow!("Model has no TF-IDF extractor"))?;

        let test_messages: Vec<String> =
            test_examples.iter().map(|ex| ex.message.clone()).collect();

        let test_labels: Vec<usize> = test_examples
            .iter()
            .map(|ex| {
                *model
                    .category_to_label
                    .get(&format!("{}", ex.label))
                    .unwrap_or(&0)
            })
            .collect();

        // Extract features and convert to f32
        let test_features = tfidf_extractor.transform(&test_messages)?;
        let test_features_f32 = Self::convert_f64_to_f32(&test_features)?;

        // Predict and calculate accuracy
        let test_predictions = classifier.predict(&test_features_f32);
        let test_accuracy = Self::calculate_accuracy(&test_predictions, &test_labels);

        Ok(test_accuracy)
    }

    /// Save trained model to disk
    ///
    /// # Arguments
    ///
    /// * `model` - Trained model
    /// * `path` - Path to save model JSON
    ///
    /// # Returns
    ///
    /// * `Ok(())` if successful
    /// * `Err` if save fails
    pub fn save_model<P: AsRef<Path>>(model: &TrainedModel, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(model)
            .map_err(|e| anyhow!("Failed to serialize model: {}", e))?;

        fs::write(path.as_ref(), json).map_err(|e| anyhow!("Failed to write model file: {}", e))?;

        Ok(())
    }

    /// Load trained model from disk
    ///
    /// # Arguments
    ///
    /// * `path` - Path to model JSON file
    ///
    /// # Returns
    ///
    /// * `Ok(TrainedModel)` if successful
    /// * `Err` if load fails
    pub fn load_model<P: AsRef<Path>>(path: P) -> Result<TrainedModel> {
        let content = fs::read_to_string(path.as_ref())
            .map_err(|e| anyhow!("Failed to read model file: {}", e))?;

        serde_json::from_str(&content).map_err(|e| anyhow!("Failed to parse model JSON: {}", e))
    }
}

impl Default for MLTrainer {
    fn default() -> Self {
        Self::new(100, Some(20), 1500)
    }
}

impl TrainedModel {
    /// Predict defect category for a single commit message
    ///
    /// # Arguments
    /// * `message` - Commit message to classify
    ///
    /// # Returns
    /// * `Ok(Some((DefectCategory, f32)))` - Predicted category and confidence
    /// * `Ok(None)` - Model components not available (deserialized model)
    /// * `Err` - Prediction error
    ///
    /// # Examples
    /// ```no_run
    /// # use organizational_intelligence_plugin::ml_trainer::TrainedModel;
    /// # fn example(model: &TrainedModel) -> anyhow::Result<()> {
    /// if let Some((category, confidence)) = model.predict("fix: null pointer in parser")? {
    ///     println!("Predicted: {:?} ({:.2})", category, confidence);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn predict(&self, message: &str) -> Result<Option<(DefectCategory, f32)>> {
        // Check if model components are available
        let tfidf = self
            .tfidf_extractor
            .as_ref()
            .ok_or_else(|| anyhow!("TF-IDF extractor not available"))?;
        let classifier = self
            .classifier
            .as_ref()
            .ok_or_else(|| anyhow!("Classifier not available"))?;

        // Extract TF-IDF features
        let features = tfidf.transform(&[message.to_string()])?;

        // Convert to f32 for Random Forest
        let (n_rows, n_cols) = (features.n_rows(), features.n_cols());
        let data_f32: Vec<f32> = (0..n_rows * n_cols)
            .map(|i| {
                let row = i / n_cols;
                let col = i % n_cols;
                features.get(row, col) as f32
            })
            .collect();

        let features_f32 = Matrix::from_vec(n_rows, n_cols, data_f32)
            .map_err(|e| anyhow!("Failed to create feature matrix: {}", e))?;

        // Predict
        let predictions = classifier.predict(&features_f32);

        if predictions.is_empty() {
            return Ok(None);
        }

        // Get predicted label index
        let label_idx = predictions[0];

        // Map label index back to DefectCategory
        let category_name = self
            .label_to_category
            .get(&label_idx)
            .ok_or_else(|| anyhow!("Unknown label index: {}", label_idx))?;

        // Parse category name to DefectCategory enum
        let category = Self::parse_category(category_name)?;

        // TODO: Get confidence from Random Forest (requires probability output)
        // For now, use a placeholder confidence
        let confidence = 0.75f32;

        Ok(Some((category, confidence)))
    }

    /// Predict top-N defect categories for a commit message
    ///
    /// # Arguments
    /// * `message` - Commit message to classify
    /// * `top_n` - Number of top categories to return
    ///
    /// # Returns
    /// * `Ok(Vec<(DefectCategory, f32)>)` - Top-N categories with confidences
    ///
    /// # Examples
    /// ```no_run
    /// # use organizational_intelligence_plugin::ml_trainer::TrainedModel;
    /// # fn example(model: &TrainedModel) -> anyhow::Result<()> {
    /// let predictions = model.predict_top_n("fix: null pointer in parser", 3)?;
    /// for (category, confidence) in predictions {
    ///     println!("{:?}: {:.2}", category, confidence);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn predict_top_n(
        &self,
        message: &str,
        _top_n: usize,
    ) -> Result<Vec<(DefectCategory, f32)>> {
        // For now, just return the single prediction
        // TODO: Implement multi-label prediction with probability outputs
        if let Some((category, confidence)) = self.predict(message)? {
            Ok(vec![(category, confidence)])
        } else {
            Ok(vec![])
        }
    }

    /// Parse category name string to DefectCategory enum
    fn parse_category(name: &str) -> Result<DefectCategory> {
        match name {
            "MemorySafety" => Ok(DefectCategory::MemorySafety),
            "ConcurrencyBugs" => Ok(DefectCategory::ConcurrencyBugs),
            "LogicErrors" => Ok(DefectCategory::LogicErrors),
            "ApiMisuse" => Ok(DefectCategory::ApiMisuse),
            "ResourceLeaks" => Ok(DefectCategory::ResourceLeaks),
            "TypeErrors" => Ok(DefectCategory::TypeErrors),
            "ConfigurationErrors" => Ok(DefectCategory::ConfigurationErrors),
            "SecurityVulnerabilities" => Ok(DefectCategory::SecurityVulnerabilities),
            "PerformanceIssues" => Ok(DefectCategory::PerformanceIssues),
            "IntegrationFailures" => Ok(DefectCategory::IntegrationFailures),
            "OperatorPrecedence" => Ok(DefectCategory::OperatorPrecedence),
            "TypeAnnotationGaps" => Ok(DefectCategory::TypeAnnotationGaps),
            "StdlibMapping" => Ok(DefectCategory::StdlibMapping),
            "ASTTransform" => Ok(DefectCategory::ASTTransform),
            "ComprehensionBugs" => Ok(DefectCategory::ComprehensionBugs),
            "IteratorChain" => Ok(DefectCategory::IteratorChain),
            "OwnershipBorrow" => Ok(DefectCategory::OwnershipBorrow),
            "TraitBounds" => Ok(DefectCategory::TraitBounds),
            _ => Err(anyhow!("Unknown category: {}", name)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::git::CommitInfo;
    use crate::training::TrainingDataExtractor;

    fn create_test_commits() -> Vec<CommitInfo> {
        vec![
            CommitInfo {
                hash: "abc1".to_string(),
                message: "fix: null pointer dereference in parser".to_string(),
                author: "dev@example.com".to_string(),
                timestamp: 1234567890,
                files_changed: 2,
                lines_added: 10,
                lines_removed: 5,
            },
            CommitInfo {
                hash: "abc2".to_string(),
                message: "fix: race condition in mutex lock".to_string(),
                author: "dev@example.com".to_string(),
                timestamp: 1234567891,
                files_changed: 1,
                lines_added: 5,
                lines_removed: 3,
            },
            CommitInfo {
                hash: "abc3".to_string(),
                message: "fix: memory leak in allocator".to_string(),
                author: "dev@example.com".to_string(),
                timestamp: 1234567892,
                files_changed: 1,
                lines_added: 8,
                lines_removed: 2,
            },
            CommitInfo {
                hash: "abc4".to_string(),
                message: "fix: configuration error in yaml parser".to_string(),
                author: "dev@example.com".to_string(),
                timestamp: 1234567893,
                files_changed: 1,
                lines_added: 3,
                lines_removed: 1,
            },
            CommitInfo {
                hash: "abc5".to_string(),
                message: "fix: type error in generic bounds".to_string(),
                author: "dev@example.com".to_string(),
                timestamp: 1234567894,
                files_changed: 2,
                lines_added: 15,
                lines_removed: 8,
            },
        ]
    }

    #[test]
    fn test_ml_trainer_creation() {
        let trainer = MLTrainer::new(100, Some(20), 1500);
        assert_eq!(trainer.n_estimators, 100);
        assert_eq!(trainer.max_depth, Some(20));
        assert_eq!(trainer.max_features, 1500);
    }

    #[test]
    fn test_ml_trainer_default() {
        let trainer = MLTrainer::default();
        assert_eq!(trainer.n_estimators, 100);
        assert_eq!(trainer.max_depth, Some(20));
        assert_eq!(trainer.max_features, 1500);
    }

    #[test]
    fn test_calculate_accuracy() {
        let predictions = vec![0, 1, 2, 0, 1];
        let labels = vec![0, 1, 2, 1, 1];
        let accuracy = MLTrainer::calculate_accuracy(&predictions, &labels);
        assert_eq!(accuracy, 0.8); // 4 out of 5 correct
    }

    #[test]
    fn test_calculate_accuracy_perfect() {
        let predictions = vec![0, 1, 2];
        let labels = vec![0, 1, 2];
        let accuracy = MLTrainer::calculate_accuracy(&predictions, &labels);
        assert_eq!(accuracy, 1.0);
    }

    #[test]
    fn test_calculate_accuracy_empty() {
        let predictions: Vec<usize> = vec![];
        let labels: Vec<usize> = vec![];
        let accuracy = MLTrainer::calculate_accuracy(&predictions, &labels);
        assert_eq!(accuracy, 0.0);
    }

    #[test]
    fn test_train_with_small_dataset() {
        let trainer = MLTrainer::new(10, Some(5), 100);

        // Create small training dataset
        let extractor = TrainingDataExtractor::new(0.70);
        let commits = create_test_commits();
        let examples = extractor
            .extract_training_data(&commits, "test-repo")
            .unwrap();

        if examples.len() < 10 {
            // Not enough data for meaningful test (need at least 10 for proper splits)
            return;
        }

        let dataset = extractor
            .create_splits(&examples, &["test-repo".to_string()])
            .unwrap();

        // Ensure splits are non-empty
        if dataset.train.is_empty() || dataset.validation.is_empty() {
            return;
        }

        // Train model
        let result = trainer.train(&dataset);
        if let Err(e) = &result {
            eprintln!("Training error: {}", e);
        }
        assert!(result.is_ok());

        let model = result.unwrap();
        assert!(model.classifier.is_some());
        assert!(model.metadata.train_accuracy > 0.0);
        assert!(model.metadata.n_classes > 0);
    }

    #[test]
    fn test_train_empty_dataset_error() {
        let trainer = MLTrainer::new(10, Some(5), 100);

        // Create empty dataset
        let dataset = TrainingDataset {
            train: vec![],
            validation: vec![],
            test: vec![],
            metadata: crate::training::DatasetMetadata {
                total_examples: 0,
                train_size: 0,
                validation_size: 0,
                test_size: 0,
                class_distribution: HashMap::new(),
                avg_confidence: 0.0,
                min_confidence: 0.75,
                repositories: vec![],
            },
        };

        let result = trainer.train(&dataset);
        assert!(result.is_err());
    }

    #[test]
    fn test_convert_f64_to_f32() {
        let matrix_f64 = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let result = MLTrainer::convert_f64_to_f32(&matrix_f64);
        assert!(result.is_ok());

        let matrix_f32 = result.unwrap();
        assert_eq!(matrix_f32.n_rows(), 2);
        assert_eq!(matrix_f32.n_cols(), 3);
        assert_eq!(matrix_f32.get(0, 0), 1.0f32);
        assert_eq!(matrix_f32.get(1, 2), 6.0f32);
    }
}
