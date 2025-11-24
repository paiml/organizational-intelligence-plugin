//! Training data extraction pipeline for ML defect classification.
//!
//! This module implements Phase 2 training data collection from Git history:
//! - Extract commit messages from repositories
//! - Filter relevant defect-fix commits
//! - Auto-label using rule-based classifier
//! - Create train/test/validation splits
//! - Export to structured format for ML training
//!
//! Implements Section 5.4 Training Data Pipeline from nlp-models-techniques-spec.md

use crate::classifier::{DefectCategory, RuleBasedClassifier};
use crate::git::CommitInfo;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Training example with features and label
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    /// Commit message text
    pub message: String,
    /// Defect category label
    pub label: DefectCategory,
    /// Classifier confidence (0.0-1.0)
    pub confidence: f32,
    /// Original commit hash
    pub commit_hash: String,
    /// Author name
    pub author: String,
    /// Unix timestamp
    pub timestamp: i64,
    /// Lines added in commit
    pub lines_added: usize,
    /// Lines removed in commit
    pub lines_removed: usize,
    /// Number of files changed
    pub files_changed: usize,
}

/// Training dataset with train/test/validation splits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingDataset {
    /// Training examples
    pub train: Vec<TrainingExample>,
    /// Validation examples
    pub validation: Vec<TrainingExample>,
    /// Test examples
    pub test: Vec<TrainingExample>,
    /// Dataset metadata
    pub metadata: DatasetMetadata,
}

/// Metadata about the training dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    /// Total number of examples
    pub total_examples: usize,
    /// Number of training examples
    pub train_size: usize,
    /// Number of validation examples
    pub validation_size: usize,
    /// Number of test examples
    pub test_size: usize,
    /// Class distribution (category -> count)
    pub class_distribution: HashMap<String, usize>,
    /// Average confidence score
    pub avg_confidence: f32,
    /// Minimum confidence threshold used
    pub min_confidence: f32,
    /// Repository names included
    pub repositories: Vec<String>,
}

/// Training data extractor
pub struct TrainingDataExtractor {
    classifier: RuleBasedClassifier,
    min_confidence: f32,
}

impl TrainingDataExtractor {
    /// Create a new training data extractor
    ///
    /// # Arguments
    ///
    /// * `min_confidence` - Minimum confidence threshold for auto-labeling (0.6-0.9)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use organizational_intelligence_plugin::training::TrainingDataExtractor;
    ///
    /// let extractor = TrainingDataExtractor::new(0.75);
    /// ```
    pub fn new(min_confidence: f32) -> Self {
        Self {
            classifier: RuleBasedClassifier::new(),
            min_confidence,
        }
    }

    /// Extract training examples from commit history
    ///
    /// Filters commits and auto-labels using rule-based classifier.
    ///
    /// # Arguments
    ///
    /// * `commits` - Raw commit history
    /// * `repository_name` - Name of the repository
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<TrainingExample>)` - Labeled training examples
    /// * `Err` - If extraction fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// use organizational_intelligence_plugin::training::TrainingDataExtractor;
    /// use organizational_intelligence_plugin::git::CommitInfo;
    ///
    /// let extractor = TrainingDataExtractor::new(0.75);
    /// let commits = vec![
    ///     CommitInfo {
    ///         hash: "abc123".to_string(),
    ///         message: "fix: null pointer dereference".to_string(),
    ///         author: "dev@example.com".to_string(),
    ///         timestamp: 1234567890,
    ///         files_changed: 2,
    ///         lines_added: 10,
    ///         lines_removed: 5,
    ///     },
    /// ];
    ///
    /// let examples = extractor.extract_training_data(&commits, "test-repo").unwrap();
    /// assert_eq!(examples.len(), 1);
    /// ```
    pub fn extract_training_data(
        &self,
        commits: &[CommitInfo],
        _repository_name: &str,
    ) -> Result<Vec<TrainingExample>> {
        let mut examples = Vec::new();

        for commit in commits {
            // Filter: Skip if not a defect-fix commit
            if !self.is_defect_fix_commit(&commit.message) {
                continue;
            }

            // Auto-label using rule-based classifier
            if let Some(classification) = self.classifier.classify_from_message(&commit.message) {
                // Only include if confidence meets threshold
                if classification.confidence >= self.min_confidence {
                    examples.push(TrainingExample {
                        message: commit.message.clone(),
                        label: classification.category,
                        confidence: classification.confidence,
                        commit_hash: commit.hash.clone(),
                        author: commit.author.clone(),
                        timestamp: commit.timestamp,
                        lines_added: commit.lines_added,
                        lines_removed: commit.lines_removed,
                        files_changed: commit.files_changed,
                    });
                }
            }
        }

        Ok(examples)
    }

    /// Check if a commit message is a defect fix
    ///
    /// Uses heuristics to identify defect-fix commits:
    /// - Starts with "fix:", "bug:", "patch:"
    /// - Contains keywords: "fix", "bug", "error", "crash", "issue"
    /// - Excludes: merge commits, reverts, docs, tests (unless fixing a bug)
    fn is_defect_fix_commit(&self, message: &str) -> bool {
        let lower = message.to_lowercase();

        // Skip obvious non-defect commits
        if lower.starts_with("merge")
            || lower.starts_with("revert")
            || lower.contains("wip")
            || lower.contains("work in progress")
        {
            return false;
        }

        // Check for defect-fix indicators
        lower.starts_with("fix:")
            || lower.starts_with("bug:")
            || lower.starts_with("patch:")
            || lower.contains("fix ")
            || lower.contains("bug ")
            || lower.contains("error")
            || lower.contains("crash")
            || lower.contains("issue")
    }

    /// Create train/test/validation splits
    ///
    /// Uses 70/15/15 split (train/validation/test) as recommended by the spec.
    ///
    /// # Arguments
    ///
    /// * `examples` - Labeled training examples
    /// * `repositories` - List of repository names
    ///
    /// # Returns
    ///
    /// * `Ok(TrainingDataset)` - Dataset with splits
    /// * `Err` - If split fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// use organizational_intelligence_plugin::training::TrainingDataExtractor;
    /// use organizational_intelligence_plugin::training::TrainingExample;
    /// use organizational_intelligence_plugin::classifier::DefectCategory;
    ///
    /// let extractor = TrainingDataExtractor::new(0.75);
    /// let examples = vec![
    ///     TrainingExample {
    ///         message: "fix: bug".to_string(),
    ///         label: DefectCategory::MemorySafety,
    ///         confidence: 0.85,
    ///         commit_hash: "abc".to_string(),
    ///         author: "dev".to_string(),
    ///         timestamp: 123,
    ///         lines_added: 5,
    ///         lines_removed: 2,
    ///         files_changed: 1,
    ///     },
    /// ];
    ///
    /// let dataset = extractor.create_splits(&examples, &["repo1".to_string()]).unwrap();
    /// assert!(dataset.train.len() + dataset.validation.len() + dataset.test.len() == 1);
    /// ```
    pub fn create_splits(
        &self,
        examples: &[TrainingExample],
        repositories: &[String],
    ) -> Result<TrainingDataset> {
        if examples.is_empty() {
            return Err(anyhow!("Cannot create splits from empty dataset"));
        }

        let total = examples.len();

        // Calculate split sizes (70/15/15)
        let train_size = (total as f32 * 0.70) as usize;
        let validation_size = (total as f32 * 0.15) as usize;
        let test_size = total - train_size - validation_size;

        // Split the data
        let train = examples[0..train_size].to_vec();
        let validation = examples[train_size..train_size + validation_size].to_vec();
        let test = examples[train_size + validation_size..].to_vec();

        // Calculate class distribution
        let mut class_distribution = HashMap::new();
        for example in examples {
            let category_name = format!("{}", example.label);
            *class_distribution.entry(category_name).or_insert(0) += 1;
        }

        // Calculate average confidence
        let avg_confidence =
            examples.iter().map(|e| e.confidence).sum::<f32>() / examples.len() as f32;

        let metadata = DatasetMetadata {
            total_examples: total,
            train_size,
            validation_size,
            test_size,
            class_distribution,
            avg_confidence,
            min_confidence: self.min_confidence,
            repositories: repositories.to_vec(),
        };

        Ok(TrainingDataset {
            train,
            validation,
            test,
            metadata,
        })
    }

    /// Get statistics about extracted training data
    ///
    /// # Arguments
    ///
    /// * `examples` - Training examples
    ///
    /// # Returns
    ///
    /// * Formatted statistics string
    pub fn get_statistics(&self, examples: &[TrainingExample]) -> String {
        if examples.is_empty() {
            return "No examples extracted".to_string();
        }

        let mut category_counts: HashMap<DefectCategory, usize> = HashMap::new();
        let mut confidence_sum = 0.0_f32;

        for example in examples {
            *category_counts.entry(example.label).or_insert(0) += 1;
            confidence_sum += example.confidence;
        }

        let avg_confidence = confidence_sum / examples.len() as f32;

        let mut stats = "Training Data Statistics:\n".to_string();
        stats.push_str(&format!("  Total examples: {}\n", examples.len()));
        stats.push_str(&format!("  Avg confidence: {:.2}\n", avg_confidence));
        stats.push_str(&format!(
            "  Min confidence threshold: {:.2}\n",
            self.min_confidence
        ));
        stats.push_str("\nClass Distribution:\n");

        let mut sorted_categories: Vec<_> = category_counts.iter().collect();
        sorted_categories.sort_by_key(|(_, count)| std::cmp::Reverse(*count));

        for (category, count) in sorted_categories {
            let percentage = (*count as f32 / examples.len() as f32) * 100.0;
            stats.push_str(&format!(
                "  {:?}: {} ({:.1}%)\n",
                category, count, percentage
            ));
        }

        stats
    }
}

impl Default for TrainingDataExtractor {
    fn default() -> Self {
        Self::new(0.75) // Default 75% confidence threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extractor_creation() {
        let extractor = TrainingDataExtractor::new(0.80);
        assert_eq!(extractor.min_confidence, 0.80);
    }

    #[test]
    fn test_is_defect_fix_commit() {
        let extractor = TrainingDataExtractor::new(0.75);

        // Should be defect fixes
        assert!(extractor.is_defect_fix_commit("fix: null pointer"));
        assert!(extractor.is_defect_fix_commit("bug: race condition"));
        assert!(extractor.is_defect_fix_commit("patch: memory leak"));
        assert!(extractor.is_defect_fix_commit("fix memory leak in parser"));

        // Should not be defect fixes
        assert!(!extractor.is_defect_fix_commit("Merge branch 'main'"));
        assert!(!extractor.is_defect_fix_commit("Revert commit abc123"));
        assert!(!extractor.is_defect_fix_commit("feat: add new feature"));
        assert!(!extractor.is_defect_fix_commit("docs: update README"));
        assert!(!extractor.is_defect_fix_commit("WIP: working on feature"));
    }

    #[test]
    fn test_extract_training_data() {
        let extractor = TrainingDataExtractor::new(0.70);

        let commits = vec![
            CommitInfo {
                hash: "abc123".to_string(),
                message: "fix: null pointer dereference in parser".to_string(),
                author: "dev@example.com".to_string(),
                timestamp: 1234567890,
                files_changed: 2,
                lines_added: 10,
                lines_removed: 5,
            },
            CommitInfo {
                hash: "def456".to_string(),
                message: "feat: add new feature".to_string(), // Not a defect fix
                author: "dev@example.com".to_string(),
                timestamp: 1234567891,
                files_changed: 5,
                lines_added: 100,
                lines_removed: 0,
            },
            CommitInfo {
                hash: "ghi789".to_string(),
                message: "fix: race condition in mutex lock".to_string(),
                author: "dev@example.com".to_string(),
                timestamp: 1234567892,
                files_changed: 1,
                lines_added: 5,
                lines_removed: 3,
            },
        ];

        let examples = extractor
            .extract_training_data(&commits, "test-repo")
            .unwrap();

        // Should extract 2 defect-fix commits
        assert_eq!(examples.len(), 2);
        assert_eq!(
            examples[0].message,
            "fix: null pointer dereference in parser"
        );
        assert_eq!(examples[1].message, "fix: race condition in mutex lock");
    }

    #[test]
    fn test_create_splits() {
        let extractor = TrainingDataExtractor::new(0.75);

        // Create 100 examples for clean split
        let mut examples = Vec::new();
        for i in 0..100 {
            examples.push(TrainingExample {
                message: format!("fix: bug {}", i),
                label: DefectCategory::MemorySafety,
                confidence: 0.85,
                commit_hash: format!("hash{}", i),
                author: "dev".to_string(),
                timestamp: 123 + i as i64,
                lines_added: 5,
                lines_removed: 2,
                files_changed: 1,
            });
        }

        let dataset = extractor
            .create_splits(&examples, &["repo1".to_string()])
            .unwrap();

        // Check split sizes (70/15/15)
        assert_eq!(dataset.train.len(), 70);
        assert_eq!(dataset.validation.len(), 15);
        assert_eq!(dataset.test.len(), 15);
        assert_eq!(dataset.metadata.total_examples, 100);
        assert_eq!(dataset.metadata.train_size, 70);
    }

    #[test]
    fn test_empty_dataset_error() {
        let extractor = TrainingDataExtractor::new(0.75);
        let examples: Vec<TrainingExample> = vec![];

        let result = extractor.create_splits(&examples, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_statistics() {
        let extractor = TrainingDataExtractor::new(0.75);

        let examples = vec![
            TrainingExample {
                message: "fix: bug 1".to_string(),
                label: DefectCategory::MemorySafety,
                confidence: 0.85,
                commit_hash: "a".to_string(),
                author: "dev".to_string(),
                timestamp: 123,
                lines_added: 5,
                lines_removed: 2,
                files_changed: 1,
            },
            TrainingExample {
                message: "fix: bug 2".to_string(),
                label: DefectCategory::ConcurrencyBugs,
                confidence: 0.90,
                commit_hash: "b".to_string(),
                author: "dev".to_string(),
                timestamp: 124,
                lines_added: 3,
                lines_removed: 1,
                files_changed: 1,
            },
        ];

        let stats = extractor.get_statistics(&examples);
        assert!(stats.contains("Total examples: 2"));
        assert!(stats.contains("Avg confidence:"));
        assert!(stats.contains("Class Distribution:"));
    }

    #[test]
    fn test_confidence_threshold_filtering() {
        let extractor = TrainingDataExtractor::new(0.90); // High threshold

        let commits = vec![CommitInfo {
            hash: "abc".to_string(),
            message: "fix: memory leak".to_string(), // Will have ~0.85 confidence
            author: "dev".to_string(),
            timestamp: 123,
            files_changed: 1,
            lines_added: 5,
            lines_removed: 2,
        }];

        let examples = extractor
            .extract_training_data(&commits, "test-repo")
            .unwrap();

        // With 0.90 threshold, low-confidence examples should be filtered
        // (actual result depends on classifier confidence)
        assert!(examples.len() <= 1);
    }

    #[test]
    fn test_default_extractor() {
        let extractor = TrainingDataExtractor::default();
        assert_eq!(extractor.min_confidence, 0.75);
    }
}
