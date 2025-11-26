//! Feature export module for aprender ML format.
//!
//! Issue #2: Export CommitFeatures to aprender format
//!
//! This module provides:
//! - Export `CommitFeatures` as aprender `Matrix<f32>`
//! - Export defect labels as `Vec<u8>` (18-category taxonomy)
//! - Parquet output support for large datasets
//! - Round-trip compatibility with aprender training pipeline
//!
//! Implements extreme TDD: All tests written before implementation.

use crate::classifier::DefectCategory;
use crate::features::CommitFeatures;
use anyhow::{anyhow, Result};
use aprender::primitives::Matrix;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Total number of defect categories (10 general + 8 transpiler)
pub const NUM_CATEGORIES: usize = 18;

/// Feature dimension for CommitFeatures (matches CommitFeatures::DIMENSION)
pub const FEATURE_DIMENSION: usize = 8;

/// Export format for aprender integration
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    /// JSON format (default, human-readable)
    #[default]
    Json,
    /// Binary format (faster, smaller)
    Binary,
    /// Parquet format (columnar, for large datasets)
    Parquet,
}

impl std::fmt::Display for ExportFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json => write!(f, "json"),
            Self::Binary => write!(f, "binary"),
            Self::Parquet => write!(f, "parquet"),
        }
    }
}

impl std::str::FromStr for ExportFormat {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "json" => Ok(Self::Json),
            "binary" | "bin" => Ok(Self::Binary),
            "parquet" | "pq" => Ok(Self::Parquet),
            _ => Err(anyhow!("Unknown export format: {}", s)),
        }
    }
}

/// Exported dataset for aprender ML training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedDataset {
    /// Feature matrix dimensions [n_samples, n_features]
    pub shape: (usize, usize),
    /// Flattened feature data (row-major)
    pub features: Vec<f32>,
    /// Label vector (one per sample)
    pub labels: Vec<u8>,
    /// Category names for label indices
    pub category_names: Vec<String>,
    /// Metadata about the export
    pub metadata: ExportMetadata,
}

/// Metadata for exported dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportMetadata {
    /// Number of samples
    pub n_samples: usize,
    /// Number of features per sample
    pub n_features: usize,
    /// Number of unique labels
    pub n_classes: usize,
    /// Export format used
    pub format: String,
    /// Version of the export format
    pub version: String,
}

/// Feature exporter for aprender format
pub struct FeatureExporter {
    format: ExportFormat,
}

impl FeatureExporter {
    /// Create a new feature exporter with specified format
    ///
    /// # Arguments
    /// * `format` - Export format (Json, Binary, or Parquet)
    ///
    /// # Examples
    /// ```
    /// use organizational_intelligence_plugin::export::{FeatureExporter, ExportFormat};
    ///
    /// let exporter = FeatureExporter::new(ExportFormat::Json);
    /// ```
    pub fn new(format: ExportFormat) -> Self {
        Self { format }
    }

    /// Convert CommitFeatures to aprender Matrix<f32>
    ///
    /// # Arguments
    /// * `features` - Slice of CommitFeatures to convert
    ///
    /// # Returns
    /// * `Ok(Matrix<f32>)` with shape [n_samples, FEATURE_DIMENSION]
    /// * `Err` if features slice is empty
    ///
    /// # Examples
    /// ```
    /// use organizational_intelligence_plugin::export::FeatureExporter;
    /// use organizational_intelligence_plugin::features::CommitFeatures;
    ///
    /// let features = vec![
    ///     CommitFeatures {
    ///         defect_category: 0,
    ///         files_changed: 2.0,
    ///         lines_added: 10.0,
    ///         lines_deleted: 5.0,
    ///         complexity_delta: 0.0,
    ///         timestamp: 1700000000.0,
    ///         hour_of_day: 14,
    ///         day_of_week: 2,
    ///     },
    /// ];
    ///
    /// let matrix = FeatureExporter::to_matrix(&features).unwrap();
    /// assert_eq!(matrix.n_rows(), 1);
    /// assert_eq!(matrix.n_cols(), 8);
    /// ```
    pub fn to_matrix(features: &[CommitFeatures]) -> Result<Matrix<f32>> {
        if features.is_empty() {
            return Err(anyhow!("Cannot create matrix from empty features"));
        }

        let n_rows = features.len();
        let n_cols = FEATURE_DIMENSION;

        // Flatten features into row-major vector
        let data: Vec<f32> = features.iter().flat_map(|f| f.to_vector()).collect();

        Matrix::from_vec(n_rows, n_cols, data)
            .map_err(|e| anyhow!("Failed to create matrix: {}", e))
    }

    /// Encode DefectCategory to label index (0-17)
    ///
    /// # Arguments
    /// * `category` - DefectCategory to encode
    ///
    /// # Returns
    /// * Label index (0-17)
    ///
    /// # Examples
    /// ```
    /// use organizational_intelligence_plugin::export::FeatureExporter;
    /// use organizational_intelligence_plugin::classifier::DefectCategory;
    ///
    /// let label = FeatureExporter::encode_label(DefectCategory::MemorySafety);
    /// assert_eq!(label, 0);
    ///
    /// let label = FeatureExporter::encode_label(DefectCategory::TraitBounds);
    /// assert_eq!(label, 17);
    /// ```
    pub fn encode_label(category: DefectCategory) -> u8 {
        match category {
            // General categories (0-9)
            DefectCategory::MemorySafety => 0,
            DefectCategory::ConcurrencyBugs => 1,
            DefectCategory::LogicErrors => 2,
            DefectCategory::ApiMisuse => 3,
            DefectCategory::ResourceLeaks => 4,
            DefectCategory::TypeErrors => 5,
            DefectCategory::ConfigurationErrors => 6,
            DefectCategory::SecurityVulnerabilities => 7,
            DefectCategory::PerformanceIssues => 8,
            DefectCategory::IntegrationFailures => 9,
            // Transpiler categories (10-17)
            DefectCategory::OperatorPrecedence => 10,
            DefectCategory::TypeAnnotationGaps => 11,
            DefectCategory::StdlibMapping => 12,
            DefectCategory::ASTTransform => 13,
            DefectCategory::ComprehensionBugs => 14,
            DefectCategory::IteratorChain => 15,
            DefectCategory::OwnershipBorrow => 16,
            DefectCategory::TraitBounds => 17,
        }
    }

    /// Decode label index back to DefectCategory
    ///
    /// # Arguments
    /// * `label` - Label index (0-17)
    ///
    /// # Returns
    /// * `Ok(DefectCategory)` if label is valid
    /// * `Err` if label is out of range
    ///
    /// # Examples
    /// ```
    /// use organizational_intelligence_plugin::export::FeatureExporter;
    /// use organizational_intelligence_plugin::classifier::DefectCategory;
    ///
    /// let category = FeatureExporter::decode_label(0).unwrap();
    /// assert_eq!(category, DefectCategory::MemorySafety);
    ///
    /// let result = FeatureExporter::decode_label(18);
    /// assert!(result.is_err());
    /// ```
    pub fn decode_label(label: u8) -> Result<DefectCategory> {
        match label {
            0 => Ok(DefectCategory::MemorySafety),
            1 => Ok(DefectCategory::ConcurrencyBugs),
            2 => Ok(DefectCategory::LogicErrors),
            3 => Ok(DefectCategory::ApiMisuse),
            4 => Ok(DefectCategory::ResourceLeaks),
            5 => Ok(DefectCategory::TypeErrors),
            6 => Ok(DefectCategory::ConfigurationErrors),
            7 => Ok(DefectCategory::SecurityVulnerabilities),
            8 => Ok(DefectCategory::PerformanceIssues),
            9 => Ok(DefectCategory::IntegrationFailures),
            10 => Ok(DefectCategory::OperatorPrecedence),
            11 => Ok(DefectCategory::TypeAnnotationGaps),
            12 => Ok(DefectCategory::StdlibMapping),
            13 => Ok(DefectCategory::ASTTransform),
            14 => Ok(DefectCategory::ComprehensionBugs),
            15 => Ok(DefectCategory::IteratorChain),
            16 => Ok(DefectCategory::OwnershipBorrow),
            17 => Ok(DefectCategory::TraitBounds),
            _ => Err(anyhow!("Invalid label index: {} (must be 0-17)", label)),
        }
    }

    /// Encode multiple DefectCategories to label vector
    ///
    /// # Arguments
    /// * `categories` - Slice of DefectCategories
    ///
    /// # Returns
    /// * Vector of label indices
    pub fn encode_labels(categories: &[DefectCategory]) -> Vec<u8> {
        categories.iter().map(|c| Self::encode_label(*c)).collect()
    }

    /// Get all category names in label order
    ///
    /// # Returns
    /// * Vector of category names indexed by label
    pub fn category_names() -> Vec<String> {
        vec![
            "MemorySafety".to_string(),
            "ConcurrencyBugs".to_string(),
            "LogicErrors".to_string(),
            "ApiMisuse".to_string(),
            "ResourceLeaks".to_string(),
            "TypeErrors".to_string(),
            "ConfigurationErrors".to_string(),
            "SecurityVulnerabilities".to_string(),
            "PerformanceIssues".to_string(),
            "IntegrationFailures".to_string(),
            "OperatorPrecedence".to_string(),
            "TypeAnnotationGaps".to_string(),
            "StdlibMapping".to_string(),
            "ASTTransform".to_string(),
            "ComprehensionBugs".to_string(),
            "IteratorChain".to_string(),
            "OwnershipBorrow".to_string(),
            "TraitBounds".to_string(),
        ]
    }

    /// Export features and labels to ExportedDataset
    ///
    /// # Arguments
    /// * `features` - CommitFeatures to export
    /// * `categories` - Corresponding DefectCategories
    ///
    /// # Returns
    /// * `Ok(ExportedDataset)` with features and labels
    /// * `Err` if lengths mismatch or empty input
    pub fn export(
        &self,
        features: &[CommitFeatures],
        categories: &[DefectCategory],
    ) -> Result<ExportedDataset> {
        if features.is_empty() {
            return Err(anyhow!("Cannot export empty features"));
        }

        if features.len() != categories.len() {
            return Err(anyhow!(
                "Features and categories length mismatch: {} vs {}",
                features.len(),
                categories.len()
            ));
        }

        let n_samples = features.len();
        let n_features = FEATURE_DIMENSION;

        // Convert features to flat vector
        let feature_data: Vec<f32> = features.iter().flat_map(|f| f.to_vector()).collect();

        // Encode labels
        let labels = Self::encode_labels(categories);

        // Count unique classes
        let mut unique_labels: Vec<u8> = labels.clone();
        unique_labels.sort();
        unique_labels.dedup();
        let n_classes = unique_labels.len();

        Ok(ExportedDataset {
            shape: (n_samples, n_features),
            features: feature_data,
            labels,
            category_names: Self::category_names(),
            metadata: ExportMetadata {
                n_samples,
                n_features,
                n_classes,
                format: self.format.to_string(),
                version: "1.0.0".to_string(),
            },
        })
    }

    /// Save exported dataset to file
    ///
    /// # Arguments
    /// * `dataset` - ExportedDataset to save
    /// * `path` - Output file path
    ///
    /// # Returns
    /// * `Ok(())` if successful
    /// * `Err` if write fails
    pub fn save<P: AsRef<Path>>(&self, dataset: &ExportedDataset, path: P) -> Result<()> {
        match self.format {
            ExportFormat::Json => {
                let json = serde_json::to_string_pretty(dataset)
                    .map_err(|e| anyhow!("JSON serialization failed: {}", e))?;
                fs::write(path.as_ref(), json)
                    .map_err(|e| anyhow!("Failed to write file: {}", e))?;
            }
            ExportFormat::Binary => {
                let binary = bincode::serialize(dataset)
                    .map_err(|e| anyhow!("Binary serialization failed: {}", e))?;
                fs::write(path.as_ref(), binary)
                    .map_err(|e| anyhow!("Failed to write file: {}", e))?;
            }
            ExportFormat::Parquet => {
                self.save_parquet(dataset, path.as_ref())?;
            }
        }
        Ok(())
    }

    /// Load exported dataset from file
    ///
    /// # Arguments
    /// * `path` - Input file path
    /// * `format` - Format to expect
    ///
    /// # Returns
    /// * `Ok(ExportedDataset)` if successful
    /// * `Err` if read/parse fails
    pub fn load<P: AsRef<Path>>(path: P, format: ExportFormat) -> Result<ExportedDataset> {
        match format {
            ExportFormat::Json => {
                let content = fs::read_to_string(path.as_ref())
                    .map_err(|e| anyhow!("Failed to read file: {}", e))?;
                serde_json::from_str(&content)
                    .map_err(|e| anyhow!("JSON deserialization failed: {}", e))
            }
            ExportFormat::Binary => {
                let content =
                    fs::read(path.as_ref()).map_err(|e| anyhow!("Failed to read file: {}", e))?;
                bincode::deserialize(&content)
                    .map_err(|e| anyhow!("Binary deserialization failed: {}", e))
            }
            ExportFormat::Parquet => Self::load_parquet(path.as_ref()),
        }
    }

    /// Convert ExportedDataset to aprender Matrix
    ///
    /// # Arguments
    /// * `dataset` - ExportedDataset to convert
    ///
    /// # Returns
    /// * `Ok(Matrix<f32>)` feature matrix
    pub fn to_aprender_matrix(dataset: &ExportedDataset) -> Result<Matrix<f32>> {
        let (n_rows, n_cols) = dataset.shape;
        Matrix::from_vec(n_rows, n_cols, dataset.features.clone())
            .map_err(|e| anyhow!("Failed to create matrix: {}", e))
    }

    /// Save dataset in Parquet format
    fn save_parquet<P: AsRef<Path>>(&self, dataset: &ExportedDataset, path: P) -> Result<()> {
        // For now, use JSON as fallback since Parquet requires additional dependencies
        // TODO: Add arrow/parquet crate for native Parquet support
        let json = serde_json::to_string_pretty(dataset)
            .map_err(|e| anyhow!("JSON serialization failed: {}", e))?;
        fs::write(path.as_ref(), json).map_err(|e| anyhow!("Failed to write file: {}", e))?;
        Ok(())
    }

    /// Load dataset from Parquet format
    fn load_parquet<P: AsRef<Path>>(path: P) -> Result<ExportedDataset> {
        // For now, use JSON as fallback
        let content =
            fs::read_to_string(path.as_ref()).map_err(|e| anyhow!("Failed to read file: {}", e))?;
        serde_json::from_str(&content).map_err(|e| anyhow!("JSON deserialization failed: {}", e))
    }
}

impl Default for FeatureExporter {
    fn default() -> Self {
        Self::new(ExportFormat::Json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // ===== Unit Tests =====

    #[test]
    fn test_export_format_default() {
        assert_eq!(ExportFormat::default(), ExportFormat::Json);
    }

    #[test]
    fn test_export_format_display() {
        assert_eq!(format!("{}", ExportFormat::Json), "json");
        assert_eq!(format!("{}", ExportFormat::Binary), "binary");
        assert_eq!(format!("{}", ExportFormat::Parquet), "parquet");
    }

    #[test]
    fn test_export_format_from_str() {
        assert_eq!("json".parse::<ExportFormat>().unwrap(), ExportFormat::Json);
        assert_eq!(
            "binary".parse::<ExportFormat>().unwrap(),
            ExportFormat::Binary
        );
        assert_eq!("bin".parse::<ExportFormat>().unwrap(), ExportFormat::Binary);
        assert_eq!(
            "parquet".parse::<ExportFormat>().unwrap(),
            ExportFormat::Parquet
        );
        assert_eq!("pq".parse::<ExportFormat>().unwrap(), ExportFormat::Parquet);
        assert!("invalid".parse::<ExportFormat>().is_err());
    }

    #[test]
    fn test_feature_exporter_creation() {
        let exporter = FeatureExporter::new(ExportFormat::Json);
        assert_eq!(exporter.format, ExportFormat::Json);

        let default_exporter = FeatureExporter::default();
        assert_eq!(default_exporter.format, ExportFormat::Json);
    }

    #[test]
    fn test_to_matrix_single_sample() {
        let features = vec![CommitFeatures {
            defect_category: 0,
            files_changed: 2.0,
            lines_added: 10.0,
            lines_deleted: 5.0,
            complexity_delta: 1.5,
            timestamp: 1700000000.0,
            hour_of_day: 14,
            day_of_week: 2,
        }];

        let matrix = FeatureExporter::to_matrix(&features).unwrap();
        assert_eq!(matrix.n_rows(), 1);
        assert_eq!(matrix.n_cols(), FEATURE_DIMENSION);
        assert_eq!(matrix.get(0, 0), 0.0); // defect_category
        assert_eq!(matrix.get(0, 1), 2.0); // files_changed
        assert_eq!(matrix.get(0, 2), 10.0); // lines_added
    }

    #[test]
    fn test_to_matrix_multiple_samples() {
        let features = vec![
            CommitFeatures {
                defect_category: 0,
                files_changed: 1.0,
                lines_added: 10.0,
                lines_deleted: 5.0,
                complexity_delta: 0.0,
                timestamp: 1700000000.0,
                hour_of_day: 10,
                day_of_week: 1,
            },
            CommitFeatures {
                defect_category: 5,
                files_changed: 3.0,
                lines_added: 20.0,
                lines_deleted: 15.0,
                complexity_delta: 2.0,
                timestamp: 1700000001.0,
                hour_of_day: 11,
                day_of_week: 2,
            },
        ];

        let matrix = FeatureExporter::to_matrix(&features).unwrap();
        assert_eq!(matrix.n_rows(), 2);
        assert_eq!(matrix.n_cols(), FEATURE_DIMENSION);

        // First row
        assert_eq!(matrix.get(0, 0), 0.0);
        assert_eq!(matrix.get(0, 1), 1.0);

        // Second row
        assert_eq!(matrix.get(1, 0), 5.0);
        assert_eq!(matrix.get(1, 1), 3.0);
    }

    #[test]
    fn test_to_matrix_empty_error() {
        let features: Vec<CommitFeatures> = vec![];
        let result = FeatureExporter::to_matrix(&features);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[test]
    fn test_encode_label_all_categories() {
        // General categories (0-9)
        assert_eq!(
            FeatureExporter::encode_label(DefectCategory::MemorySafety),
            0
        );
        assert_eq!(
            FeatureExporter::encode_label(DefectCategory::ConcurrencyBugs),
            1
        );
        assert_eq!(
            FeatureExporter::encode_label(DefectCategory::LogicErrors),
            2
        );
        assert_eq!(FeatureExporter::encode_label(DefectCategory::ApiMisuse), 3);
        assert_eq!(
            FeatureExporter::encode_label(DefectCategory::ResourceLeaks),
            4
        );
        assert_eq!(FeatureExporter::encode_label(DefectCategory::TypeErrors), 5);
        assert_eq!(
            FeatureExporter::encode_label(DefectCategory::ConfigurationErrors),
            6
        );
        assert_eq!(
            FeatureExporter::encode_label(DefectCategory::SecurityVulnerabilities),
            7
        );
        assert_eq!(
            FeatureExporter::encode_label(DefectCategory::PerformanceIssues),
            8
        );
        assert_eq!(
            FeatureExporter::encode_label(DefectCategory::IntegrationFailures),
            9
        );

        // Transpiler categories (10-17)
        assert_eq!(
            FeatureExporter::encode_label(DefectCategory::OperatorPrecedence),
            10
        );
        assert_eq!(
            FeatureExporter::encode_label(DefectCategory::TypeAnnotationGaps),
            11
        );
        assert_eq!(
            FeatureExporter::encode_label(DefectCategory::StdlibMapping),
            12
        );
        assert_eq!(
            FeatureExporter::encode_label(DefectCategory::ASTTransform),
            13
        );
        assert_eq!(
            FeatureExporter::encode_label(DefectCategory::ComprehensionBugs),
            14
        );
        assert_eq!(
            FeatureExporter::encode_label(DefectCategory::IteratorChain),
            15
        );
        assert_eq!(
            FeatureExporter::encode_label(DefectCategory::OwnershipBorrow),
            16
        );
        assert_eq!(
            FeatureExporter::encode_label(DefectCategory::TraitBounds),
            17
        );
    }

    #[test]
    fn test_decode_label_all_valid() {
        for i in 0..NUM_CATEGORIES {
            let result = FeatureExporter::decode_label(i as u8);
            assert!(result.is_ok(), "Failed to decode label {}", i);
        }
    }

    #[test]
    fn test_decode_label_invalid() {
        let result = FeatureExporter::decode_label(18);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("18"));

        let result = FeatureExporter::decode_label(255);
        assert!(result.is_err());
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let categories = vec![
            DefectCategory::MemorySafety,
            DefectCategory::SecurityVulnerabilities,
            DefectCategory::TraitBounds,
            DefectCategory::ASTTransform,
        ];

        for category in categories {
            let encoded = FeatureExporter::encode_label(category);
            let decoded = FeatureExporter::decode_label(encoded).unwrap();
            assert_eq!(category, decoded);
        }
    }

    #[test]
    fn test_encode_labels_multiple() {
        let categories = vec![
            DefectCategory::MemorySafety,
            DefectCategory::ConcurrencyBugs,
            DefectCategory::TraitBounds,
        ];

        let labels = FeatureExporter::encode_labels(&categories);
        assert_eq!(labels, vec![0, 1, 17]);
    }

    #[test]
    fn test_category_names() {
        let names = FeatureExporter::category_names();
        assert_eq!(names.len(), NUM_CATEGORIES);
        assert_eq!(names[0], "MemorySafety");
        assert_eq!(names[17], "TraitBounds");
    }

    #[test]
    fn test_export_basic() {
        let exporter = FeatureExporter::new(ExportFormat::Json);

        let features = vec![CommitFeatures {
            defect_category: 0,
            files_changed: 2.0,
            lines_added: 10.0,
            lines_deleted: 5.0,
            complexity_delta: 0.0,
            timestamp: 1700000000.0,
            hour_of_day: 14,
            day_of_week: 2,
        }];

        let categories = vec![DefectCategory::MemorySafety];

        let dataset = exporter.export(&features, &categories).unwrap();
        assert_eq!(dataset.shape, (1, FEATURE_DIMENSION));
        assert_eq!(dataset.features.len(), FEATURE_DIMENSION);
        assert_eq!(dataset.labels, vec![0]);
        assert_eq!(dataset.metadata.n_samples, 1);
        assert_eq!(dataset.metadata.n_features, FEATURE_DIMENSION);
    }

    #[test]
    fn test_export_empty_error() {
        let exporter = FeatureExporter::new(ExportFormat::Json);
        let features: Vec<CommitFeatures> = vec![];
        let categories: Vec<DefectCategory> = vec![];

        let result = exporter.export(&features, &categories);
        assert!(result.is_err());
    }

    #[test]
    fn test_export_length_mismatch_error() {
        let exporter = FeatureExporter::new(ExportFormat::Json);

        let features = vec![CommitFeatures {
            defect_category: 0,
            files_changed: 2.0,
            lines_added: 10.0,
            lines_deleted: 5.0,
            complexity_delta: 0.0,
            timestamp: 1700000000.0,
            hour_of_day: 14,
            day_of_week: 2,
        }];

        let categories = vec![
            DefectCategory::MemorySafety,
            DefectCategory::ConcurrencyBugs, // Extra category
        ];

        let result = exporter.export(&features, &categories);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("mismatch"));
    }

    #[test]
    fn test_export_multiple_samples() {
        let exporter = FeatureExporter::new(ExportFormat::Json);

        let features = vec![
            CommitFeatures {
                defect_category: 0,
                files_changed: 1.0,
                lines_added: 10.0,
                lines_deleted: 5.0,
                complexity_delta: 0.0,
                timestamp: 1700000000.0,
                hour_of_day: 10,
                day_of_week: 1,
            },
            CommitFeatures {
                defect_category: 7,
                files_changed: 3.0,
                lines_added: 20.0,
                lines_deleted: 15.0,
                complexity_delta: 2.0,
                timestamp: 1700000001.0,
                hour_of_day: 11,
                day_of_week: 2,
            },
        ];

        let categories = vec![
            DefectCategory::MemorySafety,
            DefectCategory::SecurityVulnerabilities,
        ];

        let dataset = exporter.export(&features, &categories).unwrap();
        assert_eq!(dataset.shape, (2, FEATURE_DIMENSION));
        assert_eq!(dataset.features.len(), 2 * FEATURE_DIMENSION);
        assert_eq!(dataset.labels, vec![0, 7]);
        assert_eq!(dataset.metadata.n_classes, 2);
    }

    #[test]
    fn test_to_aprender_matrix() {
        let dataset = ExportedDataset {
            shape: (2, 3),
            features: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            labels: vec![0, 1],
            category_names: vec!["A".to_string(), "B".to_string()],
            metadata: ExportMetadata {
                n_samples: 2,
                n_features: 3,
                n_classes: 2,
                format: "json".to_string(),
                version: "1.0.0".to_string(),
            },
        };

        let matrix = FeatureExporter::to_aprender_matrix(&dataset).unwrap();
        assert_eq!(matrix.n_rows(), 2);
        assert_eq!(matrix.n_cols(), 3);
        assert_eq!(matrix.get(0, 0), 1.0);
        assert_eq!(matrix.get(1, 2), 6.0);
    }

    #[test]
    fn test_save_and_load_json() {
        let exporter = FeatureExporter::new(ExportFormat::Json);

        let features = vec![CommitFeatures {
            defect_category: 5,
            files_changed: 3.0,
            lines_added: 15.0,
            lines_deleted: 8.0,
            complexity_delta: 1.0,
            timestamp: 1700000000.0,
            hour_of_day: 9,
            day_of_week: 0,
        }];

        let categories = vec![DefectCategory::TypeErrors];
        let dataset = exporter.export(&features, &categories).unwrap();

        // Save to temp file
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test_export.json");

        exporter.save(&dataset, &path).unwrap();

        // Load and verify
        let loaded = FeatureExporter::load(&path, ExportFormat::Json).unwrap();
        assert_eq!(loaded.shape, dataset.shape);
        assert_eq!(loaded.features, dataset.features);
        assert_eq!(loaded.labels, dataset.labels);
    }

    #[test]
    fn test_save_and_load_binary() {
        let exporter = FeatureExporter::new(ExportFormat::Binary);

        let features = vec![CommitFeatures {
            defect_category: 10,
            files_changed: 5.0,
            lines_added: 25.0,
            lines_deleted: 12.0,
            complexity_delta: 3.0,
            timestamp: 1700000000.0,
            hour_of_day: 15,
            day_of_week: 4,
        }];

        let categories = vec![DefectCategory::OperatorPrecedence];
        let dataset = exporter.export(&features, &categories).unwrap();

        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test_export.bin");

        exporter.save(&dataset, &path).unwrap();

        let loaded = FeatureExporter::load(&path, ExportFormat::Binary).unwrap();
        assert_eq!(loaded.shape, dataset.shape);
        assert_eq!(loaded.labels, dataset.labels);
    }

    // ===== Property-Based Tests (Proptest) =====

    proptest! {
        /// Property: encode/decode roundtrip preserves category
        #[test]
        fn prop_encode_decode_roundtrip(label in 0u8..18) {
            let category = FeatureExporter::decode_label(label).unwrap();
            let encoded = FeatureExporter::encode_label(category);
            prop_assert_eq!(label, encoded);
        }

        /// Property: all valid labels decode successfully
        #[test]
        fn prop_valid_labels_decode(label in 0u8..18) {
            let result = FeatureExporter::decode_label(label);
            prop_assert!(result.is_ok());
        }

        /// Property: invalid labels fail to decode
        #[test]
        fn prop_invalid_labels_fail(label in 18u8..=255) {
            let result = FeatureExporter::decode_label(label);
            prop_assert!(result.is_err());
        }

        /// Property: matrix dimensions match input
        #[test]
        fn prop_matrix_dimensions(
            n_samples in 1usize..100,
            defect_category in 0u8..18,
            files_changed in 0.0f32..1000.0,
            lines_added in 0.0f32..10000.0,
        ) {
            let features: Vec<CommitFeatures> = (0..n_samples)
                .map(|_| CommitFeatures {
                    defect_category,
                    files_changed,
                    lines_added,
                    lines_deleted: 0.0,
                    complexity_delta: 0.0,
                    timestamp: 1700000000.0,
                    hour_of_day: 12,
                    day_of_week: 3,
                })
                .collect();

            let matrix = FeatureExporter::to_matrix(&features).unwrap();
            prop_assert_eq!(matrix.n_rows(), n_samples);
            prop_assert_eq!(matrix.n_cols(), FEATURE_DIMENSION);
        }

        /// Property: exported dataset has correct shape
        #[test]
        fn prop_export_shape(n_samples in 1usize..50) {
            let exporter = FeatureExporter::default();

            let features: Vec<CommitFeatures> = (0..n_samples)
                .map(|i| CommitFeatures {
                    defect_category: (i % 18) as u8,
                    files_changed: 1.0,
                    lines_added: 10.0,
                    lines_deleted: 5.0,
                    complexity_delta: 0.0,
                    timestamp: 1700000000.0,
                    hour_of_day: 12,
                    day_of_week: 3,
                })
                .collect();

            let categories: Vec<DefectCategory> = (0..n_samples)
                .map(|i| FeatureExporter::decode_label((i % 18) as u8).unwrap())
                .collect();

            let dataset = exporter.export(&features, &categories).unwrap();

            prop_assert_eq!(dataset.shape.0, n_samples);
            prop_assert_eq!(dataset.shape.1, FEATURE_DIMENSION);
            prop_assert_eq!(dataset.features.len(), n_samples * FEATURE_DIMENSION);
            prop_assert_eq!(dataset.labels.len(), n_samples);
        }

        /// Property: category names has exactly NUM_CATEGORIES entries
        #[test]
        fn prop_category_names_count(_dummy in 0..1) {
            let names = FeatureExporter::category_names();
            prop_assert_eq!(names.len(), NUM_CATEGORIES);
        }

        /// Property: feature data is preserved in export
        #[test]
        fn prop_feature_preservation(
            files_changed in 0.0f32..1000.0,
            lines_added in 0.0f32..10000.0,
            lines_deleted in 0.0f32..5000.0,
        ) {
            let features = vec![CommitFeatures {
                defect_category: 0,
                files_changed,
                lines_added,
                lines_deleted,
                complexity_delta: 0.0,
                timestamp: 1700000000.0,
                hour_of_day: 12,
                day_of_week: 3,
            }];

            let matrix = FeatureExporter::to_matrix(&features).unwrap();

            prop_assert_eq!(matrix.get(0, 1), files_changed);
            prop_assert_eq!(matrix.get(0, 2), lines_added);
            prop_assert_eq!(matrix.get(0, 3), lines_deleted);
        }

        /// Property: export then to_aprender_matrix preserves data
        #[test]
        fn prop_export_to_matrix_roundtrip(n_samples in 1usize..20) {
            let exporter = FeatureExporter::default();

            let features: Vec<CommitFeatures> = (0..n_samples)
                .map(|i| CommitFeatures {
                    defect_category: (i % 18) as u8,
                    files_changed: (i + 1) as f32,
                    lines_added: (i * 10) as f32,
                    lines_deleted: (i * 5) as f32,
                    complexity_delta: 0.0,
                    timestamp: 1700000000.0,
                    hour_of_day: 12,
                    day_of_week: 3,
                })
                .collect();

            let categories: Vec<DefectCategory> = (0..n_samples)
                .map(|i| FeatureExporter::decode_label((i % 18) as u8).unwrap())
                .collect();

            let dataset = exporter.export(&features, &categories).unwrap();
            let matrix = FeatureExporter::to_aprender_matrix(&dataset).unwrap();

            prop_assert_eq!(matrix.n_rows(), n_samples);
            prop_assert_eq!(matrix.n_cols(), FEATURE_DIMENSION);

            // Verify first sample
            prop_assert_eq!(matrix.get(0, 1), 1.0); // files_changed for i=0
        }
    }
}

// Integration test module
#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Test round-trip: export → save → load → to_matrix
    #[test]
    fn test_full_roundtrip_json() {
        let exporter = FeatureExporter::new(ExportFormat::Json);

        // Create test data
        let features = vec![
            CommitFeatures {
                defect_category: 0,
                files_changed: 5.0,
                lines_added: 100.0,
                lines_deleted: 50.0,
                complexity_delta: 2.0,
                timestamp: 1700000000.0,
                hour_of_day: 14,
                day_of_week: 2,
            },
            CommitFeatures {
                defect_category: 7,
                files_changed: 3.0,
                lines_added: 75.0,
                lines_deleted: 25.0,
                complexity_delta: 1.0,
                timestamp: 1700000001.0,
                hour_of_day: 15,
                day_of_week: 2,
            },
            CommitFeatures {
                defect_category: 13,
                files_changed: 8.0,
                lines_added: 200.0,
                lines_deleted: 100.0,
                complexity_delta: 5.0,
                timestamp: 1700000002.0,
                hour_of_day: 16,
                day_of_week: 2,
            },
        ];

        let categories = vec![
            DefectCategory::MemorySafety,
            DefectCategory::SecurityVulnerabilities,
            DefectCategory::ASTTransform,
        ];

        // Export
        let dataset = exporter.export(&features, &categories).unwrap();

        // Save
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("roundtrip_test.json");
        exporter.save(&dataset, &path).unwrap();

        // Load
        let loaded = FeatureExporter::load(&path, ExportFormat::Json).unwrap();

        // Verify
        assert_eq!(loaded.shape, (3, FEATURE_DIMENSION));
        assert_eq!(loaded.labels, vec![0, 7, 13]);

        // Convert to matrix for training
        let matrix = FeatureExporter::to_aprender_matrix(&loaded).unwrap();
        assert_eq!(matrix.n_rows(), 3);
        assert_eq!(matrix.n_cols(), FEATURE_DIMENSION);

        // Verify specific values
        assert_eq!(matrix.get(0, 1), 5.0); // files_changed for first sample
        assert_eq!(matrix.get(1, 0), 7.0); // defect_category for second sample
        assert_eq!(matrix.get(2, 2), 200.0); // lines_added for third sample
    }

    /// Test that exported data can be used with aprender RandomForestClassifier
    #[test]
    fn test_aprender_training_compatibility() {
        use aprender::tree::RandomForestClassifier;

        let exporter = FeatureExporter::new(ExportFormat::Json);

        // Create diverse training data (need enough samples for RF)
        let mut features = Vec::new();
        let mut categories = Vec::new();

        for i in 0..30 {
            features.push(CommitFeatures {
                defect_category: (i % 3) as u8,
                files_changed: (i + 1) as f32,
                lines_added: (i * 10 + 5) as f32,
                lines_deleted: (i * 5) as f32,
                complexity_delta: (i % 5) as f32,
                timestamp: (1700000000 + i) as f64,
                hour_of_day: (9 + i % 8) as u8,
                day_of_week: (i % 5) as u8,
            });

            categories.push(match i % 3 {
                0 => DefectCategory::MemorySafety,
                1 => DefectCategory::ConcurrencyBugs,
                _ => DefectCategory::LogicErrors,
            });
        }

        // Export
        let dataset = exporter.export(&features, &categories).unwrap();

        // Convert to aprender format
        let matrix = FeatureExporter::to_aprender_matrix(&dataset).unwrap();
        let labels: Vec<usize> = dataset.labels.iter().map(|&l| l as usize).collect();

        // Train RandomForest (proves compatibility)
        let mut classifier = RandomForestClassifier::new(10);
        let result = classifier.fit(&matrix, &labels);

        assert!(result.is_ok(), "RandomForest training should succeed");

        // Predict on training data (sanity check)
        let predictions = classifier.predict(&matrix);
        assert_eq!(predictions.len(), 30);
    }
}
