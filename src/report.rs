// Report generation module
// Toyota Way: Start simple, deliver value

use crate::classifier::DefectCategory;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokio::fs;
use tracing::{debug, info};

/// Analysis metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisMetadata {
    pub organization: String,
    pub analysis_date: String,
    pub repositories_analyzed: usize,
    pub commits_analyzed: usize,
    pub analyzer_version: String,
}

/// Quality signals aggregated for a defect category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualitySignals {
    /// Average TDG score across all instances
    pub avg_tdg_score: Option<f32>,
    /// Maximum TDG score seen
    pub max_tdg_score: Option<f32>,
    /// Average cyclomatic complexity
    pub avg_complexity: Option<f32>,
    /// Average test coverage (0.0 to 1.0)
    pub avg_test_coverage: Option<f32>,
    /// Number of SATD (TODO/FIXME/HACK) markers found
    pub satd_instances: usize,
    /// Average lines changed per commit
    pub avg_lines_changed: f32,
    /// Number of files changed per commit on average
    pub avg_files_per_commit: f32,
}

impl Default for QualitySignals {
    fn default() -> Self {
        Self {
            avg_tdg_score: None,
            max_tdg_score: None,
            avg_complexity: None,
            avg_test_coverage: None,
            satd_instances: 0,
            avg_lines_changed: 0.0,
            avg_files_per_commit: 0.0,
        }
    }
}

/// Enhanced defect instance with quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefectInstance {
    pub commit_hash: String,
    pub message: String,
    pub author: String,
    pub timestamp: i64,
    /// Number of files affected
    pub files_affected: usize,
    /// Lines added in this commit
    pub lines_added: usize,
    /// Lines removed in this commit
    pub lines_removed: usize,
}

/// Defect pattern information
/// Represents aggregated statistics for a defect category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefectPattern {
    pub category: DefectCategory,
    pub frequency: usize,
    pub confidence: f32,
    /// Quality signals for this defect category
    pub quality_signals: QualitySignals,
    /// Enhanced examples with metrics
    pub examples: Vec<DefectInstance>,
}

/// Complete analysis report
/// Following specification Section 6: YAML Schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisReport {
    pub version: String,
    pub metadata: AnalysisMetadata,
    pub defect_patterns: Vec<DefectPattern>,
}

/// Report generator
/// Phase 1: Basic YAML output generation
pub struct ReportGenerator;

impl ReportGenerator {
    /// Create a new report generator
    ///
    /// # Examples
    /// ```
    /// use organizational_intelligence_plugin::report::ReportGenerator;
    ///
    /// let generator = ReportGenerator::new();
    /// ```
    pub fn new() -> Self {
        Self
    }

    /// Convert report to YAML string
    ///
    /// # Arguments
    /// * `report` - The analysis report to serialize
    ///
    /// # Errors
    /// Returns error if serialization fails
    ///
    /// # Examples
    /// ```
    /// use organizational_intelligence_plugin::report::{
    ///     ReportGenerator, AnalysisReport, AnalysisMetadata
    /// };
    ///
    /// let generator = ReportGenerator::new();
    /// let metadata = AnalysisMetadata {
    ///     organization: "test-org".to_string(),
    ///     analysis_date: "2025-11-15T00:00:00Z".to_string(),
    ///     repositories_analyzed: 10,
    ///     commits_analyzed: 100,
    ///     analyzer_version: "0.1.0".to_string(),
    /// };
    ///
    /// let report = AnalysisReport {
    ///     version: "1.0".to_string(),
    ///     metadata,
    ///     defect_patterns: vec![],
    /// };
    ///
    /// let yaml = generator.to_yaml(&report).expect("Should serialize");
    /// assert!(yaml.contains("version"));
    /// ```
    pub fn to_yaml(&self, report: &AnalysisReport) -> Result<String> {
        debug!("Serializing report to YAML");
        let yaml = serde_yaml::to_string(report)?;
        Ok(yaml)
    }

    /// Write report to file
    ///
    /// # Arguments
    /// * `report` - The analysis report to write
    /// * `path` - Path to output file
    ///
    /// # Errors
    /// Returns error if:
    /// - Serialization fails
    /// - File write fails
    /// - Path is invalid
    ///
    /// # Examples
    /// ```no_run
    /// use organizational_intelligence_plugin::report::{
    ///     ReportGenerator, AnalysisReport, AnalysisMetadata
    /// };
    /// use std::path::PathBuf;
    ///
    /// # async fn example() -> Result<(), anyhow::Error> {
    /// let generator = ReportGenerator::new();
    /// let metadata = AnalysisMetadata {
    ///     organization: "test-org".to_string(),
    ///     analysis_date: "2025-11-15T00:00:00Z".to_string(),
    ///     repositories_analyzed: 10,
    ///     commits_analyzed: 100,
    ///     analyzer_version: "0.1.0".to_string(),
    /// };
    ///
    /// let report = AnalysisReport {
    ///     version: "1.0".to_string(),
    ///     metadata,
    ///     defect_patterns: vec![],
    /// };
    ///
    /// generator.write_to_file(&report, &PathBuf::from("report.yaml")).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn write_to_file(&self, report: &AnalysisReport, path: &Path) -> Result<()> {
        info!("Writing report to file: {}", path.display());

        // Serialize to YAML
        let yaml = self.to_yaml(report)?;

        // Write to file
        fs::write(path, yaml).await?;

        info!("Successfully wrote report to {}", path.display());
        Ok(())
    }
}

impl Default for ReportGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_report_generator_creation() {
        let _generator = ReportGenerator::new();
        let _generator_default = ReportGenerator;
    }

    #[test]
    fn test_yaml_serialization() {
        let metadata = AnalysisMetadata {
            organization: "test-org".to_string(),
            analysis_date: "2025-11-15T00:00:00Z".to_string(),
            repositories_analyzed: 5,
            commits_analyzed: 50,
            analyzer_version: "0.1.0".to_string(),
        };

        let report = AnalysisReport {
            version: "1.0".to_string(),
            metadata,
            defect_patterns: vec![],
        };

        let generator = ReportGenerator::new();
        let yaml = generator.to_yaml(&report).expect("Should serialize");

        assert!(yaml.contains("version: '1.0'"));
        assert!(yaml.contains("organization: test-org"));
    }

    #[test]
    fn test_yaml_with_defect_patterns() {
        let metadata = AnalysisMetadata {
            organization: "test-org".to_string(),
            analysis_date: "2025-11-15T00:00:00Z".to_string(),
            repositories_analyzed: 10,
            commits_analyzed: 100,
            analyzer_version: "0.1.0".to_string(),
        };

        let patterns = vec![
            DefectPattern {
                category: DefectCategory::MemorySafety,
                frequency: 42,
                confidence: 0.85,
                quality_signals: QualitySignals {
                    avg_lines_changed: 45.2,
                    avg_files_per_commit: 2.1,
                    ..Default::default()
                },
                examples: vec![DefectInstance {
                    commit_hash: "abc123".to_string(),
                    message: "fix memory leak".to_string(),
                    author: "test@example.com".to_string(),
                    timestamp: 1234567890,
                    files_affected: 2,
                    lines_added: 30,
                    lines_removed: 15,
                }],
            },
            DefectPattern {
                category: DefectCategory::ConcurrencyBugs,
                frequency: 30,
                confidence: 0.80,
                quality_signals: QualitySignals {
                    avg_lines_changed: 67.3,
                    avg_files_per_commit: 3.5,
                    ..Default::default()
                },
                examples: vec![DefectInstance {
                    commit_hash: "def456".to_string(),
                    message: "fix race condition".to_string(),
                    author: "test@example.com".to_string(),
                    timestamp: 1234567891,
                    files_affected: 4,
                    lines_added: 50,
                    lines_removed: 17,
                }],
            },
        ];

        let report = AnalysisReport {
            version: "1.0".to_string(),
            metadata,
            defect_patterns: patterns,
        };

        let generator = ReportGenerator::new();
        let yaml = generator.to_yaml(&report).expect("Should serialize");

        assert!(yaml.contains("MemorySafety"));
        assert!(yaml.contains("ConcurrencyBugs"));
        assert!(yaml.contains("frequency: 42"));
    }

    #[tokio::test]
    async fn test_write_to_file() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let report_path = temp_dir.path().join("test-report.yaml");

        let metadata = AnalysisMetadata {
            organization: "test-org".to_string(),
            analysis_date: "2025-11-15T00:00:00Z".to_string(),
            repositories_analyzed: 5,
            commits_analyzed: 50,
            analyzer_version: "0.1.0".to_string(),
        };

        let report = AnalysisReport {
            version: "1.0".to_string(),
            metadata,
            defect_patterns: vec![],
        };

        let generator = ReportGenerator::new();
        generator
            .write_to_file(&report, &report_path)
            .await
            .expect("Should write file");

        assert!(report_path.exists());

        let content = tokio::fs::read_to_string(&report_path).await.unwrap();
        assert!(content.contains("test-org"));
    }

    #[test]
    fn test_quality_signals_default() {
        let signals = QualitySignals::default();
        assert!(signals.avg_tdg_score.is_none());
        assert!(signals.avg_complexity.is_none());
        assert_eq!(signals.satd_instances, 0);
        assert_eq!(signals.avg_lines_changed, 0.0);
    }

    #[test]
    fn test_quality_signals_with_values() {
        let signals = QualitySignals {
            avg_tdg_score: Some(2.5),
            max_tdg_score: Some(5.0),
            avg_complexity: Some(8.3),
            avg_test_coverage: Some(0.75),
            satd_instances: 10,
            avg_lines_changed: 50.5,
            avg_files_per_commit: 3.2,
        };

        assert_eq!(signals.avg_tdg_score, Some(2.5));
        assert_eq!(signals.max_tdg_score, Some(5.0));
        assert_eq!(signals.satd_instances, 10);
    }

    #[test]
    fn test_defect_instance_structure() {
        let instance = DefectInstance {
            commit_hash: "abc123".to_string(),
            message: "fix bug".to_string(),
            author: "dev@example.com".to_string(),
            timestamp: 1234567890,
            files_affected: 3,
            lines_added: 25,
            lines_removed: 10,
        };

        assert_eq!(instance.commit_hash, "abc123");
        assert_eq!(instance.files_affected, 3);
        assert_eq!(instance.lines_added, 25);
    }

    #[test]
    fn test_defect_pattern_structure() {
        let pattern = DefectPattern {
            category: DefectCategory::LogicErrors,
            frequency: 15,
            confidence: 0.70,
            quality_signals: QualitySignals::default(),
            examples: vec![],
        };

        assert_eq!(pattern.frequency, 15);
        assert_eq!(pattern.confidence, 0.70);
        assert!(pattern.examples.is_empty());
    }

    #[test]
    fn test_analysis_metadata_structure() {
        let metadata = AnalysisMetadata {
            organization: "my-org".to_string(),
            analysis_date: "2025-11-24T12:00:00Z".to_string(),
            repositories_analyzed: 20,
            commits_analyzed: 500,
            analyzer_version: "0.2.0".to_string(),
        };

        assert_eq!(metadata.organization, "my-org");
        assert_eq!(metadata.repositories_analyzed, 20);
        assert_eq!(metadata.commits_analyzed, 500);
    }

    #[test]
    fn test_report_serialization_deserialization() {
        let metadata = AnalysisMetadata {
            organization: "test".to_string(),
            analysis_date: "2025-01-01T00:00:00Z".to_string(),
            repositories_analyzed: 1,
            commits_analyzed: 10,
            analyzer_version: "0.1.0".to_string(),
        };

        let report = AnalysisReport {
            version: "1.0".to_string(),
            metadata,
            defect_patterns: vec![],
        };

        let json = serde_json::to_string(&report).unwrap();
        let deserialized: AnalysisReport = serde_json::from_str(&json).unwrap();

        assert_eq!(report.version, deserialized.version);
        assert_eq!(
            report.metadata.organization,
            deserialized.metadata.organization
        );
    }

    #[test]
    fn test_report_generator_default() {
        let generator = ReportGenerator;
        let metadata = AnalysisMetadata {
            organization: "test".to_string(),
            analysis_date: "2025-01-01T00:00:00Z".to_string(),
            repositories_analyzed: 1,
            commits_analyzed: 1,
            analyzer_version: "0.1.0".to_string(),
        };

        let report = AnalysisReport {
            version: "1.0".to_string(),
            metadata,
            defect_patterns: vec![],
        };

        let yaml = generator.to_yaml(&report).expect("Should serialize");
        assert!(yaml.contains("version"));
    }

    #[test]
    fn test_empty_defect_patterns() {
        let metadata = AnalysisMetadata {
            organization: "empty-org".to_string(),
            analysis_date: "2025-01-01T00:00:00Z".to_string(),
            repositories_analyzed: 0,
            commits_analyzed: 0,
            analyzer_version: "0.1.0".to_string(),
        };

        let report = AnalysisReport {
            version: "1.0".to_string(),
            metadata,
            defect_patterns: vec![],
        };

        let generator = ReportGenerator::new();
        let yaml = generator.to_yaml(&report).expect("Should serialize");

        assert!(yaml.contains("defect_patterns: []"));
    }
}
