// Report generation module
// Toyota Way: Start simple, deliver value

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

/// Defect pattern information
/// Phase 1: Basic structure for future classifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefectPattern {
    pub category: String,
    pub frequency: usize,
    pub percentage: f64,
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
                category: "Memory Safety".to_string(),
                frequency: 42,
                percentage: 35.0,
            },
            DefectPattern {
                category: "Concurrency".to_string(),
                frequency: 30,
                percentage: 25.0,
            },
        ];

        let report = AnalysisReport {
            version: "1.0".to_string(),
            metadata,
            defect_patterns: patterns,
        };

        let generator = ReportGenerator::new();
        let yaml = generator.to_yaml(&report).expect("Should serialize");

        assert!(yaml.contains("Memory Safety"));
        assert!(yaml.contains("Concurrency"));
        assert!(yaml.contains("frequency: 42"));
    }
}
