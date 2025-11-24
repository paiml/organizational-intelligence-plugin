// src/summarizer.rs
//! Summarization module for converting organizational reports into AI-friendly summaries
//!
//! This module provides automated PII stripping and pattern extraction to eliminate
//! manual waste (Toyota Way: Muda reduction).

use crate::report::{AnalysisReport, DefectPattern};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Configuration for summarization behavior
#[derive(Debug, Clone)]
pub struct SummaryConfig {
    /// Strip PII (author names, commit hashes, email addresses)
    pub strip_pii: bool,
    /// Show only top N defect categories by frequency
    pub top_n_categories: usize,
    /// Filter out categories with frequency below this threshold
    pub min_frequency: usize,
    /// Include anonymized examples (with PII removed)
    pub include_examples: bool,
}

impl Default for SummaryConfig {
    fn default() -> Self {
        Self {
            strip_pii: true,
            top_n_categories: 10,
            min_frequency: 5,
            include_examples: false,
        }
    }
}

/// Quality thresholds for code assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    pub tdg_minimum: f32,
    pub test_coverage_minimum: f32,
    pub max_function_length: usize,
    pub max_cyclomatic_complexity: usize,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            tdg_minimum: 85.0,
            test_coverage_minimum: 0.85,
            max_function_length: 50,
            max_cyclomatic_complexity: 10,
        }
    }
}

/// Metadata about the summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryMetadata {
    pub analysis_date: String,
    pub repositories_analyzed: usize,
    pub commits_analyzed: usize,
}

/// Summarized organizational intelligence for AI consumption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Summary {
    pub organizational_insights: OrganizationalInsights,
    pub code_quality_thresholds: QualityThresholds,
    pub metadata: SummaryMetadata,
}

impl Summary {
    /// Find a defect category by name
    pub fn find_category(&self, category_name: &str) -> Option<DefectPatternSummary> {
        self.organizational_insights
            .top_defect_categories
            .iter()
            .find(|p| p.category.to_string() == category_name)
            .map(|p| DefectPatternSummary {
                category: p.category.to_string(),
                frequency: p.frequency,
                confidence: p.confidence,
                avg_tdg_score: p.quality_signals.avg_tdg_score.unwrap_or(0.0),
                common_patterns: Vec::new(), // Not stored in summary
                prevention_strategies: Vec::new(), // Not stored in summary
            })
    }

    /// Load summary from file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let summary: Summary = serde_yaml::from_str(&content)?;
        Ok(summary)
    }
}

/// Top-level container for defect patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganizationalInsights {
    pub top_defect_categories: Vec<DefectPattern>,
}

/// Simplified defect pattern for PR review (without examples)
#[derive(Debug, Clone, PartialEq)]
pub struct DefectPatternSummary {
    pub category: String,
    pub frequency: usize,
    pub confidence: f32,
    pub avg_tdg_score: f32,
    pub common_patterns: Vec<String>,
    pub prevention_strategies: Vec<String>,
}

/// Summarize organizational analysis reports
pub struct ReportSummarizer;

impl ReportSummarizer {
    /// Summarize a full organizational report according to config
    pub fn summarize<P: AsRef<Path>>(input: P, config: SummaryConfig) -> Result<Summary> {
        // Load full report
        let content = std::fs::read_to_string(input)?;
        let report: AnalysisReport = serde_yaml::from_str(&content)?;

        // Filter and sort patterns by frequency
        let mut patterns: Vec<DefectPattern> = report
            .defect_patterns
            .into_iter()
            .filter(|p| p.frequency >= config.min_frequency)
            .collect();

        // Sort by frequency descending
        patterns.sort_by(|a, b| b.frequency.cmp(&a.frequency));

        // Take top N
        patterns.truncate(config.top_n_categories);

        // Strip PII if requested
        if config.strip_pii {
            Self::strip_pii_from_patterns(&mut patterns);
        }

        // Remove examples unless explicitly requested
        if !config.include_examples {
            for pattern in &mut patterns {
                pattern.examples.clear();
            }
        }

        // Build summary
        Ok(Summary {
            organizational_insights: OrganizationalInsights {
                top_defect_categories: patterns,
            },
            code_quality_thresholds: QualityThresholds::default(),
            metadata: SummaryMetadata {
                analysis_date: report.metadata.analysis_date,
                repositories_analyzed: report.metadata.repositories_analyzed,
                commits_analyzed: report.metadata.commits_analyzed,
            },
        })
    }

    /// Strip PII from defect patterns (author, commit hash, email)
    fn strip_pii_from_patterns(patterns: &mut [DefectPattern]) {
        for pattern in patterns {
            for example in &mut pattern.examples {
                // Clear PII fields
                example.commit_hash = "REDACTED".to_string();
                example.author = "REDACTED".to_string();
            }
        }
    }

    /// Save summary to YAML file
    pub fn save_to_file<P: AsRef<Path>>(summary: &Summary, output: P) -> Result<()> {
        let yaml = serde_yaml::to_string(summary)?;
        std::fs::write(output, yaml)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classifier::DefectCategory;
    use crate::report::{AnalysisMetadata, DefectInstance, QualitySignals};
    use tempfile::NamedTempFile;

    fn create_test_report() -> AnalysisReport {
        AnalysisReport {
            version: "1.0".to_string(),
            metadata: AnalysisMetadata {
                organization: "test-org".to_string(),
                analysis_date: "2025-11-15T12:00:00Z".to_string(),
                repositories_analyzed: 10,
                commits_analyzed: 1000,
                analyzer_version: "0.1.0".to_string(),
            },
            defect_patterns: vec![
                DefectPattern {
                    category: DefectCategory::ConfigurationErrors,
                    frequency: 25,
                    confidence: 0.85,
                    quality_signals: QualitySignals {
                        avg_tdg_score: Some(45.2),
                        max_tdg_score: Some(60.0),
                        avg_complexity: None,
                        avg_test_coverage: None,
                        satd_instances: 0,
                        avg_lines_changed: 50.0,
                        avg_files_per_commit: 3.0,
                    },
                    examples: vec![DefectInstance {
                        commit_hash: "abc123".to_string(),
                        message: "fix config bug".to_string(),
                        author: "test@example.com".to_string(),
                        timestamp: 1731662400,
                        files_affected: 3,
                        lines_added: 50,
                        lines_removed: 10,
                    }],
                },
                DefectPattern {
                    category: DefectCategory::TypeErrors,
                    frequency: 3,
                    confidence: 0.90,
                    quality_signals: QualitySignals {
                        avg_tdg_score: Some(95.0),
                        max_tdg_score: Some(98.0),
                        avg_complexity: None,
                        avg_test_coverage: None,
                        satd_instances: 0,
                        avg_lines_changed: 10.0,
                        avg_files_per_commit: 1.0,
                    },
                    examples: vec![],
                },
                DefectPattern {
                    category: DefectCategory::SecurityVulnerabilities,
                    frequency: 15,
                    confidence: 0.80,
                    quality_signals: QualitySignals {
                        avg_tdg_score: Some(55.0),
                        max_tdg_score: Some(70.0),
                        avg_complexity: None,
                        avg_test_coverage: None,
                        satd_instances: 0,
                        avg_lines_changed: 30.0,
                        avg_files_per_commit: 2.0,
                    },
                    examples: vec![],
                },
            ],
        }
    }

    #[test]
    fn test_pii_stripping_removes_sensitive_data() {
        let report = create_test_report();
        let temp_file = NamedTempFile::new().unwrap();
        let report_path = temp_file.path();

        // Save test report
        let yaml = serde_yaml::to_string(&report).unwrap();
        std::fs::write(report_path, yaml).unwrap();

        // Summarize with PII stripping
        let config = SummaryConfig {
            strip_pii: true,
            ..Default::default()
        };
        let summary = ReportSummarizer::summarize(report_path, config).unwrap();

        // Verify PII is stripped
        for pattern in &summary.organizational_insights.top_defect_categories {
            for example in &pattern.examples {
                assert_eq!(example.commit_hash, "REDACTED");
                assert_eq!(example.author, "REDACTED");
            }
        }
    }

    #[test]
    fn test_frequency_filtering() {
        let report = create_test_report();
        let temp_file = NamedTempFile::new().unwrap();
        let report_path = temp_file.path();

        let yaml = serde_yaml::to_string(&report).unwrap();
        std::fs::write(report_path, yaml).unwrap();

        // Filter out defects with frequency < 10
        let config = SummaryConfig {
            min_frequency: 10,
            ..Default::default()
        };
        let summary = ReportSummarizer::summarize(report_path, config).unwrap();

        // Should only have ConfigurationErrors (25) and SecurityVulnerabilities (15)
        assert_eq!(
            summary.organizational_insights.top_defect_categories.len(),
            2
        );

        let categories: Vec<String> = summary
            .organizational_insights
            .top_defect_categories
            .iter()
            .map(|p| p.category.to_string())
            .collect();

        assert!(categories.contains(&"ConfigurationErrors".to_string()));
        assert!(categories.contains(&"SecurityVulnerabilities".to_string()));
        assert!(!categories.contains(&"TypeErrors".to_string()));
    }

    #[test]
    fn test_top_n_selection() {
        let report = create_test_report();
        let temp_file = NamedTempFile::new().unwrap();
        let report_path = temp_file.path();

        let yaml = serde_yaml::to_string(&report).unwrap();
        std::fs::write(report_path, yaml).unwrap();

        // Only take top 2
        let config = SummaryConfig {
            top_n_categories: 2,
            min_frequency: 0,
            ..Default::default()
        };
        let summary = ReportSummarizer::summarize(report_path, config).unwrap();

        assert_eq!(
            summary.organizational_insights.top_defect_categories.len(),
            2
        );

        // Should be sorted by frequency
        assert_eq!(
            summary.organizational_insights.top_defect_categories[0].frequency,
            25
        ); // ConfigurationErrors
        assert_eq!(
            summary.organizational_insights.top_defect_categories[1].frequency,
            15
        ); // SecurityVulnerabilities
    }

    #[test]
    fn test_examples_removed_by_default() {
        let report = create_test_report();
        let temp_file = NamedTempFile::new().unwrap();
        let report_path = temp_file.path();

        let yaml = serde_yaml::to_string(&report).unwrap();
        std::fs::write(report_path, yaml).unwrap();

        let config = SummaryConfig::default();
        let summary = ReportSummarizer::summarize(report_path, config).unwrap();

        // Examples should be empty by default
        for pattern in &summary.organizational_insights.top_defect_categories {
            assert!(pattern.examples.is_empty());
        }
    }

    #[test]
    fn test_roundtrip_save_and_load() {
        let report = create_test_report();
        let report_file = NamedTempFile::new().unwrap();
        let summary_file = NamedTempFile::new().unwrap();

        // Save report
        let yaml = serde_yaml::to_string(&report).unwrap();
        std::fs::write(report_file.path(), yaml).unwrap();

        // Summarize
        let config = SummaryConfig::default();
        let summary = ReportSummarizer::summarize(report_file.path(), config).unwrap();

        // Save summary
        ReportSummarizer::save_to_file(&summary, summary_file.path()).unwrap();

        // Load summary back
        let loaded_yaml = std::fs::read_to_string(summary_file.path()).unwrap();
        let loaded_summary: Summary = serde_yaml::from_str(&loaded_yaml).unwrap();

        // Verify metadata
        assert_eq!(loaded_summary.metadata.repositories_analyzed, 10);
        assert_eq!(loaded_summary.metadata.commits_analyzed, 1000);
    }

    #[test]
    fn test_summary_config_default() {
        let config = SummaryConfig::default();
        assert!(config.strip_pii);
        assert_eq!(config.top_n_categories, 10);
        assert_eq!(config.min_frequency, 5);
        assert!(!config.include_examples);
    }

    #[test]
    fn test_quality_thresholds_default() {
        let thresholds = QualityThresholds::default();
        assert_eq!(thresholds.tdg_minimum, 85.0);
        assert_eq!(thresholds.test_coverage_minimum, 0.85);
        assert_eq!(thresholds.max_function_length, 50);
        assert_eq!(thresholds.max_cyclomatic_complexity, 10);
    }

    #[test]
    fn test_summary_find_category() {
        let report = create_test_report();
        let temp_file = NamedTempFile::new().unwrap();
        let report_path = temp_file.path();

        let yaml = serde_yaml::to_string(&report).unwrap();
        std::fs::write(report_path, yaml).unwrap();

        let config = SummaryConfig::default();
        let summary = ReportSummarizer::summarize(report_path, config).unwrap();

        // Find existing category
        let found = summary.find_category("ConfigurationErrors");
        assert!(found.is_some());
        let category = found.unwrap();
        assert_eq!(category.category, "ConfigurationErrors");
        assert_eq!(category.frequency, 25);
        assert_eq!(category.avg_tdg_score, 45.2);

        // Find non-existent category
        assert!(summary.find_category("NonExistent").is_none());
    }

    #[test]
    fn test_summary_from_file() {
        let report = create_test_report();
        let report_file = NamedTempFile::new().unwrap();
        let summary_file = NamedTempFile::new().unwrap();

        // Create and save summary
        let yaml = serde_yaml::to_string(&report).unwrap();
        std::fs::write(report_file.path(), yaml).unwrap();

        let config = SummaryConfig::default();
        let summary = ReportSummarizer::summarize(report_file.path(), config).unwrap();
        ReportSummarizer::save_to_file(&summary, summary_file.path()).unwrap();

        // Load using Summary::from_file
        let loaded = Summary::from_file(summary_file.path()).unwrap();
        assert_eq!(loaded.metadata.repositories_analyzed, 10);
        assert_eq!(loaded.metadata.commits_analyzed, 1000);
    }

    #[test]
    fn test_include_examples_config() {
        let report = create_test_report();
        let temp_file = NamedTempFile::new().unwrap();
        let report_path = temp_file.path();

        let yaml = serde_yaml::to_string(&report).unwrap();
        std::fs::write(report_path, yaml).unwrap();

        // Summarize with examples included
        let config = SummaryConfig {
            include_examples: true,
            strip_pii: false,
            ..Default::default()
        };
        let summary = ReportSummarizer::summarize(report_path, config).unwrap();

        // ConfigurationErrors has 1 example
        let config_pattern = summary
            .organizational_insights
            .top_defect_categories
            .iter()
            .find(|p| p.category.to_string() == "ConfigurationErrors")
            .unwrap();

        assert_eq!(config_pattern.examples.len(), 1);
        assert_eq!(config_pattern.examples[0].commit_hash, "abc123");
        assert_eq!(config_pattern.examples[0].author, "test@example.com");
    }

    #[test]
    fn test_defect_pattern_summary_equality() {
        let summary1 = DefectPatternSummary {
            category: "MemorySafety".to_string(),
            frequency: 10,
            confidence: 0.85,
            avg_tdg_score: 70.0,
            common_patterns: vec!["use-after-free".to_string()],
            prevention_strategies: vec!["Use smart pointers".to_string()],
        };

        let summary2 = summary1.clone();
        assert_eq!(summary1, summary2);
    }

    #[test]
    fn test_summary_metadata_serialization() {
        let metadata = SummaryMetadata {
            analysis_date: "2025-11-24".to_string(),
            repositories_analyzed: 5,
            commits_analyzed: 500,
        };

        let yaml = serde_yaml::to_string(&metadata).unwrap();
        let deserialized: SummaryMetadata = serde_yaml::from_str(&yaml).unwrap();

        assert_eq!(deserialized.analysis_date, "2025-11-24");
        assert_eq!(deserialized.repositories_analyzed, 5);
        assert_eq!(deserialized.commits_analyzed, 500);
    }

    #[test]
    fn test_organizational_insights_serialization() {
        let insights = OrganizationalInsights {
            top_defect_categories: vec![],
        };

        let yaml = serde_yaml::to_string(&insights).unwrap();
        let deserialized: OrganizationalInsights = serde_yaml::from_str(&yaml).unwrap();

        assert!(deserialized.top_defect_categories.is_empty());
    }

    #[test]
    fn test_no_pii_stripping_when_disabled() {
        let report = create_test_report();
        let temp_file = NamedTempFile::new().unwrap();
        let report_path = temp_file.path();

        let yaml = serde_yaml::to_string(&report).unwrap();
        std::fs::write(report_path, yaml).unwrap();

        // Summarize without PII stripping
        let config = SummaryConfig {
            strip_pii: false,
            include_examples: true,
            ..Default::default()
        };
        let summary = ReportSummarizer::summarize(report_path, config).unwrap();

        // Verify PII is NOT stripped
        let config_pattern = summary
            .organizational_insights
            .top_defect_categories
            .iter()
            .find(|p| p.category.to_string() == "ConfigurationErrors")
            .unwrap();

        assert_eq!(config_pattern.examples[0].commit_hash, "abc123");
        assert_eq!(config_pattern.examples[0].author, "test@example.com");
    }
}
