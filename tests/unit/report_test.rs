// Unit tests for Report Generator
// Following EXTREME TDD: Tests first

use organizational_intelligence_plugin::report::{
    AnalysisMetadata, AnalysisReport, DefectPattern, ReportGenerator,
};
use std::path::PathBuf;
use tempfile::NamedTempFile;

#[test]
fn test_report_generator_can_be_created() {
    // RED: This will fail until we implement ReportGenerator
    let _generator = ReportGenerator::new();
}

#[test]
fn test_analysis_report_structure() {
    // Test that AnalysisReport structure exists
    let metadata = AnalysisMetadata {
        organization: "test-org".to_string(),
        analysis_date: "2025-11-15T00:00:00Z".to_string(),
        repositories_analyzed: 10,
        commits_analyzed: 100,
        analyzer_version: "0.1.0".to_string(),
    };

    let report = AnalysisReport {
        version: "1.0".to_string(),
        metadata,
        defect_patterns: vec![],
    };

    assert_eq!(report.version, "1.0");
    assert_eq!(report.metadata.organization, "test-org");
}

#[test]
fn test_report_generator_to_yaml() {
    // RED: Test YAML serialization
    let generator = ReportGenerator::new();

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

    let yaml = generator.to_yaml(&report).expect("Should serialize to YAML");

    // Verify YAML contains expected fields
    assert!(yaml.contains("version"));
    assert!(yaml.contains("test-org"));
    assert!(yaml.contains("repositories_analyzed"));
}

#[tokio::test]
async fn test_report_generator_write_to_file() {
    // RED: Test writing report to file
    let generator = ReportGenerator::new();

    let metadata = AnalysisMetadata {
        organization: "test-org".to_string(),
        analysis_date: "2025-11-15T00:00:00Z".to_string(),
        repositories_analyzed: 3,
        commits_analyzed: 30,
        analyzer_version: "0.1.0".to_string(),
    };

    let report = AnalysisReport {
        version: "1.0".to_string(),
        metadata,
        defect_patterns: vec![],
    };

    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp_file.path().to_path_buf();

    generator
        .write_to_file(&report, &path)
        .await
        .expect("Should write report to file");

    // Verify file was created and contains YAML
    let content = tokio::fs::read_to_string(&path)
        .await
        .expect("Should read file");
    assert!(content.contains("version"));
    assert!(content.contains("test-org"));
}

#[test]
fn test_defect_pattern_structure() {
    // Test DefectPattern structure
    let pattern = DefectPattern {
        category: "Memory Safety".to_string(),
        frequency: 42,
        percentage: 15.5,
    };

    assert_eq!(pattern.category, "Memory Safety");
    assert_eq!(pattern.frequency, 42);
    assert_eq!(pattern.percentage, 15.5);
}

#[test]
fn test_empty_report_is_valid() {
    // Empty report should be valid
    let metadata = AnalysisMetadata {
        organization: "empty-org".to_string(),
        analysis_date: "2025-11-15T00:00:00Z".to_string(),
        repositories_analyzed: 0,
        commits_analyzed: 0,
        analyzer_version: "0.1.0".to_string(),
    };

    let report = AnalysisReport {
        version: "1.0".to_string(),
        metadata,
        defect_patterns: vec![],
    };

    assert_eq!(report.defect_patterns.len(), 0);
    assert_eq!(report.metadata.repositories_analyzed, 0);
}
