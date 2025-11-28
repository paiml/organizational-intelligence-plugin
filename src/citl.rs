//! Compiler-in-the-Loop (CITL) Integration Module
//!
//! NLP-014: Integrates Depyler's CITL diagnostic output as ground-truth training labels.
//!
//! Provides:
//! - rustc error code → DefectCategory mapping
//! - Clippy lint → DefectCategory mapping
//! - Depyler export import functionality
//! - Extended training example support

use crate::classifier::DefectCategory;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Error code class for feature extraction (Section 3.4)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ErrorCodeClass {
    Type = 0,
    Borrow = 1,
    Name = 2,
    Trait = 3,
    #[default]
    Other = 4,
}

impl ErrorCodeClass {
    pub fn as_u8(&self) -> u8 {
        *self as u8
    }
}

/// Suggestion applicability from rustc/clippy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum SuggestionApplicability {
    #[default]
    None = 0,
    MachineApplicable = 1,
    MaybeIncorrect = 2,
    HasPlaceholders = 3,
}

impl SuggestionApplicability {
    pub fn as_u8(&self) -> u8 {
        *self as u8
    }

    pub fn parse(s: &str) -> Self {
        match s {
            "MachineApplicable" => Self::MachineApplicable,
            "MaybeIncorrect" => Self::MaybeIncorrect,
            "HasPlaceholders" => Self::HasPlaceholders,
            _ => Self::None,
        }
    }
}

/// Source of a training example
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum TrainingSource {
    #[default]
    CommitMessage,
    DepylerCitl,
    Manual,
}

/// Depyler CITL export record (Section 3.2)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepylerExport {
    pub source_file: String,
    pub error_code: Option<String>,
    pub clippy_lint: Option<String>,
    pub level: String,
    pub message: String,
    pub oip_category: Option<String>,
    pub confidence: f32,
    pub span: Option<SpanInfo>,
    pub suggestion: Option<SuggestionInfo>,
    pub timestamp: i64,
    pub depyler_version: String,
}

/// Span information for diagnostic location
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpanInfo {
    pub line_start: u32,
    pub column_start: u32,
}

/// Suggestion information from compiler
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SuggestionInfo {
    pub replacement: String,
    pub applicability: String,
}

/// Import statistics
#[derive(Debug, Clone, Default)]
pub struct ImportStats {
    pub total_records: usize,
    pub imported: usize,
    pub skipped_low_confidence: usize,
    pub skipped_unknown_category: usize,
    pub by_category: HashMap<DefectCategory, usize>,
    pub by_source: HashMap<String, usize>,
    pub avg_confidence: f32,
}

/// Confidence values for error code mappings (Appendix A)
pub fn get_error_code_confidence(code: &str) -> f32 {
    match code {
        "E0308" | "E0277" => 0.95,
        "E0502" | "E0503" | "E0505" => 0.95,
        "E0382" | "E0507" => 0.90,
        "E0425" | "E0433" | "E0412" => 0.85,
        "E0599" | "E0614" | "E0615" => 0.80,
        "E0658" => 0.75,
        _ => 0.70,
    }
}

/// Map rustc error code to OIP DefectCategory (Section 3.1)
///
/// # Arguments
/// * `code` - Rustc error code (e.g., "E0308")
///
/// # Returns
/// * `Some(DefectCategory)` if mapping exists
/// * `None` for unknown codes
///
/// # Examples
/// ```
/// use organizational_intelligence_plugin::citl::rustc_to_defect_category;
/// use organizational_intelligence_plugin::classifier::DefectCategory;
///
/// assert_eq!(rustc_to_defect_category("E0308"), Some(DefectCategory::TypeErrors));
/// assert_eq!(rustc_to_defect_category("E0277"), Some(DefectCategory::TraitBounds));
/// assert_eq!(rustc_to_defect_category("UNKNOWN"), None);
/// ```
pub fn rustc_to_defect_category(code: &str) -> Option<DefectCategory> {
    match code {
        // Type system
        "E0308" => Some(DefectCategory::TypeErrors),
        "E0412" => Some(DefectCategory::TypeAnnotationGaps),

        // Ownership/borrowing
        "E0502" | "E0503" | "E0505" => Some(DefectCategory::OwnershipBorrow),
        "E0382" | "E0507" => Some(DefectCategory::MemorySafety),

        // Traits
        "E0277" => Some(DefectCategory::TraitBounds),

        // Name resolution
        "E0425" | "E0433" => Some(DefectCategory::StdlibMapping),

        // AST/structure
        "E0599" | "E0615" => Some(DefectCategory::ASTTransform),
        "E0614" => Some(DefectCategory::OperatorPrecedence),

        // Configuration
        "E0658" => Some(DefectCategory::ConfigurationErrors),

        _ => None,
    }
}

/// Map Clippy lint to OIP DefectCategory (Section 3.1)
///
/// # Arguments
/// * `lint` - Clippy lint name (e.g., "clippy::unwrap_used")
///
/// # Returns
/// * `Some(DefectCategory)` if mapping exists
/// * `None` for unknown lints
///
/// # Examples
/// ```
/// use organizational_intelligence_plugin::citl::clippy_to_defect_category;
/// use organizational_intelligence_plugin::classifier::DefectCategory;
///
/// assert_eq!(clippy_to_defect_category("clippy::unwrap_used"), Some(DefectCategory::ApiMisuse));
/// assert_eq!(clippy_to_defect_category("clippy::todo"), Some(DefectCategory::LogicErrors));
/// assert_eq!(clippy_to_defect_category("clippy::unknown"), None);
/// ```
pub fn clippy_to_defect_category(lint: &str) -> Option<DefectCategory> {
    match lint {
        "clippy::unwrap_used" | "clippy::expect_used" | "clippy::panic" => {
            Some(DefectCategory::ApiMisuse)
        }
        "clippy::todo" | "clippy::unreachable" => Some(DefectCategory::LogicErrors),
        "clippy::cognitive_complexity" => Some(DefectCategory::PerformanceIssues),
        "clippy::too_many_arguments" | "clippy::match_single_binding" => {
            Some(DefectCategory::ASTTransform)
        }
        "clippy::needless_collect" => Some(DefectCategory::IteratorChain),
        "clippy::manual_map" => Some(DefectCategory::ComprehensionBugs),
        _ => None,
    }
}

/// Get error code class for feature extraction (Section 3.4)
pub fn get_error_code_class(code: &str) -> ErrorCodeClass {
    match code {
        // Type errors
        "E0308" | "E0412" => ErrorCodeClass::Type,
        // Borrow errors
        "E0502" | "E0503" | "E0505" | "E0382" | "E0507" => ErrorCodeClass::Borrow,
        // Name resolution
        "E0425" | "E0433" => ErrorCodeClass::Name,
        // Trait errors
        "E0277" => ErrorCodeClass::Trait,
        // Other
        _ => ErrorCodeClass::Other,
    }
}

/// Import Depyler CITL corpus from JSONL file
///
/// # Arguments
/// * `path` - Path to JSONL export file
/// * `min_confidence` - Minimum confidence threshold
///
/// # Returns
/// * `Ok((Vec<DepylerExport>, ImportStats))` on success
pub fn import_depyler_corpus<P: AsRef<Path>>(
    path: P,
    min_confidence: f32,
) -> Result<(Vec<DepylerExport>, ImportStats)> {
    let content = std::fs::read_to_string(path.as_ref())
        .map_err(|e| anyhow!("Failed to read corpus file: {}", e))?;

    let mut exports = Vec::new();
    let mut stats = ImportStats::default();

    for (line_num, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }

        stats.total_records += 1;

        let export: DepylerExport = serde_json::from_str(line).map_err(|e| {
            anyhow!(
                "Failed to parse JSON at line {}: {} - content: {}",
                line_num + 1,
                e,
                line
            )
        })?;

        // Check confidence threshold
        if export.confidence < min_confidence {
            stats.skipped_low_confidence += 1;
            continue;
        }

        // Resolve category
        let category = resolve_category(&export);
        if category.is_none() {
            stats.skipped_unknown_category += 1;
            continue;
        }

        let cat = category.unwrap();
        *stats.by_category.entry(cat).or_insert(0) += 1;
        *stats
            .by_source
            .entry(export.source_file.clone())
            .or_insert(0) += 1;

        stats.imported += 1;
        exports.push(export);
    }

    // Calculate average confidence
    if stats.imported > 0 {
        stats.avg_confidence =
            exports.iter().map(|e| e.confidence).sum::<f32>() / stats.imported as f32;
    }

    Ok((exports, stats))
}

/// Convert DepylerExport records to TrainingExamples (NLP-014)
///
/// # Arguments
/// * `exports` - Vector of DepylerExport records
///
/// # Returns
/// * Vector of TrainingExamples with CITL source
pub fn convert_to_training_examples(
    exports: &[DepylerExport],
) -> Vec<crate::training::TrainingExample> {
    exports
        .iter()
        .filter_map(|export| {
            let category = resolve_category(export)?;
            let suggestion_applicability = export
                .suggestion
                .as_ref()
                .map(|s| SuggestionApplicability::parse(&s.applicability));

            Some(crate::training::TrainingExample {
                message: export.message.clone(),
                label: category,
                confidence: export.confidence,
                commit_hash: String::new(), // CITL doesn't have commit hash
                author: "depyler".to_string(),
                timestamp: export.timestamp,
                lines_added: 0,
                lines_removed: 0,
                files_changed: 1,
                // NLP-014: CITL fields
                error_code: export.error_code.clone(),
                clippy_lint: export.clippy_lint.clone(),
                has_suggestion: export.suggestion.is_some(),
                suggestion_applicability,
                source: TrainingSource::DepylerCitl,
            })
        })
        .collect()
}

/// Resolve DefectCategory from DepylerExport
fn resolve_category(export: &DepylerExport) -> Option<DefectCategory> {
    // Try pre-mapped category first
    if let Some(ref cat_str) = export.oip_category {
        if let Some(cat) = parse_defect_category(cat_str) {
            return Some(cat);
        }
    }

    // Try error code mapping
    if let Some(ref code) = export.error_code {
        if let Some(cat) = rustc_to_defect_category(code) {
            return Some(cat);
        }
    }

    // Try clippy lint mapping
    if let Some(ref lint) = export.clippy_lint {
        if let Some(cat) = clippy_to_defect_category(lint) {
            return Some(cat);
        }
    }

    None
}

/// Parse DefectCategory from string
fn parse_defect_category(s: &str) -> Option<DefectCategory> {
    match s {
        "MemorySafety" => Some(DefectCategory::MemorySafety),
        "ConcurrencyBugs" => Some(DefectCategory::ConcurrencyBugs),
        "LogicErrors" => Some(DefectCategory::LogicErrors),
        "ApiMisuse" => Some(DefectCategory::ApiMisuse),
        "ResourceLeaks" => Some(DefectCategory::ResourceLeaks),
        "TypeErrors" => Some(DefectCategory::TypeErrors),
        "ConfigurationErrors" => Some(DefectCategory::ConfigurationErrors),
        "SecurityVulnerabilities" => Some(DefectCategory::SecurityVulnerabilities),
        "PerformanceIssues" => Some(DefectCategory::PerformanceIssues),
        "IntegrationFailures" => Some(DefectCategory::IntegrationFailures),
        "OperatorPrecedence" => Some(DefectCategory::OperatorPrecedence),
        "TypeAnnotationGaps" => Some(DefectCategory::TypeAnnotationGaps),
        "StdlibMapping" => Some(DefectCategory::StdlibMapping),
        "ASTTransform" => Some(DefectCategory::ASTTransform),
        "ComprehensionBugs" => Some(DefectCategory::ComprehensionBugs),
        "IteratorChain" => Some(DefectCategory::IteratorChain),
        "OwnershipBorrow" => Some(DefectCategory::OwnershipBorrow),
        "TraitBounds" => Some(DefectCategory::TraitBounds),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== rustc_to_defect_category tests ====================

    #[test]
    fn test_rustc_type_error_e0308() {
        assert_eq!(
            rustc_to_defect_category("E0308"),
            Some(DefectCategory::TypeErrors)
        );
    }

    #[test]
    fn test_rustc_type_annotation_e0412() {
        assert_eq!(
            rustc_to_defect_category("E0412"),
            Some(DefectCategory::TypeAnnotationGaps)
        );
    }

    #[test]
    fn test_rustc_ownership_borrow_e0502() {
        assert_eq!(
            rustc_to_defect_category("E0502"),
            Some(DefectCategory::OwnershipBorrow)
        );
    }

    #[test]
    fn test_rustc_ownership_borrow_e0503() {
        assert_eq!(
            rustc_to_defect_category("E0503"),
            Some(DefectCategory::OwnershipBorrow)
        );
    }

    #[test]
    fn test_rustc_ownership_borrow_e0505() {
        assert_eq!(
            rustc_to_defect_category("E0505"),
            Some(DefectCategory::OwnershipBorrow)
        );
    }

    #[test]
    fn test_rustc_memory_safety_e0382() {
        assert_eq!(
            rustc_to_defect_category("E0382"),
            Some(DefectCategory::MemorySafety)
        );
    }

    #[test]
    fn test_rustc_memory_safety_e0507() {
        assert_eq!(
            rustc_to_defect_category("E0507"),
            Some(DefectCategory::MemorySafety)
        );
    }

    #[test]
    fn test_rustc_trait_bounds_e0277() {
        assert_eq!(
            rustc_to_defect_category("E0277"),
            Some(DefectCategory::TraitBounds)
        );
    }

    #[test]
    fn test_rustc_stdlib_mapping_e0425() {
        assert_eq!(
            rustc_to_defect_category("E0425"),
            Some(DefectCategory::StdlibMapping)
        );
    }

    #[test]
    fn test_rustc_stdlib_mapping_e0433() {
        assert_eq!(
            rustc_to_defect_category("E0433"),
            Some(DefectCategory::StdlibMapping)
        );
    }

    #[test]
    fn test_rustc_ast_transform_e0599() {
        assert_eq!(
            rustc_to_defect_category("E0599"),
            Some(DefectCategory::ASTTransform)
        );
    }

    #[test]
    fn test_rustc_ast_transform_e0615() {
        assert_eq!(
            rustc_to_defect_category("E0615"),
            Some(DefectCategory::ASTTransform)
        );
    }

    #[test]
    fn test_rustc_operator_precedence_e0614() {
        assert_eq!(
            rustc_to_defect_category("E0614"),
            Some(DefectCategory::OperatorPrecedence)
        );
    }

    #[test]
    fn test_rustc_configuration_e0658() {
        assert_eq!(
            rustc_to_defect_category("E0658"),
            Some(DefectCategory::ConfigurationErrors)
        );
    }

    #[test]
    fn test_rustc_unknown_code_returns_none() {
        assert_eq!(rustc_to_defect_category("E9999"), None);
        assert_eq!(rustc_to_defect_category("UNKNOWN"), None);
        assert_eq!(rustc_to_defect_category(""), None);
    }

    // ==================== clippy_to_defect_category tests ====================

    #[test]
    fn test_clippy_api_misuse_unwrap() {
        assert_eq!(
            clippy_to_defect_category("clippy::unwrap_used"),
            Some(DefectCategory::ApiMisuse)
        );
    }

    #[test]
    fn test_clippy_api_misuse_expect() {
        assert_eq!(
            clippy_to_defect_category("clippy::expect_used"),
            Some(DefectCategory::ApiMisuse)
        );
    }

    #[test]
    fn test_clippy_api_misuse_panic() {
        assert_eq!(
            clippy_to_defect_category("clippy::panic"),
            Some(DefectCategory::ApiMisuse)
        );
    }

    #[test]
    fn test_clippy_logic_errors_todo() {
        assert_eq!(
            clippy_to_defect_category("clippy::todo"),
            Some(DefectCategory::LogicErrors)
        );
    }

    #[test]
    fn test_clippy_logic_errors_unreachable() {
        assert_eq!(
            clippy_to_defect_category("clippy::unreachable"),
            Some(DefectCategory::LogicErrors)
        );
    }

    #[test]
    fn test_clippy_performance_cognitive_complexity() {
        assert_eq!(
            clippy_to_defect_category("clippy::cognitive_complexity"),
            Some(DefectCategory::PerformanceIssues)
        );
    }

    #[test]
    fn test_clippy_ast_transform_too_many_arguments() {
        assert_eq!(
            clippy_to_defect_category("clippy::too_many_arguments"),
            Some(DefectCategory::ASTTransform)
        );
    }

    #[test]
    fn test_clippy_ast_transform_match_single_binding() {
        assert_eq!(
            clippy_to_defect_category("clippy::match_single_binding"),
            Some(DefectCategory::ASTTransform)
        );
    }

    #[test]
    fn test_clippy_iterator_chain_needless_collect() {
        assert_eq!(
            clippy_to_defect_category("clippy::needless_collect"),
            Some(DefectCategory::IteratorChain)
        );
    }

    #[test]
    fn test_clippy_comprehension_bugs_manual_map() {
        assert_eq!(
            clippy_to_defect_category("clippy::manual_map"),
            Some(DefectCategory::ComprehensionBugs)
        );
    }

    #[test]
    fn test_clippy_unknown_lint_returns_none() {
        assert_eq!(clippy_to_defect_category("clippy::unknown_lint"), None);
        assert_eq!(clippy_to_defect_category("not_clippy"), None);
        assert_eq!(clippy_to_defect_category(""), None);
    }

    // ==================== error_code_class tests ====================

    #[test]
    fn test_error_code_class_type() {
        assert_eq!(get_error_code_class("E0308"), ErrorCodeClass::Type);
        assert_eq!(get_error_code_class("E0412"), ErrorCodeClass::Type);
    }

    #[test]
    fn test_error_code_class_borrow() {
        assert_eq!(get_error_code_class("E0502"), ErrorCodeClass::Borrow);
        assert_eq!(get_error_code_class("E0503"), ErrorCodeClass::Borrow);
        assert_eq!(get_error_code_class("E0505"), ErrorCodeClass::Borrow);
        assert_eq!(get_error_code_class("E0382"), ErrorCodeClass::Borrow);
        assert_eq!(get_error_code_class("E0507"), ErrorCodeClass::Borrow);
    }

    #[test]
    fn test_error_code_class_name() {
        assert_eq!(get_error_code_class("E0425"), ErrorCodeClass::Name);
        assert_eq!(get_error_code_class("E0433"), ErrorCodeClass::Name);
    }

    #[test]
    fn test_error_code_class_trait() {
        assert_eq!(get_error_code_class("E0277"), ErrorCodeClass::Trait);
    }

    #[test]
    fn test_error_code_class_other() {
        assert_eq!(get_error_code_class("E9999"), ErrorCodeClass::Other);
        assert_eq!(get_error_code_class("UNKNOWN"), ErrorCodeClass::Other);
    }

    #[test]
    fn test_error_code_class_as_u8() {
        assert_eq!(ErrorCodeClass::Type.as_u8(), 0);
        assert_eq!(ErrorCodeClass::Borrow.as_u8(), 1);
        assert_eq!(ErrorCodeClass::Name.as_u8(), 2);
        assert_eq!(ErrorCodeClass::Trait.as_u8(), 3);
        assert_eq!(ErrorCodeClass::Other.as_u8(), 4);
    }

    // ==================== SuggestionApplicability tests ====================

    #[test]
    fn test_suggestion_applicability_parse() {
        assert_eq!(
            SuggestionApplicability::parse("MachineApplicable"),
            SuggestionApplicability::MachineApplicable
        );
        assert_eq!(
            SuggestionApplicability::parse("MaybeIncorrect"),
            SuggestionApplicability::MaybeIncorrect
        );
        assert_eq!(
            SuggestionApplicability::parse("HasPlaceholders"),
            SuggestionApplicability::HasPlaceholders
        );
        assert_eq!(
            SuggestionApplicability::parse("Unknown"),
            SuggestionApplicability::None
        );
    }

    #[test]
    fn test_suggestion_applicability_as_u8() {
        assert_eq!(SuggestionApplicability::None.as_u8(), 0);
        assert_eq!(SuggestionApplicability::MachineApplicable.as_u8(), 1);
        assert_eq!(SuggestionApplicability::MaybeIncorrect.as_u8(), 2);
        assert_eq!(SuggestionApplicability::HasPlaceholders.as_u8(), 3);
    }

    // ==================== get_error_code_confidence tests ====================

    #[test]
    fn test_error_code_confidence_high() {
        assert!((get_error_code_confidence("E0308") - 0.95).abs() < 0.001);
        assert!((get_error_code_confidence("E0277") - 0.95).abs() < 0.001);
        assert!((get_error_code_confidence("E0502") - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_error_code_confidence_medium() {
        assert!((get_error_code_confidence("E0382") - 0.90).abs() < 0.001);
        assert!((get_error_code_confidence("E0425") - 0.85).abs() < 0.001);
        assert!((get_error_code_confidence("E0599") - 0.80).abs() < 0.001);
    }

    #[test]
    fn test_error_code_confidence_low() {
        assert!((get_error_code_confidence("E0658") - 0.75).abs() < 0.001);
        assert!((get_error_code_confidence("UNKNOWN") - 0.70).abs() < 0.001);
    }

    // ==================== DepylerExport parsing tests ====================

    #[test]
    fn test_depyler_export_parse() {
        let json = r#"{
            "source_file": "example.py",
            "error_code": "E0308",
            "clippy_lint": null,
            "level": "error",
            "message": "mismatched types",
            "oip_category": "TypeErrors",
            "confidence": 0.95,
            "span": {"line_start": 42, "column_start": 12},
            "suggestion": {"replacement": ".parse::<i32>()", "applicability": "MaybeIncorrect"},
            "timestamp": 1732752000,
            "depyler_version": "3.21.0"
        }"#;

        let export: DepylerExport = serde_json::from_str(json).unwrap();

        assert_eq!(export.source_file, "example.py");
        assert_eq!(export.error_code, Some("E0308".to_string()));
        assert_eq!(export.clippy_lint, None);
        assert_eq!(export.level, "error");
        assert!((export.confidence - 0.95).abs() < 0.001);
        assert_eq!(export.span.as_ref().unwrap().line_start, 42);
        assert_eq!(
            export.suggestion.as_ref().unwrap().applicability,
            "MaybeIncorrect"
        );
    }

    #[test]
    fn test_depyler_export_minimal() {
        let json = r#"{
            "source_file": "test.py",
            "error_code": null,
            "clippy_lint": "clippy::unwrap_used",
            "level": "warning",
            "message": "unwrap used",
            "oip_category": null,
            "confidence": 0.80,
            "span": null,
            "suggestion": null,
            "timestamp": 1732752000,
            "depyler_version": "3.21.0"
        }"#;

        let export: DepylerExport = serde_json::from_str(json).unwrap();

        assert_eq!(export.error_code, None);
        assert_eq!(export.clippy_lint, Some("clippy::unwrap_used".to_string()));
        assert_eq!(export.span, None);
        assert_eq!(export.suggestion, None);
    }

    // ==================== resolve_category tests ====================

    #[test]
    fn test_resolve_category_from_oip_category() {
        let export = DepylerExport {
            source_file: "test.py".to_string(),
            error_code: None,
            clippy_lint: None,
            level: "error".to_string(),
            message: "test".to_string(),
            oip_category: Some("MemorySafety".to_string()),
            confidence: 0.90,
            span: None,
            suggestion: None,
            timestamp: 0,
            depyler_version: "1.0".to_string(),
        };

        assert_eq!(
            resolve_category(&export),
            Some(DefectCategory::MemorySafety)
        );
    }

    #[test]
    fn test_resolve_category_from_error_code() {
        let export = DepylerExport {
            source_file: "test.py".to_string(),
            error_code: Some("E0308".to_string()),
            clippy_lint: None,
            level: "error".to_string(),
            message: "test".to_string(),
            oip_category: None,
            confidence: 0.90,
            span: None,
            suggestion: None,
            timestamp: 0,
            depyler_version: "1.0".to_string(),
        };

        assert_eq!(resolve_category(&export), Some(DefectCategory::TypeErrors));
    }

    #[test]
    fn test_resolve_category_from_clippy_lint() {
        let export = DepylerExport {
            source_file: "test.py".to_string(),
            error_code: None,
            clippy_lint: Some("clippy::unwrap_used".to_string()),
            level: "warning".to_string(),
            message: "test".to_string(),
            oip_category: None,
            confidence: 0.90,
            span: None,
            suggestion: None,
            timestamp: 0,
            depyler_version: "1.0".to_string(),
        };

        assert_eq!(resolve_category(&export), Some(DefectCategory::ApiMisuse));
    }

    #[test]
    fn test_resolve_category_unknown() {
        let export = DepylerExport {
            source_file: "test.py".to_string(),
            error_code: Some("E9999".to_string()),
            clippy_lint: None,
            level: "error".to_string(),
            message: "test".to_string(),
            oip_category: None,
            confidence: 0.90,
            span: None,
            suggestion: None,
            timestamp: 0,
            depyler_version: "1.0".to_string(),
        };

        assert_eq!(resolve_category(&export), None);
    }

    // ==================== parse_defect_category tests ====================

    #[test]
    fn test_parse_all_defect_categories() {
        let categories = vec![
            ("MemorySafety", DefectCategory::MemorySafety),
            ("ConcurrencyBugs", DefectCategory::ConcurrencyBugs),
            ("LogicErrors", DefectCategory::LogicErrors),
            ("ApiMisuse", DefectCategory::ApiMisuse),
            ("ResourceLeaks", DefectCategory::ResourceLeaks),
            ("TypeErrors", DefectCategory::TypeErrors),
            ("ConfigurationErrors", DefectCategory::ConfigurationErrors),
            (
                "SecurityVulnerabilities",
                DefectCategory::SecurityVulnerabilities,
            ),
            ("PerformanceIssues", DefectCategory::PerformanceIssues),
            ("IntegrationFailures", DefectCategory::IntegrationFailures),
            ("OperatorPrecedence", DefectCategory::OperatorPrecedence),
            ("TypeAnnotationGaps", DefectCategory::TypeAnnotationGaps),
            ("StdlibMapping", DefectCategory::StdlibMapping),
            ("ASTTransform", DefectCategory::ASTTransform),
            ("ComprehensionBugs", DefectCategory::ComprehensionBugs),
            ("IteratorChain", DefectCategory::IteratorChain),
            ("OwnershipBorrow", DefectCategory::OwnershipBorrow),
            ("TraitBounds", DefectCategory::TraitBounds),
        ];

        for (s, expected) in categories {
            assert_eq!(
                parse_defect_category(s),
                Some(expected),
                "Failed for: {}",
                s
            );
        }
    }

    #[test]
    fn test_parse_unknown_category() {
        assert_eq!(parse_defect_category("Unknown"), None);
        assert_eq!(parse_defect_category(""), None);
    }

    // ==================== TrainingSource tests ====================

    #[test]
    fn test_training_source_default() {
        assert_eq!(TrainingSource::default(), TrainingSource::CommitMessage);
    }

    #[test]
    fn test_training_source_serialization() {
        let source = TrainingSource::DepylerCitl;
        let json = serde_json::to_string(&source).unwrap();
        let parsed: TrainingSource = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, TrainingSource::DepylerCitl);
    }

    // ==================== import_depyler_corpus tests ====================

    #[test]
    fn test_import_depyler_corpus_file_not_found() {
        let result = import_depyler_corpus("/nonexistent/path.jsonl", 0.75);
        assert!(result.is_err());
    }

    #[test]
    fn test_import_stats_default() {
        let stats = ImportStats::default();
        assert_eq!(stats.total_records, 0);
        assert_eq!(stats.imported, 0);
        assert_eq!(stats.skipped_low_confidence, 0);
        assert_eq!(stats.skipped_unknown_category, 0);
        assert!(stats.by_category.is_empty());
        assert!((stats.avg_confidence - 0.0).abs() < 0.001);
    }

    // ==================== convert_to_training_examples tests ====================

    #[test]
    fn test_convert_to_training_examples_basic() {
        let exports = vec![DepylerExport {
            source_file: "test.py".to_string(),
            error_code: Some("E0308".to_string()),
            clippy_lint: None,
            level: "error".to_string(),
            message: "mismatched types".to_string(),
            oip_category: None,
            confidence: 0.95,
            span: None,
            suggestion: None,
            timestamp: 1732752000,
            depyler_version: "3.21.0".to_string(),
        }];

        let examples = convert_to_training_examples(&exports);
        assert_eq!(examples.len(), 1);
        assert_eq!(examples[0].label, DefectCategory::TypeErrors);
        assert_eq!(examples[0].message, "mismatched types");
        assert!((examples[0].confidence - 0.95).abs() < 0.001);
        assert_eq!(examples[0].error_code, Some("E0308".to_string()));
        assert_eq!(examples[0].source, TrainingSource::DepylerCitl);
    }

    #[test]
    fn test_convert_to_training_examples_with_suggestion() {
        let exports = vec![DepylerExport {
            source_file: "test.py".to_string(),
            error_code: Some("E0308".to_string()),
            clippy_lint: None,
            level: "error".to_string(),
            message: "type error".to_string(),
            oip_category: None,
            confidence: 0.90,
            span: None,
            suggestion: Some(SuggestionInfo {
                replacement: ".parse::<i32>()".to_string(),
                applicability: "MachineApplicable".to_string(),
            }),
            timestamp: 1732752000,
            depyler_version: "3.21.0".to_string(),
        }];

        let examples = convert_to_training_examples(&exports);
        assert_eq!(examples.len(), 1);
        assert!(examples[0].has_suggestion);
        assert_eq!(
            examples[0].suggestion_applicability,
            Some(SuggestionApplicability::MachineApplicable)
        );
    }

    #[test]
    fn test_convert_to_training_examples_filters_unknown() {
        let exports = vec![
            DepylerExport {
                source_file: "test.py".to_string(),
                error_code: Some("E0308".to_string()),
                clippy_lint: None,
                level: "error".to_string(),
                message: "known error".to_string(),
                oip_category: None,
                confidence: 0.90,
                span: None,
                suggestion: None,
                timestamp: 0,
                depyler_version: "1.0".to_string(),
            },
            DepylerExport {
                source_file: "test.py".to_string(),
                error_code: Some("E9999".to_string()), // Unknown error code
                clippy_lint: None,
                level: "error".to_string(),
                message: "unknown error".to_string(),
                oip_category: None,
                confidence: 0.90,
                span: None,
                suggestion: None,
                timestamp: 0,
                depyler_version: "1.0".to_string(),
            },
        ];

        let examples = convert_to_training_examples(&exports);
        assert_eq!(examples.len(), 1);
        assert_eq!(examples[0].message, "known error");
    }
}
