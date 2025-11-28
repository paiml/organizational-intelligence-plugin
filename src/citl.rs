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

// ==================== Alimentar DataLoader Integration ====================

/// Merge strategy for combining CITL data with existing training data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MergeStrategy {
    /// Append CITL examples to existing data
    #[default]
    Append,
    /// Replace existing data with CITL examples
    Replace,
    /// Weight CITL examples higher (multiplier applied)
    Weighted(u32),
}

/// Configuration for CITL DataLoader
#[derive(Debug, Clone)]
pub struct CitlLoaderConfig {
    /// Batch size for data loading
    pub batch_size: usize,
    /// Whether to shuffle the data
    pub shuffle: bool,
    /// Minimum confidence threshold
    pub min_confidence: f32,
    /// Merge strategy for combining with existing data
    pub merge_strategy: MergeStrategy,
    /// Weight multiplier for CITL examples (used with Weighted strategy)
    pub weight: f32,
}

impl Default for CitlLoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 128,
            shuffle: true,
            min_confidence: 0.75,
            merge_strategy: MergeStrategy::Append,
            weight: 1.0,
        }
    }
}

/// CITL DataLoader using alimentar for efficient data loading
pub struct CitlDataLoader {
    config: CitlLoaderConfig,
}

impl CitlDataLoader {
    /// Create a new CITL DataLoader with default configuration
    pub fn new() -> Self {
        Self {
            config: CitlLoaderConfig::default(),
        }
    }

    /// Create a new CITL DataLoader with custom configuration
    pub fn with_config(config: CitlLoaderConfig) -> Self {
        Self { config }
    }

    /// Set batch size
    pub fn batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = size;
        self
    }

    /// Enable/disable shuffling
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.config.shuffle = shuffle;
        self
    }

    /// Set minimum confidence threshold
    pub fn min_confidence(mut self, confidence: f32) -> Self {
        self.config.min_confidence = confidence;
        self
    }

    /// Set merge strategy
    pub fn merge_strategy(mut self, strategy: MergeStrategy) -> Self {
        self.config.merge_strategy = strategy;
        self
    }

    /// Load CITL corpus from Parquet file using alimentar
    ///
    /// Returns an iterator over batches of TrainingExamples
    pub fn load_parquet<P: AsRef<Path>>(&self, path: P) -> Result<CitlBatchIterator> {
        use alimentar::{ArrowDataset, DataLoader};

        let dataset = ArrowDataset::from_parquet(path.as_ref())
            .map_err(|e| anyhow!("Failed to load Parquet: {}", e))?;

        let mut loader = DataLoader::new(dataset).batch_size(self.config.batch_size);

        if self.config.shuffle {
            loader = loader.shuffle(true);
        }

        Ok(CitlBatchIterator {
            inner: Box::new(loader.into_iter()),
            min_confidence: self.config.min_confidence,
        })
    }

    /// Load CITL corpus from JSONL file (streaming)
    pub fn load_jsonl<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<(Vec<crate::training::TrainingExample>, ImportStats)> {
        let (exports, stats) = import_depyler_corpus(path, self.config.min_confidence)?;
        let examples = convert_to_training_examples(&exports);
        Ok((examples, stats))
    }

    /// Get the configuration
    pub fn config(&self) -> &CitlLoaderConfig {
        &self.config
    }
}

impl Default for CitlDataLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over batches of training examples from alimentar
pub struct CitlBatchIterator {
    inner: Box<dyn Iterator<Item = arrow::array::RecordBatch> + Send>,
    min_confidence: f32,
}

impl Iterator for CitlBatchIterator {
    type Item = Vec<crate::training::TrainingExample>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|batch| {
            // Convert Arrow RecordBatch to TrainingExamples
            convert_batch_to_examples(&batch, self.min_confidence)
        })
    }
}

/// Convert an Arrow RecordBatch to TrainingExamples
fn convert_batch_to_examples(
    batch: &arrow::array::RecordBatch,
    min_confidence: f32,
) -> Vec<crate::training::TrainingExample> {
    use arrow::array::{Array, Float32Array, Int64Array, StringArray};

    let num_rows = batch.num_rows();
    let mut examples = Vec::with_capacity(num_rows);

    // Downcast columns upfront for efficiency
    let message_arr = batch
        .column_by_name("message")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());
    let error_code_arr = batch
        .column_by_name("error_code")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());
    let clippy_lint_arr = batch
        .column_by_name("clippy_lint")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());
    let confidence_arr = batch
        .column_by_name("confidence")
        .and_then(|c| c.as_any().downcast_ref::<Float32Array>());
    let timestamp_arr = batch
        .column_by_name("timestamp")
        .and_then(|c| c.as_any().downcast_ref::<Int64Array>());
    let oip_category_arr = batch
        .column_by_name("oip_category")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>());

    for i in 0..num_rows {
        // Extract confidence
        let confidence = confidence_arr.map(|a| a.value(i)).unwrap_or(0.0);

        if confidence < min_confidence {
            continue;
        }

        // Extract message
        let message = message_arr
            .and_then(|a| {
                if a.is_null(i) {
                    None
                } else {
                    Some(a.value(i).to_string())
                }
            })
            .unwrap_or_default();

        // Extract error_code
        let error_code = error_code_arr.and_then(|a| {
            if a.is_null(i) {
                None
            } else {
                Some(a.value(i).to_string())
            }
        });

        // Extract clippy_lint
        let clippy_lint = clippy_lint_arr.and_then(|a| {
            if a.is_null(i) {
                None
            } else {
                Some(a.value(i).to_string())
            }
        });

        // Extract timestamp
        let timestamp = timestamp_arr.map(|a| a.value(i)).unwrap_or(0);

        // Resolve category
        let oip_category =
            oip_category_arr.and_then(|a| if a.is_null(i) { None } else { Some(a.value(i)) });

        let category = oip_category
            .and_then(parse_defect_category)
            .or_else(|| error_code.as_deref().and_then(rustc_to_defect_category))
            .or_else(|| clippy_lint.as_deref().and_then(clippy_to_defect_category));

        if let Some(label) = category {
            examples.push(crate::training::TrainingExample {
                message,
                label,
                confidence,
                commit_hash: String::new(),
                author: "depyler".to_string(),
                timestamp,
                lines_added: 0,
                lines_removed: 0,
                files_changed: 1,
                error_code,
                clippy_lint,
                has_suggestion: false,
                suggestion_applicability: None,
                source: TrainingSource::DepylerCitl,
            });
        }
    }

    examples
}

/// Validate CITL export schema (FR-8)
pub fn validate_citl_schema<P: AsRef<Path>>(path: P) -> Result<SchemaValidation> {
    use alimentar::{ArrowDataset, Dataset};
    use arrow::datatypes::FieldRef;

    let ext = path.as_ref().extension().and_then(|e| e.to_str());

    let schema = match ext {
        Some("parquet") => {
            let dataset = ArrowDataset::from_parquet(path.as_ref())
                .map_err(|e| anyhow!("Failed to load Parquet: {}", e))?;
            dataset.schema()
        }
        Some("jsonl") | Some("json") => {
            // For JSONL, we validate the first line
            let content = std::fs::read_to_string(path.as_ref())?;
            let first_line = content
                .lines()
                .next()
                .ok_or_else(|| anyhow!("Empty file"))?;
            let _: DepylerExport = serde_json::from_str(first_line)
                .map_err(|e| anyhow!("Invalid JSONL schema: {}", e))?;
            return Ok(SchemaValidation {
                is_valid: true,
                missing_fields: vec![],
                extra_fields: vec![],
                format: "jsonl".to_string(),
            });
        }
        _ => return Err(anyhow!("Unsupported file format: {:?}", ext)),
    };

    // Required fields for CITL schema
    let required_fields = ["message", "confidence", "timestamp"];
    let optional_fields = [
        "error_code",
        "clippy_lint",
        "oip_category",
        "suggestion",
        "span",
    ];

    let schema_fields: Vec<&str> = schema
        .fields()
        .iter()
        .map(|f: &FieldRef| f.name().as_str())
        .collect();

    let missing: Vec<String> = required_fields
        .iter()
        .filter(|f| !schema_fields.contains(*f))
        .map(|s: &&str| (*s).to_string())
        .collect();

    let known_fields: Vec<&str> = required_fields
        .iter()
        .chain(optional_fields.iter())
        .copied()
        .collect();

    let extra: Vec<String> = schema_fields
        .iter()
        .filter(|f| !known_fields.contains(*f))
        .map(|s: &&str| (*s).to_string())
        .collect();

    Ok(SchemaValidation {
        is_valid: missing.is_empty(),
        missing_fields: missing,
        extra_fields: extra,
        format: "parquet".to_string(),
    })
}

/// Schema validation result
#[derive(Debug, Clone)]
pub struct SchemaValidation {
    /// Whether the schema is valid
    pub is_valid: bool,
    /// Missing required fields
    pub missing_fields: Vec<String>,
    /// Extra fields not in the expected schema
    pub extra_fields: Vec<String>,
    /// Detected format
    pub format: String,
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

    // ==================== MergeStrategy tests ====================

    #[test]
    fn test_merge_strategy_default() {
        let strategy = MergeStrategy::default();
        assert!(matches!(strategy, MergeStrategy::Append));
    }

    #[test]
    fn test_merge_strategy_append() {
        let strategy = MergeStrategy::Append;
        assert!(matches!(strategy, MergeStrategy::Append));
    }

    #[test]
    fn test_merge_strategy_replace() {
        let strategy = MergeStrategy::Replace;
        assert!(matches!(strategy, MergeStrategy::Replace));
    }

    #[test]
    fn test_merge_strategy_weighted() {
        let strategy = MergeStrategy::Weighted(2);
        if let MergeStrategy::Weighted(multiplier) = strategy {
            assert_eq!(multiplier, 2);
        } else {
            panic!("Expected MergeStrategy::Weighted");
        }
    }

    // ==================== CitlLoaderConfig tests ====================

    #[test]
    fn test_citl_loader_config_default() {
        let config = CitlLoaderConfig::default();
        assert_eq!(config.batch_size, 128);
        assert!((config.min_confidence - 0.75).abs() < 0.001);
        assert!(matches!(config.merge_strategy, MergeStrategy::Append));
        assert!(config.shuffle);
        assert!((config.weight - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_citl_loader_config_custom() {
        let config = CitlLoaderConfig {
            batch_size: 512,
            min_confidence: 0.9,
            merge_strategy: MergeStrategy::Replace,
            shuffle: false,
            weight: 2.0,
        };
        assert_eq!(config.batch_size, 512);
        assert!((config.min_confidence - 0.9).abs() < 0.001);
        assert!(!config.shuffle);
        assert!((config.weight - 2.0).abs() < 0.001);
    }

    // ==================== CitlDataLoader tests ====================

    #[test]
    fn test_citl_data_loader_new() {
        let loader = CitlDataLoader::new();
        assert_eq!(loader.config().batch_size, 128);
    }

    #[test]
    fn test_citl_data_loader_with_config() {
        let config = CitlLoaderConfig {
            batch_size: 256,
            min_confidence: 0.8,
            ..CitlLoaderConfig::default()
        };
        let loader = CitlDataLoader::with_config(config);
        assert_eq!(loader.config().batch_size, 256);
        assert!((loader.config().min_confidence - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_citl_data_loader_default() {
        let loader = CitlDataLoader::default();
        assert_eq!(loader.config().batch_size, 128);
    }

    #[test]
    fn test_citl_data_loader_load_jsonl_not_found() {
        let loader = CitlDataLoader::new();
        let result = loader.load_jsonl("nonexistent.jsonl");
        assert!(result.is_err());
    }

    #[test]
    fn test_citl_data_loader_load_parquet_not_found() {
        let loader = CitlDataLoader::new();
        let result = loader.load_parquet("nonexistent.parquet");
        assert!(result.is_err());
    }

    #[test]
    fn test_citl_data_loader_load_jsonl_valid() {
        use std::io::Write;
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("valid.jsonl");
        let mut file = std::fs::File::create(&file_path).unwrap();

        // Write valid CITL entries
        writeln!(file, r#"{{"source_file":"test.py","error_code":"E0308","clippy_lint":null,"level":"error","message":"type mismatch","oip_category":null,"confidence":0.95,"span":null,"suggestion":null,"timestamp":1732752000,"depyler_version":"1.0"}}"#).unwrap();
        writeln!(file, r#"{{"source_file":"test.py","error_code":null,"clippy_lint":"clippy::unwrap_used","level":"warning","message":"unwrap used","oip_category":null,"confidence":0.85,"span":null,"suggestion":null,"timestamp":1732752001,"depyler_version":"1.0"}}"#).unwrap();

        let loader = CitlDataLoader::new();
        let result = loader.load_jsonl(&file_path);
        assert!(result.is_ok());

        let (examples, stats) = result.unwrap();
        assert_eq!(examples.len(), 2);
        assert_eq!(stats.total_records, 2);
        assert_eq!(stats.imported, 2);
    }

    #[test]
    fn test_citl_data_loader_load_parquet_valid() {
        use arrow::array::{Float32Array, Int64Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use parquet::arrow::ArrowWriter;
        use std::fs::File;
        use std::sync::Arc;

        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("valid.parquet");

        // Create schema
        let schema = Arc::new(Schema::new(vec![
            Field::new("message", DataType::Utf8, false),
            Field::new("confidence", DataType::Float32, false),
            Field::new("error_code", DataType::Utf8, true),
            Field::new("timestamp", DataType::Int64, false),
        ]));

        // Create data
        let message_arr = StringArray::from(vec!["type mismatch", "api misuse"]);
        let confidence_arr = Float32Array::from(vec![0.95, 0.88]);
        let error_code_arr = StringArray::from(vec![Some("E0308"), None]);
        let timestamp_arr = Int64Array::from(vec![1732752000, 1732752001]);

        let batch = arrow::array::RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(message_arr),
                Arc::new(confidence_arr),
                Arc::new(error_code_arr),
                Arc::new(timestamp_arr),
            ],
        )
        .unwrap();

        // Write parquet file
        let file = File::create(&file_path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        // Load using CitlDataLoader (returns iterator)
        let loader = CitlDataLoader::new();
        let result = loader.load_parquet(&file_path);
        assert!(result.is_ok());

        // Collect examples from iterator
        let iter = result.unwrap();
        let all_examples: Vec<_> = iter.flatten().collect();
        // Only 1 example should be valid (E0308 maps to TypeErrors, the other has no error_code mapping)
        assert_eq!(all_examples.len(), 1);
        assert_eq!(all_examples[0].label, DefectCategory::TypeErrors);
    }

    // ==================== SchemaValidation tests ====================

    #[test]
    fn test_schema_validation_valid() {
        let validation = SchemaValidation {
            is_valid: true,
            missing_fields: vec![],
            extra_fields: vec![],
            format: "parquet".to_string(),
        };
        assert!(validation.is_valid);
        assert!(validation.missing_fields.is_empty());
    }

    #[test]
    fn test_schema_validation_invalid() {
        let validation = SchemaValidation {
            is_valid: false,
            missing_fields: vec!["message".to_string(), "confidence".to_string()],
            extra_fields: vec![],
            format: "parquet".to_string(),
        };
        assert!(!validation.is_valid);
        assert_eq!(validation.missing_fields.len(), 2);
    }

    #[test]
    fn test_validate_citl_schema_unsupported_format() {
        let result = validate_citl_schema("test.csv");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_citl_schema_jsonl_valid() {
        use std::io::Write;
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("test.jsonl");
        let mut file = std::fs::File::create(&file_path).unwrap();
        writeln!(file, r#"{{"source_file":"test.py","error_code":"E0308","clippy_lint":null,"level":"error","message":"test","oip_category":null,"confidence":0.9,"span":null,"suggestion":null,"timestamp":0,"depyler_version":"1.0"}}"#).unwrap();

        let result = validate_citl_schema(&file_path).unwrap();
        assert!(result.is_valid);
        assert_eq!(result.format, "jsonl");
    }

    #[test]
    fn test_validate_citl_schema_empty_file() {
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("empty.jsonl");
        let _file = std::fs::File::create(&file_path).unwrap();

        let result = validate_citl_schema(&file_path);
        assert!(result.is_err());
    }

    // ==================== convert_batch_to_examples tests ====================

    #[test]
    fn test_convert_batch_empty() {
        use arrow::array::RecordBatch;
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("message", DataType::Utf8, false),
            Field::new("confidence", DataType::Float32, false),
        ]));

        let batch = RecordBatch::new_empty(schema);
        let examples = convert_batch_to_examples(&batch, 0.0);
        assert!(examples.is_empty());
    }

    #[test]
    fn test_convert_batch_with_data() {
        use arrow::array::{Float32Array, RecordBatch, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("message", DataType::Utf8, false),
            Field::new("confidence", DataType::Float32, false),
            Field::new("error_code", DataType::Utf8, true),
            Field::new("timestamp", DataType::Int64, false),
        ]));

        let message_arr = StringArray::from(vec!["type mismatch"]);
        let confidence_arr = Float32Array::from(vec![0.95]);
        let error_code_arr = StringArray::from(vec![Some("E0308")]);
        let timestamp_arr = arrow::array::Int64Array::from(vec![1732752000]);

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(message_arr),
                Arc::new(confidence_arr),
                Arc::new(error_code_arr),
                Arc::new(timestamp_arr),
            ],
        )
        .unwrap();

        let examples = convert_batch_to_examples(&batch, 0.5);
        assert_eq!(examples.len(), 1);
        assert_eq!(examples[0].message, "type mismatch");
        assert_eq!(examples[0].label, DefectCategory::TypeErrors);
        assert!((examples[0].confidence - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_convert_batch_filters_low_confidence() {
        use arrow::array::{Float32Array, RecordBatch, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("message", DataType::Utf8, false),
            Field::new("confidence", DataType::Float32, false),
            Field::new("error_code", DataType::Utf8, true),
        ]));

        let message_arr = StringArray::from(vec!["low conf", "high conf"]);
        let confidence_arr = Float32Array::from(vec![0.3, 0.9]);
        let error_code_arr = StringArray::from(vec![Some("E0308"), Some("E0308")]);

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(message_arr),
                Arc::new(confidence_arr),
                Arc::new(error_code_arr),
            ],
        )
        .unwrap();

        let examples = convert_batch_to_examples(&batch, 0.5);
        assert_eq!(examples.len(), 1);
        assert_eq!(examples[0].message, "high conf");
    }
}
