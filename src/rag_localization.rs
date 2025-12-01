//! RAG-Enhanced Fault Localization
//!
//! This module integrates trueno-rag with SBFL fault localization to provide:
//! - Semantic search over historical bugs
//! - Similar code pattern retrieval
//! - Fix suggestion generation
//! - Contextual explanations
//!
//! Toyota Way Alignment:
//! - Genchi Genbutsu: Retrieve actual historical bugs, not hypothetical patterns
//! - Kaizen: Bug knowledge base improves continuously from each fix
//! - Jidoka: Human-readable explanations with context
//! - Muda: Only query RAG for top-N suspicious statements (avoid waste)
//! - Muri: Configurable retrieval limits prevent information overload

use crate::tarantula::{
    FaultLocalizationResult, SbflFormula, SbflLocalizer, StatementCoverage, SuspiciousnessRanking,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;
use trueno_rag::{
    chunk::{Chunk, ChunkId},
    index::{BM25Index, SparseIndex},
    DocumentId,
};

/// Errors that can occur during RAG-enhanced fault localization
#[derive(Debug, Error)]
pub enum RagLocalizationError {
    #[error("Failed to build RAG pipeline: {0}")]
    PipelineBuild(String),

    #[error("Failed to index document: {0}")]
    IndexError(String),

    #[error("Failed to query RAG pipeline: {0}")]
    QueryError(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Result type for RAG localization operations
pub type Result<T> = std::result::Result<T, RagLocalizationError>;

/// Defect category for bug classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DefectCategory {
    MemorySafety,
    Concurrency,
    TypeErrors,
    Performance,
    Security,
    Configuration,
    ApiMisuse,
    IntegrationFailure,
    DocumentationGap,
    TestingGap,
    OperatorPrecedence,
    TypeAnnotationGap,
    StdlibMapping,
    AstTransform,
    ComprehensionBug,
    IteratorChain,
    OwnershipBorrow,
    TraitBounds,
}

impl DefectCategory {
    /// Get all defect categories
    pub fn all() -> &'static [DefectCategory] {
        &[
            DefectCategory::MemorySafety,
            DefectCategory::Concurrency,
            DefectCategory::TypeErrors,
            DefectCategory::Performance,
            DefectCategory::Security,
            DefectCategory::Configuration,
            DefectCategory::ApiMisuse,
            DefectCategory::IntegrationFailure,
            DefectCategory::DocumentationGap,
            DefectCategory::TestingGap,
            DefectCategory::OperatorPrecedence,
            DefectCategory::TypeAnnotationGap,
            DefectCategory::StdlibMapping,
            DefectCategory::AstTransform,
            DefectCategory::ComprehensionBug,
            DefectCategory::IteratorChain,
            DefectCategory::OwnershipBorrow,
            DefectCategory::TraitBounds,
        ]
    }

    /// Get display name
    pub fn display_name(&self) -> &'static str {
        match self {
            DefectCategory::MemorySafety => "Memory Safety",
            DefectCategory::Concurrency => "Concurrency",
            DefectCategory::TypeErrors => "Type Errors",
            DefectCategory::Performance => "Performance",
            DefectCategory::Security => "Security",
            DefectCategory::Configuration => "Configuration",
            DefectCategory::ApiMisuse => "API Misuse",
            DefectCategory::IntegrationFailure => "Integration Failure",
            DefectCategory::DocumentationGap => "Documentation Gap",
            DefectCategory::TestingGap => "Testing Gap",
            DefectCategory::OperatorPrecedence => "Operator Precedence",
            DefectCategory::TypeAnnotationGap => "Type Annotation Gap",
            DefectCategory::StdlibMapping => "Stdlib Mapping",
            DefectCategory::AstTransform => "AST Transform",
            DefectCategory::ComprehensionBug => "Comprehension Bug",
            DefectCategory::IteratorChain => "Iterator Chain",
            DefectCategory::OwnershipBorrow => "Ownership/Borrow",
            DefectCategory::TraitBounds => "Trait Bounds",
        }
    }
}

impl std::fmt::Display for DefectCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// Bug document for RAG indexing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BugDocument {
    /// Unique bug identifier (e.g., commit hash or issue number)
    pub id: String,
    /// Bug title/summary
    pub title: String,
    /// Full bug description
    pub description: String,
    /// Commit that fixed the bug
    pub fix_commit: String,
    /// The actual code change (diff)
    pub fix_diff: String,
    /// Files affected by the bug
    pub affected_files: Vec<String>,
    /// Defect category
    pub category: DefectCategory,
    /// Severity level (1-5, 5 being most severe)
    pub severity: u8,
    /// Symptoms that indicate this bug
    pub symptoms: Vec<String>,
    /// Root cause description
    pub root_cause: String,
    /// Fix pattern description
    pub fix_pattern: String,
}

impl BugDocument {
    /// Create a new bug document
    pub fn new(id: impl Into<String>, title: impl Into<String>, category: DefectCategory) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            description: String::new(),
            fix_commit: String::new(),
            fix_diff: String::new(),
            affected_files: Vec::new(),
            category,
            severity: 3,
            symptoms: Vec::new(),
            root_cause: String::new(),
            fix_pattern: String::new(),
        }
    }

    /// Set description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set fix commit
    pub fn with_fix_commit(mut self, commit: impl Into<String>) -> Self {
        self.fix_commit = commit.into();
        self
    }

    /// Set fix diff
    pub fn with_fix_diff(mut self, diff: impl Into<String>) -> Self {
        self.fix_diff = diff.into();
        self
    }

    /// Add affected file
    pub fn with_affected_file(mut self, file: impl Into<String>) -> Self {
        self.affected_files.push(file.into());
        self
    }

    /// Set severity
    pub fn with_severity(mut self, severity: u8) -> Self {
        self.severity = severity.clamp(1, 5);
        self
    }

    /// Add symptom
    pub fn with_symptom(mut self, symptom: impl Into<String>) -> Self {
        self.symptoms.push(symptom.into());
        self
    }

    /// Set root cause
    pub fn with_root_cause(mut self, cause: impl Into<String>) -> Self {
        self.root_cause = cause.into();
        self
    }

    /// Set fix pattern
    pub fn with_fix_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.fix_pattern = pattern.into();
        self
    }

    /// Convert to trueno-rag Chunk for indexing
    pub fn to_rag_chunk(&self) -> Chunk {
        let content = format!(
            "{}\n\n{}\n\nSymptoms:\n{}\n\nRoot Cause:\n{}\n\nFix Pattern:\n{}",
            self.title,
            self.description,
            self.symptoms.join("\n- "),
            self.root_cause,
            self.fix_pattern
        );

        Chunk::new(DocumentId::new(), content, 0, self.description.len().max(1))
    }
}

/// Similar bug retrieved from RAG
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarBug {
    /// Bug ID
    pub id: String,
    /// Similarity score (0.0 to 1.0)
    pub similarity: f32,
    /// Defect category
    pub category: DefectCategory,
    /// Bug summary
    pub summary: String,
    /// Fix commit hash
    pub fix_commit: String,
}

/// Suggested fix from RAG retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestedFix {
    /// Fix pattern name
    pub pattern: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Example code showing the fix
    pub example: String,
    /// Source bug ID this pattern came from
    pub source_bug_id: String,
}

/// RAG-enhanced ranking with additional context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagEnhancedRanking {
    /// Original SBFL ranking
    pub sbfl_ranking: SuspiciousnessRanking,
    /// Similar historical bugs
    pub similar_bugs: Vec<SimilarBug>,
    /// Suggested fixes
    pub suggested_fixes: Vec<SuggestedFix>,
    /// Contextual explanation
    pub context_explanation: String,
    /// Combined score (SBFL + RAG)
    pub combined_score: f32,
}

/// Result of RAG-enhanced fault localization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagEnhancedResult {
    /// Enhanced rankings with RAG context
    pub rankings: Vec<RagEnhancedRanking>,
    /// Original SBFL result
    pub sbfl_result: FaultLocalizationResult,
    /// Fusion strategy used
    pub fusion_strategy: String,
    /// Number of bugs in knowledge base
    pub knowledge_base_size: usize,
}

/// Bug knowledge base for RAG retrieval
#[derive(Debug)]
pub struct BugKnowledgeBase {
    /// Indexed bug documents
    bugs: Vec<BugDocument>,
    /// BM25 index for text search
    bm25_index: BM25Index,
    /// Chunk ID to bug ID mapping
    chunk_to_bug: HashMap<ChunkId, String>,
}

impl BugKnowledgeBase {
    /// Create a new empty knowledge base
    pub fn new() -> Self {
        Self {
            bugs: Vec::new(),
            bm25_index: BM25Index::new(),
            chunk_to_bug: HashMap::new(),
        }
    }

    /// Add a bug to the knowledge base
    pub fn add_bug(&mut self, bug: BugDocument) {
        let chunk = bug.to_rag_chunk();
        let chunk_id = chunk.id;

        // Index the chunk content using SparseIndex trait
        self.bm25_index.add(&chunk);
        self.chunk_to_bug.insert(chunk_id, bug.id.clone());
        self.bugs.push(bug);
    }

    /// Get the number of bugs in the knowledge base
    pub fn len(&self) -> usize {
        self.bugs.len()
    }

    /// Check if the knowledge base is empty
    pub fn is_empty(&self) -> bool {
        self.bugs.is_empty()
    }

    /// Search for similar bugs using BM25
    pub fn search(&self, query: &str, top_k: usize) -> Vec<SimilarBug> {
        let results: Vec<(ChunkId, f32)> = self.bm25_index.search(query, top_k);

        // Normalize scores to 0-1 range
        let max_score = results.iter().map(|(_, s)| *s).fold(0.0_f32, f32::max);
        let normalizer = if max_score > 0.0 { max_score } else { 1.0 };

        results
            .into_iter()
            .filter_map(|(chunk_id, score): (ChunkId, f32)| {
                // Look up bug ID from chunk mapping
                let bug_id = self.chunk_to_bug.get(&chunk_id)?;
                let bug = self.bugs.iter().find(|b| &b.id == bug_id)?;

                Some(SimilarBug {
                    id: bug.id.clone(),
                    similarity: (score / normalizer).clamp(0.0, 1.0),
                    category: bug.category,
                    summary: bug.title.clone(),
                    fix_commit: bug.fix_commit.clone(),
                })
            })
            .collect()
    }

    /// Get fix patterns for similar bugs
    pub fn get_fix_patterns(&self, bug_ids: &[String]) -> Vec<SuggestedFix> {
        bug_ids
            .iter()
            .filter_map(|id| {
                self.bugs
                    .iter()
                    .find(|b| &b.id == id)
                    .map(|bug| SuggestedFix {
                        pattern: format!("Fix pattern for {}", bug.category),
                        confidence: 0.7,
                        example: bug.fix_pattern.clone(),
                        source_bug_id: bug.id.clone(),
                    })
            })
            .collect()
    }

    /// Get bug by ID
    pub fn get_bug(&self, id: &str) -> Option<&BugDocument> {
        self.bugs.iter().find(|b| b.id == id)
    }

    /// Get all bugs in a category
    pub fn get_by_category(&self, category: DefectCategory) -> Vec<&BugDocument> {
        self.bugs
            .iter()
            .filter(|b| b.category == category)
            .collect()
    }

    /// Import bugs from YAML file
    pub fn import_from_yaml(path: &std::path::Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let bugs: Vec<BugDocument> = serde_yaml::from_str(&content).map_err(|e| {
            RagLocalizationError::Serialization(serde_json::Error::io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                e.to_string(),
            )))
        })?;

        let mut kb = Self::new();
        for bug in bugs {
            kb.add_bug(bug);
        }
        Ok(kb)
    }

    /// Export bugs to YAML file
    pub fn export_to_yaml(&self, path: &std::path::Path) -> Result<()> {
        let content = serde_yaml::to_string(&self.bugs).map_err(|e| {
            RagLocalizationError::Serialization(serde_json::Error::io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                e.to_string(),
            )))
        })?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

impl Default for BugKnowledgeBase {
    fn default() -> Self {
        Self::new()
    }
}

/// Fusion strategy wrapper for SBFL + RAG combination
#[derive(Debug, Clone, Copy)]
pub enum LocalizationFusion {
    /// Reciprocal Rank Fusion (recommended)
    RRF { k: f32 },
    /// Linear weighted combination
    Linear { sbfl_weight: f32 },
    /// Distribution-based score fusion
    DBSF,
    /// Use only SBFL, RAG for context only
    SbflOnly,
}

impl Default for LocalizationFusion {
    fn default() -> Self {
        LocalizationFusion::RRF { k: 60.0 }
    }
}

impl LocalizationFusion {
    /// Combine SBFL and RAG scores
    pub fn combine(
        &self,
        sbfl_score: f32,
        rag_score: f32,
        sbfl_rank: usize,
        rag_rank: usize,
    ) -> f32 {
        match self {
            LocalizationFusion::RRF { k } => {
                // RRF: sum of reciprocal ranks
                let sbfl_rrf = 1.0 / (k + sbfl_rank as f32);
                let rag_rrf = 1.0 / (k + rag_rank as f32);
                sbfl_rrf + rag_rrf
            }
            LocalizationFusion::Linear { sbfl_weight } => {
                // Linear: weighted combination of normalized scores
                let rag_weight = 1.0 - sbfl_weight;
                sbfl_score * sbfl_weight + rag_score * rag_weight
            }
            LocalizationFusion::DBSF => {
                // DBSF: average of scores (simplified)
                (sbfl_score + rag_score) / 2.0
            }
            LocalizationFusion::SbflOnly => sbfl_score,
        }
    }

    /// Get display name
    pub fn name(&self) -> &'static str {
        match self {
            LocalizationFusion::RRF { .. } => "RRF",
            LocalizationFusion::Linear { .. } => "Linear",
            LocalizationFusion::DBSF => "DBSF",
            LocalizationFusion::SbflOnly => "SBFL Only",
        }
    }
}

/// Configuration for RAG-enhanced fault localization
#[derive(Debug, Clone)]
pub struct RagLocalizationConfig {
    /// SBFL formula to use
    pub sbfl_formula: SbflFormula,
    /// Number of top statements to enhance with RAG
    pub top_n: usize,
    /// Number of similar bugs to retrieve
    pub similar_bugs_k: usize,
    /// Number of fix suggestions to retrieve
    pub fix_suggestions_k: usize,
    /// Fusion strategy
    pub fusion: LocalizationFusion,
    /// Include detailed explanations
    pub include_explanations: bool,
}

impl Default for RagLocalizationConfig {
    fn default() -> Self {
        Self {
            sbfl_formula: SbflFormula::Ochiai,
            top_n: 10,
            similar_bugs_k: 5,
            fix_suggestions_k: 3,
            fusion: LocalizationFusion::default(),
            include_explanations: true,
        }
    }
}

impl RagLocalizationConfig {
    /// Create new config with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set SBFL formula
    pub fn with_formula(mut self, formula: SbflFormula) -> Self {
        self.sbfl_formula = formula;
        self
    }

    /// Set top-N statements to enhance
    pub fn with_top_n(mut self, n: usize) -> Self {
        self.top_n = n;
        self
    }

    /// Set number of similar bugs to retrieve
    pub fn with_similar_bugs(mut self, k: usize) -> Self {
        self.similar_bugs_k = k;
        self
    }

    /// Set number of fix suggestions
    pub fn with_fix_suggestions(mut self, k: usize) -> Self {
        self.fix_suggestions_k = k;
        self
    }

    /// Set fusion strategy
    pub fn with_fusion(mut self, fusion: LocalizationFusion) -> Self {
        self.fusion = fusion;
        self
    }

    /// Enable/disable explanations
    pub fn with_explanations(mut self, include: bool) -> Self {
        self.include_explanations = include;
        self
    }
}

/// RAG-enhanced fault localizer
pub struct RagFaultLocalizer {
    /// SBFL localizer
    sbfl: SbflLocalizer,
    /// Bug knowledge base
    knowledge_base: BugKnowledgeBase,
    /// Configuration
    config: RagLocalizationConfig,
}

impl RagFaultLocalizer {
    /// Create a new RAG fault localizer
    pub fn new(knowledge_base: BugKnowledgeBase, config: RagLocalizationConfig) -> Self {
        let sbfl = SbflLocalizer::new()
            .with_formula(config.sbfl_formula)
            .with_top_n(config.top_n)
            .with_explanations(config.include_explanations);

        Self {
            sbfl,
            knowledge_base,
            config,
        }
    }

    /// Create with default configuration
    pub fn with_knowledge_base(knowledge_base: BugKnowledgeBase) -> Self {
        Self::new(knowledge_base, RagLocalizationConfig::default())
    }

    /// Localize faults with RAG enhancement
    pub fn localize(
        &self,
        coverage: &[StatementCoverage],
        total_passed: usize,
        total_failed: usize,
    ) -> RagEnhancedResult {
        tracing::info!(
            "Running RAG-enhanced fault localization on {} statements",
            coverage.len()
        );

        // Step 1: Run SBFL localization
        let sbfl_result = self.sbfl.localize(coverage, total_passed, total_failed);

        // Step 2: Enhance top-N rankings with RAG
        let mut enhanced_rankings = Vec::new();

        for (sbfl_rank, ranking) in sbfl_result.rankings.iter().enumerate() {
            // Build query from statement context
            let query = self.build_query(ranking);

            // Search for similar bugs
            let similar_bugs = self
                .knowledge_base
                .search(&query, self.config.similar_bugs_k);

            // Get fix patterns from similar bugs
            let bug_ids: Vec<String> = similar_bugs.iter().map(|b| b.id.clone()).collect();
            let suggested_fixes = self.knowledge_base.get_fix_patterns(&bug_ids);

            // Calculate RAG-based score (average similarity of top bugs)
            let rag_score = if similar_bugs.is_empty() {
                0.0
            } else {
                similar_bugs.iter().map(|b| b.similarity).sum::<f32>() / similar_bugs.len() as f32
            };

            // Combine scores using fusion strategy
            let rag_rank = if rag_score > 0.0 {
                sbfl_rank
            } else {
                sbfl_rank + 100
            };
            let combined_score =
                self.config
                    .fusion
                    .combine(ranking.suspiciousness, rag_score, sbfl_rank, rag_rank);

            // Generate contextual explanation
            let context_explanation = if self.config.include_explanations {
                self.generate_explanation(ranking, &similar_bugs)
            } else {
                String::new()
            };

            enhanced_rankings.push(RagEnhancedRanking {
                sbfl_ranking: ranking.clone(),
                similar_bugs,
                suggested_fixes,
                context_explanation,
                combined_score,
            });
        }

        // Re-sort by combined score
        enhanced_rankings.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Update ranks
        for (i, ranking) in enhanced_rankings.iter_mut().enumerate() {
            ranking.sbfl_ranking.rank = i + 1;
        }

        RagEnhancedResult {
            rankings: enhanced_rankings,
            sbfl_result,
            fusion_strategy: self.config.fusion.name().to_string(),
            knowledge_base_size: self.knowledge_base.len(),
        }
    }

    /// Build a search query from a suspicious ranking
    fn build_query(&self, ranking: &SuspiciousnessRanking) -> String {
        // Use file name and explanation to build query
        let file_name = ranking
            .statement
            .file
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        format!(
            "{} line {} {}",
            file_name, ranking.statement.line, ranking.explanation
        )
    }

    /// Generate contextual explanation
    fn generate_explanation(
        &self,
        ranking: &SuspiciousnessRanking,
        similar_bugs: &[SimilarBug],
    ) -> String {
        if similar_bugs.is_empty() {
            return format!(
                "Statement at {}:{} has suspiciousness score {:.3}. No similar historical bugs found in knowledge base.",
                ranking.statement.file.display(),
                ranking.statement.line,
                ranking.suspiciousness
            );
        }

        let top_bug = &similar_bugs[0];
        let bug_count = similar_bugs.len();

        format!(
            "This pattern matches historical bug \"{}\" ({}) with {:.0}% similarity. \
             {} similar bugs found in knowledge base. \
             Most common category: {}.",
            top_bug.id,
            top_bug.summary,
            top_bug.similarity * 100.0,
            bug_count,
            top_bug.category
        )
    }

    /// Get the knowledge base
    pub fn knowledge_base(&self) -> &BugKnowledgeBase {
        &self.knowledge_base
    }

    /// Get mutable knowledge base
    pub fn knowledge_base_mut(&mut self) -> &mut BugKnowledgeBase {
        &mut self.knowledge_base
    }
}

/// Builder for creating RAG-enhanced fault localizer
pub struct RagFaultLocalizerBuilder {
    knowledge_base: Option<BugKnowledgeBase>,
    config: RagLocalizationConfig,
}

impl RagFaultLocalizerBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            knowledge_base: None,
            config: RagLocalizationConfig::default(),
        }
    }

    /// Set the knowledge base
    pub fn knowledge_base(mut self, kb: BugKnowledgeBase) -> Self {
        self.knowledge_base = Some(kb);
        self
    }

    /// Set SBFL formula
    pub fn formula(mut self, formula: SbflFormula) -> Self {
        self.config.sbfl_formula = formula;
        self
    }

    /// Set top-N statements
    pub fn top_n(mut self, n: usize) -> Self {
        self.config.top_n = n;
        self
    }

    /// Set similar bugs count
    pub fn similar_bugs(mut self, k: usize) -> Self {
        self.config.similar_bugs_k = k;
        self
    }

    /// Set fix suggestions count
    pub fn fix_suggestions(mut self, k: usize) -> Self {
        self.config.fix_suggestions_k = k;
        self
    }

    /// Set fusion strategy
    pub fn fusion(mut self, fusion: LocalizationFusion) -> Self {
        self.config.fusion = fusion;
        self
    }

    /// Enable explanations
    pub fn with_explanations(mut self) -> Self {
        self.config.include_explanations = true;
        self
    }

    /// Build the localizer
    pub fn build(self) -> RagFaultLocalizer {
        RagFaultLocalizer::new(self.knowledge_base.unwrap_or_default(), self.config)
    }
}

impl Default for RagFaultLocalizerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Integration helper for generating reports
pub struct RagReportGenerator;

impl RagReportGenerator {
    /// Generate YAML report
    pub fn to_yaml(result: &RagEnhancedResult) -> Result<String> {
        serde_yaml::to_string(result).map_err(|e| {
            RagLocalizationError::Serialization(serde_json::Error::io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                e.to_string(),
            )))
        })
    }

    /// Generate JSON report
    pub fn to_json(result: &RagEnhancedResult) -> Result<String> {
        serde_json::to_string_pretty(result).map_err(RagLocalizationError::Serialization)
    }

    /// Generate terminal report
    pub fn to_terminal(result: &RagEnhancedResult) -> String {
        let mut output = String::new();

        output.push_str("╔══════════════════════════════════════════════════════════════╗\n");
        output.push_str("║        RAG-ENHANCED FAULT LOCALIZATION REPORT                ║\n");
        output.push_str("╠══════════════════════════════════════════════════════════════╣\n");
        output.push_str(&format!(
            "║ SBFL Formula: {:?}                                           \n",
            result.sbfl_result.formula_used
        ));
        output.push_str(&format!(
            "║ Fusion Strategy: {}                                          \n",
            result.fusion_strategy
        ));
        output.push_str(&format!(
            "║ Knowledge Base: {} bugs                                      \n",
            result.knowledge_base_size
        ));
        output.push_str(&format!(
            "║ Tests: {} passed, {} failed                                  \n",
            result.sbfl_result.total_passed_tests, result.sbfl_result.total_failed_tests
        ));
        output.push_str("╠══════════════════════════════════════════════════════════════╣\n");
        output.push_str("║  TOP SUSPICIOUS STATEMENTS (RAG-Enhanced)                    ║\n");
        output.push_str("╠══════════════════════════════════════════════════════════════╣\n");

        for ranking in result.rankings.iter().take(10) {
            let file = ranking
                .sbfl_ranking
                .statement
                .file
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown");
            let line = ranking.sbfl_ranking.statement.line;
            let score = ranking.combined_score;

            // Score bar visualization
            let bar_len = (score * 20.0).min(20.0) as usize;
            let bar = "█".repeat(bar_len) + &"░".repeat(20 - bar_len);

            output.push_str(&format!(
                "║  #{:<2} {}:{:<6} {} {:.2}   ║\n",
                ranking.sbfl_ranking.rank, file, line, bar, score
            ));

            // Show similar bugs if any
            if !ranking.similar_bugs.is_empty() {
                let top_bug = &ranking.similar_bugs[0];
                output.push_str(&format!(
                    "║      → Similar: {} ({:.0}%)                      ║\n",
                    top_bug.summary,
                    top_bug.similarity * 100.0
                ));
            }
        }

        output.push_str("╚══════════════════════════════════════════════════════════════╝\n");

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tarantula::StatementId;

    // ============ BugDocument Tests ============

    #[test]
    fn test_bug_document_creation() {
        let bug = BugDocument::new(
            "bug-001",
            "Null pointer dereference",
            DefectCategory::MemorySafety,
        );
        assert_eq!(bug.id, "bug-001");
        assert_eq!(bug.title, "Null pointer dereference");
        assert_eq!(bug.category, DefectCategory::MemorySafety);
        assert_eq!(bug.severity, 3); // default
    }

    #[test]
    fn test_bug_document_builder() {
        let bug = BugDocument::new("bug-002", "Race condition", DefectCategory::Concurrency)
            .with_description("Thread safety issue in handler")
            .with_fix_commit("abc123")
            .with_affected_file("src/handler.rs")
            .with_severity(5)
            .with_symptom("Random test failures")
            .with_root_cause("Missing mutex lock")
            .with_fix_pattern("Add Arc<Mutex<T>> wrapper");

        assert_eq!(bug.description, "Thread safety issue in handler");
        assert_eq!(bug.fix_commit, "abc123");
        assert_eq!(bug.affected_files, vec!["src/handler.rs"]);
        assert_eq!(bug.severity, 5);
        assert_eq!(bug.symptoms, vec!["Random test failures"]);
        assert_eq!(bug.root_cause, "Missing mutex lock");
        assert_eq!(bug.fix_pattern, "Add Arc<Mutex<T>> wrapper");
    }

    #[test]
    fn test_bug_document_to_rag_chunk() {
        let bug = BugDocument::new("bug-003", "Buffer overflow", DefectCategory::MemorySafety)
            .with_description("Stack buffer overflow in parser")
            .with_affected_file("src/parser.rs");

        let chunk = bug.to_rag_chunk();
        assert!(chunk.content.contains("Buffer overflow"));
        assert!(chunk.content.contains("Stack buffer overflow"));
    }

    // ============ DefectCategory Tests ============

    #[test]
    fn test_defect_category_all() {
        let categories = DefectCategory::all();
        assert_eq!(categories.len(), 18);
    }

    #[test]
    fn test_defect_category_display() {
        assert_eq!(DefectCategory::MemorySafety.display_name(), "Memory Safety");
        assert_eq!(
            DefectCategory::OwnershipBorrow.display_name(),
            "Ownership/Borrow"
        );
    }

    // ============ BugKnowledgeBase Tests ============

    #[test]
    fn test_knowledge_base_new() {
        let kb = BugKnowledgeBase::new();
        assert!(kb.is_empty());
        assert_eq!(kb.len(), 0);
    }

    #[test]
    fn test_knowledge_base_add_bug() {
        let mut kb = BugKnowledgeBase::new();
        let bug = BugDocument::new("bug-001", "Test bug", DefectCategory::TypeErrors);
        kb.add_bug(bug);

        assert!(!kb.is_empty());
        assert_eq!(kb.len(), 1);
    }

    #[test]
    fn test_knowledge_base_get_bug() {
        let mut kb = BugKnowledgeBase::new();
        let bug = BugDocument::new("bug-001", "Test bug", DefectCategory::TypeErrors);
        kb.add_bug(bug);

        let retrieved = kb.get_bug("bug-001");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().title, "Test bug");

        assert!(kb.get_bug("nonexistent").is_none());
    }

    #[test]
    fn test_knowledge_base_get_by_category() {
        let mut kb = BugKnowledgeBase::new();
        kb.add_bug(BugDocument::new(
            "bug-001",
            "Bug 1",
            DefectCategory::MemorySafety,
        ));
        kb.add_bug(BugDocument::new(
            "bug-002",
            "Bug 2",
            DefectCategory::Concurrency,
        ));
        kb.add_bug(BugDocument::new(
            "bug-003",
            "Bug 3",
            DefectCategory::MemorySafety,
        ));

        let memory_bugs = kb.get_by_category(DefectCategory::MemorySafety);
        assert_eq!(memory_bugs.len(), 2);

        let concurrency_bugs = kb.get_by_category(DefectCategory::Concurrency);
        assert_eq!(concurrency_bugs.len(), 1);
    }

    // ============ LocalizationFusion Tests ============

    #[test]
    fn test_fusion_rrf() {
        let fusion = LocalizationFusion::RRF { k: 60.0 };
        let score = fusion.combine(0.9, 0.8, 0, 1);
        // RRF: 1/(60+0) + 1/(60+1) = 1/60 + 1/61
        let expected = 1.0 / 60.0 + 1.0 / 61.0;
        assert!((score - expected).abs() < 0.001);
    }

    #[test]
    fn test_fusion_linear() {
        let fusion = LocalizationFusion::Linear { sbfl_weight: 0.7 };
        let score = fusion.combine(1.0, 0.5, 0, 0);
        // Linear: 1.0 * 0.7 + 0.5 * 0.3 = 0.7 + 0.15 = 0.85
        assert!((score - 0.85).abs() < 0.001);
    }

    #[test]
    fn test_fusion_dbsf() {
        let fusion = LocalizationFusion::DBSF;
        let score = fusion.combine(0.8, 0.6, 0, 0);
        // DBSF: (0.8 + 0.6) / 2 = 0.7
        assert!((score - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_fusion_sbfl_only() {
        let fusion = LocalizationFusion::SbflOnly;
        let score = fusion.combine(0.9, 0.5, 0, 0);
        assert!((score - 0.9).abs() < 0.001);
    }

    // ============ RagLocalizationConfig Tests ============

    #[test]
    fn test_config_defaults() {
        let config = RagLocalizationConfig::default();
        assert_eq!(config.top_n, 10);
        assert_eq!(config.similar_bugs_k, 5);
        assert_eq!(config.fix_suggestions_k, 3);
        assert!(config.include_explanations);
    }

    #[test]
    fn test_config_builder() {
        let config = RagLocalizationConfig::new()
            .with_formula(SbflFormula::Tarantula)
            .with_top_n(20)
            .with_similar_bugs(10)
            .with_fix_suggestions(5)
            .with_fusion(LocalizationFusion::Linear { sbfl_weight: 0.8 })
            .with_explanations(false);

        assert!(matches!(config.sbfl_formula, SbflFormula::Tarantula));
        assert_eq!(config.top_n, 20);
        assert_eq!(config.similar_bugs_k, 10);
        assert_eq!(config.fix_suggestions_k, 5);
        assert!(!config.include_explanations);
    }

    // ============ RagFaultLocalizer Tests ============

    #[test]
    fn test_rag_localizer_creation() {
        let kb = BugKnowledgeBase::new();
        let localizer = RagFaultLocalizer::with_knowledge_base(kb);
        assert!(localizer.knowledge_base().is_empty());
    }

    #[test]
    fn test_rag_localizer_builder() {
        let mut kb = BugKnowledgeBase::new();
        kb.add_bug(BugDocument::new(
            "bug-001",
            "Test",
            DefectCategory::TypeErrors,
        ));

        let localizer = RagFaultLocalizerBuilder::new()
            .knowledge_base(kb)
            .formula(SbflFormula::Ochiai)
            .top_n(5)
            .similar_bugs(3)
            .fusion(LocalizationFusion::RRF { k: 60.0 })
            .with_explanations()
            .build();

        assert_eq!(localizer.knowledge_base().len(), 1);
    }

    #[test]
    fn test_rag_localizer_localize() {
        let mut kb = BugKnowledgeBase::new();
        kb.add_bug(
            BugDocument::new(
                "bug-001",
                "Null pointer in parser",
                DefectCategory::MemorySafety,
            )
            .with_description("Parser crashes on null input")
            .with_fix_pattern("Add null check"),
        );

        let localizer = RagFaultLocalizer::with_knowledge_base(kb);

        let coverage = vec![
            StatementCoverage::new(StatementId::new("src/parser.rs", 10), 5, 8),
            StatementCoverage::new(StatementId::new("src/parser.rs", 20), 90, 2),
        ];

        let result = localizer.localize(&coverage, 100, 10);

        assert!(!result.rankings.is_empty());
        assert_eq!(result.knowledge_base_size, 1);
        assert_eq!(result.fusion_strategy, "RRF");
    }

    #[test]
    fn test_rag_localizer_empty_kb() {
        let kb = BugKnowledgeBase::new();
        let localizer = RagFaultLocalizer::with_knowledge_base(kb);

        let coverage = vec![StatementCoverage::new(
            StatementId::new("src/test.rs", 10),
            5,
            8,
        )];

        let result = localizer.localize(&coverage, 100, 10);

        assert!(!result.rankings.is_empty());
        assert!(result.rankings[0].similar_bugs.is_empty());
        assert_eq!(result.knowledge_base_size, 0);
    }

    // ============ RagReportGenerator Tests ============

    #[test]
    fn test_report_generator_terminal() {
        let mut kb = BugKnowledgeBase::new();
        kb.add_bug(BugDocument::new(
            "bug-001",
            "Test bug",
            DefectCategory::TypeErrors,
        ));

        let localizer = RagFaultLocalizer::with_knowledge_base(kb);
        let coverage = vec![StatementCoverage::new(
            StatementId::new("src/test.rs", 10),
            5,
            8,
        )];
        let result = localizer.localize(&coverage, 100, 10);

        let report = RagReportGenerator::to_terminal(&result);
        assert!(report.contains("RAG-ENHANCED"));
        assert!(report.contains("SBFL Formula"));
    }

    #[test]
    fn test_report_generator_json() {
        let kb = BugKnowledgeBase::new();
        let localizer = RagFaultLocalizer::with_knowledge_base(kb);
        let coverage = vec![StatementCoverage::new(
            StatementId::new("src/test.rs", 10),
            5,
            8,
        )];
        let result = localizer.localize(&coverage, 100, 10);

        let json = RagReportGenerator::to_json(&result).unwrap();
        assert!(json.contains("rankings"));
        assert!(json.contains("fusion_strategy"));
    }

    // ============ SimilarBug Tests ============

    #[test]
    fn test_similar_bug_serialization() {
        let bug = SimilarBug {
            id: "bug-001".to_string(),
            similarity: 0.85,
            category: DefectCategory::MemorySafety,
            summary: "Null pointer".to_string(),
            fix_commit: "abc123".to_string(),
        };

        let json = serde_json::to_string(&bug).unwrap();
        assert!(json.contains("bug-001"));
        assert!(json.contains("0.85"));
    }

    // ============ SuggestedFix Tests ============

    #[test]
    fn test_suggested_fix_serialization() {
        let fix = SuggestedFix {
            pattern: "Add null check".to_string(),
            confidence: 0.9,
            example: "if x.is_some() { ... }".to_string(),
            source_bug_id: "bug-001".to_string(),
        };

        let json = serde_json::to_string(&fix).unwrap();
        assert!(json.contains("Add null check"));
        assert!(json.contains("0.9"));
    }
}
