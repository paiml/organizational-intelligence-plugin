// CLI argument parsing for OIP
// Following EXTREME TDD: Minimal implementation to make tests compile

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "oip")]
#[command(about = "Organizational Intelligence Plugin - Defect Pattern Analysis", long_about = None)]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// Enable verbose logging
    #[arg(long, global = true)]
    pub verbose: bool,

    /// Configuration file path
    #[arg(long, global = true)]
    pub config: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Analyze GitHub organization for defect patterns
    Analyze {
        /// Organization name
        #[arg(long, required = true)]
        org: String,

        /// Output file path
        #[arg(long, short, default_value = "defects.yaml")]
        output: PathBuf,

        /// Maximum concurrent repository analysis
        #[arg(long, default_value = "10")]
        max_concurrent: usize,

        /// Path to trained ML model (optional, uses rule-based if not provided)
        #[arg(long)]
        model: Option<PathBuf>,

        /// Confidence threshold for ML predictions (0.0-1.0)
        #[arg(long, default_value = "0.65")]
        ml_confidence: f32,
    },

    /// Summarize analysis report for AI consumption (Phase 2)
    Summarize {
        /// Input YAML report from 'analyze' command
        #[arg(long, short, required = true)]
        input: PathBuf,

        /// Output summary file
        #[arg(long, short, required = true)]
        output: PathBuf,

        /// Strip PII (author names, commit hashes, email addresses)
        #[arg(long, default_value = "true")]
        strip_pii: bool,

        /// Top N defect categories to include
        #[arg(long, default_value = "10")]
        top_n: usize,

        /// Minimum frequency to include
        #[arg(long, default_value = "5")]
        min_frequency: usize,

        /// Include anonymized examples (with PII redacted if strip-pii is true)
        #[arg(long, default_value = "false")]
        include_examples: bool,
    },

    /// Review PR with organizational context (Phase 3)
    ReviewPr {
        /// Baseline summary from weekly analysis
        #[arg(long, short, required = true)]
        baseline: PathBuf,

        /// Files changed in PR (comma-separated)
        #[arg(long, short, required = true)]
        files: String,

        /// Output format: markdown, json
        #[arg(long, default_value = "markdown")]
        format: String,

        /// Output file (stdout if not specified)
        #[arg(long, short)]
        output: Option<PathBuf>,
    },

    /// Extract training data from Git repository (Phase 2 ML)
    ExtractTrainingData {
        /// Path to Git repository
        #[arg(long, short, required = true)]
        repo: PathBuf,

        /// Output JSON file
        #[arg(long, short, default_value = "training-data.json")]
        output: PathBuf,

        /// Minimum confidence threshold (0.0-1.0)
        #[arg(long, default_value = "0.75")]
        min_confidence: f32,

        /// Maximum commits to analyze
        #[arg(long, default_value = "1000")]
        max_commits: usize,

        /// Create train/validation/test splits
        #[arg(long, default_value = "true")]
        create_splits: bool,

        /// Show visualization (requires --features viz)
        #[arg(long, default_value = "false")]
        viz: bool,
    },

    /// Train ML classifier on extracted training data (Phase 2 ML)
    TrainClassifier {
        /// Input training data JSON file
        #[arg(long, short, required = true)]
        input: PathBuf,

        /// Output model file (optional)
        #[arg(long, short)]
        output: Option<PathBuf>,

        /// Number of trees in Random Forest
        #[arg(long, default_value = "100")]
        n_estimators: usize,

        /// Maximum tree depth
        #[arg(long, default_value = "20")]
        max_depth: usize,

        /// Maximum TF-IDF features
        #[arg(long, default_value = "1500")]
        max_features: usize,
    },

    /// Export CommitFeatures to aprender-compatible format (Issue #2)
    Export {
        /// Path to Git repository to analyze
        #[arg(long, short, required = true)]
        repo: PathBuf,

        /// Output file path
        #[arg(long, short, default_value = "features.json")]
        output: PathBuf,

        /// Export format: json, binary, parquet
        #[arg(long, short, default_value = "json")]
        format: String,

        /// Maximum commits to analyze
        #[arg(long, default_value = "1000")]
        max_commits: usize,

        /// Minimum confidence threshold for classification (0.0-1.0)
        #[arg(long, default_value = "0.70")]
        min_confidence: f32,
    },

    /// Import Depyler CITL corpus for ground-truth training labels (NLP-014)
    ImportDepyler {
        /// Path to Depyler JSONL export file
        #[arg(long, short, required = true)]
        input: PathBuf,

        /// Output training data JSON file
        #[arg(long, short, default_value = "citl-training.json")]
        output: PathBuf,

        /// Minimum confidence threshold (0.0-1.0)
        #[arg(long, default_value = "0.75")]
        min_confidence: f32,

        /// Merge with existing training data file (optional)
        #[arg(long, short)]
        merge: Option<PathBuf>,

        /// Create train/validation/test splits
        #[arg(long, default_value = "true")]
        create_splits: bool,
    },

    /// Localize faults using Tarantula SBFL (Spectrum-Based Fault Localization)
    Localize {
        /// Path to LCOV coverage file from passing tests
        #[arg(long, required = true)]
        passed_coverage: PathBuf,

        /// Path to LCOV coverage file from failing tests
        #[arg(long, required = true)]
        failed_coverage: PathBuf,

        /// Number of passing tests
        #[arg(long, default_value = "1")]
        passed_count: usize,

        /// Number of failing tests
        #[arg(long, default_value = "1")]
        failed_count: usize,

        /// SBFL formula: tarantula, ochiai, dstar2, dstar3
        #[arg(long, default_value = "tarantula")]
        formula: String,

        /// Top N suspicious statements to report
        #[arg(long, default_value = "10")]
        top_n: usize,

        /// Output file path
        #[arg(long, short, default_value = "fault-localization.yaml")]
        output: PathBuf,

        /// Output format: yaml, json, terminal
        #[arg(long, short, default_value = "yaml")]
        format: String,

        /// Include TDG scores from pmat (requires pmat)
        #[arg(long, default_value = "false")]
        enrich_tdg: bool,

        /// Repository path for TDG enrichment
        #[arg(long)]
        repo: Option<PathBuf>,

        /// Enable RAG-enhanced localization with trueno-rag
        #[arg(long, default_value = "false")]
        rag: bool,

        /// Path to bug knowledge base YAML file (for RAG)
        #[arg(long)]
        knowledge_base: Option<PathBuf>,

        /// Fusion strategy for RAG: rrf, linear, dbsf, sbfl-only
        #[arg(long, default_value = "rrf")]
        fusion: String,

        /// Number of similar bugs to retrieve (for RAG)
        #[arg(long, default_value = "5")]
        similar_bugs: usize,

        /// Enable weighted ensemble model (Phase 6)
        #[arg(long, default_value = "false")]
        ensemble: bool,

        /// Path to trained ensemble model file
        #[arg(long)]
        ensemble_model: Option<PathBuf>,

        /// Include churn metrics from git history (for ensemble)
        #[arg(long, default_value = "false")]
        include_churn: bool,

        /// Enable calibrated probability output (Phase 7)
        #[arg(long, default_value = "false")]
        calibrated: bool,

        /// Path to trained calibration model file
        #[arg(long)]
        calibration_model: Option<PathBuf>,

        /// Confidence threshold for calibrated predictions (0.0-1.0)
        #[arg(long, default_value = "0.5")]
        confidence_threshold: f32,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_structure_exists() {
        // Verify the CLI structure compiles
        // This is a sanity check test
        let _cli_type_check: Option<Cli> = None;
        let _commands_type_check: Option<Commands> = None;
    }
}
