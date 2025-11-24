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
