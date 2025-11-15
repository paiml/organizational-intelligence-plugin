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
