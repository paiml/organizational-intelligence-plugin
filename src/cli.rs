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
