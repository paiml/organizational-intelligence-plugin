// Integration tests for CLI
// Following EXTREME TDD: Test first

use clap::Parser;
use organizational_intelligence_plugin::{Cli, Commands};

#[test]
fn test_cli_requires_subcommand() {
    // RED: Test that CLI requires a subcommand
    let args = vec!["oip"];
    let result = Cli::try_parse_from(args);

    // Should fail without subcommand
    assert!(result.is_err(), "CLI should require a subcommand");
}

#[test]
fn test_cli_analyze_command_requires_org() {
    // RED: Test that analyze command requires --org
    let args = vec!["oip", "analyze"];
    let result = Cli::try_parse_from(args);

    // Should fail without --org argument
    assert!(result.is_err(), "Analyze command should require --org");
}

#[test]
fn test_cli_analyze_command_with_org() {
    // GREEN: Test successful parsing
    let args = vec!["oip", "analyze", "--org", "rust-lang"];
    let result = Cli::try_parse_from(args);

    assert!(result.is_ok(), "Should parse valid analyze command");

    let cli = result.unwrap();
    match cli.command {
        Commands::Analyze { org, .. } => {
            assert_eq!(org, "rust-lang");
        }
    }
}

#[test]
fn test_cli_analyze_command_with_output() {
    // GREEN: Test output parameter
    let args = vec![
        "oip",
        "analyze",
        "--org",
        "test-org",
        "--output",
        "custom.yaml",
    ];
    let result = Cli::try_parse_from(args);

    assert!(result.is_ok());

    let cli = result.unwrap();
    match cli.command {
        Commands::Analyze { org, output, .. } => {
            assert_eq!(org, "test-org");
            assert_eq!(output.to_str().unwrap(), "custom.yaml");
        }
    }
}

#[test]
fn test_cli_global_verbose_flag() {
    // Test global verbose flag
    let args = vec!["oip", "--verbose", "analyze", "--org", "test"];
    let result = Cli::try_parse_from(args);

    assert!(result.is_ok());

    let cli = result.unwrap();
    assert!(cli.verbose, "Verbose flag should be set");
}
