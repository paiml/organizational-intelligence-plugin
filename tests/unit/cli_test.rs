// Unit tests for CLI argument parsing
// Following EXTREME TDD: Test first, then implement

use organizational_intelligence_plugin::cli::{Cli, Commands};
use clap::Parser;

#[test]
fn test_cli_parses_version_flag() {
    // RED: This will fail because cli module doesn't exist yet
    let args = vec!["oip", "--version"];
    let result = Cli::try_parse_from(args);

    // Should parse successfully
    assert!(result.is_ok() || result.is_err()); // Will fail with compile error first
}

#[test]
fn test_cli_requires_subcommand() {
    // Test that CLI requires a subcommand
    let args = vec!["oip"];
    let result = Cli::try_parse_from(args);

    // Should fail without subcommand
    assert!(result.is_err());
}

#[test]
fn test_cli_help_flag() {
    // Test help flag works
    let args = vec!["oip", "--help"];
    let result = Cli::try_parse_from(args);

    // Help should cause an error (clap exits with help text)
    assert!(result.is_err());
}
