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
        _ => panic!("Expected Analyze command"),
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
        _ => panic!("Expected Analyze command"),
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

#[test]
fn test_cli_summarize_command_requires_input_and_output() {
    // Test that summarize command requires both input and output
    let args = vec!["oip", "summarize"];
    let result = Cli::try_parse_from(args);

    assert!(
        result.is_err(),
        "Summarize command should require --input and --output"
    );
}

#[test]
fn test_cli_summarize_command_with_required_args() {
    // Test successful parsing of summarize command
    let args = vec![
        "oip",
        "summarize",
        "--input",
        "report.yaml",
        "--output",
        "summary.yaml",
    ];
    let result = Cli::try_parse_from(args);

    assert!(result.is_ok(), "Should parse valid summarize command");

    let cli = result.unwrap();
    match cli.command {
        Commands::Summarize { input, output, .. } => {
            assert_eq!(input.to_str().unwrap(), "report.yaml");
            assert_eq!(output.to_str().unwrap(), "summary.yaml");
        }
        _ => panic!("Expected Summarize command"),
    }
}

#[test]
fn test_cli_summarize_command_with_all_options() {
    // Test all summarize command options
    let args = vec![
        "oip",
        "summarize",
        "--input",
        "input.yaml",
        "--output",
        "output.yaml",
        "--strip-pii",
        "--top-n",
        "5",
        "--min-frequency",
        "3",
        "--include-examples",
    ];
    let result = Cli::try_parse_from(args);

    assert!(result.is_ok());

    let cli = result.unwrap();
    match cli.command {
        Commands::Summarize {
            input,
            output,
            strip_pii,
            top_n,
            min_frequency,
            include_examples,
        } => {
            assert_eq!(input.to_str().unwrap(), "input.yaml");
            assert_eq!(output.to_str().unwrap(), "output.yaml");
            assert!(strip_pii, "strip-pii should be true");
            assert_eq!(top_n, 5);
            assert_eq!(min_frequency, 3);
            assert!(include_examples, "include-examples should be true");
        }
        _ => panic!("Expected Summarize command"),
    }
}
