//! Organizational Intelligence Plugin - Binary Entry Point
//! Toyota Way: Thin entry point, testable business logic in cli_handlers module

use anyhow::Result;
use clap::Parser;
use organizational_intelligence_plugin::cli_handlers;
use organizational_intelligence_plugin::{Cli, Commands};
use std::env;
use tracing::Level;

#[tokio::main]
async fn main() -> Result<()> {
    // Parse CLI arguments
    let cli = Cli::parse();

    // Initialize logging
    let level = if cli.verbose {
        Level::DEBUG
    } else {
        Level::INFO
    };
    tracing_subscriber::fmt::fmt().with_max_level(level).init();

    tracing::info!(
        "🚀 Organizational Intelligence Plugin v{}",
        env!("CARGO_PKG_VERSION")
    );

    // Handle commands by calling handlers (all business logic is testable)
    match cli.command {
        Commands::ReviewPr {
            baseline,
            files,
            format,
            output,
        } => cli_handlers::handle_review_pr(baseline, files, format, output).await,

        Commands::Summarize {
            input,
            output,
            strip_pii,
            top_n,
            min_frequency,
            include_examples,
        } => {
            cli_handlers::handle_summarize(
                input,
                output,
                strip_pii,
                top_n,
                min_frequency,
                include_examples,
            )
            .await
        }

        Commands::Analyze {
            org,
            output,
            max_concurrent,
        } => {
            let github_token = env::var("GITHUB_TOKEN").ok();
            let analyzer_version = env!("CARGO_PKG_VERSION").to_string();
            cli_handlers::handle_analyze(
                org,
                output,
                max_concurrent,
                github_token,
                analyzer_version,
            )
            .await
        }

        Commands::ExtractTrainingData {
            repo,
            output,
            min_confidence,
            max_commits,
            create_splits,
        } => {
            cli_handlers::handle_extract_training_data(
                repo,
                output,
                min_confidence,
                max_commits,
                create_splits,
            )
            .await
        }

        Commands::TrainClassifier {
            input,
            output,
            n_estimators,
            max_depth,
            max_features,
        } => {
            cli_handlers::handle_train_classifier(
                input,
                output,
                n_estimators,
                max_depth,
                max_features,
            )
            .await
        }
    }
}
