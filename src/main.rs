// Organizational Intelligence Plugin - Main entry point
// Toyota Way: Start simple, deliver value incrementally

use anyhow::Result;
use chrono::Utc;
use clap::Parser;
use organizational_intelligence_plugin::github::GitHubMiner;
use organizational_intelligence_plugin::report::{
    AnalysisMetadata, AnalysisReport, ReportGenerator,
};
use organizational_intelligence_plugin::{Cli, Commands};
use std::env;
use tracing::{error, info, warn, Level};

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

    info!(
        "🚀 Organizational Intelligence Plugin v{}",
        env!("CARGO_PKG_VERSION")
    );

    // Handle commands
    match cli.command {
        Commands::Analyze {
            org,
            output,
            max_concurrent,
        } => {
            info!("Analyzing organization: {}", org);
            info!("Output file: {}", output.display());
            info!("Max concurrent: {}", max_concurrent);

            // Initialize GitHub client
            let github_token = env::var("GITHUB_TOKEN").ok();
            if github_token.is_none() {
                warn!("GITHUB_TOKEN not set - using unauthenticated requests (lower rate limits)");
                info!("Set GITHUB_TOKEN environment variable for higher rate limits");
            }

            let miner = GitHubMiner::new(github_token);

            // Fetch organization repositories
            info!("Fetching repositories for organization: {}", org);
            match miner.fetch_organization_repos(&org).await {
                Ok(repos) => {
                    info!("✅ Successfully fetched {} repositories", repos.len());

                    println!("\n📊 Organization Analysis: {}", org);
                    println!("   Repositories found: {}", repos.len());

                    // Display top 5 repositories by stars
                    let mut sorted_repos = repos.clone();
                    sorted_repos.sort_by(|a, b| b.stars.cmp(&a.stars));

                    println!("\n⭐ Top repositories by stars:");
                    for (i, repo) in sorted_repos.iter().take(5).enumerate() {
                        println!(
                            "   {}. {} ({} ⭐) - {}",
                            i + 1,
                            repo.name,
                            repo.stars,
                            repo.language.as_deref().unwrap_or("Unknown")
                        );
                    }

                    // Generate YAML report
                    info!("Generating YAML report");
                    let report_generator = ReportGenerator::new();

                    let metadata = AnalysisMetadata {
                        organization: org.clone(),
                        analysis_date: Utc::now().to_rfc3339(),
                        repositories_analyzed: repos.len(),
                        commits_analyzed: 0, // Phase 1: Not analyzing commits yet
                        analyzer_version: env!("CARGO_PKG_VERSION").to_string(),
                    };

                    let report = AnalysisReport {
                        version: "1.0".to_string(),
                        metadata,
                        defect_patterns: vec![], // Phase 1: No classifier yet
                    };

                    // Write report to file
                    report_generator.write_to_file(&report, &output).await?;

                    info!("✅ Report written to {}", output.display());
                    println!("\n📄 Report saved to: {}", output.display());

                    println!("\n📋 Phase 1 Progress:");
                    println!("   1. ✅ CLI structure (DONE)");
                    println!("   2. ✅ GitHub API integration (DONE)");
                    println!("   3. ✅ YAML output generation (DONE)");
                    println!("   4. 🔄 Git history analysis (TODO)");
                    println!("   5. 🔄 Rule-based classifier (TODO)");
                    println!("\n🎯 Following EXTREME TDD - one feature at a time");

                    Ok(())
                }
                Err(e) => {
                    error!("Failed to fetch repositories: {}", e);
                    eprintln!("❌ Error: {}", e);
                    Err(e)
                }
            }
        }
    }
}
