// Organizational Intelligence Plugin - Main entry point
// Toyota Way: Start simple, deliver value incrementally

use anyhow::Result;
use chrono::Utc;
use clap::Parser;
use organizational_intelligence_plugin::analyzer::OrgAnalyzer;
use organizational_intelligence_plugin::github::GitHubMiner;
use organizational_intelligence_plugin::report::{
    AnalysisMetadata, AnalysisReport, ReportGenerator,
};
use organizational_intelligence_plugin::{Cli, Commands};
use std::env;
use tempfile::TempDir;
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

                    // Analyze defect patterns
                    info!("Analyzing defect patterns in top repositories");
                    println!("\n🔍 Analyzing defect patterns...");

                    let temp_dir = TempDir::new()?;
                    let analyzer = OrgAnalyzer::new(temp_dir.path());

                    let mut all_patterns = vec![];
                    let mut total_commits = 0;

                    // Analyze top repositories (limit by max_concurrent)
                    let repos_to_analyze = sorted_repos.iter().take(max_concurrent);

                    for (i, repo) in repos_to_analyze.enumerate() {
                        println!(
                            "   [{}/{}] Analyzing: {}",
                            i + 1,
                            max_concurrent.min(sorted_repos.len()),
                            repo.name
                        );

                        let repo_url = format!("https://github.com/{}/{}", org, repo.name);

                        match analyzer
                            .analyze_repository(&repo_url, &repo.name, 100)
                            .await
                        {
                            Ok(patterns) => {
                                total_commits += 100;
                                all_patterns.extend(patterns);
                                info!("✅ Analyzed {}", repo.name);
                            }
                            Err(e) => {
                                warn!("Failed to analyze {}: {}", repo.name, e);
                                println!("     ⚠️  Skipping {} (error: {})", repo.name, e);
                            }
                        }
                    }

                    println!("   ✅ Analysis complete!");

                    // Generate YAML report
                    info!("Generating YAML report");
                    let report_generator = ReportGenerator::new();

                    let metadata = AnalysisMetadata {
                        organization: org.clone(),
                        analysis_date: Utc::now().to_rfc3339(),
                        repositories_analyzed: max_concurrent.min(sorted_repos.len()),
                        commits_analyzed: total_commits,
                        analyzer_version: env!("CARGO_PKG_VERSION").to_string(),
                    };

                    let report = AnalysisReport {
                        version: "1.0".to_string(),
                        metadata,
                        defect_patterns: all_patterns,
                    };

                    // Write report to file
                    report_generator.write_to_file(&report, &output).await?;

                    info!("✅ Report written to {}", output.display());
                    println!("\n📄 Report saved to: {}", output.display());

                    println!("\n🎯 Phase 1 MVP Complete!");
                    println!("   ✅ CLI structure");
                    println!("   ✅ GitHub API integration");
                    println!("   ✅ YAML output generation");
                    println!("   ✅ Git history analysis");
                    println!("   ✅ Rule-based defect classifier");
                    println!("   ✅ Pattern aggregation");

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
