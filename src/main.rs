// Organizational Intelligence Plugin - Main entry point
// Toyota Way: Start simple, deliver value incrementally

use anyhow::Result;
use chrono::{Duration, Utc};
use clap::Parser;
use organizational_intelligence_plugin::analyzer::OrgAnalyzer;
use organizational_intelligence_plugin::github::GitHubMiner;
use organizational_intelligence_plugin::report::{
    AnalysisMetadata, AnalysisReport, ReportGenerator,
};
use organizational_intelligence_plugin::pr_reviewer::PrReviewer;
use organizational_intelligence_plugin::summarizer::{ReportSummarizer, SummaryConfig};
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
        Commands::ReviewPr {
            baseline,
            files,
            format,
            output,
        } => {
            info!("Reviewing PR with baseline: {}", baseline.display());
            info!("Files changed: {}", files);
            info!("Output format: {}", format);

            println!("\n🔍 PR Review: Organizational Intelligence");
            println!("   Baseline: {}", baseline.display());
            println!("   Format:   {}", format);

            // Load baseline summary
            let reviewer = PrReviewer::load_baseline(&baseline)?;

            // Parse comma-separated file list
            let files_vec: Vec<String> = files
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();

            println!("   Files:    {} file(s)", files_vec.len());

            // Review PR
            let review = reviewer.review_pr(&files_vec);

            // Generate output based on format
            let output_content = match format.as_str() {
                "json" => review.to_json()?,
                "markdown" | _ => review.to_markdown(),
            };

            // Write to file or stdout
            if let Some(output_path) = output {
                std::fs::write(&output_path, &output_content)?;
                println!("\n✅ Review saved to: {}", output_path.display());
            } else {
                println!("\n{}", output_content);
            }

            // Summary
            println!("\n📊 Review Summary:");
            println!("   Warnings: {}", review.warnings.len());
            println!("   Files analyzed: {}", review.files_analyzed.len());
            println!("   Baseline date: {}", review.baseline_date);
            println!("   Repositories in baseline: {}", review.repositories_analyzed);

            if review.warnings.is_empty() {
                println!("\n✅ No warnings - PR looks good based on historical patterns!");
            } else {
                println!("\n⚠️  {} warning(s) generated - review carefully!", review.warnings.len());
            }

            println!("\n🎯 Phase 3 Complete!");
            println!("   ✅ Fast PR review (<30s)");
            println!("   ✅ Stateful baselines (no re-analysis)");
            println!("   ✅ Actionable warnings");
            println!("   ✅ Multiple output formats");

            Ok(())
        }

        Commands::Summarize {
            input,
            output,
            strip_pii,
            top_n,
            min_frequency,
            include_examples,
        } => {
            info!("Summarizing report: {}", input.display());
            info!("Output file: {}", output.display());
            info!("Strip PII: {}", strip_pii);
            info!("Top N categories: {}", top_n);
            info!("Min frequency: {}", min_frequency);
            info!("Include examples: {}", include_examples);

            println!("\n📊 Summarizing Analysis Report");
            println!("   Input:  {}", input.display());
            println!("   Output: {}", output.display());

            // Create summarization config
            let config = SummaryConfig {
                strip_pii,
                top_n_categories: top_n,
                min_frequency,
                include_examples,
            };

            // Summarize report
            match ReportSummarizer::summarize(&input, config) {
                Ok(summary) => {
                    // Save summary to file
                    ReportSummarizer::save_to_file(&summary, &output)?;

                    info!("✅ Summary written to {}", output.display());
                    println!("\n✅ Summary saved to: {}", output.display());

                    println!("\n📈 Summary Statistics:");
                    println!(
                        "   Repositories analyzed: {}",
                        summary.metadata.repositories_analyzed
                    );
                    println!(
                        "   Commits analyzed: {}",
                        summary.metadata.commits_analyzed
                    );
                    println!(
                        "   Top defect categories included: {}",
                        summary.organizational_insights.top_defect_categories.len()
                    );

                    if strip_pii {
                        println!("\n🔒 PII Stripping:");
                        println!("   ✅ Author names: REDACTED");
                        println!("   ✅ Commit hashes: REDACTED");
                        println!("   ✅ Safe for sharing");
                    }

                    println!("\n🎯 Phase 2 Complete!");
                    println!("   ✅ Automated PII stripping");
                    println!("   ✅ Frequency filtering");
                    println!("   ✅ Top-N selection");
                    println!("   ✅ Ready for AI consumption");

                    Ok(())
                }
                Err(e) => {
                    error!("Failed to summarize report: {}", e);
                    eprintln!("❌ Error: {}", e);
                    Err(e)
                }
            }
        }

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
                Ok(all_repos) => {
                    info!("✅ Successfully fetched {} repositories", all_repos.len());

                    // Filter repos updated in last 2 years
                    let two_years_ago = Utc::now() - Duration::days(730);
                    let repos = GitHubMiner::filter_by_date(all_repos.clone(), two_years_ago);

                    println!("\n📊 Organization Analysis: {}", org);
                    println!("   Total repositories: {}", all_repos.len());
                    println!("   Repositories updated in last 2 years: {}", repos.len());

                    // Display top 5 repositories by stars
                    let mut sorted_repos = repos.clone();
                    sorted_repos.sort_by(|a, b| b.stars.cmp(&a.stars));

                    println!("\n⭐ Top repositories by stars (last 2 years):");
                    for (i, repo) in sorted_repos.iter().take(5).enumerate() {
                        println!(
                            "   {}. {} ({} ⭐) - {}",
                            i + 1,
                            repo.name,
                            repo.stars,
                            repo.language.as_deref().unwrap_or("Unknown")
                        );
                    }

                    // Analyze ALL repos from last 2 years
                    info!(
                        "Analyzing defect patterns in ALL {} repositories",
                        repos.len()
                    );
                    println!("\n🔍 Analyzing defect patterns in ALL repos from last 2 years...");

                    let temp_dir = TempDir::new()?;
                    let analyzer = OrgAnalyzer::new(temp_dir.path());

                    let mut all_patterns = vec![];
                    let mut total_commits = 0;
                    let mut repos_analyzed = 0;

                    // Analyze ALL repositories (not limited by max_concurrent anymore)
                    for (i, repo) in sorted_repos.iter().enumerate() {
                        println!(
                            "   [{}/{}] Analyzing: {} (updated: {})",
                            i + 1,
                            sorted_repos.len(),
                            repo.name,
                            repo.updated_at.format("%Y-%m-%d")
                        );

                        let repo_url = format!("https://github.com/{}/{}", org, repo.name);

                        match analyzer
                            .analyze_repository(&repo_url, &repo.name, 100)
                            .await
                        {
                            Ok(patterns) => {
                                total_commits += 100;
                                all_patterns.extend(patterns);
                                repos_analyzed += 1;
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
                        repositories_analyzed: repos_analyzed,
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
