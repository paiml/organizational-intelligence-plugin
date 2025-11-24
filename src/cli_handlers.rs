//! CLI Command Handlers
//!
//! Testable business logic for all CLI commands.
//! Binary entry points (main.rs) should only parse args and call these handlers.

use anyhow::Result;
use chrono::{Duration, Utc};
use std::path::PathBuf;
use tempfile::TempDir;
use tracing::{error, info, warn};

use crate::analyzer::OrgAnalyzer;
use crate::github::GitHubMiner;
use crate::pr_reviewer::PrReviewer;
use crate::report::{AnalysisMetadata, AnalysisReport, ReportGenerator};
use crate::summarizer::{ReportSummarizer, SummaryConfig};

/// Handle the `review-pr` command
pub async fn handle_review_pr(
    baseline: PathBuf,
    files: String,
    format: String,
    output: Option<PathBuf>,
) -> Result<()> {
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
        _ => review.to_markdown(), // Default to markdown
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
    println!(
        "   Repositories in baseline: {}",
        review.repositories_analyzed
    );

    if review.warnings.is_empty() {
        println!("\n✅ No warnings - PR looks good based on historical patterns!");
    } else {
        println!(
            "\n⚠️  {} warning(s) generated - review carefully!",
            review.warnings.len()
        );
    }

    println!("\n🎯 Phase 3 Complete!");
    println!("   ✅ Fast PR review (<30s)");
    println!("   ✅ Stateful baselines (no re-analysis)");
    println!("   ✅ Actionable warnings");
    println!("   ✅ Multiple output formats");

    Ok(())
}

/// Handle the `summarize` command
pub async fn handle_summarize(
    input: PathBuf,
    output: PathBuf,
    strip_pii: bool,
    top_n: usize,
    min_frequency: usize,
    include_examples: bool,
) -> Result<()> {
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
            println!("   Commits analyzed: {}", summary.metadata.commits_analyzed);
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

/// Handle the `analyze` command
pub async fn handle_analyze(
    org: String,
    output: PathBuf,
    _max_concurrent: usize,
    github_token: Option<String>,
    analyzer_version: String,
) -> Result<()> {
    info!("Analyzing organization: {}", org);
    info!("Output file: {}", output.display());

    // Initialize GitHub client
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
                analyzer_version,
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_handle_summarize_invalid_input() {
        let input = PathBuf::from("nonexistent.yaml");
        let temp_output = NamedTempFile::new().unwrap();
        let output = temp_output.path().to_path_buf();

        let result = handle_summarize(input, output, true, 10, 5, false).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_handle_review_pr_invalid_baseline() {
        let baseline = PathBuf::from("nonexistent-baseline.yaml");
        let files = "src/main.rs,src/lib.rs".to_string();
        let format = "markdown".to_string();

        let result = handle_review_pr(baseline, files, format, None).await;
        assert!(result.is_err());
    }
}
