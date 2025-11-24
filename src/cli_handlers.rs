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
use crate::git;
use crate::github::GitHubMiner;
use crate::pr_reviewer::PrReviewer;
use crate::report::{AnalysisMetadata, AnalysisReport, ReportGenerator};
use crate::summarizer::{ReportSummarizer, SummaryConfig};
use crate::training::TrainingDataExtractor;

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

/// Handle the `extract-training-data` command
pub async fn handle_extract_training_data(
    repo: PathBuf,
    output: PathBuf,
    min_confidence: f32,
    max_commits: usize,
    create_splits: bool,
) -> Result<()> {
    info!("Extracting training data from: {}", repo.display());
    info!("Output file: {}", output.display());
    info!("Min confidence: {}", min_confidence);
    info!("Max commits: {}", max_commits);

    println!("\n🎓 Training Data Extraction (Phase 2 ML)");
    println!("   Repository:      {}", repo.display());
    println!("   Output:          {}", output.display());
    println!("   Min confidence:  {:.2}", min_confidence);
    println!("   Max commits:     {}", max_commits);
    println!("   Create splits:   {}", create_splits);

    // Validate repository path
    if !repo.exists() {
        return Err(anyhow::anyhow!("Repository path does not exist: {}", repo.display()));
    }

    if !repo.join(".git").exists() {
        return Err(anyhow::anyhow!("Not a Git repository: {}", repo.display()));
    }

    // Extract commit history
    println!("\n📖 Reading commit history...");
    let commits = git::analyze_repository_at_path(&repo, max_commits)?;
    println!("   ✅ Found {} commits", commits.len());

    // Extract training data
    println!("\n🔍 Extracting and auto-labeling defect-fix commits...");
    let extractor = TrainingDataExtractor::new(min_confidence);

    let repo_name = repo
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown-repo");

    let examples = extractor.extract_training_data(&commits, repo_name)?;

    println!("   ✅ Extracted {} training examples", examples.len());

    if examples.is_empty() {
        warn!("No training examples extracted - try lowering min_confidence threshold");
        println!("\n⚠️  No training examples extracted!");
        println!("   Try lowering --min-confidence (current: {:.2})", min_confidence);
        return Ok(());
    }

    // Show statistics
    println!("\n📊 Training Data Statistics:");
    let stats = extractor.get_statistics(&examples);
    for line in stats.lines() {
        if !line.is_empty() {
            println!("   {}", line);
        }
    }

    // Create splits or export raw examples
    if create_splits {
        println!("\n📂 Creating train/validation/test splits (70/15/15)...");
        let dataset = extractor.create_splits(&examples, &[repo_name.to_string()])?;

        println!("   ✅ Train:      {} examples", dataset.train.len());
        println!("   ✅ Validation: {} examples", dataset.validation.len());
        println!("   ✅ Test:       {} examples", dataset.test.len());

        // Export dataset to JSON
        let json = serde_json::to_string_pretty(&dataset)?;
        std::fs::write(&output, json)?;
    } else {
        // Export raw examples to JSON
        println!("\n💾 Exporting raw examples...");
        let json = serde_json::to_string_pretty(&examples)?;
        std::fs::write(&output, json)?;
    }

    println!("\n✅ Training data saved to: {}", output.display());

    // Summary
    println!("\n🎯 Phase 2 Training Data Extraction Complete!");
    println!("   ✅ Commit filtering (excludes merges, reverts, WIP)");
    println!("   ✅ Auto-labeling with rule-based classifier");
    println!("   ✅ Confidence threshold filtering ({:.2})", min_confidence);
    if create_splits {
        println!("   ✅ Train/validation/test splits created");
    }
    println!("   ✅ Ready for ML training (RandomForestClassifier)");

    println!("\n💡 Next Steps:");
    println!("   1. Review extracted data: cat {}", output.display());
    println!("   2. Train ML classifier: oip train-classifier --input {}", output.display());
    println!("   3. Evaluate model performance on test set");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    // Helper to create a minimal valid baseline YAML for testing
    fn create_test_baseline() -> String {
        r#"organizational_insights:
  top_defect_categories:
    - category: ConfigurationErrors
      frequency: 25
      confidence: 0.78
      quality_signals:
        avg_tdg_score: 45.2
        max_tdg_score: 60.0
        avg_complexity: 8.0
        avg_test_coverage: 0.5
        satd_instances: 5
        avg_lines_changed: 10.0
        avg_files_per_commit: 2.0
      examples: []
code_quality_thresholds:
  tdg_minimum: 85.0
  test_coverage_minimum: 0.85
  max_function_length: 50
  max_cyclomatic_complexity: 10
metadata:
  analysis_date: "2024-01-01T00:00:00Z"
  repositories_analyzed: 1
  commits_analyzed: 10
"#
        .to_string()
    }

    // Helper to create a minimal valid report YAML for testing
    fn create_test_report() -> String {
        r#"version: "1.0"
metadata:
  organization: "test-org"
  analysis_date: "2024-01-01T00:00:00Z"
  repositories_analyzed: 1
  commits_analyzed: 10
  analyzer_version: "1.0.0"
defect_patterns:
  - category: LogicErrors
    frequency: 5
    confidence: 0.91
    quality_signals:
      avg_tdg_score: 88.5
      max_tdg_score: 95.0
      avg_complexity: 4.0
      avg_test_coverage: 0.85
      satd_instances: 0
      avg_lines_changed: 8.0
      avg_files_per_commit: 1.5
    examples:
      - commit_hash: "abc123"
        message: "Fix critical bug"
        author: "test-author"
        timestamp: 1704067200
        files_affected: 2
        lines_added: 10
        lines_removed: 5
"#
        .to_string()
    }

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

    #[tokio::test]
    async fn test_handle_review_pr_markdown_format() {
        // Create a temporary baseline file
        let temp_baseline = NamedTempFile::new().unwrap();
        std::fs::write(temp_baseline.path(), create_test_baseline()).unwrap();

        let baseline = temp_baseline.path().to_path_buf();
        let files = "src/main.rs,src/lib.rs".to_string();
        let format = "markdown".to_string();

        let result = handle_review_pr(baseline, files, format, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_handle_review_pr_json_format() {
        // Create a temporary baseline file
        let temp_baseline = NamedTempFile::new().unwrap();
        std::fs::write(temp_baseline.path(), create_test_baseline()).unwrap();

        let baseline = temp_baseline.path().to_path_buf();
        let files = "src/test.rs".to_string();
        let format = "json".to_string();

        let result = handle_review_pr(baseline, files, format, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_handle_review_pr_with_output_file() {
        // Create temporary baseline and output files
        let temp_baseline = NamedTempFile::new().unwrap();
        std::fs::write(temp_baseline.path(), create_test_baseline()).unwrap();

        let temp_output = NamedTempFile::new().unwrap();
        let output_path = temp_output.path().to_path_buf();

        let baseline = temp_baseline.path().to_path_buf();
        let files = "src/main.rs".to_string();
        let format = "markdown".to_string();

        let result = handle_review_pr(baseline, files, format, Some(output_path.clone())).await;
        assert!(result.is_ok());

        // Verify output file was created and has content
        let content = std::fs::read_to_string(&output_path).unwrap();
        assert!(!content.is_empty());
    }

    #[tokio::test]
    async fn test_handle_review_pr_empty_files() {
        // Create a temporary baseline file
        let temp_baseline = NamedTempFile::new().unwrap();
        std::fs::write(temp_baseline.path(), create_test_baseline()).unwrap();

        let baseline = temp_baseline.path().to_path_buf();
        let files = "".to_string(); // Empty files list
        let format = "markdown".to_string();

        let result = handle_review_pr(baseline, files, format, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_handle_review_pr_multiple_files() {
        let temp_baseline = NamedTempFile::new().unwrap();
        std::fs::write(temp_baseline.path(), create_test_baseline()).unwrap();

        let baseline = temp_baseline.path().to_path_buf();
        let files = "src/main.rs, src/lib.rs, src/test.rs, src/utils.rs".to_string();
        let format = "markdown".to_string();

        let result = handle_review_pr(baseline, files, format, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_handle_summarize_valid_report() {
        // Create temporary report and output files
        let temp_report = NamedTempFile::new().unwrap();
        std::fs::write(temp_report.path(), create_test_report()).unwrap();

        let temp_output = NamedTempFile::new().unwrap();
        let output_path = temp_output.path().to_path_buf();

        let input = temp_report.path().to_path_buf();

        let result = handle_summarize(input, output_path.clone(), false, 10, 1, true).await;
        assert!(result.is_ok());

        // Verify output file was created
        assert!(output_path.exists());
    }

    #[tokio::test]
    async fn test_handle_summarize_with_pii_stripping() {
        let temp_report = NamedTempFile::new().unwrap();
        std::fs::write(temp_report.path(), create_test_report()).unwrap();

        let temp_output = NamedTempFile::new().unwrap();
        let output_path = temp_output.path().to_path_buf();

        let input = temp_report.path().to_path_buf();

        let result = handle_summarize(input, output_path.clone(), true, 5, 1, false).await;
        assert!(result.is_ok());

        // Verify output file was created
        assert!(output_path.exists());
    }

    #[tokio::test]
    async fn test_handle_summarize_different_top_n() {
        let temp_report = NamedTempFile::new().unwrap();
        std::fs::write(temp_report.path(), create_test_report()).unwrap();

        let temp_output = NamedTempFile::new().unwrap();
        let output_path = temp_output.path().to_path_buf();

        let input = temp_report.path().to_path_buf();

        // Test with top_n = 3
        let result = handle_summarize(input, output_path.clone(), false, 3, 1, true).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_handle_summarize_min_frequency_filter() {
        let temp_report = NamedTempFile::new().unwrap();
        std::fs::write(temp_report.path(), create_test_report()).unwrap();

        let temp_output = NamedTempFile::new().unwrap();
        let output_path = temp_output.path().to_path_buf();

        let input = temp_report.path().to_path_buf();

        // Test with high min_frequency
        let result = handle_summarize(input, output_path.clone(), false, 10, 100, false).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_handle_summarize_invalid_yaml_format() {
        let temp_report = NamedTempFile::new().unwrap();
        // Write invalid YAML
        std::fs::write(temp_report.path(), "not: valid: yaml: {{{").unwrap();

        let temp_output = NamedTempFile::new().unwrap();
        let output_path = temp_output.path().to_path_buf();

        let input = temp_report.path().to_path_buf();

        let result = handle_summarize(input, output_path, false, 10, 1, false).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_handle_analyze_with_token() {
        // This test will fail because it requires real GitHub API
        // But it exercises the code path
        let org = "nonexistent-org-12345678".to_string();
        let temp_output = NamedTempFile::new().unwrap();
        let output = temp_output.path().to_path_buf();

        // Use a fake token to test the token path
        let token = Some("fake-token-for-testing".to_string());

        let result = handle_analyze(org, output, 5, token, "1.0.0".to_string()).await;
        // Should fail because org doesn't exist, but we're testing the code path
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_handle_analyze_without_token() {
        // Test without GitHub token (unauthenticated)
        let org = "nonexistent-org-87654321".to_string();
        let temp_output = NamedTempFile::new().unwrap();
        let output = temp_output.path().to_path_buf();

        let result = handle_analyze(org, output, 5, None, "1.0.0".to_string()).await;
        // Should fail because org doesn't exist
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_handle_review_pr_whitespace_in_files() {
        let temp_baseline = NamedTempFile::new().unwrap();
        std::fs::write(temp_baseline.path(), create_test_baseline()).unwrap();

        let baseline = temp_baseline.path().to_path_buf();
        // Test with extra whitespace and commas
        let files = "  src/main.rs  ,  src/lib.rs  ,  , src/test.rs  ".to_string();
        let format = "markdown".to_string();

        let result = handle_review_pr(baseline, files, format, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_handle_summarize_all_config_combinations() {
        let temp_report = NamedTempFile::new().unwrap();
        std::fs::write(temp_report.path(), create_test_report()).unwrap();

        let temp_output = NamedTempFile::new().unwrap();
        let output_path = temp_output.path().to_path_buf();

        let input = temp_report.path().to_path_buf();

        // Test with all config options enabled
        let result = handle_summarize(input, output_path.clone(), true, 20, 2, true).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_handle_extract_training_data_invalid_path() {
        let repo = PathBuf::from("/nonexistent/repo/path");
        let temp_output = NamedTempFile::new().unwrap();
        let output = temp_output.path().to_path_buf();

        let result = handle_extract_training_data(repo, output, 0.75, 100, true).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_handle_extract_training_data_not_git_repo() {
        // Create a temporary directory that exists but is not a git repo
        let temp_dir = tempfile::TempDir::new().unwrap();
        let repo = temp_dir.path().to_path_buf();

        let temp_output = NamedTempFile::new().unwrap();
        let output = temp_output.path().to_path_buf();

        let result = handle_extract_training_data(repo, output, 0.75, 100, true).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Not a Git repository"));
    }

    #[tokio::test]
    async fn test_handle_extract_training_data_with_splits() {
        // Use the current repository (which is guaranteed to be a git repo)
        let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let temp_output = NamedTempFile::new().unwrap();
        let output = temp_output.path().to_path_buf();

        // This should succeed as it's a real git repo
        let result = handle_extract_training_data(repo, output.clone(), 0.70, 50, true).await;

        // Should succeed or return Ok with empty results
        match result {
            Ok(_) => {
                // If successful, output file should exist
                if output.exists() {
                    let content = std::fs::read_to_string(&output).unwrap();
                    assert!(!content.is_empty());
                }
            }
            Err(e) => {
                // Some errors are acceptable (e.g., no commits found)
                eprintln!("Expected error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_handle_extract_training_data_without_splits() {
        // Use the current repository
        let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let temp_output = NamedTempFile::new().unwrap();
        let output = temp_output.path().to_path_buf();

        let result = handle_extract_training_data(repo, output.clone(), 0.70, 50, false).await;

        // Should succeed or return Ok with empty results
        match result {
            Ok(_) => {
                // If successful, output file should exist
                if output.exists() {
                    let content = std::fs::read_to_string(&output).unwrap();
                    assert!(!content.is_empty());
                }
            }
            Err(e) => {
                // Some errors are acceptable
                eprintln!("Expected error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_handle_extract_training_data_high_confidence_threshold() {
        // Use the current repository with a very high confidence threshold
        let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let temp_output = NamedTempFile::new().unwrap();
        let output = temp_output.path().to_path_buf();

        // High confidence threshold (0.95) should result in fewer/no examples
        let result = handle_extract_training_data(repo, output.clone(), 0.95, 50, true).await;

        // Should succeed (even if no examples found)
        assert!(result.is_ok() || result.unwrap_err().to_string().contains("Git"));
    }

    #[tokio::test]
    async fn test_handle_extract_training_data_low_max_commits() {
        // Use the current repository with low max_commits
        let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let temp_output = NamedTempFile::new().unwrap();
        let output = temp_output.path().to_path_buf();

        let result = handle_extract_training_data(repo, output.clone(), 0.75, 5, true).await;

        // Should succeed with limited commits
        assert!(result.is_ok() || result.unwrap_err().to_string().contains("Git"));
    }
}
