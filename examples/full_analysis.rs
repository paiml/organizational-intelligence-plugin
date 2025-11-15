// Example: Full organizational defect pattern analysis
// Demonstrates complete workflow: GitHub org → Git analysis → Classification → YAML report
//
// Usage:
//   cargo run --example full_analysis
//
// This example shows the complete Phase 1 pipeline

use anyhow::Result;
use chrono::Utc;
use organizational_intelligence_plugin::analyzer::OrgAnalyzer;
use organizational_intelligence_plugin::github::GitHubMiner;
use organizational_intelligence_plugin::report::{
    AnalysisMetadata, AnalysisReport, ReportGenerator,
};
use std::path::PathBuf;
use tempfile::TempDir;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🚀 Organizational Intelligence Plugin - Full Analysis Example\n");
    println!("Demonstrates: GitHub API → Git Mining → Classification → YAML Report\n");

    // Step 1: Fetch repositories from GitHub organization
    println!("📥 Step 1: Fetching repositories from GitHub...");
    let github_miner = GitHubMiner::new(None); // Unauthenticated for demo
    let org_name = "tokio-rs";

    let repos = github_miner.fetch_organization_repos(org_name).await?;
    println!("✅ Found {} repositories in {}\n", repos.len(), org_name);

    // Display top repositories
    let mut sorted_repos = repos.clone();
    sorted_repos.sort_by(|a, b| b.stars.cmp(&a.stars));

    println!("⭐ Top 3 repositories:");
    for (i, repo) in sorted_repos.iter().take(3).enumerate() {
        println!("   {}. {} ({} ⭐)", i + 1, repo.name, repo.stars);
    }
    println!();

    // Step 2: Analyze repositories for defect patterns
    println!("🔍 Step 2: Analyzing commit history for defect patterns...");

    // Create temporary directory for cloning
    let temp_dir = TempDir::new()?;
    let analyzer = OrgAnalyzer::new(temp_dir.path());

    // Analyze top repository (limit to 100 commits for demo speed)
    let top_repo = &sorted_repos[0];
    println!("   Analyzing: {} (up to 100 commits)", top_repo.name);

    let patterns = analyzer
        .analyze_repository(
            &format!("https://github.com/{}/{}", org_name, top_repo.name),
            &top_repo.name,
            100,
        )
        .await?;

    if patterns.is_empty() {
        println!("   ℹ️  No defect patterns detected in last 100 commits");
    } else {
        println!("   ✅ Found {} defect categories\n", patterns.len());

        // Display patterns
        println!("📊 Defect Patterns Found:");
        for pattern in &patterns {
            println!(
                "   • {} (n={}, confidence={:.0}%)",
                pattern.category.as_str(),
                pattern.frequency,
                pattern.confidence * 100.0
            );
            if !pattern.examples.is_empty() {
                println!("     Example: {}", pattern.examples[0]);
            }
        }
        println!();
    }

    // Step 3: Generate YAML report
    println!("📝 Step 3: Generating YAML report...");
    let report_generator = ReportGenerator::new();

    let metadata = AnalysisMetadata {
        organization: org_name.to_string(),
        analysis_date: Utc::now().to_rfc3339(),
        repositories_analyzed: 1, // We analyzed 1 repo in this demo
        commits_analyzed: 100,
        analyzer_version: env!("CARGO_PKG_VERSION").to_string(),
    };

    let report = AnalysisReport {
        version: "1.0".to_string(),
        metadata,
        defect_patterns: patterns,
    };

    // Write to file
    let output_path = PathBuf::from("full-analysis-report.yaml");
    report_generator
        .write_to_file(&report, &output_path)
        .await?;

    println!("✅ Report saved to: {}", output_path.display());

    // Display YAML preview
    let yaml_content = tokio::fs::read_to_string(&output_path).await?;
    println!("\n📄 Report Preview (first 20 lines):");
    println!("---");
    for (i, line) in yaml_content.lines().enumerate() {
        if i >= 20 {
            println!(
                "   ... (truncated, see {} for full report)",
                output_path.display()
            );
            break;
        }
        println!("{}", line);
    }
    println!("---");

    // Summary
    println!("\n🎯 Full Analysis Complete!");
    println!("   ✅ GitHub API integration");
    println!("   ✅ Git repository cloning");
    println!("   ✅ Commit history analysis");
    println!("   ✅ Rule-based defect classification");
    println!("   ✅ YAML report generation");
    println!("\n   Phase 1 MVP complete!");
    println!("   Next: User feedback mechanism for Phase 2 ML training");

    Ok(())
}
