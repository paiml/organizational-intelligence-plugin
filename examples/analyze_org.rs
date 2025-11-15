// Example: Analyze a GitHub organization
// Demonstrates the full workflow of OIP
//
// Usage:
//   cargo run --example analyze_org
//
// This example analyzes a small GitHub organization (tokio-rs)
// and generates a YAML report

use anyhow::Result;
use chrono::Utc;
use organizational_intelligence_plugin::github::GitHubMiner;
use organizational_intelligence_plugin::report::{
    AnalysisMetadata, AnalysisReport, ReportGenerator,
};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🚀 Organizational Intelligence Plugin - Example");
    println!("   Analyzing organization: tokio-rs\n");

    // Initialize GitHub client (unauthenticated for this example)
    let miner = GitHubMiner::new(None);

    // Fetch organization repositories
    println!("📥 Fetching repositories...");
    let repos = miner.fetch_organization_repos("tokio-rs").await?;

    println!("✅ Found {} repositories\n", repos.len());

    // Display top 3 by stars
    let mut sorted_repos = repos.clone();
    sorted_repos.sort_by(|a, b| b.stars.cmp(&a.stars));

    println!("⭐ Top repositories:");
    for (i, repo) in sorted_repos.iter().take(3).enumerate() {
        println!(
            "   {}. {} ({} ⭐) - {}",
            i + 1,
            repo.name,
            repo.stars,
            repo.language.as_deref().unwrap_or("Unknown")
        );
    }

    // Generate report
    println!("\n📊 Generating YAML report...");
    let report_generator = ReportGenerator::new();

    let metadata = AnalysisMetadata {
        organization: "tokio-rs".to_string(),
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

    // Write to file
    let output_path = PathBuf::from("tokio-rs-analysis.yaml");
    report_generator
        .write_to_file(&report, &output_path)
        .await?;

    println!("✅ Report saved to: {}", output_path.display());

    // Display a sample of the YAML
    let yaml_content = tokio::fs::read_to_string(&output_path).await?;
    println!("\n📄 Report preview:");
    println!("---");
    for (i, line) in yaml_content.lines().take(15).enumerate() {
        println!("{}", line);
        if i == 14 && yaml_content.lines().count() > 15 {
            println!("   ... (truncated)");
        }
    }
    println!("---");

    println!("\n🎯 Example complete!");
    println!("   Phase 1 MVP features demonstrated:");
    println!("   ✅ GitHub API integration");
    println!("   ✅ Repository fetching");
    println!("   ✅ YAML report generation");

    Ok(())
}
