# Example: Analyze Organization

This example demonstrates fetching repository data from a GitHub organization and generating a YAML analysis report.

## Running the Example

```bash
cargo run --example analyze_org
```

**Note**: For higher API rate limits, set the `GITHUB_TOKEN` environment variable:

```bash
GITHUB_TOKEN=ghp_xxxx cargo run --example analyze_org
```

## What It Does

1. **Fetches GitHub Organization Data** - Uses the GitHub API to retrieve all repositories for an organization
2. **Ranks by Popularity** - Sorts repositories by star count
3. **Generates YAML Report** - Creates a structured analysis report with metadata

## Sample Output

```
🚀 Organizational Intelligence Plugin - Example
   Analyzing organization: tokio-rs

📥 Fetching repositories...
✅ Found 47 repositories

⭐ Top repositories:
   1. tokio (26,842 ⭐) - Rust
   2. mio (6,524 ⭐) - Rust
   3. axum (19,847 ⭐) - Rust

📊 Generating YAML report...
✅ Report saved to: tokio-rs-analysis.yaml

📄 Report preview:
---
version: "1.0"
metadata:
  organization: tokio-rs
  analysis_date: "2024-12-01T10:30:00Z"
  repositories_analyzed: 47
  commits_analyzed: 0
  analyzer_version: "0.2.0"
defect_patterns: []
---

🎯 Example complete!
   Phase 1 MVP features demonstrated:
   ✅ GitHub API integration
   ✅ Repository fetching
   ✅ YAML report generation
```

## Output File

The example creates `tokio-rs-analysis.yaml` in the current directory:

```yaml
version: "1.0"
metadata:
  organization: tokio-rs
  analysis_date: "2024-12-01T10:30:00.123456Z"
  repositories_analyzed: 47
  commits_analyzed: 0
  analyzer_version: "0.2.0"
defect_patterns: []
```

## API Usage

```rust
use organizational_intelligence_plugin::github::GitHubMiner;
use organizational_intelligence_plugin::report::{
    AnalysisMetadata, AnalysisReport, ReportGenerator,
};

// Initialize GitHub client
let miner = GitHubMiner::new(Some("ghp_your_token".to_string()));

// Fetch repositories
let repos = miner.fetch_organization_repos("tokio-rs").await?;

// Generate report
let report_generator = ReportGenerator::new();
let report = AnalysisReport {
    version: "1.0".to_string(),
    metadata: AnalysisMetadata { /* ... */ },
    defect_patterns: vec![],
};

report_generator.write_to_file(&report, &output_path).await?;
```

## See Also

- [Full Analysis Pipeline](./full-analysis.md) - Complete workflow including commit analysis
- [CLI: Analyze Command](../cli/analyze.md) - Production CLI usage
