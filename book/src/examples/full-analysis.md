# Example: Full Analysis Pipeline

This example demonstrates the complete workflow: GitHub org → Git analysis → Classification → YAML report.

## Running the Example

```bash
cargo run --example full_analysis
```

**Note**: This example clones repositories, so it may take a few minutes. For faster iteration, set `GITHUB_TOKEN`:

```bash
GITHUB_TOKEN=ghp_xxxx cargo run --example full_analysis
```

## What It Does

1. **Fetches Repositories** - Retrieves organization repos from GitHub API
2. **Clones Top Repository** - Clones to a temporary directory
3. **Analyzes Commit History** - Examines last 100 commits for defect patterns
4. **Classifies Defects** - Uses rule-based classifier to categorize fixes
5. **Generates YAML Report** - Creates comprehensive defect pattern report

## Sample Output

```
🚀 Organizational Intelligence Plugin - Full Analysis Example

Demonstrates: GitHub API → Git Mining → Classification → YAML Report

📥 Step 1: Fetching repositories from GitHub...
✅ Found 47 repositories in tokio-rs

⭐ Top 3 repositories:
   1. tokio (26,842 ⭐)
   2. mio (6,524 ⭐)
   3. axum (19,847 ⭐)

🔍 Step 2: Analyzing commit history for defect patterns...
   Analyzing: tokio (up to 100 commits)
   ✅ Found 8 defect categories

📊 Defect Patterns Found:
   • Concurrency Bugs (n=12, confidence=89%)
     Example: a1b2c3d: fix: race condition in task scheduler
   • Memory Safety (n=8, confidence=92%)
     Example: d4e5f6g: fix: use-after-free in buffer pool
   • Performance Issues (n=6, confidence=85%)
     Example: h7i8j9k: perf: optimize poll loop

📝 Step 3: Generating YAML report...
✅ Report saved to: full-analysis-report.yaml

📄 Report Preview (first 20 lines):
---
version: "1.0"
metadata:
  organization: tokio-rs
  analysis_date: "2024-12-01T10:30:00Z"
  repositories_analyzed: 1
  commits_analyzed: 100
  analyzer_version: "0.2.0"
defect_patterns:
  - category: "Concurrency Bugs"
    frequency: 12
    confidence: 0.89
    examples:
      - commit_hash: "a1b2c3d"
        message: "fix: race condition in task scheduler"
---

🎯 Full Analysis Complete!
   ✅ GitHub API integration
   ✅ Git repository cloning
   ✅ Commit history analysis
   ✅ Rule-based defect classification
   ✅ YAML report generation

   Phase 1 MVP complete!
   Next: User feedback mechanism for Phase 2 ML training
```

## Output File

The example creates `full-analysis-report.yaml`:

```yaml
version: "1.0"
metadata:
  organization: tokio-rs
  analysis_date: "2024-12-01T10:30:00.123456Z"
  repositories_analyzed: 1
  commits_analyzed: 100
  analyzer_version: "0.2.0"
defect_patterns:
  - category: "Concurrency Bugs"
    frequency: 12
    confidence: 0.89
    examples:
      - commit_hash: "a1b2c3d4e5f6"
        message: "fix: race condition in task scheduler"
        timestamp: "2024-11-15T14:22:00Z"
  - category: "Memory Safety"
    frequency: 8
    confidence: 0.92
    examples:
      - commit_hash: "d4e5f6g7h8i9"
        message: "fix: use-after-free in buffer pool"
        timestamp: "2024-11-10T09:15:00Z"
```

## Key Components

### OrgAnalyzer

The `OrgAnalyzer` handles repository cloning and commit analysis:

```rust
use organizational_intelligence_plugin::analyzer::OrgAnalyzer;
use tempfile::TempDir;

let temp_dir = TempDir::new()?;
let analyzer = OrgAnalyzer::new(temp_dir.path());

let patterns = analyzer
    .analyze_repository(
        "https://github.com/tokio-rs/tokio",
        "tokio",
        100,  // max commits
    )
    .await?;
```

### DefectPattern

Each pattern includes:
- `category`: The defect category (e.g., "Memory Safety")
- `frequency`: Number of occurrences
- `confidence`: Classification confidence (0.0-1.0)
- `examples`: Sample commits with hash, message, timestamp

## See Also

- [Analyze Organization](./analyze-organization.md) - Simpler example without commit analysis
- [Classify Defects](./classify-defects.md) - Focus on the classifier component
- [CLI: Analyze Command](../cli/analyze.md) - Production CLI usage
