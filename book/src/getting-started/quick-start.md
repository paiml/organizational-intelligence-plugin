# Quick Start

Get up and running with OIP in less than 5 minutes!

## Prerequisites

- Rust 1.75.0 or later
- GitHub personal access token (for analyzing organizations)
- Git installed locally

## Installation

```bash
# Clone the repository
git clone https://github.com/paiml/organizational-intelligence-plugin.git
cd organizational-intelligence-plugin

# Build the release binary
cargo build --release

# Verify installation
./target/release/oip --version
```

## First Analysis: Local Repository

Analyze a local Git repository (no GitHub token required):

```bash
./target/release/oip analyze \
  --org dummy \
  --local ../your-repo \
  --output defects.yaml \
  --max-commits 500
```

**Output**: `defects.yaml` containing defect patterns found in the last 500 commits.

## Example Output

```yaml
analyzer_metadata:
  version: "0.1.0"
  timestamp: "2025-11-24T23:00:00Z"
  total_commits_analyzed: 500

defect_summary:
  total_defects: 234
  categories:
    - category: ASTTransform
      count: 113
      percentage: 48.3
    - category: OwnershipBorrow
      count: 42
      percentage: 17.9
    - category: StdlibMapping
      count: 20
      percentage: 8.5
```

## View Results

```bash
# Pretty-print the YAML
cat defects.yaml | less

# Count defects by category
grep "category:" defects.yaml | sort | uniq -c
```

## Next Steps

- **[Installation](./installation.md)** - Detailed installation and configuration
- **[First Analysis](./first-analysis.md)** - In-depth analysis walkthrough
- **[CLI Usage](../cli/analyze.md)** - Master all CLI commands

## Common Issues

### "GITHUB_TOKEN not set"

**Solution**: For local repositories, this is just a warning. Ignore it, or set:

```bash
export GITHUB_TOKEN=your_token_here
```

### "Failed to find .git directory"

**Solution**: Ensure you're analyzing a Git repository:

```bash
cd ../your-repo && git status  # Verify it's a Git repo
```

### Compilation errors

**Solution**: Update Rust and rebuild:

```bash
rustup update
cargo clean
cargo build --release
```

---

**🎉 Congratulations! You've run your first OIP analysis.**

Proceed to [Installation](./installation.md) for detailed setup options, or jump to [CLI Usage](../cli/analyze.md) to explore all commands.
