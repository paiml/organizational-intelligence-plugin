<div align="center">
  <img src="logo.png" alt="OIP Logo" width="128" height="128">

  # Organizational Intelligence Plugin (OIP)

  [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
  [![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
  [![Grade: A](https://img.shields.io/badge/pmat_repo--score-A_(90%2F100)-success.svg)](https://github.com/paiml/pmat)
  [![TDG Score](https://img.shields.io/badge/TDG-94.1%2F100_(A)-brightgreen.svg)](https://github.com/paiml/pmat)
  [![Tests](https://img.shields.io/badge/tests-472_passing-brightgreen.svg)](https://github.com/paiml/organizational-intelligence-plugin)

  A plugin for [pmat](https://github.com/paiml/pmat) that analyzes GitHub organizations to detect defect patterns, measure code quality, and generate actionable intelligence for software development teams.

  [Installation](#installation) •
  [Quick Start](#usage) •
  [Documentation](#development) •
  [Contributing](#contributing)

</div>

---

## Overview

Organizational Intelligence Plugin (OIP) mines Git history and integrates with pmat's Technical Debt Gradient (TDG) analysis to:

- **Detect defect patterns** across 10 categories (Configuration Errors, Security Vulnerabilities, Type Errors, etc.)
- **Measure code quality** via pmat TDG integration (0-100 score, higher is better)
- **Generate privacy-safe summaries** with automated PII stripping
- **Provide fast PR reviews** using stateful baselines (<30s vs 10+ minutes)
- **Enable data-driven decisions** for technical debt prioritization

### Key Features

✅ **Phase 1 - Core Analysis** (`oip analyze`)
- Analyze GitHub organizations for defect patterns
- Integrate pmat TDG quality scores
- Generate comprehensive YAML reports

✅ **Phase 2 - Summarization** (`oip summarize`)
- Automated PII stripping (commit hashes, author emails)
- Frequency filtering and top-N category selection
- Privacy-safe summaries ready for AI consumption

✅ **Phase 3 - PR Review** (`oip review-pr`)
- Fast PR reviews using stateful baselines (<30s)
- Context-aware warnings based on organizational history
- Multiple output formats (Markdown, JSON)

🚀 **Phase 1 GPU Extension** (`oip-gpu`) - **NEW!**
- GPU-accelerated correlation analysis for defect patterns
- SIMD-optimized feature extraction (trueno backend)
- Benchmark suite with criterion (10-50× speedup targets)
- Complete GitHub → Features → Storage pipeline
- See: [GPU Quick Start](docs/GPU_QUICKSTART.md) | [Full Spec](docs/specifications/GPU-correlation-predictions-spec.md)

### Toyota Way Principles

This tool is built following Toyota Production System principles:

- **Genchi Genbutsu** (Go and See): Analyzes actual commit history, not surveys
- **Kaizen** (Continuous Improvement): Weekly reports track improvement over time
- **Jidoka** (Build Quality In): Identifies defect patterns to fix root causes
- **Muda/Muri/Mura Elimination**: Automates manual work, prevents overburden, smooths workflow

## Installation

### Prerequisites

- Rust 1.70 or higher
- Git
- [pmat](https://github.com/paiml/pmat) (optional, for TDG integration)
- GitHub Personal Access Token (for higher rate limits)

### From Source

```bash
git clone https://github.com/paiml/organizational-intelligence-plugin
cd organizational-intelligence-plugin
cargo build --release

# Binary available at target/release/oip
export PATH=$PATH:$(pwd)/target/release
```

### Via Cargo (once published)

```bash
cargo install organizational-intelligence-plugin
```

### Setting Up GitHub Token

```bash
# Create a GitHub Personal Access Token at:
# https://github.com/settings/tokens
# Required scopes: repo (for private repos) or public_repo (for public only)

export GITHUB_TOKEN=ghp_your_token_here

# Add to ~/.bashrc or ~/.zshrc for persistence
echo 'export GITHUB_TOKEN=ghp_your_token_here' >> ~/.bashrc
```

## Usage

### Quick Start

```bash
# 1. Analyze your organization
oip analyze --org YOUR_ORG --output analysis.yaml

# 2. Generate privacy-safe summary
oip summarize --input analysis.yaml --output summary.yaml --strip-pii

# 3. Review a PR (requires baseline)
oip review-pr --baseline summary.yaml --files src/config.rs,src/auth.rs
```

### Phase 1: Analyze Organization

```bash
# Analyze all repositories in an organization
oip analyze --org paiml --output paiml-analysis.yaml

# With verbose logging
oip analyze --org paiml --output paiml-analysis.yaml --verbose

# Limit concurrent analysis (default: 10)
oip analyze --org paiml --output paiml-analysis.yaml --max-concurrent 5
```

**Output**: YAML report with:
- Defect patterns by category (frequency, confidence)
- TDG quality scores (requires pmat)
- Code churn metrics (lines changed, files per commit)
- Example commits (⚠️ contains PII - use Phase 2 to strip)

### Phase 2: Summarize for Sharing

```bash
# Generate privacy-safe summary
oip summarize \
  --input analysis.yaml \
  --output summary.yaml \
  --strip-pii \
  --top-n 10 \
  --min-frequency 5

# Include anonymized examples
oip summarize \
  --input analysis.yaml \
  --output summary.yaml \
  --strip-pii \
  --include-examples
```

**Output**: Clean YAML with:
- Top N defect categories by frequency
- PII redacted (commit_hash: REDACTED, author: REDACTED)
- Quality thresholds (TDG 85+, coverage 85%+)
- Safe for sharing with AI tools

### Phase 3: PR Review

```bash
# One-time: Create baseline (run weekly)
oip analyze --org myorg --output baseline.yaml
oip summarize --input baseline.yaml --output baseline-summary.yaml

# On every PR: Fast review (<30s)
oip review-pr \
  --baseline baseline-summary.yaml \
  --files src/config.rs,src/auth.rs \
  --format markdown \
  --output pr-review.md

# Output to stdout for CI integration
oip review-pr \
  --baseline baseline-summary.yaml \
  --files $(git diff --name-only HEAD~1) \
  --format json
```

### Phase 1 GPU: Accelerated Analysis

```bash
# Analyze repository with GPU-accelerated features
oip-gpu analyze --repo rust-lang/rust --output features.db

# Run performance benchmarks
oip-gpu benchmark --suite all

# Force SIMD backend (CPU)
oip-gpu analyze --repo owner/repo --backend simd --output out.db
```

**Output**: Feature vectors ready for GPU correlation analysis

See [GPU Quick Start](docs/GPU_QUICKSTART.md) for detailed examples.

**Output**: Context-aware warnings based on organizational defect patterns

## Examples

### Sprint Planning

```bash
# Generate current analysis
oip analyze --org myorg --output sprint-data.yaml

# Identify high-priority technical debt
# (High frequency + Low TDG score = urgent refactoring needed)
cat sprint-data.yaml | grep -A10 "frequency: 2[0-9]"
```

### Weekly Baseline Updates

```bash
#!/bin/bash
# weekly-baseline.sh - Run via cron every Monday

export GITHUB_TOKEN=your_token
ORG=myorg
DATE=$(date +%Y-%m-%d)

oip analyze --org $ORG --output "baselines/full-$DATE.yaml"
oip summarize \
  --input "baselines/full-$DATE.yaml" \
  --output "baselines/summary-$DATE.yaml" \
  --strip-pii

echo "✅ Baseline updated: baselines/summary-$DATE.yaml"
```

### CI/CD Integration

```yaml
# .github/workflows/pr-review.yml
name: Organizational Intelligence PR Review

on: [pull_request]

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install OIP
        run: cargo install organizational-intelligence-plugin

      - name: Review PR
        run: |
          FILES=$(gh pr diff ${{ github.event.pull_request.number }} --name-only | tr '\n' ',')
          oip review-pr \
            --baseline .oip/baseline.yaml \
            --files "$FILES" \
            --format markdown
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

## Configuration

### Makefile Targets

```bash
make help             # Show all available targets
make lint             # Quick lint check
make test-fast        # Fast unit tests (<5s)
make test-all         # All tests including integration
make coverage         # Generate HTML coverage report
make build            # Build release binary
```

### Quality Gates

All code must pass:
- ✅ `make lint` - No clippy warnings
- ✅ `make test-fast` - All unit tests pass
- ✅ `make coverage` - 85%+ line coverage (currently: **86.65%** ✅)
- ✅ pmat TDG score 85+ (currently: 96.4/100 ✅)

## Development

### Project Structure

```
organizational-intelligence-plugin/
├── src/
│   ├── analyzer.rs       # Organization analysis orchestration
│   ├── classifier.rs     # Defect pattern classification (10 categories)
│   ├── cli.rs           # Command-line interface (clap)
│   ├── git.rs           # Git history mining
│   ├── github.rs        # GitHub API integration (octocrab)
│   ├── pmat.rs          # pmat TDG integration
│   ├── pr_reviewer.rs   # PR review with stateful baselines
│   ├── report.rs        # YAML report generation
│   ├── summarizer.rs    # PII stripping and summarization
│   └── main.rs          # Entry point
├── tests/
│   └── cli_tests.rs     # CLI integration tests
├── docs/
│   └── how-to-integrate-as-plugin-with-pmat-improve-prompts-spec.md
├── Makefile             # Development workflow automation
└── Cargo.toml           # Rust dependencies
```

### Running Tests

```bash
# Fast unit tests (recommended for development)
make test-fast

# All tests including network integration tests
make test-all

# Coverage report with HTML output
make coverage
make coverage-open  # Opens in browser
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow EXTREME TDD: Write tests first (RED-GREEN-REFACTOR)
4. Ensure all quality gates pass: `make lint && make test-fast`
5. Commit using conventional commits (`feat:`, `fix:`, `docs:`, etc.)
6. Push to your fork and submit a Pull Request

**Code Standards**:
- Minimum 85% test coverage for new code
- All clippy warnings must be resolved
- Follow Rust API guidelines
- Document public APIs with examples

## Roadmap

### Completed (Phase 1-3)

- ✅ Phase 1: Core analysis with pmat TDG integration
- ✅ Phase 2: Automated summarization with PII stripping
- ✅ Phase 3: Fast PR reviews using stateful baselines

### Proposed (Phase 4)

- ⚪ AI Prompt Integration: Generate context-aware prompts for AI tools
- ⚪ DefectAwarePromptGenerator for paiml-mcp-agent-toolkit
- ⚪ MCP (Model Context Protocol) integration
- ⚪ Automated code review comments on GitHub PRs

See [docs/how-to-integrate-as-plugin-with-pmat-improve-prompts-spec.md](docs/how-to-integrate-as-plugin-with-pmat-improve-prompts-spec.md) for detailed design specifications.

## Troubleshooting

### GitHub Rate Limits

**Problem**: `API rate limit exceeded for...`

**Solution**: Set `GITHUB_TOKEN` environment variable
```bash
export GITHUB_TOKEN=ghp_your_token_here
```

### pmat Not Found

**Problem**: `pmat analyze tdg` fails

**Solution**: Install pmat
```bash
cargo install pmat
```

### Analysis Takes Too Long

**Problem**: Analyzing large organizations is slow

**Solution**: Reduce max-concurrent flag
```bash
oip analyze --org large-org --output report.yaml --max-concurrent 3
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built following Toyota Production System principles
- Inspired by empirical software engineering research
- Integrates with [pmat](https://github.com/paiml/pmat) for quality analysis
- Uses [octocrab](https://github.com/XAMPPRocky/octocrab) for GitHub API
- Uses [git2-rs](https://github.com/rust-lang/git2-rs) for Git operations

## Support

- 📖 Documentation: [docs/](docs/)
- 🐛 Issues: [GitHub Issues](https://github.com/paiml/organizational-intelligence-plugin/issues)

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{organizational_intelligence_plugin,
  title = {Organizational Intelligence Plugin},
  author = {paiml},
  year = {2025},
  url = {https://github.com/paiml/organizational-intelligence-plugin},
  note = {A plugin for pmat that analyzes GitHub organizations for defect patterns}
}
```

---

**Status**: Phase 1-3 Complete | **Grade**: TDG 96.4/100 (A+) | **Coverage**: 86.65% (422 tests) ✅

Built with ❤️ following the Toyota Way
