# Organizational Intelligence Plugin: pmat Integration Design

**Document Type**: Technical Design Specification
**Status**: Phase 1-3 Complete, Phase 4 Proposed
**Last Updated**: 2025-11-15
**Target Audience**: Contributors, Integrators, Technical Planning

---

## ⚠️ Important: Current Implementation Status

| Phase | Status | What Exists | What's Proposed |
|-------|--------|-------------|-----------------|
| **Phase 1** | ✅ **COMPLETE** | `oip analyze`, pmat TDG integration, YAML reports | - |
| **Phase 2** | ✅ **COMPLETE** | `oip summarize` (automated PII stripping) | - |
| **Phase 3** | ✅ **COMPLETE** | `oip review-pr` (fast PR reviews <30s) | - |
| **Phase 4** | ⚪ **PROPOSED** | - | AI prompt integration |

**If you're looking for a user guide for existing features, see `README.md`.**

This document describes the **vision and design** for integrating OIP with pmat and AI prompt engineering. It includes implemented features (Phase 1) and proposed enhancements (Phase 2-4) with estimated effort and implementation details.

---

## Overview

**Vision**: Use historical defect patterns from organizational intelligence to generate context-aware AI prompts that prevent common mistakes.

**Key Insight**: By analyzing years of commit history, we can identify recurring defect patterns and feed this knowledge into AI-assisted development, creating prompts that guide developers away from known pitfalls.

**Current Reality**: Phase 1 provides the data foundation. Phases 2-4 (proposed below) will automate the transformation of this data into actionable AI prompts.

## Table of Contents

1. [What Exists Today (Phase 1)](#what-exists-today-phase-1)
2. [Development Roadmap](#development-roadmap)
3. [Integration Architecture](#integration-architecture)
4. [Phase 2: Summarization (IN PROGRESS)](#phase-2-summarization-in-progress)
5. [Phase 3: PR Review (PROPOSED)](#phase-3-pr-review-proposed)
6. [Phase 4: AI Integration (PROPOSED)](#phase-4-ai-integration-proposed)
7. [Implementation Guide](#implementation-guide)

---

## What Exists Today (Phase 1)

✅ **IMPLEMENTED AND WORKING**

### Current Capabilities

```bash
# What you can do RIGHT NOW:

# 1. Analyze organization for defect patterns
cargo run -- analyze --org YOUR_ORG --output report.yaml

# 2. Get detailed YAML report with:
#    - Defect patterns by category
#    - TDG quality scores
#    - Code churn metrics
#    - Example commit messages (with PII)
```

### Current Output Format

```yaml
version: "1.0"
metadata:
  organization: paiml
  analysis_date: "2025-11-15T12:00:00Z"
  repositories_analyzed: 25
  commits_analyzed: 2500
defect_patterns:
- category: ConfigurationErrors
  frequency: 25
  confidence: 0.78
  quality_signals:
    avg_tdg_score: 96.4      # ✅ Working
    max_tdg_score: 98.0       # ✅ Working
    avg_lines_changed: 45.2   # ✅ Working
    avg_files_per_commit: 2.1 # ✅ Working
  examples:                  # ⚠️ Contains PII (author, commit hash)
  - commit_hash: "abc123"
    message: "fix config bug"
    author: "dev@company.com"
```

### Current Limitations (Addressed in Phase 2-4)

❌ **Manual work required** to anonymize PII from reports
❌ **No automated summarization** for AI prompt generation
❌ **No PR review integration**
❌ **No direct AI prompt generation**

**Next**: These limitations are addressed in the proposed phases below.

---

## Development Roadmap

### Phase-by-Phase Plan

| Phase | Status | Effort | Features | Value Delivered |
|-------|--------|--------|----------|-----------------|
| **1: Core Analysis** | ✅ DONE | - | Defect detection, TDG integration, YAML reports | Organizational intelligence baseline |
| **2: Summarization** | ✅ DONE | 2-4 hours | `oip summarize`, automated PII stripping, prompt-ready output | Eliminates manual YAML editing waste |
| **3: PR Review** | ✅ DONE | 4-6 hours | `oip review-pr`, stateful baselines, <30s feedback | Fast, actionable PR reviews |
| **4: AI Integration** | ⚪ PROPOSED | 8-16 hours | DefectAwarePromptGenerator, MCP integration | Context-aware AI prompts |

**Total Estimated Effort**: 14-28 hours across all phases

### Dependencies

```
Phase 1 (DONE)
    ↓
Phase 2 (DONE)
    ↓
Phase 3 (DONE) ← You are here
    ↓
Phase 4 (Requires Phases 1-3)
```

### Toyota Way Principles Applied

1. **Phase 1**: Built foundation with pmat integration (Jidoka - build quality in)
2. **Phase 2**: Eliminate manual waste in summarization (Muda reduction)
3. **Phase 3**: Eliminate overburden in PR reviews (Muri reduction)
4. **Phase 4**: Deliver customer value (context-aware prompts)

### Critical Learning from Toyota Way Review

**Original Problem**: Document promised features as if they existed, creating confusion.

**Root Cause (Five Whys)**:
1. Why confusion? → Manual steps hidden
2. Why manual? → Summarization not automated
3. Why not automated? → Not designed into tool
4. Why not designed in? → Focused on data collection, not synthesis
5. **Root**: "Intelligence" requires synthesis, not just collection

**Solution**: Implement `oip summarize` (Phase 2) to automate the entire workflow.

---

## Integration Architecture

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    pmat (Main Tool)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │ TDG Analysis │  │ SATD Detect  │  │ Coverage Report │ │
│  └──────────────┘  └──────────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ Plugin API
                              ▼
┌─────────────────────────────────────────────────────────────┐
│         Organizational Intelligence Plugin (OIP)            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. Analyze GitHub org → Defect patterns              │  │
│  │ 2. Integrate pmat TDG → Quality scores               │  │
│  │ 3. Generate insights → Prioritized recommendations   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ YAML Report
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              AI Prompt Generator                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Convert defect patterns → Context-aware prompts      │  │
│  │ High-frequency defects → Prevention instructions     │  │
│  │ Low TDG modules → Refactoring guidance               │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
               ┌────────────────────────────────┐
               │  paiml-mcp-agent-toolkit       │
               │  Enhanced AI Prompts           │
               └────────────────────────────────┘
```

---

## Phase 2: Summarization (IN PROGRESS)

**Status**: 🟡 **IN PROGRESS**
**Effort**: 2-4 hours
**Goal**: Eliminate manual waste in summarization workflow

### The Problem (Identified via Toyota Way Review)

**Current workflow (Phase 1)**:
1. Run `oip analyze` → Get YAML with PII
2. **MANUAL STEP**: Hand-edit YAML to remove PII and extract patterns
3. Use edited summary in AI prompts

**Waste (Muda)**: Manual editing is error-prone, inconsistent, prevents automation

### The Solution: `oip summarize` Command

**Proposed command**:
```bash
# Automated PII stripping and pattern extraction
oip summarize --input org-report.yaml --output summary.yaml --strip-pii
```

**Implementation Design**:

```rust
// src/summarizer.rs (NEW MODULE)

/// Configuration for summarization
pub struct SummaryConfig {
    pub strip_pii: bool,           // Remove author, commit hashes
    pub top_n_categories: usize,   // Show only top N defects
    pub min_frequency: usize,      // Filter low-frequency patterns
    pub include_examples: bool,    // Include anonymized examples
}

/// Summarize organizational analysis for AI consumption
pub struct ReportSummarizer;

impl ReportSummarizer {
    pub fn summarize(input: &Path, config: SummaryConfig) -> Result<Summary> {
        // 1. Load full report
        let report = OrganizationalReport::from_file(input)?;

        // 2. Filter to top N categories by frequency
        let top_patterns = report.defect_patterns
            .into_iter()
            .filter(|p| p.frequency >= config.min_frequency)
            .sorted_by_key(|p| p.frequency)
            .rev()
            .take(config.top_n_categories)
            .collect();

        // 3. Strip PII if requested
        let patterns = if config.strip_pii {
            Self::strip_pii(top_patterns)
        } else {
            top_patterns
        };

        // 4. Generate summary
        Ok(Summary {
            organizational_insights: patterns,
            code_quality_thresholds: QualityThresholds::default(),
            metadata: SummaryMetadata::from_report(&report),
        })
    }

    fn strip_pii(patterns: Vec<DefectPattern>) -> Vec<DefectPattern> {
        patterns.into_iter().map(|mut p| {
            // Remove examples entirely (contain PII)
            p.examples.clear();
            p
        }).collect()
    }
}
```

**CLI Integration**:
```rust
// src/cli.rs (UPDATE)

#[derive(Parser)]
pub enum Commands {
    Analyze { /* existing */ },

    /// NEW: Summarize analysis report for AI consumption
    Summarize {
        /// Input YAML report from 'analyze' command
        #[arg(short, long)]
        input: PathBuf,

        /// Output summary file
        #[arg(short, long)]
        output: PathBuf,

        /// Strip PII (author, commit hashes)
        #[arg(long, default_value = "true")]
        strip_pii: bool,

        /// Top N defect categories to include
        #[arg(long, default_value = "10")]
        top_n: usize,

        /// Minimum frequency to include
        #[arg(long, default_value = "5")]
        min_frequency: usize,
    },
}
```

**Test Plan** (EXTREME TDD):
1. Test PII stripping removes authors, commit hashes
2. Test frequency filtering works correctly
3. Test top-N selection
4. Test output format is valid YAML
5. Test roundtrip (analyze → summarize → load)

**Expected Output**:
```yaml
# After: oip summarize --input full.yaml --output summary.yaml --strip-pii
organizational_insights:
  top_defect_categories:
  - category: ConfigurationErrors
    frequency: 25
    avg_tdg_score: 45.2
    common_patterns:          # Extracted from examples
    - "Missing validation"
    prevention_strategies:    # Generated from analysis
    - "Explicit validation"
  # NO commit hashes, NO author emails, NO PII
```

**Value Delivered**:
- ✅ Eliminates manual editing waste
- ✅ Consistent, reproducible summaries
- ✅ Safe for sharing (no PII leakage)
- ✅ Enables automation in CI/CD

---

## Phase 3: PR Review (COMPLETE)

**Status**: ✅ **COMPLETE**
**Effort**: 4-6 hours (actual)
**Dependencies**: Requires Phase 2 ✅
**Goal**: Fast PR reviews (<30s) without re-analyzing entire org ✅

### The Problem (Identified via Toyota Way Review)

**Naive approach**:
```yaml
# .github/workflows/review.yml
- name: Analyze org on every PR  # ❌ 10+ minutes!
  run: oip analyze --org myorg
```

**Overburden (Muri)**:
- Developer opens 1-file PR
- CI re-analyzes 50 repos with 10,000 commits
- Waits 10+ minutes
- After 1 week: "I disabled the bot"

### The Solution: Stateful Baselines

**Implemented Architecture** (November 2025):

```bash
# One-time: Establish baseline (expensive, run weekly)
oip analyze --org myorg --output baseline.yaml
oip summarize --input baseline.yaml --output baseline-summary.yaml

# On every PR: Fast review (cheap, <30s)
oip review-pr \
  --baseline baseline-summary.yaml \
  --files-changed src/config.rs,src/auth.rs \
  --output pr-review.md
```

**Implementation Design**:

```rust
// src/pr_reviewer.rs (NEW MODULE)

pub struct PrReviewer {
    baseline: Summary,  // Pre-computed org patterns
}

impl PrReviewer {
    pub fn load_baseline(path: &Path) -> Result<Self> {
        let baseline = Summary::from_file(path)?;
        Ok(Self { baseline })
    }

    pub fn review_pr(&self, files_changed: &[String]) -> PrReview {
        let mut warnings = Vec::new();

        for file in files_changed {
            // Check file extension against defect patterns
            if file.ends_with(".yaml") || file.ends_with(".toml") {
                if let Some(pattern) = self.baseline.find_category("ConfigurationErrors") {
                    if pattern.frequency > 10 && pattern.avg_tdg_score < 60.0 {
                        warnings.push(Warning {
                            file: file.clone(),
                            category: "ConfigurationErrors",
                            message: format!(
                                "This org has {} config errors (TDG: {:.1}). Ensure validation!",
                                pattern.frequency, pattern.avg_tdg_score
                            ),
                            prevention_tips: pattern.prevention_strategies.clone(),
                        });
                    }
                }
            }

            // Similar checks for other file types...
        }

        PrReview { warnings }
    }
}
```

**CLI Integration**:
```rust
/// NEW: Review PR with organizational context
ReviewPr {
    /// Baseline summary from weekly analysis
    #[arg(short, long)]
    baseline: PathBuf,

    /// Files changed in PR (comma-separated)
    #[arg(short, long)]
    files_changed: String,

    /// Output format: markdown, json, github-comment
    #[arg(short, long, default_value = "markdown")]
    format: String,

    /// Output file (or stdout if not specified)
    #[arg(short, long)]
    output: Option<PathBuf>,
}
```

**Expected Output** (markdown format):
```markdown
# PR Review: Organizational Intelligence

## ⚠️ Warnings Based on Historical Patterns

### src/config.rs
**Category**: ConfigurationErrors (25 occurrences, TDG: 45.2)

**Common Issues in This Org**:
- Missing validation for required fields
- Type coercion without error handling

**Prevention Strategies**:
✅ Explicit validation with descriptive errors
✅ Schema validation before parsing
✅ Document all defaults in docstrings

**Quality Gates**:
```bash
pmat analyze tdg --path src/config.rs --threshold 85
cargo test --all-features
```

---

**Analysis Date**: 2025-11-08 (baseline is 7 days old)
**Repositories Analyzed**: 25
**Recommendation**: Review carefully - config errors are this org's top defect!
```

**Performance**:
- Baseline load: <100ms
- File analysis: <50ms per file
- Total: <30s for typical PR

**Value Delivered**:
- ✅ Fast feedback (<30s vs 10+ minutes)
- ✅ Actionable warnings (not generic)
- ✅ Respects developer time (no overburden)

### What Was Actually Built (November 2025)

**Implementation Results**:
- ✅ `src/pr_reviewer.rs` (441 lines) - Core review logic
- ✅ CLI integration with `review-pr` command
- ✅ 11 comprehensive unit tests (100% passing)
- ✅ Performance: **0.125 seconds** per review (well under 30s target)
- ✅ Output formats: Markdown and JSON
- ✅ File pattern matching for config, integration, and code files
- ✅ Warning thresholds based on frequency and TDG scores

**Example Usage**:
```bash
# Fast PR review (0.125s actual performance)
oip review-pr \
  --baseline baseline-summary.yaml \
  --files "src/config.yaml,src/api_client.rs,README.md" \
  --format markdown

# Output:
# ⚠️ 2 Warnings Based on Historical Patterns
#
# src/config.yaml
# Category: ConfigurationErrors (25 occurrences, TDG: 45.2)
# This org has 25 config errors (avg TDG: 45.2). Ensure validation!
#
# src/api_client.rs
# Category: IntegrationFailures (18 occurrences, TDG: 52.3)
# Integration issues detected 18 times (avg TDG: 52.3). Check timeouts and retries!
```

**Quality Metrics**:
- Code Coverage: 100% of new code tested
- Test Execution: <0.03s for all 11 tests
- Zero compiler warnings
- Follows EXTREME TDD methodology

---

## Phase 4: AI Integration (PROPOSED)

**Status**: ⚪ **PROPOSED**
**Effort**: 8-16 hours
**Dependencies**: Requires Phases 2-3
**Goal**: Generate context-aware AI prompts automatically

### Using OIP as a pmat Plugin

**Status**: ⚪ **PROPOSED** (integration not yet implemented)

### Step 1: Install Both Tools

```bash
# Install pmat
cargo install pmat

# Build OIP
cd organizational-intelligence-plugin
cargo build --release

# Add to PATH (or use full path)
export PATH=$PATH:$(pwd)/target/release
```

### Step 2: Create pmat Plugin Configuration

Create `~/.pmat/plugins/oip-plugin.toml`:

```toml
[plugin]
name = "organizational-intelligence"
description = "Analyze organizational defect patterns and code quality"
executable = "oip"  # or full path to binary
version = "0.1.0"

[[plugin.commands]]
name = "analyze-org"
description = "Analyze GitHub organization for defect patterns"
args = ["analyze", "--org", "{org}", "--output", "{output}"]

[[plugin.commands]]
name = "generate-prompts"
description = "Generate AI prompts from defect analysis"
args = ["prompts", "--input", "{report}", "--output", "{prompts_dir}"]

[plugin.integration]
# Where OIP reads pmat output
pmat_tdg_input = true
pmat_satd_input = true
pmat_coverage_input = true

# Where OIP writes for pmat to consume
output_format = "yaml"
output_location = ".pmat/oip-reports/"
```

### Step 3: Run via pmat

```bash
# Option A: Direct plugin invocation
pmat plugin run organizational-intelligence -- analyze --org paiml --output org-report.yaml

# Option B: Integrated workflow
pmat analyze comprehensive --org paiml --with-plugins organizational-intelligence
```

---

## Generating AI Prompts from Defect Patterns

**Status**: ⚪ **PROPOSED** (requires Phase 2 implementation)

### The Problem: Generic AI Prompts Miss Context

**Before (Generic Prompt):**
```
Write a configuration parser for YAML files.
```

**Issues:**
- No awareness of common pitfalls
- Misses organization-specific patterns
- No quality guidance

### The Solution: Context-Aware Prompts from OIP

**After (OIP-Enhanced Prompt):**
```
Write a configuration parser for YAML files.

CONTEXT: Our organization has a pattern of configuration errors:
- Frequency: 25 occurrences (highest defect category)
- Code Quality: TDG score 45.2 (below 50 threshold - needs improvement)
- Common Issues:
  1. Missing validation for required fields (8 instances)
  2. Type coercion errors (6 instances)
  3. Default value handling (5 instances)

REQUIREMENTS:
✅ MUST: Validate all required fields before parsing
✅ MUST: Explicit type checking with clear error messages
✅ MUST: Document all default values in code comments
✅ MUST: Write unit tests for validation logic (current coverage: 58%)
✅ MUST: Keep TDG score above 85 (use pmat validate before commit)

ANTI-PATTERNS TO AVOID (from our git history):
❌ DON'T: Silently use defaults when required fields missing
❌ DON'T: Catch-all exception handling without logging
❌ DON'NOT: Skip validation "for performance"

QUALITY GATES:
- pmat analyze tdg --threshold 85
- cargo test --all-features (>85% coverage required)
- No TODO/FIXME/HACK comments in production code

This approach has reduced config errors by 60% in modules where applied.
```

### How to Generate This Automatically

**Step 1: Analyze Organization**
```bash
cargo run -- analyze --org paiml --output paiml-analysis.yaml
```

**Step 2: Extract Defect Patterns (CLI to be implemented)**
```bash
# Future: oip prompts generate
cargo run -- prompts generate \
  --input paiml-analysis.yaml \
  --output prompts/ \
  --format claude-code
```

**Step 3: Use in paiml-mcp-agent-toolkit**

See [Real-World Example](#real-world-example-paiml-mcp-agent-toolkit) below.

---

## Real-World Example: paiml-mcp-agent-toolkit

**Status**: ⚪ **PROPOSED** (illustrative example of Phase 4)

### Scenario: Improving AI Code Generation Prompts

**Goal**: Make paiml-mcp-agent-toolkit generate better code by learning from paiml's historical defects.

### Step 1: Analyze paiml Organization

```bash
cd organizational-intelligence-plugin
export GITHUB_TOKEN=your_token

# Analyze all paiml repos from last 2 years
cargo run -- analyze --org paiml --output ../paiml-mcp-agent-toolkit/data/paiml-defect-analysis.yaml
```

**Output**: `paiml-defect-analysis.yaml` with defect patterns

### Step 2: Extract Key Insights (No PII)

From the analysis, extract generic patterns (no commit hashes, no author names):

```yaml
# paiml-mcp-agent-toolkit/data/defect-patterns-summary.yaml
organizational_insights:
  top_defect_categories:
  - category: ConfigurationErrors
    frequency: 25
    avg_tdg_score: 45.2
    common_patterns:
    - "Missing validation for required fields"
    - "Type coercion without error handling"
    - "Undocumented default values"
    prevention_strategies:
    - "Explicit validation with descriptive errors"
    - "Schema validation before parsing"
    - "Document all defaults in docstrings"

  - category: IntegrationFailures
    frequency: 18
    avg_tdg_score: 52.3
    common_patterns:
    - "HTTP timeout not configured"
    - "Missing retry logic"
    - "No circuit breaker pattern"
    prevention_strategies:
    - "Always set explicit timeouts"
    - "Implement exponential backoff"
    - "Add health check endpoints"

  - category: LogicErrors
    frequency: 12
    avg_tdg_score: 88.5
    common_patterns:
    - "Off-by-one errors in loops"
    - "Incorrect boolean conditions"
    - "Edge case not handled"
    prevention_strategies:
    - "Property-based testing for boundaries"
    - "Explicit boolean variable names"
    - "Test happy path + 3 edge cases minimum"

  code_quality_thresholds:
    tdg_minimum: 85.0
    test_coverage_minimum: 0.85
    max_function_length: 50
    max_cyclomatic_complexity: 10
```

### Step 3: Create Prompt Templates for paiml-mcp-agent-toolkit

**File**: `paiml-mcp-agent-toolkit/prompts/code-generation-with-context.md`

```markdown
# Context-Aware Code Generation Prompt

You are an AI assistant helping write code for the paiml organization.

## Organizational Quality Standards

Based on analysis of {{repositories_analyzed}} repositories with {{commits_analyzed}} commits:

### Code Quality Requirements
- **Minimum TDG Score**: {{tdg_minimum}} (0-100, higher is better)
- **Test Coverage**: {{test_coverage_minimum * 100}}%+
- **Max Function Length**: {{max_function_length}} lines
- **Max Complexity**: {{max_cyclomatic_complexity}} (cyclomatic)

### Common Defect Patterns to Avoid

{{#each top_defect_categories}}
#### {{category}} ({{frequency}} historical occurrences, avg TDG: {{avg_tdg_score}})

**Common Issues:**
{{#each common_patterns}}
- {{this}}
{{/each}}

**Prevention:**
{{#each prevention_strategies}}
✅ {{this}}
{{/each}}

---
{{/each}}

## Quality Gates (Run Before Committing)

```bash
# Check code quality
pmat analyze tdg --threshold {{tdg_minimum}}

# Run tests with coverage
cargo test --all-features
cargo llvm-cov report --summary-only

# Check for technical debt markers
pmat analyze satd --path .

# Validate integration
make test-integration
```

## Code Generation Instructions

When generating code:

1. **Start with tests** (TDD approach)
2. **Validate inputs explicitly** (prevent configuration errors)
3. **Handle errors gracefully** (prevent integration failures)
4. **Document edge cases** (prevent logic errors)
5. **Keep functions focused** (maintain low complexity)
6. **Use descriptive names** (prevent type errors)

## Example: Good vs Bad

### ❌ BAD (Based on Historical Defects)
```rust
// Missing validation, no error handling, unclear defaults
fn parse_config(data: &str) -> Config {
    let parsed = serde_yaml::from_str(data).unwrap(); // Don't do this!
    parsed
}
```

### ✅ GOOD (Prevents Historical Defects)
```rust
/// Parse configuration with validation
///
/// # Errors
/// Returns error if:
/// - YAML is malformed
/// - Required fields missing: api_key, endpoint
/// - Port out of range (1024-65535)
///
/// # Defaults
/// - timeout_ms: 30000 (30 seconds)
/// - retry_count: 3
fn parse_config(data: &str) -> Result<Config> {
    let mut config: Config = serde_yaml::from_str(data)
        .context("Failed to parse YAML configuration")?;

    // Explicit validation (prevents config errors)
    config.validate()
        .context("Configuration validation failed")?;

    // Document defaults
    config.timeout_ms = config.timeout_ms.unwrap_or(30000);
    config.retry_count = config.retry_count.unwrap_or(3);

    Ok(config)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_parse_valid_config() { /* ... */ }

    #[test]
    fn test_parse_missing_required_field() { /* ... */ }

    #[test]
    fn test_parse_invalid_port() { /* ... */ }
}
```

---

**Analysis Date**: {{analysis_date}}
**Repositories Analyzed**: {{repositories_analyzed}}
**Commits Analyzed**: {{commits_analyzed}}
```

### Step 4: Implement Prompt Generator in paiml-mcp-agent-toolkit

**File**: `paiml-mcp-agent-toolkit/src/prompts/defect_aware_prompts.rs`

```rust
use serde::{Deserialize, Serialize};
use std::path::Path;
use anyhow::Result;

/// Defect pattern from OIP analysis
#[derive(Debug, Deserialize)]
pub struct DefectPattern {
    pub category: String,
    pub frequency: usize,
    pub avg_tdg_score: f32,
    pub common_patterns: Vec<String>,
    pub prevention_strategies: Vec<String>,
}

/// OIP analysis summary
#[derive(Debug, Deserialize)]
pub struct DefectAnalysis {
    pub top_defect_categories: Vec<DefectPattern>,
    pub code_quality_thresholds: QualityThresholds,
    pub repositories_analyzed: usize,
    pub commits_analyzed: usize,
    pub analysis_date: String,
}

#[derive(Debug, Deserialize)]
pub struct QualityThresholds {
    pub tdg_minimum: f32,
    pub test_coverage_minimum: f32,
    pub max_function_length: usize,
    pub max_cyclomatic_complexity: usize,
}

/// Generate context-aware prompts from defect analysis
pub struct DefectAwarePromptGenerator {
    analysis: DefectAnalysis,
}

impl DefectAwarePromptGenerator {
    /// Load OIP analysis
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let analysis: DefectAnalysis = serde_yaml::from_str(&content)?;
        Ok(Self { analysis })
    }

    /// Generate prompt for a specific task type
    pub fn generate_prompt(&self, task: &str, context: &str) -> String {
        let mut prompt = String::new();

        // Base task
        prompt.push_str(&format!("# Task\n{}\n\n", task));

        // Add context
        prompt.push_str(&format!("# Context\n{}\n\n", context));

        // Add organizational learnings
        prompt.push_str("# Organizational Quality Standards\n\n");
        prompt.push_str(&format!(
            "Based on analysis of {} repositories with {} commits:\n\n",
            self.analysis.repositories_analyzed,
            self.analysis.commits_analyzed
        ));

        // Quality requirements
        prompt.push_str("## Quality Requirements\n");
        prompt.push_str(&format!(
            "- Minimum TDG Score: {:.1}\n",
            self.analysis.code_quality_thresholds.tdg_minimum
        ));
        prompt.push_str(&format!(
            "- Test Coverage: {:.0}%+\n",
            self.analysis.code_quality_thresholds.test_coverage_minimum * 100.0
        ));

        // Add relevant defect patterns
        prompt.push_str("\n## Common Defect Patterns to Avoid\n\n");
        for pattern in &self.analysis.top_defect_categories {
            if pattern.frequency >= 10 {  // Only show frequent patterns
                prompt.push_str(&format!(
                    "### {} ({} occurrences, TDG: {:.1})\n",
                    pattern.category, pattern.frequency, pattern.avg_tdg_score
                ));

                prompt.push_str("\n**Common Issues:**\n");
                for issue in &pattern.common_patterns {
                    prompt.push_str(&format!("- {}\n", issue));
                }

                prompt.push_str("\n**Prevention:**\n");
                for strategy in &pattern.prevention_strategies {
                    prompt.push_str(&format!("✅ {}\n", strategy));
                }

                prompt.push_str("\n");
            }
        }

        // Quality gates
        prompt.push_str("\n## Quality Gates (Before Committing)\n\n");
        prompt.push_str("```bash\n");
        prompt.push_str(&format!(
            "pmat analyze tdg --threshold {}\n",
            self.analysis.code_quality_thresholds.tdg_minimum
        ));
        prompt.push_str("cargo test --all-features\n");
        prompt.push_str("cargo llvm-cov report --summary-only\n");
        prompt.push_str("```\n");

        prompt
    }

    /// Generate prompt for preventing a specific defect category
    pub fn generate_prevention_prompt(&self, defect_category: &str) -> Option<String> {
        self.analysis
            .top_defect_categories
            .iter()
            .find(|p| p.category == defect_category)
            .map(|pattern| {
                let mut prompt = String::new();
                prompt.push_str(&format!(
                    "# Preventing {}\n\n",
                    pattern.category
                ));
                prompt.push_str(&format!(
                    "**Historical Frequency**: {} occurrences\n",
                    pattern.frequency
                ));
                prompt.push_str(&format!(
                    "**Average Code Quality**: TDG {:.1}/100\n\n",
                    pattern.avg_tdg_score
                ));

                prompt.push_str("## Common Patterns (What to Avoid):\n");
                for issue in &pattern.common_patterns {
                    prompt.push_str(&format!("❌ {}\n", issue));
                }

                prompt.push_str("\n## Prevention Strategies:\n");
                for strategy in &pattern.prevention_strategies {
                    prompt.push_str(&format!("✅ {}\n", strategy));
                }

                prompt
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_defect_analysis() {
        // Test with sample YAML
        let yaml = r#"
top_defect_categories:
- category: ConfigurationErrors
  frequency: 25
  avg_tdg_score: 45.2
  common_patterns:
  - "Missing validation"
  prevention_strategies:
  - "Explicit validation"
code_quality_thresholds:
  tdg_minimum: 85.0
  test_coverage_minimum: 0.85
  max_function_length: 50
  max_cyclomatic_complexity: 10
repositories_analyzed: 25
commits_analyzed: 2500
analysis_date: "2025-11-15"
"#;

        let analysis: DefectAnalysis = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(analysis.top_defect_categories.len(), 1);
        assert_eq!(analysis.repositories_analyzed, 25);
    }
}
```

### Step 5: Use in MCP Server

**File**: `paiml-mcp-agent-toolkit/src/mcp_server.rs`

```rust
use crate::prompts::defect_aware_prompts::DefectAwarePromptGenerator;

pub struct McpServer {
    prompt_generator: DefectAwarePromptGenerator,
}

impl McpServer {
    pub fn new() -> Result<Self> {
        // Load defect analysis on startup
        let prompt_generator = DefectAwarePromptGenerator::from_file(
            "data/defect-patterns-summary.yaml"
        )?;

        Ok(Self { prompt_generator })
    }

    pub async fn handle_code_generation(&self, task: &str) -> String {
        // Generate context-aware prompt
        let enhanced_prompt = self.prompt_generator.generate_prompt(
            task,
            "Generating code for paiml organization"
        );

        // Send to Claude/GPT with enhanced context
        // This now includes organizational learnings!
        enhanced_prompt
    }
}
```

---

## Advanced Use Cases

**Status**: ⚪ **ALL PROPOSED** (require Phases 2-4)

### Use Case 1: Automated Code Review Comments

**Dependencies**: Phase 2 (summarize), Phase 3 (review-pr)

**Integration**: GitHub Actions + OIP + AI

```yaml
# .github/workflows/ai-code-review.yml
name: AI Code Review with Defect Context

on: [pull_request]

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Analyze Organization Defect Patterns
        run: |
          cargo run --manifest-path ../organizational-intelligence-plugin/Cargo.toml \
            -- analyze --org ${{ github.repository_owner }} \
            --output defect-analysis.yaml

      - name: Generate AI Review Prompt
        run: |
          # Extract files changed in PR
          FILES=$(gh pr diff ${{ github.event.pull_request.number }} --name-only)

          # Generate context-aware review prompt
          cat > review-prompt.txt <<EOF
          Review these files for common defect patterns:

          $(cat defect-analysis.yaml | yq '.defect_patterns[] |
            select(.frequency > 10) |
            "- " + .category + ": " + .quality_signals.avg_tdg_score')

          Files to review:
          $FILES
          EOF

      - name: AI Review via Claude Code
        run: |
          claude-code review --context review-prompt.txt --files $FILES
```

### Use Case 2: Developer Onboarding

**Dependencies**: Phase 2 (summarize)
**Goal**: New developers learn from historical mistakes

```bash
# Generate onboarding guide from defect patterns
cargo run -- analyze --org paiml --output defects.yaml

# Create developer guide (manual or automated)
cat > docs/new-developer-guide.md <<EOF
# Welcome to paiml!

## Learn from Our Mistakes (So You Don't Repeat Them)

Based on analyzing our last 2 years of development:

$(yq '.defect_patterns[] |
  "### Avoid: " + .category + "\n" +
  "This happened " + (.frequency | tostring) + " times.\n" +
  (.examples[0].message | "Example: " + .)
' defects.yaml)

## Quality Standards

All code must meet:
- TDG Score: 85+ (check with pmat analyze tdg)
- Test Coverage: 85%+ (check with make coverage)
- No SATD markers in production code (TODO/FIXME/HACK)

Happy coding!
EOF
```

### Use Case 3: Sprint Planning Intelligence

**Dependencies**: Phase 1 (analyze) - ✅ **WORKS TODAY**
**Goal**: Prioritize technical debt with data

```bash
# Generate sprint priorities
cargo run -- analyze --org paiml --output sprint-data.yaml

# Extract high-priority issues
yq '.defect_patterns[] |
  select(.frequency > 15 and .quality_signals.avg_tdg_score < 60) |
  {
    category: .category,
    frequency: .frequency,
    tdg: .quality_signals.avg_tdg_score,
    priority: "URGENT"
  }
' sprint-data.yaml > sprint-priorities.yaml

# Use in sprint planning meeting
echo "These modules need refactoring THIS sprint:"
cat sprint-priorities.yaml
```

---

## Prompt Templates

**Status**: ⚪ **PROPOSED** (require Phase 2 for data, Phase 4 for automation)

### Template 1: Feature Development

```markdown
# Feature: {{feature_name}}

## Context
Implementing {{feature_description}} for {{project_name}}.

## Organizational Learnings
{{#load_defect_analysis}}
Based on analyzing {{repositories_analyzed}} repositories:

### Most Common Defects in Similar Features
{{#filter_by_relevance feature_type}}
{{#top 3 defect_patterns}}
- **{{category}}** ({{frequency}} times, TDG: {{avg_tdg_score}})
  {{#each prevention_strategies}}
  - ✅ {{this}}
  {{/each}}
{{/top}}
{{/filter_by_relevance}}
{{/load_defect_analysis}}

## Requirements
1. Functional requirements: {{requirements}}
2. Quality requirements:
   - TDG Score: 85+
   - Test Coverage: 85%+
   - No SATD markers
3. Must avoid historical patterns above

## Implementation Approach
[AI generates code here with context awareness]
```

### Template 2: Bug Fix

```markdown
# Bug Fix: {{bug_description}}

## Current Behavior
{{current_behavior}}

## Expected Behavior
{{expected_behavior}}

## Root Cause Analysis
{{#load_defect_analysis}}
{{#find_similar_defects bug_category}}
This appears to be a {{category}} defect.

### Historical Pattern
We've seen this {{frequency}} times. Common causes:
{{#each common_patterns}}
- {{this}}
{{/each}}

### Prevention
{{#each prevention_strategies}}
✅ {{this}}
{{/each}}
{{/find_similar_defects}}
{{/load_defect_analysis}}

## Fix Approach
1. Address root cause (not just symptoms)
2. Add regression test
3. Check for similar issues in codebase
4. Update defect patterns document

## Verification
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] TDG score maintained/improved
- [ ] No new SATD markers
```

### Template 3: Refactoring

```markdown
# Refactoring: {{module_name}}

## Current State
{{#load_defect_analysis}}
{{#analyze_module module_name}}
- Defect Frequency: {{defect_count}}
- Average TDG: {{avg_tdg_score}}
- Test Coverage: {{coverage}}%
{{/analyze_module}}
{{/load_defect_analysis}}

## Why Refactor?
{{#if avg_tdg_score < 60}}
🚨 TDG score ({{avg_tdg_score}}) is below threshold (60)
{{/if}}
{{#if defect_count > 10}}
🚨 High defect frequency ({{defect_count}})
{{/if}}
{{#if coverage < 0.70}}
🚨 Low test coverage ({{coverage}}%)
{{/if}}

## Refactoring Strategy
1. **Start with tests**: Improve coverage to 85%+
2. **Fix defect patterns**: Address top {{defect_count}} issues
3. **Improve structure**: Reduce complexity, improve TDG
4. **Validate**: Run all quality gates

## Success Criteria
- [ ] TDG Score: 85+
- [ ] Test Coverage: 85%+
- [ ] Defect patterns: Reduced by 50%+
- [ ] All tests pass

## Before/After Metrics
```

---

## Implementation Checklist

**Note**: This checklist reflects the proposed roadmap. Phase 1 is complete; Phases 2-4 are proposed.

### Phase 1: Basic Integration ✅ **COMPLETE**
- [x] Install OIP in your workspace
- [x] Run analysis on your organization
- [ ] Extract key defect patterns (no PII) - **MANUAL** (Phase 2 will automate)
- [ ] Create summary YAML for paiml-mcp-agent-toolkit - **MANUAL** (Phase 2 will automate)

### Phase 2: Summarization ✅ **COMPLETE** (2-4 hours)
- [x] Implement `oip summarize` command
- [x] Add PII stripping logic
- [x] Add frequency filtering
- [x] Write comprehensive tests (EXTREME TDD)
- [x] Validate YAML output format

### Phase 3: PR Review ✅ **COMPLETE** (4-6 hours)
- [x] Implement `oip review-pr` command (src/pr_reviewer.rs)
- [x] Add baseline loading logic (PrReviewer::load_baseline)
- [x] Add file-based pattern matching (is_config_file, is_integration_file, is_code_file)
- [x] Generate markdown and JSON reports
- [x] Test with real PR scenarios (0.125s performance)
- [x] Write 11 comprehensive unit tests (100% passing)

### Phase 4: AI Integration ⚪ **PROPOSED** (8-16 hours)
- [ ] Implement DefectAwarePromptGenerator in paiml-mcp-agent-toolkit
- [ ] Create prompt templates for various scenarios
- [ ] Integrate with MCP server
- [ ] Add GitHub Actions code review automation
- [ ] Test effectiveness with real code generation tasks

---

## Best Practices

### 1. Privacy: No PII in Prompts

**✅ GOOD (Generic Patterns):**
```yaml
common_patterns:
- "Missing validation for required fields"
- "HTTP timeout not configured"
```

**❌ BAD (Contains PII):**
```yaml
examples:
- commit: "abc123"
  author: "john.doe@company.com"  # PII!
  message: "Fixed config bug in customer module"  # Potentially sensitive
```

### 2. Keep Prompts Actionable

**✅ GOOD:**
```
Prevention: Always set explicit timeouts
Example: reqwest::Client::builder().timeout(Duration::from_secs(30))
```

**❌ BAD:**
```
Prevention: Be careful with timeouts
```

### 3. Update Regularly

```bash
# Weekly analysis
0 9 * * 1 cargo run -- analyze --org paiml --output latest.yaml

# Monthly prompt regeneration
0 10 1 * * ./scripts/regenerate-prompts.sh
```

### 4. Validate Effectiveness

Track metrics before/after using enhanced prompts:
- Defect frequency (should decrease)
- TDG scores (should increase)
- Developer velocity (should improve)

---

## Troubleshooting

### Issue: OIP Analysis Takes Too Long

**Solution**: Use `--max-concurrent` to limit parallel repo analysis
```bash
cargo run -- analyze --org paiml --output report.yaml --max-concurrent 5
```

### Issue: Too Much Data for Prompts

**Solution**: Filter to top N defect categories
```bash
yq '.defect_patterns[] | select(.frequency > 10)' report.yaml
```

### Issue: pmat Not Found

**Solution**: Ensure pmat is in PATH
```bash
which pmat
export PATH=$PATH:~/.cargo/bin
```

---

## Future Enhancements

**Note**: These are exploratory ideas beyond the 4-phase roadmap.

### Planned Features

1. **Real-time Integration**
   - OIP as pmat subcommand: `pmat org analyze`
   - Streaming updates to MCP server

2. **ML-Based Pattern Recognition**
   - Train on historical defects
   - Predict likely defects in new code

3. **Interactive Prompt Builder**
   - Web UI to customize prompts
   - A/B testing for prompt effectiveness

4. **Multi-Org Comparison**
   - Benchmark against industry standards
   - Share anonymized patterns

---

## Summary

### What You Can Do Today (Phase 1)

✅ **IMPLEMENTED**:
```bash
# Analyze your organization
cargo run -- analyze --org YOUR_ORG --output report.yaml

# Review the YAML report
# Contains: defect patterns, TDG scores, churn metrics, examples (with PII)
```

**Limitations**:
- Manual PII removal required before sharing
- No automated summarization
- No PR review integration
- No AI prompt generation

### What's Coming (Phase 4)

✅ **Phase 2 (DONE)**: `oip summarize` - Automated PII stripping (2-4 hours)
✅ **Phase 3 (DONE)**: `oip review-pr` - Fast PR reviews <30s (4-6 hours, 0.125s actual)
⚪ **Phase 4 (PROPOSED)**: AI integration with paiml-mcp-agent-toolkit (8-16 hours)

### Key Takeaways

1. **Phase 1 is production-ready**: Use `oip analyze` today for organizational intelligence
2. **Phases 2-4 eliminate waste**: Automation removes manual steps (Toyota Way)
3. **Privacy-first design**: PII stripping is a first-class feature in Phase 2
4. **Fast feedback**: Phase 3 enables <30s PR reviews via stateful baselines
5. **Context-aware AI**: Phase 4 generates prompts that prevent historical mistakes

### Start Here (Today)

```bash
# 1. Clone and build
git clone https://github.com/paiml/organizational-intelligence-plugin
cd organizational-intelligence-plugin
cargo build --release

# 2. Analyze your organization (Phase 1)
export GITHUB_TOKEN=your_token
cargo run -- analyze --org YOUR_ORG --output report.yaml

# 3. Summarize for AI consumption (Phase 2)
cargo run -- summarize \
  --input report.yaml \
  --output summary.yaml \
  --strip-pii \
  --top-n 10 \
  --min-frequency 5

# 4. Review PRs with organizational context (Phase 3)
cargo run -- review-pr \
  --baseline summary.yaml \
  --files "src/config.yaml,src/api.rs" \
  --format markdown
```

### Contributing to Phase 2

This is a **design specification** that guides implementation. If you want to help implement `oip summarize`:

1. See Phase 2 section above for detailed implementation design
2. Follow EXTREME TDD methodology (RED-GREEN-REFACTOR)
3. Ensure all quality gates pass: `make lint`, `make test-fast`, `make coverage`
4. Submit PR with tests achieving 85%+ coverage

**Questions?** Open an issue at: https://github.com/paiml/organizational-intelligence-plugin

---

**Document Version**: 1.0
**Last Updated**: 2025-11-15
**Maintained By**: paiml organization
**License**: MIT
