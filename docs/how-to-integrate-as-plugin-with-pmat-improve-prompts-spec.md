# How to Integrate Organizational Intelligence as a pmat Plugin

## Overview

This guide shows how to integrate the Organizational Intelligence Plugin (OIP) with pmat and use its insights to generate better AI prompts, particularly for improving the paiml-mcp-agent-toolkit.

**Key Insight**: OIP analyzes your organization's defect patterns and code quality. These insights can be fed into AI prompt engineering to create context-aware, defect-preventing prompts.

## Table of Contents

1. [Integration Architecture](#integration-architecture)
2. [Using OIP as a pmat Plugin](#using-oip-as-a-pmat-plugin)
3. [Generating AI Prompts from Defect Patterns](#generating-ai-prompts-from-defect-patterns)
4. [Real-World Example: paiml-mcp-agent-toolkit](#real-world-example-paiml-mcp-agent-toolkit)
5. [Advanced Use Cases](#advanced-use-cases)
6. [Prompt Templates](#prompt-templates)

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

## Using OIP as a pmat Plugin

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

### Use Case 1: Automated Code Review Comments

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

### Phase 1: Basic Integration (1-2 hours)
- [ ] Install OIP in your workspace
- [ ] Run analysis on your organization
- [ ] Extract key defect patterns (no PII)
- [ ] Create summary YAML for paiml-mcp-agent-toolkit

### Phase 2: Prompt Enhancement (2-3 hours)
- [ ] Implement DefectAwarePromptGenerator
- [ ] Create prompt templates
- [ ] Integrate with MCP server
- [ ] Test with sample tasks

### Phase 3: Automation (2-4 hours)
- [ ] Set up weekly defect analysis
- [ ] Create GitHub Actions workflow
- [ ] Implement automatic prompt updates
- [ ] Add metrics tracking

### Phase 4: Advanced Features (4-8 hours)
- [ ] Code review automation
- [ ] Developer onboarding guide generation
- [ ] Sprint planning intelligence
- [ ] Trend analysis dashboard

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

**Key Takeaways:**

1. **OIP provides organizational intelligence** that makes AI prompts context-aware
2. **Integration is straightforward**: Run analysis → Extract patterns → Enhance prompts
3. **Benefits are measurable**: Track defect reduction, TDG improvement, velocity
4. **Privacy is preserved**: Use generic patterns, no PII
5. **Continuous improvement**: Weekly analysis keeps prompts relevant

**Start Here:**
```bash
# 1. Analyze your org
cargo run -- analyze --org YOUR_ORG --output analysis.yaml

# 2. Extract top 3 defect categories
yq '.defect_patterns | sort_by(.frequency) | reverse | .[0:3]' analysis.yaml

# 3. Add to your AI prompts
# "Avoid these patterns: [paste here]"
```

**Questions?** Open an issue at: https://github.com/paiml/organizational-intelligence-plugin

---

**Document Version**: 1.0
**Last Updated**: 2025-11-15
**Maintained By**: paiml organization
**License**: MIT
