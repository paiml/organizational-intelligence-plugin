# Organizational Intelligence Plugin - Technical Specification

**Version:** 1.0.0
**Status:** Draft
**Last Updated:** 2025-11-15

## Executive Summary

The Organizational Intelligence Plugin (OIP) is a high-performance Rust CLI tool and MCP (Model Context Protocol) server designed to analyze the combined git history of GitHub organizations to identify, classify, and report common defect patterns. By leveraging distributed computing, AST-based analysis, and machine learning techniques, OIP enables engineering teams to proactively identify and prevent recurring bugs across codebases.

**Design Philosophy: Toyota Way Principles**

This specification applies Toyota Production System principles to software architecture:

- **Genchi Genbutsu (Go and See)**: Start with simple, observable systems. Add complexity only after profiling reveals bottlenecks. Every architecture decision includes validation criteria.

- **Kaizen (Continuous Improvement)**: The system evolves through validation gates. Proceed to next phase only when success criteria are met. Adapt based on observed data, not assumptions.

- **Respect for People**: Empower the engineering team with guidelines, not mandates. Focus quality on outcomes (defect escape rate, MTTR) rather than process compliance. Every classification includes explanations to help developers learn.

- **Jidoka (Automation with Human Intelligence)**: ML classifier provides suggestions with confidence scores. Human feedback improves the system. Users can always override automated classifications.

- **Five Whys**: Every major design decision (storage tiers, ML approach, quality gates) is questioned repeatedly to ensure we address root causes, not symptoms.

**This specification avoids premature optimization and complexity without evidence.** We start simple, measure continuously, and evolve based on data.

## 1. Overview

### 1.1 Purpose

OIP searches across organizational git repositories to discover patterns in historical defects, enabling:
- **Proactive defect prevention** through pattern recognition
- **Data-driven quality improvement** via defect classification
- **Knowledge transfer** by documenting common failure modes
- **Integration with pmat** (PAIML MCP Agent Toolkit) for automated quality enforcement

### 1.2 Core Capabilities

1. **Multi-Repository Mining**: Concurrent analysis of GitHub organization repositories
2. **Defect Pattern Classification**: ML-based categorization into 5-10 common bug types
3. **Frequency Analysis**: Statistical ranking of defect patterns by occurrence
4. **Structured YAML Output**: Machine-readable defect reports for downstream tooling
5. **Efficient Storage**: Multi-tier caching system based on pmat TDG architecture
6. **Dual Interface**: Standalone Rust CLI and MCP server for LLM integration
7. **Extreme TDD Workflow**: Test-first development with 85%+ coverage and mutation validation

### 1.3 Integration Requirements

**A. PMAT Tooling Enforcement (Requirement A)**
- Quality gates defined in `pmat-quality.toml`
- Roadmap validation via pmat TDG
- Pre-commit hooks for quality enforcement
- Makefile-driven validation pipeline

**B. Shell and Makefile Standards (Requirement B)**
- All shell scripts validated against `../bashrc` standards
- Makefile targets unit tested using bats or similar
- Shellcheck compliance for all bash code
- Reproducible build environment

## 2. Architecture

### 2.1 System Architecture

**Design Philosophy: Start Simple, Evolve Based on Evidence (Genchi Genbutsu)**

The architecture follows the Toyota Way principle of avoiding premature optimization. We begin with the simplest design that delivers value, then evolve based on observed performance data.

```
┌─────────────────────────────────────────────────────────┐
│                    User Interfaces                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │ CLI Tool │  │   MCP    │  │   TUI    │             │
│  │          │  │  Server  │  │ (Future) │             │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘             │
└───────┼─────────────┼─────────────┼────────────────────┘
        │             │             │
        └─────────────┴─────────────┘
                      │
┌─────────────────────┴─────────────────────────────────┐
│              Service Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │   GitHub     │  │   Pattern    │  │   Defect    │ │
│  │   Miner      │  │  Classifier  │  │  Analyzer   │ │
│  └──────────────┘  └──────────────┘  └─────────────┘ │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │     AST      │  │Data Labeling │  │   Report    │ │
│  │   Parser     │  │  Framework   │  │  Generator  │ │
│  └──────────────┘  └──────────────┘  └─────────────┘ │
└────────────────────────┬──────────────────────────────┘
                         │
┌────────────────────────┴──────────────────────────────┐
│           Storage Layer (Simple First)                 │
│  ┌──────────────┐  ┌──────────────────────────────┐  │
│  │   L1: Hot    │→ │  L2: Persistent Storage      │  │
│  │   (DashMap)  │  │  (Parquet + Optional Index)  │  │
│  │   <200ns     │  │  ~1-10ms (with OS cache)     │  │
│  └──────────────┘  └──────────────────────────────┘  │
│                                                        │
│  L3: Future Evolution (Add Only When Validated)       │
│  - SQL cache (if query patterns demand)               │
│  - S3/Object storage (if scale demands)               │
└────────────────────────────────────────────────────────┘
```

**Metrics-Driven Evolution:**
- Instrumentation added from day one to measure cache hit rates, query patterns, latency distributions
- L3 tier introduced only after profiling reveals specific bottlenecks (see Section 2.3.4)

### 2.2 Component Descriptions

#### 2.2.1 GitHub Miner
- **Purpose**: Concurrent git history analysis across organization repositories
- **Technology**:
  - `octocrab` for GitHub API access
  - `git2-rs` for local git operations
  - `tokio` for async I/O and concurrency
  - `rayon` for CPU-bound parallel processing
- **Features**:
  - Rate-limiting and backoff strategies
  - Incremental repository cloning
  - Commit-level blame analysis
  - PR and issue correlation

#### 2.2.2 AST Parser
- **Purpose**: Language-agnostic abstract syntax tree generation
- **Technology**:
  - `tree-sitter` with multi-language grammar support
  - `syn` for Rust-specific parsing
  - Custom parsers for domain-specific languages
- **Output**: Normalized AST representations for defect analysis

#### 2.2.3 Pattern Classifier

**Philosophy: Data-First, Incremental Complexity (Kaizen)**

The classifier evolves through three phases, each validated before proceeding:

**Phase 1: Rule-Based Foundation (MVP)**
- **Purpose**: Establish baseline classification and *data collection infrastructure*
- **Approach**:
  - Heuristic rules based on commit message patterns (e.g., "fix race condition" → Concurrency Bug)
  - File path analysis (e.g., `unsafe` blocks in Rust → Memory Safety)
  - Issue/PR label correlation (e.g., "bug:concurrency" → Concurrency Bug)
  - **Critical**: Every classification includes a confidence score and "needs review" flag
- **Success Criteria**: 60%+ accuracy on manually labeled validation set (100 commits)
- **Data Collection**: Build feedback loop where users can confirm/correct classifications

**Phase 2: Interpretable ML (Validated Evolution)**
- **Trigger**: After collecting 1000+ labeled examples with user feedback
- **Approach**:
  - Gradient Boosting (XGBoost/LightGBM) on engineered features from AST diffs
  - Features: Change size, complexity delta, modified AST node types, identifier patterns
  - Model explainability via SHAP values (critical for "Respect for People")
  - **Why not GNN yet?** Simpler models provide baseline; explainability aids developer learning
- **Success Criteria**: 75%+ accuracy, SHAP values provide actionable insights
- **Validation**: A/B test against rule-based system for 2 weeks

**Phase 3: Advanced ML (Evidence-Based)**
- **Trigger**: Phase 2 accuracy plateaus AND we have 5000+ labeled examples
- **Evaluation**:
  - Benchmark GNN vs. current model on holdout set
  - Measure accuracy improvement vs. inference latency increase
  - Assess explainability trade-off (can we explain GNN predictions?)
- **Decision**: Proceed only if improvement >10% AND explainability acceptable

**Defect Taxonomy (10 Categories):**
  1. **Memory Safety Violations**: Use-after-free, null pointer dereference, buffer overflow
  2. **Concurrency Bugs**: Data races, deadlocks, atomicity violations
  3. **Logic Errors**: Off-by-one, incorrect conditionals, state machine violations
  4. **API Misuse**: Incorrect parameter ordering, missing error handling, lifecycle violations
  5. **Resource Leaks**: File handles, network connections, memory leaks
  6. **Type Errors**: Type confusion, casting errors, serialization failures
  7. **Configuration Errors**: Missing env vars, incorrect settings, deployment misconfigurations
  8. **Security Vulnerabilities**: SQL injection, XSS, authentication bypass
  9. **Performance Issues**: N+1 queries, inefficient algorithms, memory bloat
  10. **Integration Failures**: Version incompatibilities, breaking API changes

**User Feedback Loop (Critical for Long-Term Success):**
```rust
pub struct Classification {
    category: DefectCategory,
    confidence: f32,  // 0.0 to 1.0
    explanation: String,  // Why this classification? (SHAP values, rules matched)
    needs_review: bool,  // Flag for human validation
    feedback: Option<UserFeedback>,
}

pub struct UserFeedback {
    correct: bool,
    actual_category: Option<DefectCategory>,
    comments: String,
}
```

#### 2.2.4 Defect Analyzer
- **Purpose**: Historical pattern mining and statistical analysis
- **Techniques**:
  - Temporal analysis (defect trends over time)
  - Spatial analysis (defect hotspots in codebase)
  - Contributor correlation (common patterns by team/developer)
  - Semantic code change mining (CPatMiner-inspired approach)

#### 2.2.5 Report Generator
- **Purpose**: Structured YAML output generation (Requirement E)
- **Output Schema**:
```yaml
version: "1.0"
organization: "example-org"
analysis_date: "2025-11-15T00:00:00Z"
repositories_analyzed: 150
total_defects_analyzed: 5432
time_range:
  start: "2020-01-01T00:00:00Z"
  end: "2025-11-15T00:00:00Z"

defect_patterns:
  - category: "Concurrency Bugs"
    frequency: 1234
    percentage: 22.7
    severity_distribution:
      critical: 45
      high: 234
      medium: 678
      low: 277
    examples:
      - repository: "repo-name"
        commit: "abc123def456"
        file: "src/main.rs"
        line_range: [45, 67]
        description: "Data race in shared counter"
        fix_commit: "def456abc123"
    common_locations:
      - "async handlers"
      - "shared state management"
    prevention_strategies:
      - "Use Arc&lt;Mutex&lt;T&gt;&gt; for shared mutable state"
      - "Prefer message passing over shared memory"
    references:
      - "Understanding and Detecting Real-World Safety Issues in Rust"

  - category: "Memory Safety Violations"
    frequency: 987
    percentage: 18.2
    # ... similar structure
```

### 2.3 Storage System (Requirement C)

**Philosophy: Simplicity First, Validate Before Evolving**

Inspired by pmat TDG architecture but starting with minimal complexity.

#### 2.3.1 Two-Tier Storage (MVP)

**L1: Hot Cache (In-Memory)**
```rust
use dashmap::DashMap;
use blake3::Hash;

pub struct HotCache {
    store: DashMap<Hash, DefectRecord>,
    max_size_mb: u64,
    ttl_seconds: u64,
    // Metrics for validation
    metrics: Arc<CacheMetrics>,
}

pub struct DefectRecord {
    identity: DefectIdentity,
    classification: Classification,  // Now includes explanation + confidence
    metadata: DefectMetadata,
    semantic_signature: Blake3Hash,
}

pub struct CacheMetrics {
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    latency_histogram: Histogram,  // To validate if we need L2
}
```

**L2: Persistent Storage (Parquet)**
```rust
pub trait StorageBackend: Send + Sync {
    async fn put(&self, key: &[u8], value: &[u8]) -> Result<()>;
    async fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>>;

    // Analytical queries on Parquet files
    async fn query_by_category(&self, category: DefectCategory) -> Result<Vec<DefectRecord>>;
    async fn query_by_frequency(&self, min_count: u32) -> Result<Vec<DefectPattern>>;
    async fn query_time_range(&self, start: DateTime, end: DateTime) -> Result<Vec<DefectRecord>>;

    // Metrics for validation
    fn get_metrics(&self) -> StorageMetrics;
}

pub struct ParquetStorage {
    base_path: PathBuf,
    // Optional: Maintain lightweight index for common queries
    index: Option<BTreeMap<DefectCategory, Vec<RecordLocation>>>,
    compression: CompressionCodec,  // LZ4 by default
    metrics: Arc<StorageMetrics>,
}

pub struct StorageMetrics {
    read_latency: Histogram,
    write_latency: Histogram,
    query_patterns: HashMap<String, u64>,  // Track which queries are common
}
```

**Why Parquet for L2?**
- Columnar format optimized for analytical queries (group by category, time range)
- Built-in compression (LZ4/Snappy)
- Standard format for data science tools (DuckDB, Polars, Pandas)
- OS filesystem cache provides "warm" tier behavior for recent files

**Trade-off Analysis:**
- ✅ Simpler than maintaining LibSQL database
- ✅ Natural fit for append-only defect records
- ✅ Easy to partition by org/repo/date
- ⚠️ Slower random access than SQL (mitigated by L1 cache + optional index)
- ⚠️ Less flexible querying (mitigated by DuckDB for ad-hoc queries)

#### 2.3.2 Optional Index Layer

For frequently accessed query patterns, maintain a lightweight index:
```rust
pub struct DefectIndex {
    // In-memory B-tree for fast category lookups
    by_category: BTreeMap<DefectCategory, Vec<RecordLocation>>,
    by_repo: BTreeMap<String, Vec<RecordLocation>>,
    by_month: BTreeMap<String, Vec<RecordLocation>>,  // "2025-01"
}

pub struct RecordLocation {
    file_path: PathBuf,
    row_group: u32,
    row_offset: u64,
}
```

Index is rebuilt on startup (fast for Parquet metadata) and updated on writes.

#### 2.3.3 Future L3: Object Storage (Conditional)

**Validation Trigger:** Add S3/object storage ONLY if:
- Dataset exceeds 100GB on disk
- Multiple machines need access
- Cost analysis shows S3 is cheaper than local SSD

**Implementation:**
- Parquet files uploaded to S3 with partitioning: `s3://bucket/org={org}/year={year}/month={month}/`
- Local caching of recently accessed partitions
- Pre-signed URLs for direct analytical tool access

#### 2.3.4 Future SQL Cache (Conditional)

**Validation Trigger:** Add LibSQL/SQLite middle tier ONLY if profiling shows:
- Complex relational queries are common (e.g., "find all concurrency bugs fixed by developer X in repo Y")
- Query latency p95 > 100ms
- Index layer insufficient for query patterns

**Five Whys Before Adding:**
1. Why is the query slow? (Profile first)
2. Why can't we optimize Parquet access? (Try DuckDB, partition pruning)
3. Why can't we precompute this query result? (Caching layer)
4. Why do we need SQL semantics? (Validate assumption)
5. Why not use an external analytical DB? (DuckDB, ClickHouse)

**Decision Framework:**
| Metric | Threshold | Action |
|--------|-----------|--------|
| p95 query latency | > 100ms | Investigate optimization |
| Complex join queries | > 10/day | Consider SQL cache |
| Cache hit rate | < 80% | Improve caching strategy first |
| Development velocity | Blocked | Validate if SQL helps |

### 2.4 Concurrency Model (Requirement G)

**Distributed Computing Best Practices:**
1. **Repository-level parallelism**: Each repo analyzed in separate async task
2. **Commit-level concurrency**: Batch processing with `rayon` parallel iterators
3. **AST parsing pipeline**: Lock-free work-stealing queue for parse jobs
4. **Storage writes**: Buffered batch writes to minimize lock contention

**Rust Concurrency Primitives:**
```rust
use tokio::task::JoinSet;
use rayon::prelude::*;
use crossbeam::channel::{bounded, Sender, Receiver};
use dashmap::DashMap;

// Repository mining with bounded concurrency
async fn mine_organization(org: &str, max_concurrent: usize) -> Result<AnalysisResult> {
    let semaphore = Arc::new(Semaphore::new(max_concurrent));
    let mut join_set = JoinSet::new();

    for repo in fetch_repos(org).await? {
        let permit = semaphore.clone().acquire_owned().await?;
        join_set.spawn(async move {
            let result = analyze_repository(repo).await;
            drop(permit);
            result
        });
    }

    // Collect results with proper error handling
    let mut results = Vec::new();
    while let Some(result) = join_set.join_next().await {
        results.push(result??);
    }

    aggregate_results(results)
}

// CPU-bound parallel processing with rayon
fn classify_defects(commits: Vec&lt;CommitData&gt;) -> Vec&lt;ClassifiedDefect&gt; {
    commits
        .par_iter()
        .filter_map(|commit| classify_single_defect(commit))
        .collect()
}
```

## 3. Development Workflow (Requirement H)

**Philosophy: Empower the Team, Measure Outcomes (Respect for People + Kaizen)**

The workflow is designed as a set of strong defaults and recommendations, not rigid mandates. The team is empowered to adapt based on context, with quality measured by outcomes (defect escape rate, MTTR) rather than just process compliance.

### 3.1 Recommended TDD Workflow

**Ticket → Test → Code → CI Validation → Runnable Example**

1. **Ticket Creation**
   - GitHub issue with acceptance criteria
   - Link to specification section
   - Test scenarios defined upfront (when applicable)
   - **Context matters**: Exploratory prototyping may defer tests; critical paths demand tests first

2. **Test-First Development (Recommended for Core Logic)**
   ```rust
   // tests/unit/classifier_test.rs
   #[test]
   fn test_classify_memory_safety_violation() {
       let commit = create_test_commit_with_use_after_free();
       let classifier = DefectClassifier::new();

       let result = classifier.classify(&commit).unwrap();

       assert_eq!(result.category, DefectCategory::MemorySafety);
       assert!(result.confidence > 0.8);
       // CRITICAL: Assert explanation is present for developer learning
       assert!(!result.explanation.is_empty());
   }
   ```

3. **Implementation**
   ```rust
   // src/classifier.rs
   impl DefectClassifier {
       pub fn classify(&self, commit: &CommitData) -> Result&lt;Classification&gt; {
           // Implementation follows test requirements
       }
   }
   ```

4. **Pre-Commit Quality Gates (Fast Feedback)**
   ```bash
   # Runs in <30 seconds to avoid developer friction
   make pre-commit
   ```

   ```makefile
   pre-commit: fmt-check lint-fast
       cargo nextest run --no-fail-fast --test-threads=8 --run-ignored=default
       @echo "✅ Pre-commit checks passed"

   fmt-check:
       cargo fmt --check

   lint-fast:
       # Only check changed files for speed
       cargo clippy --all-targets -- -D warnings
   ```

5. **CI Pipeline (Comprehensive Validation)**
   ```bash
   # Runs in CI, not blocking local development
   make ci-validate
   ```

   ```makefile
   ci-validate: lint-full test-fast coverage-report mutation-test-ci
       @echo "✅ All CI quality gates passed"

   coverage-report:
       cargo llvm-cov --workspace --html --lcov --output-path=coverage.lcov
       # Report coverage as metric, don't hard-fail on threshold
       # Let team discuss if coverage drops significantly

   mutation-test-ci:
       # Budget-limited mutation testing (5 min max)
       # Focus on critical modules identified in pmat-quality.toml
       pmat mutation-test --budget-minutes=5 --critical-modules=classifier,storage
       cargo mutants --timeout=60 --jobs=4 -F skip-slow --test-tool=nextest
   ```

6. **Fuzzing (Critical Paths Only)**
   ```bash
   # Run weekly in CI, not per-commit
   make fuzz-critical
   ```

7. **Runnable Example (Validation)**
   ```bash
   cargo run --example analyze_defects -- --org example-org --output report.yaml
   ```

**Key Philosophy Shifts:**

- **Coverage as a signal, not a gate**: 85% is a *goal* and discussion trigger, not a hard blocker
  - If coverage drops to 70%, the team discusses *why* and if it's acceptable (e.g., generated code, exploratory work)
  - Focus: Are critical paths tested? Not: Did we hit a number?

- **Mutation testing in CI, not pre-commit**: Expensive operations belong in CI
  - Budget-limited (5 min) to respect time
  - Focused on modules marked "critical" in config

- **Empower adaptation**: Team can skip TDD for prototyping, then add tests before merge
  - Trust developers to make context-appropriate decisions
  - Review discussions focus on outcomes: "Did we catch this bug type before?"

### 3.2 Quality Standards and Metrics

**Philosophy: Outcome-Focused, Data-Driven (Toyota Way)**

Quality is measured by outcomes (defect escape rate, time to resolution) and supported by process metrics (coverage, mutation score) as signals, not gates.

**Makefile Targets:**
```makefile
.PHONY: pre-commit ci-validate test-fast test-all coverage-report lint-fast lint-full

# Fast pre-commit hook (<30 seconds)
pre-commit: fmt-check lint-fast test-fast
	@echo "✅ Pre-commit checks passed (fast feedback)"

# Comprehensive CI validation
ci-validate: lint-full test-all coverage-report mutation-test-ci
	@echo "✅ All CI quality gates passed"
	@echo "📊 Review metrics in coverage.lcov and mutation-report.json"

test-fast:
	cargo nextest run --no-fail-fast --test-threads=8 --timeout=180

test-all:
	cargo test --all-features --workspace --timeout=300

coverage-report:
	cargo llvm-cov --workspace --html --lcov --output-path=coverage.lcov
	@echo "📈 Coverage report: coverage.lcov"
	@echo "🎯 Target: 85% (goal, not hard gate)"
	# Generate coverage summary as JSON for tracking trends
	cargo llvm-cov --workspace --json --output-path=coverage.json

lint-fast:
	cargo clippy --all-targets -- -D warnings

lint-full:
	cargo fmt --check
	cargo clippy --all-targets --all-features -- -D warnings -D clippy::pedantic

mutation-test-ci:
	pmat mutation-test --budget-minutes=5 --critical-modules=classifier,storage --json-output=mutation-report.json
	@echo "🧬 Mutation testing complete. Target: 95% for critical modules"

fuzz-critical:
	# Run weekly, not per-commit
	cargo +nightly fuzz run classifier_fuzz -- -max_total_time=600 -jobs=4
	cargo +nightly fuzz run storage_fuzz -- -max_total_time=300 -jobs=4
```

**Quality Thresholds (pmat-quality.toml) - As Guidelines:**
```toml
[general]
philosophy = "Guidelines with team discretion, not rigid enforcement"

[complexity]
# Signals for review discussion, not hard failures
cyclomatic_threshold = 15  # Warn, don't fail
cognitive_threshold = 20   # Warn, don't fail
max_nesting_depth = 5
max_function_lines = 100
enforcement = "warn"  # Options: "warn", "error"

[coverage]
# Target as goal and discussion trigger
target_coverage = 85.0
fail_under = 70.0  # Only fail if dramatic drop
enforce_on_new_code = true
exclude_patterns = ["*/tests/*", "*/examples/*", "*/benches/*"]

# Critical modules must maintain high coverage
[coverage.critical_modules]
classifier = 90.0
storage = 85.0
github_miner = 80.0

[satd]
# Allow TODOs with ticket references
enabled = true
zero_tolerance = false  # Changed from rigid zero tolerance
patterns = ["TODO", "FIXME", "HACK", "XXX", "BUG"]
allow_with_ticket = true  # "TODO(#123): ..." is acceptable
max_age_days = 30  # Flag TODOs older than 30 days

[mutation_testing]
# Focus on critical paths
critical_modules = ["classifier", "storage", "github_miner"]
target_survival_rate = 0.05  # 95% mutation kill rate
budget_minutes = 5

[performance]
max_compilation_time_seconds = 300
max_test_time_seconds = 300
```

### 3.3 Outcome Metrics (The Real Quality Indicators)

**Track These in CI Dashboard:**

```yaml
# .github/metrics-config.yaml
outcome_metrics:
  primary:
    - name: "Defect Escape Rate"
      description: "Bugs found in production vs. caught in dev/CI"
      target: "< 5% escape to production"
      collection: "GitHub issues labeled 'bug' + 'production'"

    - name: "Mean Time to Resolution (MTTR)"
      description: "Time from bug report to fix deployed"
      target: "< 48 hours for critical, < 1 week for others"
      collection: "GitHub issue close time - create time"

    - name: "Test Effectiveness"
      description: "% of bugs that would have been caught by existing tests"
      target: "> 80%"
      collection: "Manual tag on bug reports"

  supporting:
    - name: "Coverage Trend"
      target: "Stable or increasing"
      alert_on: "Drops > 10% week-over-week"

    - name: "Mutation Score (Critical Modules)"
      target: "> 95%"
      alert_on: "Falls below 90%"

    - name: "CI Duration"
      target: "< 10 minutes"
      alert_on: "Exceeds 15 minutes"
```

**Weekly Team Review:**
- Review defect escape rate and MTTR
- Discuss: "What bug types are escaping? How do we catch them earlier?"
- Adjust process based on data, not dogma

## 4. Technical Requirements

### 4.1 Rust CLI Implementation (Requirement F)

**CLI Framework:** `clap` v4 with derive macros

**Core Commands:**
```bash
# Analyze entire organization
oip analyze --org rust-lang --output defects.yaml

# Analyze specific repositories
oip analyze --repos rust-lang/rust,rust-lang/cargo --output defects.yaml

# Classify existing defect database
oip classify --input defects.db --output classified.yaml

# Generate report from cached data
oip report --org rust-lang --format yaml --output report.yaml

# Start MCP server
oip serve --port 3000 --transport stdio

# Export data for analysis
oip export --org rust-lang --format parquet --output defects.parquet
```

**CLI Structure:**
```rust
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "oip")]
#[command(about = "Organizational Intelligence Plugin - Defect Pattern Analysis")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[arg(long, global = true)]
    config: Option&lt;PathBuf&gt;,

    #[arg(long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    Analyze(AnalyzeArgs),
    Classify(ClassifyArgs),
    Report(ReportArgs),
    Serve(ServeArgs),
    Export(ExportArgs),
}

#[derive(Args)]
struct AnalyzeArgs {
    #[arg(long, required = true)]
    org: String,

    #[arg(long)]
    repos: Option&lt;Vec&lt;String&gt;&gt;,

    #[arg(long, short, default_value = "defects.yaml")]
    output: PathBuf,

    #[arg(long, default_value = "10")]
    max_concurrent: usize,
}
```

### 4.2 MCP Server Implementation (Requirement F)

**MCP SDK:** `pmcp` v1.4.2

**Exposed Tools:**
```rust
pub struct OipMcpServer {
    context: Arc&lt;OipContext&gt;,
    classifier: Arc&lt;DefectClassifier&gt;,
    storage: Arc&lt;dyn StorageBackend&gt;,
}

impl OipMcpServer {
    async fn register_tools(&self) -> Result&lt;()&gt; {
        self.register(AnalyzeOrganizationTool::new(self.context.clone())).await?;
        self.register(ClassifyDefectTool::new(self.classifier.clone())).await?;
        self.register(QueryPatternsTool::new(self.storage.clone())).await?;
        self.register(GenerateReportTool::new(self.context.clone())).await?;
        Ok(())
    }
}

// Example MCP tool
#[async_trait]
impl McpTool for AnalyzeOrganizationTool {
    fn metadata(&self) -> ToolMetadata {
        ToolMetadata {
            name: "oip_analyze_organization".to_string(),
            description: "Analyze GitHub organization for defect patterns".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "organization": {"type": "string"},
                    "max_repos": {"type": "number"},
                    "time_range_days": {"type": "number"}
                },
                "required": ["organization"]
            }),
        }
    }

    async fn execute(&self, params: Value) -> Result&lt;Value, McpError&gt; {
        let org: String = serde_json::from_value(params["organization"].clone())?;
        let result = self.context.analyze_org(&org).await?;
        Ok(serde_json::to_value(result)?)
    }
}
```

### 4.3 Dependencies

**Core Dependencies:**
```toml
[dependencies]
# Async runtime
tokio = { version = "1.40", features = ["full"] }
async-trait = "0.1"

# CLI framework
clap = { version = "4.5", features = ["derive", "env"] }

# GitHub integration
octocrab = "0.39"
git2 = "0.19"

# AST parsing
tree-sitter = "0.22"
syn = "2.0"

# Storage
dashmap = "6.0"
libsql = "0.6"
sled = { version = "0.34", optional = true }

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_yaml = "0.9"
serde_json = "1.0"

# Hashing and compression
blake3 = "1.5"
lz4 = "1.24"

# Concurrency
rayon = "1.10"
crossbeam = "0.8"

# MCP integration
pmcp = "1.4.2"

# Error handling
anyhow = "1.0"
thiserror = "1.0"

# Logging
tracing = "0.1"
tracing-subscriber = "0.3"

[dev-dependencies]
cargo-nextest = "0.9"
proptest = "1.4"
criterion = "0.5"
tempfile = "3.10"
mockall = "0.12"
```

## 5. Academic Research Foundation

The following peer-reviewed publications inform the design and implementation:

### 5.1 Defect Pattern Mining

1. **CPatMiner: Automatic Mining of Code Change Patterns from Open Source Repositories**
   - Liu et al., MSR 2016
   - Graph-based semantic code change pattern mining
   - https://github.com/xgdsmileboy/CPatMiner
   - **Application**: AST diff analysis for defect pattern extraction

2. **Graph-based Mining of In-the-Wild, Fine-grained, Semantic Code Change Patterns**
   - Nguyen et al., ICSE 2019
   - Fine-grained semantic change pattern detection
   - https://2019.icse-conferences.org/details/icse-2019-Technical-Papers/39/
   - **Application**: Change graph construction for defect classification

3. **Comparative analysis of real issues in open-source machine learning projects**
   - Empirical Software Engineering, 2024
   - ML vs non-ML defect taxonomy
   - https://link.springer.com/article/10.1007/s10664-024-10467-3
   - **Application**: Defect category taxonomy design

### 5.2 Graph Neural Networks for Code Analysis

4. **DeMuVGN: Effective Software Defect Prediction Model by Learning Multi-view Software Dependency via Graph Neural Networks**
   - Qin et al., arXiv:2410.19550, October 2024
   - Multi-view software dependency graphs
   - https://arxiv.org/abs/2410.19550
   - **Application**: GNN-based defect pattern classification

5. **Graph Neural Network for Source Code Defect Prediction**
   - IEEE Software Engineering, 2022
   - AST-based GCNN for defect detection
   - https://ieeexplore.ieee.org/document/9684879/
   - **Application**: AST feature extraction for ML classifier

6. **Code Revert Prediction with Graph Neural Networks: A Case Study at J.P. Morgan Chase**
   - arXiv:2403.09507, March 2024
   - Industrial application of GNN for defect prediction
   - https://arxiv.org/abs/2403.09507
   - **Application**: Real-world validation of GNN approach

### 5.3 AST-Based Analysis

7. **Abstract Syntax Tree for Programming Language Understanding and Representation: How Far Are We?**
   - Sun et al., arXiv:2312.00413, December 2023
   - Comprehensive AST representation evaluation
   - https://arxiv.org/abs/2312.00413
   - **Application**: AST encoding strategy selection

8. **An AST-based Code Change Representation and its Performance in Just-in-time Vulnerability Prediction**
   - arXiv:2303.16591, March 2023
   - Code Change Tree for diff representation
   - https://arxiv.org/abs/2303.16591
   - **Application**: Defect-inducing change detection

### 5.4 Rust Safety and Concurrency

9. **Understanding and Detecting Real-World Safety Issues in Rust**
   - IEEE Transactions on Software Engineering, 2024
   - Empirical study of Rust memory safety violations
   - https://dl.acm.org/doi/abs/10.1109/TSE.2024.3380393
   - **Application**: Memory safety defect category design

10. **An Empirical Study of Rust-for-Linux: The Success, Dissatisfaction, and Compromise**
    - USENIX ATC 2024
    - Real-world Rust concurrency patterns and bugs
    - https://www.usenix.org/system/files/atc24-li-hongyu.pdf
    - **Application**: Concurrency defect pattern taxonomy

### 5.5 Supporting Research

**Mutation Testing:**
- "Mutation Analysis: Answering the Fuzzing Challenge" (2022) - Oracle quality evaluation
- "Systematic Assessment of Fuzzers using Mutation Analysis" (USENIX Security 2023)

**Property-Based Testing:**
- "Coverage Guided, Property Based Testing" (FuzzChick) - Combining PBT with fuzzing
- "Guiding Greybox Fuzzing with Mutation Testing" (ISSTA 2023)

**Static Analysis:**
- "Combining Large Language Models with Static Analyzers" (arXiv 2025)
- Analysis Tools curated list: https://github.com/analysis-tools-dev/static-analysis

## 6. Output Format Specification (Requirement E)

### 6.1 YAML Schema

```yaml
# YAML Schema Version 1.0
version: "1.0"
schema: "https://example.com/oip-schema-v1.json"

metadata:
  organization: string
  analysis_date: ISO8601 datetime
  analysis_duration_seconds: number
  repositories_analyzed: number
  commits_analyzed: number
  total_defects_analyzed: number
  analyzer_version: string

time_range:
  start: ISO8601 datetime
  end: ISO8601 datetime

defect_patterns:
  - category: string  # One of 10 defined categories
    subcategory: string  # Optional fine-grained classification
    frequency: number
    percentage: number
    trend: string  # "increasing", "decreasing", "stable"

    severity_distribution:
      critical: number
      high: number
      medium: number
      low: number

    temporal_distribution:
      - month: string  # "2025-01"
        count: number

    spatial_distribution:
      - file_pattern: string  # "src/**/*.rs"
        count: number
        percentage: number

    examples:
      - repository: string
        commit_sha: string
        commit_message: string
        author: string
        date: ISO8601 datetime
        file: string
        line_range: [number, number]
        description: string
        fix_commit_sha: string  # Optional
        time_to_fix_hours: number  # Optional
        code_snippet:
          before: string
          after: string

    common_locations:
      - pattern: string
        count: number

    contributing_factors:
      - factor: string
        correlation: number  # 0.0 to 1.0

    prevention_strategies:
      - strategy: string
        effectiveness: string  # "high", "medium", "low"
        references: [string]

    references:
      - title: string
        url: string
        citation: string

statistics:
  mean_time_to_fix_hours: number
  median_time_to_fix_hours: number
  defect_density_per_kloc: number
  most_affected_languages:
    - language: string
      defect_count: number
  most_affected_repositories:
    - repository: string
      defect_count: number

quality_metrics:
  test_coverage_correlation: number
  code_review_effectiveness: number
  ci_cd_impact: number

recommendations:
  - priority: string  # "high", "medium", "low"
    category: string
    recommendation: string
    expected_impact: string
    implementation_effort: string
```

### 6.2 Example Output

```yaml
version: "1.0"
schema: "https://paiml.com/oip-schema-v1.json"

metadata:
  organization: "rust-lang"
  analysis_date: "2025-11-15T00:00:00Z"
  analysis_duration_seconds: 3847
  repositories_analyzed: 45
  commits_analyzed: 125394
  total_defects_analyzed: 2847
  analyzer_version: "0.1.0"

time_range:
  start: "2020-01-01T00:00:00Z"
  end: "2025-11-15T00:00:00Z"

defect_patterns:
  - category: "Concurrency Bugs"
    subcategory: "Data Race"
    frequency: 687
    percentage: 24.1
    trend: "decreasing"

    severity_distribution:
      critical: 34
      high: 156
      medium: 398
      low: 99

    temporal_distribution:
      - month: "2024-01"
        count: 12
      - month: "2024-02"
        count: 8
      - month: "2024-03"
        count: 5

    spatial_distribution:
      - file_pattern: "src/concurrent/**/*.rs"
        count: 234
        percentage: 34.1
      - file_pattern: "src/async/**/*.rs"
        count: 187
        percentage: 27.2

    examples:
      - repository: "rust-lang/rust"
        commit_sha: "abc123def456"
        commit_message: "Fix data race in parallel iterator"
        author: "contributor@example.com"
        date: "2024-03-15T14:23:00Z"
        file: "src/concurrent/parallel_iter.rs"
        line_range: [145, 167]
        description: "Unsynchronized access to shared counter in parallel fold operation"
        fix_commit_sha: "def456abc789"
        time_to_fix_hours: 4.5
        code_snippet:
          before: |
            let counter = 0;
            items.par_iter().for_each(|item| {
                counter += process(item);  // Data race!
            });
          after: |
            let counter = AtomicUsize::new(0);
            items.par_iter().for_each(|item| {
                counter.fetch_add(process(item), Ordering::Relaxed);
            });

    common_locations:
      - pattern: "parallel iterators with shared state"
        count: 234
      - pattern: "async task spawning without synchronization"
        count: 187

    contributing_factors:
      - factor: "complex concurrent logic"
        correlation: 0.78
      - factor: "insufficient test coverage of concurrent paths"
        correlation: 0.65

    prevention_strategies:
      - strategy: "Use Arc&lt;Mutex&lt;T&gt;&gt; or atomic types for shared mutable state"
        effectiveness: "high"
        references:
          - "Understanding and Detecting Real-World Safety Issues in Rust (IEEE TSE 2024)"
      - strategy: "Prefer message passing (channels) over shared memory"
        effectiveness: "high"
        references:
          - "An Empirical Study of Rust-for-Linux (USENIX ATC 2024)"
      - strategy: "Use Miri and ThreadSanitizer for concurrency testing"
        effectiveness: "medium"
        references: []

    references:
      - title: "Understanding and Detecting Real-World Safety Issues in Rust"
        url: "https://dl.acm.org/doi/abs/10.1109/TSE.2024.3380393"
        citation: "IEEE Transactions on Software Engineering, 2024"

  - category: "Memory Safety Violations"
    frequency: 456
    percentage: 16.0
    # ... similar structure

statistics:
  mean_time_to_fix_hours: 12.4
  median_time_to_fix_hours: 6.2
  defect_density_per_kloc: 0.23
  most_affected_languages:
    - language: "Rust"
      defect_count: 1823
    - language: "C++"
      defect_count: 687
  most_affected_repositories:
    - repository: "rust-lang/rust"
      defect_count: 1245

quality_metrics:
  test_coverage_correlation: -0.67  # Higher coverage = fewer defects
  code_review_effectiveness: 0.82
  ci_cd_impact: 0.75

recommendations:
  - priority: "high"
    category: "Concurrency Bugs"
    recommendation: "Implement organization-wide Miri testing in CI/CD pipelines"
    expected_impact: "Reduce concurrency bugs by 40-60%"
    implementation_effort: "medium"

  - priority: "high"
    category: "Memory Safety Violations"
    recommendation: "Enforce stricter unsafe code review policies"
    expected_impact: "Reduce memory safety violations by 30-50%"
    implementation_effort: "low"
```

## 7. Integration with PMAT (Requirement A)

### 7.1 Roadmap Validation

**pmat.toml:**
```toml
[project]
name = "organizational-intelligence-plugin"
version = "0.1.0"
roadmap_file = "ROADMAP.md"

[roadmap]
enforce_ticket_tracking = true
require_test_per_feature = true
block_commits_without_ticket = true

[quality_gates]
enabled = true
config_file = "pmat-quality.toml"
```

### 7.2 Quality Gates

**pmat-quality.toml:**
```toml
[general]
project_name = "organizational-intelligence-plugin"
enforcement_level = "strict"

[complexity]
cyclomatic_threshold = 15
cognitive_threshold = 20
max_nesting_depth = 5
max_function_lines = 100

[coverage]
minimum_coverage = 85.0
enforce_on_new_code = true
fail_on_decrease = true
exclude_patterns = ["*/tests/*", "*/benches/*", "*/examples/*"]

[satd]
enabled = true
zero_tolerance = true
patterns = ["TODO", "FIXME", "HACK", "XXX", "BUG", "DEPRECATED"]
allow_with_ticket = true

[dependencies]
check_outdated = true
check_vulnerabilities = true
max_dependency_depth = 5

[documentation]
require_public_api_docs = true
require_examples = true
check_broken_links = true

[performance]
max_compilation_time_seconds = 300
max_test_time_seconds = 300
max_binary_size_mb = 50
```

### 7.3 Pre-commit Hooks

```bash
#!/bin/bash
# .git/hooks/pre-commit (installed via scripts/install-git-hooks.sh)

set -e

echo "🔍 Running pre-commit quality gates..."

# Run pmat validation
pmat validate --config pmat-quality.toml

# Lint all shell scripts
find . -name "*.sh" -exec shellcheck {} \;

# Format check
cargo fmt --check

# Clippy
cargo clippy --all-targets --all-features -- -D warnings

# Fast tests
make test-fast

echo "✅ Pre-commit checks passed"
```

## 8. Shell and Makefile Standards (Requirement B)

### 8.1 Shell Script Standards

All shell scripts must:
1. Pass `shellcheck` with zero warnings
2. Use `set -euo pipefail` for error handling
3. Include function documentation
4. Be unit tested with `bats` (Bash Automated Testing System)

**Example: scripts/analyze-org.sh**
```bash
#!/usr/bin/env bash
set -euo pipefail

# Description: Wrapper script for OIP organization analysis
# Usage: ./scripts/analyze-org.sh --org rust-lang --output report.yaml

main() {
    local org=""
    local output="defects.yaml"

    while [[ $# -gt 0 ]]; do
        case $1 in
            --org)
                org="$2"
                shift 2
                ;;
            --output)
                output="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1" >&2
                exit 1
                ;;
        esac
    done

    if [[ -z "$org" ]]; then
        echo "Error: --org is required" >&2
        exit 1
    fi

    cargo run --release -- analyze --org "$org" --output "$output"
}

main "$@"
```

**Unit Test: tests/scripts/test_analyze_org.bats**
```bash
#!/usr/bin/env bats

load test_helper

@test "analyze-org.sh fails without --org argument" {
    run ./scripts/analyze-org.sh --output test.yaml
    [ "$status" -eq 1 ]
    [[ "$output" == *"--org is required"* ]]
}

@test "analyze-org.sh constructs correct cargo command" {
    # Mock cargo to print command instead of running
    export PATH="$BATS_TEST_DIRNAME/mocks:$PATH"

    run ./scripts/analyze-org.sh --org test-org --output result.yaml
    [ "$status" -eq 0 ]
    [[ "$output" == *"analyze --org test-org --output result.yaml"* ]]
}
```

### 8.2 Makefile Testing

**Makefile:**
```makefile
.PHONY: all build test clean validate

CARGO := cargo
CARGO_FLAGS := --release
OUTPUT_DIR := target/release

all: validate build

build:
	$(CARGO) build $(CARGO_FLAGS)

test-fast:
	$(CARGO) nextest run --no-fail-fast --test-threads=8

test-all: test-fast
	$(CARGO) test --all-features --workspace

coverage:
	$(CARGO) llvm-cov --workspace --html --fail-under-lines=85

lint:
	$(CARGO) fmt --check
	$(CARGO) clippy --all-targets --all-features -- -D warnings
	shellcheck scripts/*.sh

validate: lint test-fast coverage
	@echo "✅ All validation gates passed"

clean:
	$(CARGO) clean
	rm -rf target/

# Unit test this Makefile with bats
test-makefile:
	bats tests/makefile/test_makefile.bats
```

**Makefile Unit Test: tests/makefile/test_makefile.bats**
```bash
#!/usr/bin/env bats

@test "make build invokes cargo build with correct flags" {
    make -n build | grep -q "cargo build --release"
}

@test "make validate runs lint, test-fast, and coverage" {
    output=$(make -n validate)
    echo "$output" | grep -q "cargo fmt --check"
    echo "$output" | grep -q "cargo clippy"
    echo "$output" | grep -q "cargo nextest run"
    echo "$output" | grep -q "cargo llvm-cov"
}

@test "make clean removes target directory" {
    make -n clean | grep -q "rm -rf target/"
}
```

## 9. Performance Requirements

### 9.1 Throughput Targets

- **Repository clone**: 1000 repos/hour (with concurrency=50)
- **Commit analysis**: 10,000 commits/minute (parallel processing)
- **AST parsing**: 1000 files/second (tree-sitter)
- **Pattern classification**: 500 defects/second (ML inference)
- **Report generation**: <5 seconds for 10,000 defects

### 9.2 Latency Targets

- **L1 cache hit**: <200ns (DashMap)
- **L2 cache hit**: <500μs (LibSQL)
- **L3 cache hit**: <100ms (cold storage)
- **CLI startup**: <100ms (lazy initialization)
- **MCP tool response**: <2 seconds (95th percentile)

### 9.3 Resource Constraints

- **Memory**: <4GB for organization with 100 repos
- **Disk**: <10GB for 1M commits (with compression)
- **CPU**: Linear scaling up to physical cores
- **Network**: Respect GitHub API rate limits (5000 req/hour)

## 10. Deployment and Operations

### 10.1 Installation

```bash
# From crates.io
cargo install organizational-intelligence-plugin

# From source
git clone https://github.com/paiml/organizational-intelligence-plugin
cd organizational-intelligence-plugin
make build
cargo install --path .

# Verify installation
oip --version
```

### 10.2 Configuration

**$HOME/.config/oip/config.toml:**
```toml
[github]
token_file = "~/.config/oip/github-token"
api_url = "https://api.github.com"
max_concurrent_requests = 50
rate_limit_buffer = 100

[storage]
backend = "libsql"  # or "sled", "rocksdb", "inmemory"
path = "~/.local/share/oip/oip.db"
cache_size_mb = 1024
compression = true

[analysis]
max_concurrent_repos = 10
max_commits_per_repo = 10000
languages = ["rust", "python", "javascript", "go", "java"]

[classifier]
model_path = "~/.local/share/oip/models/classifier-v1.onnx"
confidence_threshold = 0.7

[output]
default_format = "yaml"
include_examples = true
max_examples_per_category = 5
```

### 10.3 Monitoring

**Metrics exposed via prometheus:**
```rust
use prometheus::{Counter, Histogram, Registry};

lazy_static! {
    static ref REPOS_ANALYZED: Counter = Counter::new(
        "oip_repos_analyzed_total",
        "Total number of repositories analyzed"
    ).unwrap();

    static ref DEFECTS_CLASSIFIED: Counter = Counter::new(
        "oip_defects_classified_total",
        "Total number of defects classified"
    ).unwrap();

    static ref ANALYSIS_DURATION: Histogram = Histogram::new(
        "oip_analysis_duration_seconds",
        "Time to analyze a repository"
    ).unwrap();
}
```

## 11. Testing Strategy

### 11.1 Test Pyramid

```
                  /\
                 /  \
                / E2E \          <-- 5%: Full system tests
               /--------\
              /          \
             / Integration \     <-- 15%: Service integration
            /--------------\
           /                \
          /   Unit Tests     \   <-- 80%: Unit + property tests
         /____________________\
```

### 11.2 Test Types

**Unit Tests (80%)**
```rust
// tests/unit/classifier_test.rs
#[test]
fn test_classify_memory_safety_violation() {
    let classifier = DefectClassifier::new();
    let commit = create_test_commit_use_after_free();

    let result = classifier.classify(&commit).unwrap();

    assert_eq!(result.category, DefectCategory::MemorySafety);
    assert!(result.confidence > 0.8);
}
```

**Property Tests (included in unit %)**
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_classifier_always_returns_valid_category(
        commit in arb_commit_data()
    ) {
        let classifier = DefectClassifier::new();
        let result = classifier.classify(&commit);

        if let Ok(classification) = result {
            prop_assert!(classification.category.is_valid());
            prop_assert!(classification.confidence >= 0.0);
            prop_assert!(classification.confidence <= 1.0);
        }
    }
}
```

**Integration Tests (15%)**
```rust
// tests/integration/github_miner_test.rs
#[tokio::test]
async fn test_mine_organization_end_to_end() {
    let miner = GitHubMiner::new(test_config());
    let storage = InMemoryStorage::new();

    let result = miner.mine_organization("test-org", &storage).await.unwrap();

    assert!(result.repos_analyzed > 0);
    assert!(result.commits_analyzed > 0);
    assert!(result.defects_found > 0);
}
```

**E2E Tests (5%)**
```rust
// tests/e2e/cli_test.rs
#[test]
fn test_full_analysis_workflow() {
    let output = Command::new("cargo")
        .args(&["run", "--", "analyze", "--org", "test-org", "--output", "test.yaml"])
        .output()
        .unwrap();

    assert!(output.status.success());
    assert!(Path::new("test.yaml").exists());

    let report: Report = serde_yaml::from_str(
        &fs::read_to_string("test.yaml").unwrap()
    ).unwrap();

    assert!(report.defect_patterns.len() > 0);
}
```

### 11.3 Mutation Testing

```bash
# PMAT mutation testing (AST-based)
pmat mutation-test \
    --critical-modules=classifier,analyzer,miner \
    --budget-minutes=5 \
    --min-survival-rate=0.95

# cargo-mutants (industry standard)
cargo mutants \
    --timeout=60 \
    --jobs=4 \
    --exclude-tests \
    -F skip-slow
```

### 11.4 Fuzzing

```rust
// fuzz/fuzz_targets/classifier_fuzz.rs
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(commit) = parse_fuzz_input(data) {
        let classifier = DefectClassifier::new();
        let _ = classifier.classify(&commit);
    }
});
```

```bash
# Run fuzzing for 5 minutes
cargo +nightly fuzz run classifier_fuzz -- -max_total_time=300
```

## 12. Roadmap (Kaizen-Driven with Validation Gates)

**Philosophy:** Each phase ends with a validation gate. Proceed only if success criteria are met. Adapt based on learnings.

### Phase 1: Foundation + Data Collection (Weeks 1-4)

**Focus:** Deliver minimum viable value + build data foundation for future ML

**Deliverables:**
- [ ] Core CLI structure with clap
- [ ] GitHub API integration (octocrab) with rate limiting
- [ ] Basic git history analysis (git2)
- [ ] **Rule-based defect classifier** with confidence scores
- [ ] **User feedback mechanism** (confirm/correct classifications)
- [ ] YAML output generation
- [ ] **L1 + L2 storage (DashMap + Parquet)** - simple two-tier
- [ ] **Comprehensive metrics collection** (cache hits, query patterns, classifier accuracy)
- [ ] Initial test suite (focus on critical paths)
- [ ] Pre-commit hooks (<30s) + CI pipeline
- [ ] Quality gates integration (pmat) as guidelines

**Validation Gate 1 (End of Week 4):**
```yaml
success_criteria:
  functional:
    - Can analyze 10+ repo organization in <1 hour
    - Rule-based classifier achieves 60%+ accuracy on 100 manually labeled commits
    - YAML output validates against schema
    - Feedback loop operational (users can correct classifications)

  data_collection:
    - Collected 500+ labeled defect examples from user feedback
    - Identified top 3 most common defect categories

  technical:
    - Pre-commit hooks run in <30 seconds
    - CI pipeline runs in <10 minutes
    - Storage metrics show cache hit rate >70%

  quality:
    - Critical path test coverage >85%
    - Zero critical bugs in dog-fooding
    - Makefile targets unit tested with bats

go_no_go_decision:
  proceed_if: "All functional + data_collection criteria met"
  adapt_if: "Technical criteria missed - investigate but don't block"
  pause_if: "Classifier accuracy <50% - need better heuristics"
```

**Learnings to Capture:**
- Which defect categories are hardest to classify?
- What query patterns are most common? (informs Phase 3 optimization)
- Is Parquet storage sufficient or do we need SQL layer?

---

### Phase 2: Interpretable ML (Weeks 5-8)

**Trigger:** ≥1000 labeled examples collected from Phase 1 feedback

**Focus:** Replace rule-based classifier with explainable ML

**Deliverables:**
- [ ] AST parsing (tree-sitter integration)
- [ ] Feature engineering from AST diffs (complexity delta, node types changed)
- [ ] **Gradient Boosting classifier** (XGBoost/LightGBM) NOT GNN yet
- [ ] **SHAP explainability** for classifications
- [ ] 10-category taxonomy refinement based on data
- [ ] Frequency analysis and trend visualization
- [ ] A/B test framework (ML vs. rules)
- [ ] Property-based testing for classifier
- [ ] Enhanced CI: mutation testing for classifier module

**Validation Gate 2 (End of Week 8):**
```yaml
success_criteria:
  model_performance:
    - ML classifier achieves 75%+ accuracy (15% improvement over rules)
    - SHAP values provide actionable insights (validated with 5 users)
    - A/B test shows ML preferred by users 2:1

  explainability:
    - Every classification includes top 3 contributing features
    - Users report SHAP explanations help them learn (survey)

  data:
    - Now have 2500+ labeled examples
    - Taxonomy refined based on observed patterns

  technical:
    - Inference latency <100ms p95
    - Model size <50MB
    - CI mutation testing passes (>95% kill rate)

go_no_go_decision:
  proceed_to_phase_3: "Model performance + explainability criteria met"
  iterate_phase_2: "Accuracy 70-75% - try more features"
  revert_to_rules: "Accuracy <70% - need more data or different approach"

  evaluate_gnn:
    trigger: "Only if accuracy plateaus AND we have 5000+ examples"
    benchmark: "Compare GNN vs. current model on holdout set"
    threshold: "Proceed only if GNN improves >10% AND explainability acceptable"
```

---

### Phase 3: Performance & Scale (Weeks 9-12)

**Trigger:** Phase 2 validation passed + need to scale beyond single machine

**Focus:** Optimize based on observed bottlenecks (Genchi Genbutsu)

**Pre-Phase Profiling (Week 9 Day 1):**
- Run 1-week profiling campaign on 100 repo organization
- Measure: storage latency, cache hit rates, CPU bottlenecks, memory usage
- **Make optimization decisions based on data, not assumptions**

**Conditional Deliverables (based on profiling):**
- [ ] **If** cache hit rate <80%: Improve caching strategy or add SQL tier
- [ ] **If** storage p95 >100ms: Add index layer or optimize Parquet partitioning
- [ ] **If** dataset >100GB: Implement S3 cold storage
- [ ] **If** CPU-bound: Distributed processing optimization with rayon/tokio tuning
- [ ] **If** memory >4GB per 100 repos: Memory optimization, reduce cache size
- [ ] Incremental analysis support (only analyze new commits)
- [ ] Performance benchmarking suite
- [ ] Load testing (validate 1000 repos goal)
- [ ] Fuzzing for critical paths (weekly CI job)

**Validation Gate 3 (End of Week 12):**
```yaml
success_criteria:
  performance:
    - Can analyze 1000 repo organization in <10 hours (100 repos/hour)
    - Memory <4GB for 100 repos
    - Storage p95 latency <100ms
    - Cache hit rate >80%

  scalability:
    - Linear scaling up to physical cores validated
    - Incremental analysis reduces re-work by >80%

  data_informed:
    - Optimization decisions documented with before/after metrics
    - Only added complexity where profiling justified it

go_no_go_decision:
  proceed: "Performance targets met"
  iterate: "Targets missed - profile deeper, apply Five Whys"
```

---

### Phase 4: MCP Integration (Weeks 13-16)

**Focus:** Make OIP accessible to LLMs via MCP

**Deliverables:**
- [ ] MCP server implementation (pmcp SDK)
- [ ] Tool registration: analyze_org, classify_commit, query_patterns, generate_report
- [ ] Stdio transport support
- [ ] Integration tests with Claude
- [ ] Documentation: MCP tool usage examples
- [ ] pmat-book integration chapter
- [ ] End-to-end validation with real Claude workflows

**Validation Gate 4 (End of Week 16):**
```yaml
success_criteria:
  integration:
    - Claude can successfully analyze an org and interpret results
    - MCP tools respond in <2s p95
    - Error messages are actionable for LLM context

  usability:
    - 5 pmat workflows successfully use OIP via MCP
    - Documentation complete and validated

go_no_go_decision:
  proceed: "Integration validated with real workflows"
  iterate: "Latency or usability issues - refine"
```

---

### Phase 5: Production Hardening (Weeks 17-20)

**Focus:** Production-ready reliability and operations

**Deliverables:**
- [ ] Error recovery and retry strategies
- [ ] Observability: Prometheus metrics, structured logging, tracing
- [ ] CI/CD: Automated releases, rollback capability
- [ ] Security audit (dependency scan, SAST, manual review)
- [ ] Performance regression tests (guard against slowdowns)
- [ ] User documentation and tutorials
- [ ] Public beta release

**Validation Gate 5 (End of Week 20):**
```yaml
success_criteria:
  reliability:
    - 99% success rate on 1000 repo analysis
    - Mean time to recovery <10 minutes for failures
    - Zero data loss in failure scenarios

  observability:
    - All critical paths instrumented with metrics
    - Incidents debuggable from logs alone

  security:
    - Zero high/critical vulnerabilities
    - Dependency audit passing

  adoption:
    - 10+ beta users providing feedback
    - 80%+ satisfaction score

go_no_go_decision:
  public_release: "All criteria met"
  extended_beta: "Reliability or security issues - fix first"
```

---

### Continuous Validation Framework

**Weekly Retrospectives:**
- Review outcome metrics: defect escape rate, MTTR, user satisfaction
- Discuss: "What surprised us this week? What assumptions were wrong?"
- Adjust roadmap based on learnings

**Monthly Architecture Reviews:**
- "Are we adding complexity without evidence?"
- "What can we remove or simplify?"
- Apply Five Whys to any performance or quality issues

## 13. Success Metrics

### 13.1 Quality Metrics
- **Test coverage**: ≥85% line coverage
- **Mutation score**: ≥95% for critical modules
- **Clippy warnings**: 0 (strict mode)
- **SATD (TODO/FIXME)**: 0 in production code
- **Documentation coverage**: 100% of public API

### 13.2 Performance Metrics
- **Analysis throughput**: ≥1000 repos/hour
- **Classification accuracy**: ≥85% F1-score
- **Memory efficiency**: <4GB for 100 repos
- **CLI responsiveness**: <100ms startup
- **MCP latency**: <2s at p95

### 13.3 Adoption Metrics
- **GitHub stars**: >500 in first 6 months
- **Downloads**: >1000 in first quarter
- **Active users**: >50 organizations
- **Integration**: Used in ≥3 pmat workflows
- **Contributions**: >10 external contributors

## 14. Risk Mitigation

### 14.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| GitHub API rate limits | High | High | Implement aggressive caching, support GitHub Apps auth (higher limits) |
| ML classifier accuracy too low | Medium | High | Start with rule-based fallback, collect labeled data for supervised training |
| Performance doesn't scale | Medium | Medium | Profile early, optimize hot paths, use rayon for parallelism |
| Storage bloat | Medium | Medium | Implement L3 archival, compression, configurable retention |
| AST parsing failures | High | Low | Graceful degradation, support multiple parsers, fallback to regex |

### 14.2 Process Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Quality gates too strict, slow dev | Low | Medium | Iteratively tune thresholds, fast feedback loop with cargo-nextest |
| Mutation testing too slow | Medium | Low | Budget-based testing (5 min max), parallelize with cargo-mutants |
| Test maintenance burden | Low | Medium | Property tests reduce test count, focus on critical paths |
| Documentation drift | Medium | Low | Pre-commit hooks for doc validation, mdBook in CI |

## 15. Future Enhancements

### 15.1 Advanced Features
- **Real-time monitoring**: GitHub webhook integration for continuous analysis
- **Custom taxonomies**: User-defined defect categories via config
- **Recommendation engine**: Automated PR suggestions for defect prevention
- **Team insights**: Developer/team-level defect pattern analysis
- **IDE integration**: VSCode extension for inline defect warnings
- **Slack/Discord bots**: Automated defect pattern reports

### 15.2 Multi-Language Support
- Expand beyond Rust to Python, Go, JavaScript, Java, C++
- Language-specific defect taxonomies
- Cross-language pattern detection

### 15.3 ML Improvements
- Transfer learning from pre-trained code models (CodeBERT, GraphCodeBERT)
- Active learning for continuous classifier improvement
- Explainable AI for defect classification reasoning

## 16. References

### 16.1 Key Publications

1. Liu et al., "CPatMiner: Automatic Mining of Code Change Patterns", MSR 2016
2. Nguyen et al., "Graph-based Mining of In-the-Wild, Fine-grained, Semantic Code Change Patterns", ICSE 2019
3. "Comparative analysis of real issues in open-source machine learning projects", Empirical Software Engineering, 2024
4. Qin et al., "DeMuVGN: Effective Software Defect Prediction Model", arXiv:2410.19550, 2024
5. "Graph Neural Network for Source Code Defect Prediction", IEEE TSE, 2022
6. "Code Revert Prediction with Graph Neural Networks: J.P. Morgan Case Study", arXiv:2403.09507, 2024
7. Sun et al., "Abstract Syntax Tree for Programming Language Understanding", arXiv:2312.00413, 2023
8. "An AST-based Code Change Representation for Vulnerability Prediction", arXiv:2303.16591, 2023
9. "Understanding and Detecting Real-World Safety Issues in Rust", IEEE TSE, 2024
10. "An Empirical Study of Rust-for-Linux", USENIX ATC, 2024

### 16.2 Additional Resources

- MSR Conference Proceedings (2019-2024)
- ICSE Technical Papers (2019-2024)
- USENIX ATC, Security (2023-2024)
- Analysis Tools Directory: https://github.com/analysis-tools-dev/static-analysis
- Fuzzing Papers Collection: https://github.com/wcventure/FuzzingPaper
- PMAT Documentation: https://paiml.com/pmat-book

## 17. Appendix

### 17.1 Glossary

- **AST**: Abstract Syntax Tree - hierarchical representation of source code structure
- **DashMap**: Concurrent HashMap implementation for Rust
- **GNN**: Graph Neural Network - deep learning on graph-structured data
- **LibSQL**: Modern SQLite-compatible database with enhanced features
- **MCP**: Model Context Protocol - standard for LLM-tool integration
- **PMAT**: PAIML MCP Agent Toolkit - quality enforcement framework
- **SATD**: Self-Admitted Technical Debt (TODO, FIXME comments)
- **TDG**: Test Data Generator - component of PMAT with tiered storage
- **TDD**: Test-Driven Development - write tests before implementation

### 17.2 CLI Examples

```bash
# Analyze a single organization
oip analyze --org rust-lang --output rust-defects.yaml

# Analyze with time constraints
oip analyze --org nodejs --since 2024-01-01 --until 2024-12-31

# Analyze specific repos only
oip analyze --repos "rust-lang/rust,rust-lang/cargo" --output report.yaml

# Generate report from cached data
oip report --org rust-lang --format yaml --output report.yaml

# Export to Parquet for data science
oip export --org rust-lang --format parquet --output defects.parquet

# Start MCP server for Claude integration
oip serve --transport stdio

# Classify a single commit
oip classify --repo rust-lang/rust --commit abc123 --output classification.yaml

# Clear cache
oip cache clear --org rust-lang

# Show statistics
oip stats --org rust-lang
```

### 17.3 MCP Tool Examples

```json
// Request: Analyze organization
{
  "tool": "oip_analyze_organization",
  "params": {
    "organization": "rust-lang",
    "max_repos": 50,
    "time_range_days": 365
  }
}

// Response
{
  "organization": "rust-lang",
  "repos_analyzed": 45,
  "commits_analyzed": 125394,
  "defects_found": 2847,
  "top_categories": [
    {"category": "Concurrency Bugs", "count": 687},
    {"category": "Memory Safety", "count": 456}
  ],
  "report_path": "/tmp/rust-lang-defects.yaml"
}
```

---

## Appendix A: Response to Toyota Way Review

This appendix documents how the specification addresses the critical Toyota Way review.

### A.1 Storage Architecture (3-Tier → 2-Tier)

**Critique:** The original 3-tier cache (DashMap → LibSQL → S3/Parquet) introduced unnecessary complexity before validating actual performance needs.

**Resolution:**
- **Simplified to 2-tier** (Section 2.3): DashMap (L1) → Parquet (L2)
- **Conditional L3/SQL tier** (Sections 2.3.3, 2.3.4): Added only if profiling reveals specific bottlenecks
- **Decision framework** (Section 2.3.4): Five Whys analysis before adding SQL cache
- **Metrics-driven validation** (Section 2.3.1): CacheMetrics and StorageMetrics embedded from day one
- **Genchi Genbutsu applied**: Phase 3 starts with 1-week profiling to inform optimization decisions

**Outcome:** Reduced initial complexity by 33%, faster time-to-value, evolutionary architecture based on observed needs.

### A.2 ML Classifier (GNN → Gradient Boosting → Conditional GNN)

**Critique:** Jumping to GNNs without labeled data was premature. Explainability critical for "Respect for People."

**Resolution:**
- **Phase 1** (Section 2.2.3): Rule-based classifier + **data collection infrastructure** + user feedback loop
- **Phase 2**: Gradient Boosting with SHAP explainability (NOT GNN yet)
  - Success criteria: 75%+ accuracy, actionable SHAP values
  - A/B test against rules to validate improvement
- **Phase 3** (conditional): Evaluate GNN only if:
  - Phase 2 plateaus AND 5000+ labeled examples
  - GNN improves >10% AND explainability acceptable
- **Classification struct** (Section 2.2.3): Every prediction includes explanation and confidence
- **User feedback** (Section 2.2.3): Users confirm/correct classifications → builds training data

**Outcome:** Data-first approach, interpretable models prioritized, GNN evaluated based on evidence not hype.

### A.3 Development Workflow (Rigid TDD → Empowered Guidelines)

**Critique:** Mandating extreme TDD with 85% coverage in pre-commit hooks creates friction and stifles adaptation.

**Resolution:**
- **Reframed as guidelines** (Section 3.1): "Recommended TDD Workflow" not mandatory
- **Fast pre-commit** (Section 3.2): <30 seconds, moved expensive ops to CI
- **Coverage as signal** (Section 3.2): 85% goal, not hard gate. Team discusses drops >10%
- **Mutation testing in CI** (Section 3.2): Budget-limited (5 min), not blocking commits
- **Outcome metrics prioritized** (Section 3.3):
  - Defect escape rate (<5%)
  - MTTR (<48 hours critical)
  - Test effectiveness (>80%)
- **Context-aware** (Section 3.1): Team can defer tests for prototyping, add before merge
- **Respect for People** (Section 3.2): Trust developers to make appropriate decisions

**Outcome:** Reduced developer friction, outcome-focused quality, team empowerment.

### A.4 Validation Gates (Kaizen in Practice)

**Added:** Section 12 - Roadmap with comprehensive validation gates

Each phase now includes:
- **Success criteria**: Functional, data, technical, quality metrics
- **Go/No-Go decisions**: Proceed/Iterate/Pause based on evidence
- **Learnings to capture**: Questions to answer before next phase
- **Conditional deliverables**: Add complexity only if profiling justifies

**Example (Phase 3):**
- Pre-phase profiling week to measure actual bottlenecks
- Conditional features: "If cache hit rate <80% → add SQL tier"
- Document before/after metrics for all optimizations

**Outcome:** Evidence-based evolution, no premature optimization, continuous learning.

### A.5 Key Philosophical Shifts

| Original Spec | Toyota Way Revision | Rationale |
|---------------|---------------------|-----------|
| 3-tier storage from day one | 2-tier, evolve based on metrics | Avoid premature optimization (Genchi Genbutsu) |
| GNN classifier in Phase 2 | Rule-based → Gradient Boosting → conditional GNN | Data-first, explainability matters (Respect for People) |
| 85% coverage mandate in pre-commit | 85% goal, outcome metrics prioritized | Reduce friction, focus on results (Kaizen) |
| Fixed roadmap | Validation gates with go/no-go decisions | Adapt based on learnings (Kaizen) |
| Process compliance | Outcome metrics (defect escape rate, MTTR) | Measure what matters (Toyota Way) |

### A.6 Continuous Improvement Mechanisms

**Weekly Retrospectives** (Section 12):
- "What surprised us? What assumptions were wrong?"
- Adjust roadmap based on data

**Monthly Architecture Reviews** (Section 12):
- "Are we adding complexity without evidence?"
- "What can we remove or simplify?"
- Apply Five Whys to issues

**Metrics Dashboard** (Section 3.3):
- Primary: Defect escape rate, MTTR, test effectiveness
- Supporting: Coverage trend, mutation score, CI duration
- Alert on drops >10% week-over-week

**Outcome:** Living specification that evolves with team learnings.

---

**Document Control**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-11-15 | OIP Team | Initial specification |
| 1.1.0 | 2025-11-15 | OIP Team | Toyota Way review response: simplified storage, data-first ML, empowered workflow, validation gates |

**Approval Status**: Revised Draft - Incorporating Toyota Way Principles

**Reviewers**: Toyota Way Design Review Team

**Next Review Date**: 2025-11-22
