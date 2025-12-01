# Tarantula-Style Fault Localization - Technical Specification

**Version:** 1.3.0
**Status:** Phases 1-7 Complete (Ensemble + Calibration Implemented)
**Last Updated:** 2025-12-01
**Authors:** Organizational Intelligence Team

---

## Executive Summary

This specification defines the integration of Tarantula-style spectrum-based fault localization (SBFL) into the Organizational Intelligence Plugin. By combining git history analysis with test coverage correlation, we enable precise identification of defect-prone code regions, moving beyond simple commit classification to statistical fault localization.

**Design Philosophy: Toyota Way Principles**

- **Genchi Genbutsu (Go and See)**: Start with the simplest SBFL formula (Tarantula), measure effectiveness, then evolve to Ochiai/DStar only when data justifies complexity.

- **Kaizen (Continuous Improvement)**: Each approach represents an evolution gate. Proceed to more sophisticated techniques only after validating improvement on real defects.

- **Respect for People**: Every suspiciousness score includes human-readable explanations. Developers learn *why* code is suspicious, not just *that* it is.

- **Jidoka (Automation with Human Intelligence)**: Automated ranking with confidence intervals. Human judgment determines action thresholds.

- **Heijunka (Level the Workload)**: Batch processing of coverage data to avoid runtime overhead during test execution.

- **Muda (Eliminate Waste)**: We avoid the waste of developer time chasing false positives by requiring high confidence thresholds before alerting, and by not running expensive Mutation analysis until simpler methods fail.

- **Muri (Avoid Overburden)**: We prevent system and cognitive overburden by prioritizing lightweight techniques first and presenting only the most relevant findings to the developer.

---

## 1. Background: What is Tarantula?

### 1.1 The Problem

When a test fails, developers face the challenge of identifying which lines of code caused the failure. In large codebases, manual inspection is prohibitively expensive. Fault localization techniques automate this process by ranking code elements by their likelihood of containing the fault.

### 1.2 Tarantula's Insight

Tarantula, introduced by Jones et al. [1], observes that **faulty statements tend to be executed more frequently by failing tests and less frequently by passing tests**. This insight forms the basis of Spectrum-Based Fault Localization (SBFL).

### 1.3 The Tarantula Formula

For each program statement `s`, the suspiciousness score is calculated as:

```
                    failed(s) / totalFailed
suspiciousness(s) = ─────────────────────────────────────────────
                    (passed(s) / totalPassed) + (failed(s) / totalFailed)
```

Where:
- `failed(s)` = number of failing tests that execute statement `s`
- `passed(s)` = number of passing tests that execute statement `s`
- `totalFailed` = total number of failing tests
- `totalPassed` = total number of passing tests

**Intuition**: A statement executed by all failing tests and no passing tests receives maximum suspiciousness (1.0). A statement executed by all passing tests and no failing tests receives minimum suspiciousness (0.0).

---

## 2. Five Approaches to Fault Localization

This specification evaluates five complementary approaches, ordered by implementation complexity and data requirements. Following Toyota Way principles, we implement incrementally, validating each phase before proceeding.

### 2.1 Approach 1: Classic Tarantula (SBFL Foundation)

**Phase**: MVP
**Data Required**: Test coverage + pass/fail results
**Complexity**: Low

**Description**: Pure implementation of the original Tarantula formula using statement-level coverage data from `cargo-llvm-cov`.

**Implementation**:
```rust
pub struct TarantulaScore {
    pub statement: StatementId,
    pub suspiciousness: f32,  // 0.0 to 1.0
    pub failed_coverage: usize,
    pub passed_coverage: usize,
    pub confidence: f32,  // Based on test sample size
}

pub fn tarantula(
    failed: usize,
    passed: usize,
    total_failed: usize,
    total_passed: usize,
) -> f32 {
    let failed_ratio = if total_failed > 0 {
        failed as f32 / total_failed as f32
    } else {
        0.0
    };
    let passed_ratio = if total_passed > 0 {
        passed as f32 / total_passed as f32
    } else {
        0.0
    };

    let denominator = passed_ratio + failed_ratio;
    if denominator == 0.0 {
        0.0
    } else {
        failed_ratio / denominator
    }
}
```

**Success Criteria**: Localize 40%+ of faults within Top-10 ranked statements on Defects4J-equivalent Rust test suite.

**Toyota Way Alignment**: Start simple. Tarantula is well-understood and provides baseline for comparison.

---

### 2.2 Approach 2: Ochiai Formula (Enhanced SBFL)

**Phase**: Validated Evolution
**Data Required**: Same as Tarantula
**Complexity**: Low
**Trigger**: After Tarantula baseline established AND research indicates Ochiai superiority [2]

**Description**: Ochiai, borrowed from molecular biology, has been empirically shown to outperform Tarantula in many scenarios [3]. It uses a different normalization approach.

**Formula**:
```
                         failed(s)
suspiciousness(s) = ─────────────────────────────────
                    √(totalFailed × (failed(s) + passed(s)))
```

**Implementation**:
```rust
pub fn ochiai(
    failed: usize,
    passed: usize,
    total_failed: usize,
) -> f32 {
    let denominator = ((total_failed * (failed + passed)) as f32).sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        failed as f32 / denominator
    }
}
```

**Why Ochiai?**: Studies on Defects4J show Ochiai consistently ranks in the top tier of SBFL formulas [4], often localizing 10-15% more faults in Top-5 than Tarantula.

**Success Criteria**: Demonstrate >10% improvement over Tarantula on same test suite.

---

### 2.3 Approach 3: DStar (Parameterized SBFL)

**Phase**: Advanced SBFL
**Data Required**: Same as Tarantula
**Complexity**: Medium (hyperparameter tuning required)
**Trigger**: After Ochiai validated AND edge cases identified where both underperform

**Description**: DStar [5] introduces a configurable exponent `*` that can be tuned for different fault types.

**Formula**:
```
                           failed(s)^*
suspiciousness(s) = ────────────────────────────────────
                    passed(s) + (totalFailed - failed(s))
```

Where `*` (star) is typically set to 2 or 3.

**Implementation**:
```rust
pub fn dstar(
    failed: usize,
    passed: usize,
    total_failed: usize,
    star: u32,  // Typically 2 or 3
) -> f32 {
    let numerator = (failed as f32).powi(star as i32);
    let denominator = passed as f32 + (total_failed - failed) as f32;

    if denominator == 0.0 {
        if numerator > 0.0 { f32::INFINITY } else { 0.0 }
    } else {
        numerator / denominator
    }
}
```

**Advantage**: The exponent amplifies the signal from failing tests, making DStar more aggressive at ranking statements executed predominantly by failing tests.

**Success Criteria**: Identify optimal `*` value for Rust codebases via grid search on labeled defects.

---

### 2.4 Approach 4: Mutation-Based Fault Localization (MBFL)

**Phase**: Hybrid Evolution
**Data Required**: Test coverage + mutation analysis results
**Complexity**: High (requires mutation testing infrastructure)
**Trigger**: SBFL accuracy plateaus AND mutation testing already integrated (via `cargo-mutants`)

**Description**: Metallaxis-FL [6] uses mutation analysis to correlate mutant behavior with fault locations. The insight: **if a mutant at location L is killed primarily by failing tests, L is likely faulty**.

**How It Works**:
1. Generate mutants at each statement
2. Run test suite against each mutant
3. Track which tests kill each mutant
4. Score statements by the ratio of failing-test-killed mutants

**Formula**:
```
                    Σ (killScore(m) for m in mutants(s))
suspiciousness(s) = ────────────────────────────────────
                              |mutants(s)|

where killScore(m) = failedKills(m) / (failedKills(m) + passedKills(m))
```

**Implementation Sketch**:
```rust
pub struct MutantInfo {
    pub location: StatementId,
    pub killed_by_failed: usize,
    pub killed_by_passed: usize,
}

pub fn metallaxis(mutants: &[MutantInfo]) -> HashMap<StatementId, f32> {
    let mut scores: HashMap<StatementId, Vec<f32>> = HashMap::new();

    for mutant in mutants {
        let total_kills = mutant.killed_by_failed + mutant.killed_by_passed;
        let score = if total_kills > 0 {
            mutant.killed_by_failed as f32 / total_kills as f32
        } else {
            0.0
        };
        scores.entry(mutant.location).or_default().push(score);
    }

    scores.into_iter()
        .map(|(loc, scores)| {
            let avg = scores.iter().sum::<f32>() / scores.len() as f32;
            (loc, avg)
        })
        .collect()
}
```

**Advantage**: MBFL can detect faults that SBFL misses, particularly in code with high coincidental correctness [7].

**Disadvantage**: Expensive—requires running tests against many mutants.

**Success Criteria**: Localize 15%+ additional faults not found by SBFL techniques.

---

### 2.5 Approach 5: Learning-Based Fault Localization (DeepFL Integration)

**Phase**: Future Evolution
**Data Required**: SBFL scores + MBFL scores + code metrics + historical defect data
**Complexity**: Very High (requires ML training pipeline)
**Trigger**: 5000+ labeled fault examples AND MBFL integrated

**Description**: DeepFL [8] uses deep learning to combine multiple fault diagnosis dimensions into a unified ranking. Rather than choosing one formula, it learns optimal feature combinations.

**Features Integrated**:
1. **Spectrum features**: Tarantula, Ochiai, DStar scores
2. **Mutation features**: Metallaxis scores, mutant kill rates
3. **Complexity features**: Cyclomatic complexity, nesting depth, LOC
4. **Textual features**: Code-to-bug-report similarity (via embeddings)
5. **Historical features**: Churn rate, defect density, author experience

**Architecture**:
```
┌──────────────────────────────────────────────────────────────┐
│                    Feature Extraction                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐│
│  │ SBFL     │ │ MBFL     │ │ Code     │ │ Git History      ││
│  │ Scores   │ │ Scores   │ │ Metrics  │ │ (churn, authors) ││
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────────┬─────────┘│
└───────┼────────────┼────────────┼────────────────┼───────────┘
        └────────────┴────────────┴────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   MLP / BiRNN     │
                    │   (TensorFlow)    │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Ranking Output   │
                    │  (Suspiciousness) │
                    └───────────────────┘
```

**Why DeepFL?**: Experimental results show DeepFL localizes 50+ more faults within Top-1 compared to traditional techniques on Defects4J [8].

**Success Criteria**: Cross-project prediction accuracy within 10% of same-project accuracy.

---

### 2.6 Approach 6: RAG-Enhanced Fault Localization (trueno-rag Integration) ✅

**Phase**: 5 (Complete)
**Data Required**: SBFL scores + Historical bug database + Code embeddings
**Complexity**: Medium-High (requires trueno-rag pipeline setup)
**Trigger**: Phase 2 complete AND bug knowledge base available
**Dependency**: `trueno-rag` crate from paiml ecosystem

**Description**: Integrate Retrieval-Augmented Generation (RAG) capabilities from `trueno-rag` to enhance fault localization with semantic search over historical bugs, similar code patterns, and contextual explanations.

**Key Insight**: While SBFL provides mathematical suspiciousness scores based on coverage statistics, RAG enables **semantic understanding** of why code is buggy and how similar bugs were fixed in the past.

#### 2.6.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                RAG-Enhanced Fault Localization Pipeline              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────────────┐  │
│  │   Tarantula  │   │     SZZ      │   │     trueno-rag         │  │
│  │    (SBFL)    │   │ (Historical) │   │  (Semantic Search)     │  │
│  │              │   │              │   │                        │  │
│  │ Coverage     │   │ Git Blame    │   │ ┌────────────────────┐ │  │
│  │ Statistics   │   │ Analysis     │   │ │ Bug Report Index   │ │  │
│  └──────┬───────┘   └──────┬───────┘   │ │ (BM25 + Vector)    │ │  │
│         │                  │           │ └────────────────────┘ │  │
│         │ Suspicion        │ History   │ ┌────────────────────┐ │  │
│         │ Score            │ Score     │ │ Code Pattern Index │ │  │
│         │                  │           │ │ (Semantic Chunks)  │ │  │
│         │                  │           │ └────────────────────┘ │  │
│         │                  │           │ ┌────────────────────┐ │  │
│         │                  │           │ │ Fix History Index  │ │  │
│         │                  │           │ │ (Commit Messages)  │ │  │
│         │                  │           │ └─────────┬──────────┘ │  │
│         │                  │           └───────────┼────────────┘  │
│         │                  │                       │               │
│         │                  │                       │ Similarity    │
│         │                  │                       │ Score         │
│         └────────┬─────────┴───────────────────────┘               │
│                  │                                                  │
│                  ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              trueno-rag FusionStrategy                       │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────┐   │   │
│  │  │   RRF   │ │  DBSF   │ │ Linear  │ │ Intersection    │   │   │
│  │  │ k=60    │ │ Z-norm  │ │ α=0.7   │ │ (High Conf)     │   │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────────────┘   │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                             │                                       │
│                             ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Enhanced Output                           │   │
│  │  • Final Suspiciousness Ranking                             │   │
│  │  • Similar Historical Bugs (with links)                     │   │
│  │  • Suggested Fix Patterns                                   │   │
│  │  • Contextual Explanation                                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

#### 2.6.2 trueno-rag Integration Points

**Index Types**:
```rust
use trueno_rag::{
    Document, RagPipelineBuilder, RecursiveChunker,
    FusionStrategy, BM25Index, VectorStore,
};

/// Bug report document for RAG indexing
pub struct BugDocument {
    pub id: String,           // Bug ID or commit hash
    pub title: String,        // Bug summary
    pub description: String,  // Full bug description
    pub fix_commit: String,   // Commit that fixed it
    pub fix_diff: String,     // The actual code change
    pub affected_files: Vec<String>,
    pub category: DefectCategory,
}

/// Code chunk for semantic similarity
pub struct CodeChunk {
    pub file: PathBuf,
    pub start_line: usize,
    pub end_line: usize,
    pub content: String,
    pub embedding: Vec<f32>,
}
```

**Retrieval Pipeline**:
```rust
/// RAG-enhanced fault localizer
pub struct RagFaultLocalizer {
    sbfl: SbflLocalizer,
    rag_pipeline: RagPipeline,
    fusion: FusionStrategy,
}

impl RagFaultLocalizer {
    /// Localize fault and retrieve similar bugs
    pub fn localize_with_context(
        &self,
        coverage: &[StatementCoverage],
        total_passed: usize,
        total_failed: usize,
    ) -> RagEnhancedResult {
        // Step 1: SBFL ranking
        let sbfl_result = self.sbfl.localize(coverage, total_passed, total_failed);

        // Step 2: For top-N suspicious statements, query RAG
        let mut enhanced_rankings = Vec::new();
        for ranking in sbfl_result.rankings.iter().take(10) {
            // Read code context around suspicious line
            let code_context = self.read_context(&ranking.statement, 5);

            // Query bug database
            let similar_bugs = self.rag_pipeline
                .query(&code_context, 5)
                .unwrap_or_default();

            // Query fix patterns
            let fix_patterns = self.rag_pipeline
                .query(&format!("fix for: {}", code_context), 3)
                .unwrap_or_default();

            enhanced_rankings.push(RagEnhancedRanking {
                sbfl_ranking: ranking.clone(),
                similar_bugs,
                fix_patterns,
                explanation: self.generate_explanation(ranking, &similar_bugs),
            });
        }

        RagEnhancedResult {
            rankings: enhanced_rankings,
            sbfl_result,
        }
    }
}
```

#### 2.6.3 Fusion Strategy Selection

| Strategy | Use Case | When to Apply |
|----------|----------|---------------|
| **RRF (k=60)** | Default multi-signal fusion | Combining SBFL + History + RAG similarity |
| **DBSF** | Normalized score fusion | When signals have different scales |
| **Linear (α=0.7)** | Weighted combination | When SBFL is trusted more than RAG |
| **Intersection** | High-confidence results | When only agreeing signals are acceptable |

**Recommended Configuration**:
```rust
// For fault localization, use RRF with conservative weighting
let fusion = FusionStrategy::RRF { k: 60.0 };

// Alternative: weighted toward SBFL
let fusion = FusionStrategy::Linear { dense_weight: 0.7 }; // 70% SBFL, 30% RAG
```

#### 2.6.4 Bug Knowledge Base Schema

```yaml
# bugs/memory-safety/null-pointer-001.yaml
id: "null-pointer-001"
category: "memory-safety"
subcategory: "null-dereference"
severity: "high"

description: |
  Null pointer dereference in parser when handling empty input.
  The parser assumes non-null input but doesn't validate.

symptoms:
  - "SIGSEGV in parser.rs"
  - "Test test_empty_input fails"

root_cause: |
  Missing null check before dereferencing optional value.
  Pattern: `let value = optional.unwrap();` without prior check.

fix_pattern: |
  Use pattern matching or Option::map:
  ```rust
  // Before (buggy)
  let value = optional.unwrap();

  // After (fixed)
  let value = optional.ok_or(Error::NullInput)?;
  ```

related_commits:
  - introducing: "abc123"
  - fixing: "def456"

test_case: |
  #[test]
  fn test_handles_null_input() {
      assert!(parser.parse(None).is_err());
  }
```

#### 2.6.5 Expected Output Enhancement

**Current (SBFL Only)**:
```yaml
rankings:
  - rank: 1
    statement:
      file: src/parser.rs
      line: 87
    suspiciousness: 0.92
    explanation: "Executed by 95% of failing tests..."
```

**Enhanced (SBFL + RAG)**:
```yaml
rankings:
  - rank: 1
    statement:
      file: src/parser.rs
      line: 87
    suspiciousness: 0.92
    explanation: "Executed by 95% of failing tests..."

    # NEW: RAG-enhanced fields
    similar_bugs:
      - id: "null-pointer-001"
        similarity: 0.89
        category: "memory-safety"
        summary: "Null pointer dereference in parser"
        fix_commit: "def456"

      - id: "buffer-overflow-003"
        similarity: 0.72
        category: "memory-safety"
        summary: "Buffer overflow in unsafe block"
        fix_commit: "ghi789"

    suggested_fixes:
      - pattern: "Add null check before unwrap"
        confidence: 0.85
        example: |
          let value = optional.ok_or(Error::NullInput)?;

      - pattern: "Use Option::map for safe transformation"
        confidence: 0.71
        example: |
          let value = optional.map(|v| process(v));

    context_explanation: |
      This pattern matches historical bug "null-pointer-001" which was
      fixed in commit def456 by adding input validation. The code
      structure (unwrap in parser context) has caused 3 similar bugs
      in this repository.
```

#### 2.6.6 Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Bug Retrieval Precision | 70% | % of retrieved bugs relevant to fault |
| Fix Suggestion Accuracy | 50% | % of suggested fixes applicable |
| Developer Time Saved | 30% | Time to fix with vs without RAG |
| Explanation Helpfulness | 4/5 | Developer survey rating |

#### 2.6.7 Implementation Phases

**Phase A: Bug Knowledge Base Setup**
- [ ] Define bug document schema
- [ ] Import historical bugs from git history (SZZ-labeled)
- [ ] Index bug reports with trueno-rag BM25 + vector store
- [ ] Validate retrieval precision on known bugs

**Phase B: Code Pattern Indexing**
- [ ] Chunk code using `RecursiveChunker` or `SemanticChunker`
- [ ] Generate embeddings for code chunks
- [ ] Build similarity index for pattern matching
- [ ] Validate similar code retrieval

**Phase C: Fusion Integration**
- [ ] Implement `RagFaultLocalizer` combining SBFL + RAG
- [ ] Evaluate fusion strategies (RRF, Linear, DBSF)
- [ ] Tune fusion parameters on labeled dataset
- [ ] Compare against baseline SBFL

**Phase D: Fix Suggestion Generation**
- [ ] Index fix patterns from bug-fixing commits
- [ ] Implement fix pattern retrieval
- [ ] Generate contextual explanations
- [ ] Evaluate suggestion quality

#### 2.6.8 Dependencies

```toml
[dependencies]
trueno-rag = "0.1"  # RAG pipeline
trueno = "0.7"      # Tensor operations for embeddings
trueno-db = "0.3"   # Vector storage
```

#### 2.6.9 Toyota Way Alignment

| Principle | Application |
|-----------|-------------|
| **Genchi Genbutsu** | Retrieve actual historical bugs, not hypothetical patterns |
| **Kaizen** | Bug knowledge base improves continuously from each fix |
| **Jidoka** | Human-readable explanations with context |
| **Muda** | Only query RAG for top-N suspicious statements (avoid waste) |
| **Muri** | Configurable retrieval limits prevent information overload |

---

### 2.7 Approach 7: Weighted Ensemble Risk Score (aprender Integration) ✅

**Phase**: 6 (Complete)
**Data Required**: SBFL + TDG + Churn + Complexity + RAG Similarity
**Complexity**: Medium (requires aprender weak supervision)
**Trigger**: Phase 5 complete AND sufficient labeled defect history
**Dependency**: `aprender` crate (weak supervision, active learning)

**Description**: Train an aprender model to learn optimal weights for combining multiple defect signals, replacing hand-tuned fusion with data-driven combination.

#### 2.7.1 Problem Statement

Current approaches use fixed fusion strategies (RRF, Linear) with manually tuned weights. Different codebases have different defect characteristics:

- Some codebases: High churn correlates strongly with defects
- Others: Low TDG is the primary indicator
- Some: SBFL alone is sufficient

**Goal**: Automatically learn the optimal combination for each codebase.

#### 2.7.2 Signal Sources

| Signal | Source | Range | Description |
|--------|--------|-------|-------------|
| `sbfl_score` | Tarantula/Ochiai | 0.0-1.0 | Coverage-based suspiciousness |
| `tdg_score` | pmat | 0.0-1.0 | Technical Debt Grade (inverted) |
| `churn_score` | Git history | 0.0-1.0 | Normalized commit frequency |
| `complexity_score` | AST analysis | 0.0-1.0 | Normalized cyclomatic complexity |
| `rag_similarity` | trueno-rag | 0.0-1.0 | Similarity to historical bugs |

#### 2.7.3 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Weighted Ensemble Risk Model (aprender)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐│
│  │   SBFL     │ │    TDG     │ │   Churn    │ │ Complexity │ │    RAG     ││
│  │  Score     │ │   Score    │ │   Score    │ │   Score    │ │ Similarity ││
│  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘│
│        │              │              │              │              │        │
│        ▼              ▼              ▼              ▼              ▼        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                 aprender::weak_supervision::LabelModel               │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │  Labeling Functions (Noisy Signal Aggregation)              │   │   │
│  │  │                                                              │   │   │
│  │  │  LF₁: sbfl > 0.7         → likely_defect (weight: learned)  │   │   │
│  │  │  LF₂: tdg < 0.5          → likely_defect (weight: learned)  │   │   │
│  │  │  LF₃: churn_percentile > 0.9 → likely_defect (weight: learned) │   │
│  │  │  LF₄: complexity > 15    → likely_defect (weight: learned)  │   │   │
│  │  │  LF₅: rag_similarity > 0.8 → likely_defect (weight: learned)│   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                                │                                    │   │
│  │                                ▼                                    │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │  Learned Weights: w = [0.35, 0.15, 0.20, 0.12, 0.18]       │   │   │
│  │  │  (Varies per codebase based on historical defect patterns)  │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                          │
│                                 ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Combined Risk Score                             │   │
│  │  Risk = Σ wᵢ × signalᵢ = w₁·SBFL + w₂·TDG + w₃·Churn + ...        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 2.7.4 Implementation

```rust
use aprender::weak_supervision::{LabelModel, LabelingFunction, LFOutput};

/// Labeling function for SBFL signal
pub struct SbflLabelingFunction {
    threshold: f32,
}

impl LabelingFunction for SbflLabelingFunction {
    fn apply(&self, features: &FileFeatures) -> LFOutput {
        if features.sbfl_score > self.threshold {
            LFOutput::Positive  // Likely defect
        } else if features.sbfl_score < 0.2 {
            LFOutput::Negative  // Likely clean
        } else {
            LFOutput::Abstain   // Uncertain
        }
    }
}

/// Combined ensemble model
pub struct WeightedEnsembleModel {
    label_model: LabelModel,
    labeling_functions: Vec<Box<dyn LabelingFunction>>,
}

impl WeightedEnsembleModel {
    pub fn new() -> Self {
        let lfs: Vec<Box<dyn LabelingFunction>> = vec![
            Box::new(SbflLabelingFunction { threshold: 0.7 }),
            Box::new(TdgLabelingFunction { max_grade: 0.5 }),
            Box::new(ChurnLabelingFunction { percentile: 0.9 }),
            Box::new(ComplexityLabelingFunction { max_complexity: 15.0 }),
            Box::new(RagSimilarityLabelingFunction { threshold: 0.8 }),
        ];

        Self {
            label_model: LabelModel::new(lfs.len()),
            labeling_functions: lfs,
        }
    }

    /// Fit model on unlabeled data to learn LF weights
    pub fn fit(&mut self, files: &[FileFeatures]) -> Result<()> {
        // Generate LF outputs for each file
        let lf_outputs: Vec<Vec<LFOutput>> = files.iter()
            .map(|f| self.labeling_functions.iter()
                .map(|lf| lf.apply(f))
                .collect())
            .collect();

        // aprender learns optimal weights via EM algorithm
        self.label_model.fit(&lf_outputs)?;

        Ok(())
    }

    /// Predict defect probability for a file
    pub fn predict(&self, features: &FileFeatures) -> f32 {
        let lf_outputs: Vec<LFOutput> = self.labeling_functions.iter()
            .map(|lf| lf.apply(features))
            .collect();

        self.label_model.predict_proba(&lf_outputs)
    }

    /// Get learned weights for interpretability
    pub fn get_weights(&self) -> Vec<(String, f32)> {
        vec![
            ("SBFL".into(), self.label_model.weights[0]),
            ("TDG".into(), self.label_model.weights[1]),
            ("Churn".into(), self.label_model.weights[2]),
            ("Complexity".into(), self.label_model.weights[3]),
            ("RAG Similarity".into(), self.label_model.weights[4]),
        ]
    }
}
```

#### 2.7.5 CLI Integration

```bash
oip localize \
  --passed-coverage passed.lcov \
  --failed-coverage failed.lcov \
  --ensemble \
  --ensemble-model model.apr \
  --include-tdg \
  --include-churn \
  --repo .
```

New flags:
- `--ensemble`: Enable weighted ensemble model
- `--ensemble-model <PATH>`: Path to trained aprender model
- `--include-tdg`: Include TDG scores from pmat
- `--include-churn`: Include churn metrics from git history
- `--train-ensemble <OUTPUT>`: Train new ensemble model from labeled data

#### 2.7.6 Training Workflow

```bash
# Step 1: Extract features from repository
oip extract-ensemble-features \
  --repo . \
  --output features.json

# Step 2: Train ensemble model (unsupervised weak supervision)
oip train-ensemble \
  --input features.json \
  --output ensemble-model.apr

# Step 3: Use trained model for localization
oip localize \
  --ensemble \
  --ensemble-model ensemble-model.apr \
  --passed-coverage passed.lcov \
  --failed-coverage failed.lcov
```

#### 2.7.7 Success Criteria

| Metric | Baseline (SBFL only) | Target (Ensemble) |
|--------|---------------------|-------------------|
| Top-1 Accuracy | 35% | 50%+ |
| Top-5 Accuracy | 60% | 75%+ |
| MAP (Mean Average Precision) | 0.45 | 0.60+ |
| False Positive Rate | 40% | 25% |

#### 2.7.8 Toyota Way Alignment

| Principle | Application |
|-----------|-------------|
| **Jidoka** | Learned weights are interpretable - developers see why |
| **Kaizen** | Model improves as more defect history accumulates |
| **Genchi Genbutsu** | Weights derived from actual codebase patterns |
| **Heijunka** | Batch training amortizes cost across many predictions |

---

### 2.8 Approach 8: Calibrated Defect Probability with Confidence Intervals ✅

**Phase**: 7 (Complete)
**Data Required**: Ensemble scores + Labeled validation set
**Complexity**: Medium (requires aprender calibration + Bayesian inference)
**Trigger**: Phase 6 complete AND validation dataset available
**Dependency**: `aprender` crate (calibration, Bayesian regression)

**Description**: Transform raw defect scores into calibrated probabilities with uncertainty quantification, enabling principled decision-making under uncertainty.

#### 2.8.1 Problem Statement

Current fault localization produces **scores**, not **probabilities**:

- "Suspiciousness: 0.73" - What does this mean?
- Is 0.73 high? Depends on the distribution!
- How confident are we in this score?

**Goal**: Provide calibrated probabilities with confidence intervals:
```
P(defect | file) = 0.73 ± 0.12 (95% CI)
```

A developer knows: "73% chance of defect, and we're fairly confident (narrow CI)."

#### 2.8.2 Calibration Theory

**Calibration** ensures that predicted probabilities match observed frequencies:
- If model predicts 70% defect probability for 100 files
- Approximately 70 of them should actually be defective

**Techniques**:
1. **Platt Scaling**: Sigmoid transformation of scores
2. **Isotonic Regression**: Non-parametric monotonic transformation
3. **Temperature Scaling**: Single parameter scaling for neural networks

#### 2.8.3 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              Calibrated Defect Prediction Pipeline (aprender)                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Raw Score Sources                                 │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐               │   │
│  │  │ Ensemble │ │   SBFL   │ │   TDG    │ │   RAG    │               │   │
│  │  │  Score   │ │  Score   │ │  Score   │ │ Similarity│              │   │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘               │   │
│  │       └───────────┬┴───────────┬┴───────────┬┘                      │   │
│  └───────────────────┼────────────┼────────────┼───────────────────────┘   │
│                      │            │            │                            │
│                      ▼            ▼            ▼                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │            aprender::bayesian::BayesianLogisticRegression            │   │
│  │                                                                      │   │
│  │   P(y=1|x) = σ(w·x + b)  where w ~ N(μ_w, Σ_w)                      │   │
│  │                                                                      │   │
│  │   Output: (mean prediction, variance)                                │   │
│  └──────────────────────────────┬───────────────────────────────────────┘   │
│                                 │                                           │
│                                 ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │               aprender::calibration::IsotonicRegression              │   │
│  │                                                                      │   │
│  │   Transforms raw probability to calibrated probability               │   │
│  │   Fitted on held-out validation set                                 │   │
│  └──────────────────────────────┬───────────────────────────────────────┘   │
│                                 │                                           │
│                                 ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Calibrated Output                                │   │
│  │                                                                      │   │
│  │   ┌─────────────────────────────────────────────────────────┐      │   │
│  │   │  File: src/parser.rs:50                                  │      │   │
│  │   │  P(defect) = 0.73                                        │      │   │
│  │   │  95% CI: [0.61, 0.85]                                    │      │   │
│  │   │  Confidence: HIGH (narrow interval)                      │      │   │
│  │   │                                                          │      │   │
│  │   │  Contributing Factors:                                   │      │   │
│  │   │    • SBFL score: 0.89 (contribution: 35%)               │      │   │
│  │   │    • TDG below C: (contribution: 22%)                   │      │   │
│  │   │    • Similar to BUG-042 (contribution: 18%)             │      │   │
│  │   │    • High churn (contribution: 15%)                     │      │   │
│  │   │    • Complexity > 15 (contribution: 10%)                │      │   │
│  │   └─────────────────────────────────────────────────────────┘      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 2.8.4 Implementation

```rust
use aprender::calibration::{PlattScaling, IsotonicRegression, CalibrationMethod};
use aprender::bayesian::regression::BayesianLogisticRegression;

/// Prediction with uncertainty quantification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibratedPrediction {
    /// Calibrated probability of defect
    pub probability: f32,
    /// 95% confidence interval
    pub confidence_interval: (f32, f32),
    /// Confidence level based on CI width
    pub confidence_level: ConfidenceLevel,
    /// Factor contributions (for explainability)
    pub contributing_factors: Vec<FactorContribution>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfidenceLevel {
    High,      // CI width < 0.15
    Medium,    // CI width 0.15-0.30
    Low,       // CI width > 0.30
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorContribution {
    pub factor_name: String,
    pub contribution_pct: f32,
    pub raw_value: f32,
}

/// Calibrated defect predictor using Bayesian inference
pub struct CalibratedDefectPredictor {
    /// Bayesian model provides uncertainty estimates
    bayesian_model: BayesianLogisticRegression,
    /// Post-hoc calibration
    calibrator: IsotonicRegression,
    /// Feature names for explainability
    feature_names: Vec<String>,
}

impl CalibratedDefectPredictor {
    pub fn new(n_features: usize) -> Self {
        Self {
            bayesian_model: BayesianLogisticRegression::new(n_features)
                .with_prior_precision(1.0)  // Regularization strength
                .with_n_samples(100),        // MCMC samples for uncertainty
            calibrator: IsotonicRegression::new(),
            feature_names: vec![
                "sbfl_score".into(),
                "tdg_score".into(),
                "churn_score".into(),
                "complexity_score".into(),
                "rag_similarity".into(),
            ],
        }
    }

    /// Train on labeled data
    pub fn fit(&mut self, features: &[Vec<f32>], labels: &[bool]) -> Result<()> {
        // Split into train and calibration sets
        let (train_x, train_y, cal_x, cal_y) = self.split_data(features, labels, 0.2);

        // Fit Bayesian model on training set
        self.bayesian_model.fit(&train_x, &train_y)?;

        // Get raw probabilities on calibration set
        let raw_probs: Vec<f32> = cal_x.iter()
            .map(|x| self.bayesian_model.predict_proba(x).0)
            .collect();

        // Fit calibrator on calibration set
        self.calibrator.fit(&raw_probs, &cal_y)?;

        Ok(())
    }

    /// Predict with uncertainty quantification
    pub fn predict(&self, features: &[f32]) -> CalibratedPrediction {
        // Get mean and variance from Bayesian model
        let (mean, variance) = self.bayesian_model.predict_with_uncertainty(features);

        // Calibrate the mean prediction
        let calibrated_prob = self.calibrator.transform(mean);

        // Compute confidence interval
        let std_dev = variance.sqrt();
        let z_95 = 1.96;
        let ci_low = (calibrated_prob - z_95 * std_dev).max(0.0);
        let ci_high = (calibrated_prob + z_95 * std_dev).min(1.0);

        // Determine confidence level based on CI width
        let ci_width = ci_high - ci_low;
        let confidence_level = if ci_width < 0.15 {
            ConfidenceLevel::High
        } else if ci_width < 0.30 {
            ConfidenceLevel::Medium
        } else {
            ConfidenceLevel::Low
        };

        // Compute factor contributions using coefficient magnitudes
        let coefficients = self.bayesian_model.get_mean_coefficients();
        let contributions = self.compute_contributions(features, &coefficients);

        CalibratedPrediction {
            probability: calibrated_prob,
            confidence_interval: (ci_low, ci_high),
            confidence_level,
            contributing_factors: contributions,
        }
    }

    fn compute_contributions(&self, features: &[f32], coefficients: &[f32]) -> Vec<FactorContribution> {
        let weighted: Vec<f32> = features.iter()
            .zip(coefficients.iter())
            .map(|(f, c)| (f * c).abs())
            .collect();

        let total: f32 = weighted.iter().sum();

        self.feature_names.iter()
            .zip(features.iter())
            .zip(weighted.iter())
            .map(|((name, &raw_value), &w)| FactorContribution {
                factor_name: name.clone(),
                contribution_pct: if total > 0.0 { w / total * 100.0 } else { 0.0 },
                raw_value,
            })
            .collect()
    }
}
```

#### 2.8.5 CLI Integration

```bash
oip localize \
  --passed-coverage passed.lcov \
  --failed-coverage failed.lcov \
  --calibrated \
  --calibration-model calibration.apr \
  --confidence-threshold 0.7 \
  --format terminal
```

New flags:
- `--calibrated`: Enable calibrated probability output
- `--calibration-model <PATH>`: Path to trained calibration model
- `--confidence-threshold <FLOAT>`: Only report files above this probability
- `--min-confidence <LEVEL>`: Filter by confidence level (high/medium/low)

#### 2.8.6 Output Format

**Terminal Output**:
```
╔══════════════════════════════════════════════════════════════════════════════╗
║              CALIBRATED DEFECT PREDICTION REPORT                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Calibration Method: Isotonic Regression                                      ║
║  Model: Bayesian Logistic Regression (n_samples=100)                          ║
║  Reliability: 0.94 (Brier Score: 0.12)                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  #1  src/parser.rs:50                                                         ║
║      P(defect) = 0.73 ± 0.12  [0.61, 0.85]   Confidence: HIGH               ║
║      ├─ SBFL score: 0.89             (35%)                                   ║
║      ├─ TDG below threshold          (22%)                                   ║
║      ├─ Similar to BUG-042           (18%)                                   ║
║      ├─ High churn (95th percentile) (15%)                                   ║
║      └─ Complexity: 18               (10%)                                   ║
║                                                                               ║
║  #2  src/handler.rs:100                                                       ║
║      P(defect) = 0.45 ± 0.22  [0.23, 0.67]   Confidence: MEDIUM             ║
║      ├─ TDG below threshold          (40%)                                   ║
║      ├─ High churn (88th percentile) (30%)                                   ║
║      ├─ SBFL score: 0.55             (20%)                                   ║
║      └─ Complexity: 12               (10%)                                   ║
║                                                                               ║
║  #3  src/util.rs:25                                                           ║
║      P(defect) = 0.31 ± 0.35  [0.00, 0.66]   Confidence: LOW                ║
║      ⚠️  Wide confidence interval - need more data                           ║
║                                                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

**YAML Output**:
```yaml
predictions:
  - file: src/parser.rs
    line: 50
    probability: 0.73
    confidence_interval: [0.61, 0.85]
    confidence_level: High
    contributing_factors:
      - factor: SBFL
        contribution_pct: 35
        raw_value: 0.89
      - factor: TDG
        contribution_pct: 22
        raw_value: 0.45
      - factor: RAG_Similarity
        contribution_pct: 18
        raw_value: 0.82
calibration_metrics:
  method: IsotonicRegression
  brier_score: 0.12
  reliability_diagram_area: 0.94
```

#### 2.8.7 Calibration Validation

```rust
/// Reliability diagram metrics
pub struct CalibrationMetrics {
    /// Expected Calibration Error
    pub ece: f32,
    /// Maximum Calibration Error
    pub mce: f32,
    /// Brier Score (lower is better)
    pub brier_score: f32,
    /// Area under reliability diagram
    pub reliability_area: f32,
}

impl CalibratedDefectPredictor {
    /// Evaluate calibration quality on held-out test set
    pub fn evaluate_calibration(&self, test_x: &[Vec<f32>], test_y: &[bool]) -> CalibrationMetrics {
        let predictions: Vec<f32> = test_x.iter()
            .map(|x| self.predict(x).probability)
            .collect();

        // Bin predictions and compute reliability
        let n_bins = 10;
        let mut bins: Vec<(f32, f32, usize)> = vec![(0.0, 0.0, 0); n_bins];

        for (pred, &actual) in predictions.iter().zip(test_y.iter()) {
            let bin_idx = ((pred * n_bins as f32) as usize).min(n_bins - 1);
            bins[bin_idx].0 += pred;           // sum of predictions
            bins[bin_idx].1 += actual as f32;  // sum of actuals
            bins[bin_idx].2 += 1;              // count
        }

        // Expected Calibration Error
        let ece: f32 = bins.iter()
            .filter(|(_, _, count)| *count > 0)
            .map(|(sum_pred, sum_actual, count)| {
                let avg_pred = sum_pred / *count as f32;
                let avg_actual = sum_actual / *count as f32;
                (*count as f32 / predictions.len() as f32) * (avg_pred - avg_actual).abs()
            })
            .sum();

        CalibrationMetrics {
            ece,
            mce: 0.0,  // Compute similarly
            brier_score: compute_brier_score(&predictions, test_y),
            reliability_area: 1.0 - ece,
        }
    }
}
```

#### 2.8.8 Success Criteria

| Metric | Target | Description |
|--------|--------|-------------|
| Expected Calibration Error (ECE) | < 0.05 | Predictions match observed frequencies |
| Brier Score | < 0.15 | Overall prediction accuracy |
| CI Coverage | ~95% | 95% CI should contain truth 95% of time |
| Actionability | 80%+ files with High confidence | Most predictions should be confident |

#### 2.8.9 Toyota Way Alignment

| Principle | Application |
|-----------|-------------|
| **Respect for People** | Developers know *how confident* to be in predictions |
| **Jidoka** | Explicit uncertainty prevents over-reliance on automation |
| **Genchi Genbutsu** | Calibration requires actual labeled data, not assumptions |
| **Muri** | Low-confidence predictions flagged for human judgment |
| **Kaizen** | Calibration improves as more labeled data accumulates |

---

## 3. Integration with Git History Analysis

### 3.1 SZZ Algorithm for Ground Truth Labeling

The SZZ algorithm [9] traces bug-fixing commits back to identify bug-introducing commits:

```
Bug Report → Bug-Fixing Commit → git blame → Bug-Introducing Commit
```

**Integration**:
```rust
pub struct SzzResult {
    pub bug_fixing_commit: String,
    pub bug_introducing_commits: Vec<String>,
    pub faulty_lines: Vec<(String, usize)>,  // (file, line)
    pub confidence: SzzConfidence,
}

pub enum SzzConfidence {
    High,    // Direct line trace
    Medium,  // Refactoring-aware trace
    Low,     // Heuristic fallback
}
```

### 3.2 Combining SBFL with Git History

The key insight: **git history provides ground truth labels for training and validating fault localization techniques**.

**Pipeline**:
1. **SZZ**: Identify bug-introducing commits → label faulty files/lines
2. **Coverage**: Collect test coverage from CI runs
3. **SBFL**: Calculate suspiciousness scores
4. **Validation**: Compare ranked statements against SZZ-labeled faults
5. **Feedback Loop**: Improve techniques based on miss analysis

---

## 4. Toyota Way Implementation Report

### 4.1 Phase 1: Foundation (Genchi Genbutsu) ✅ COMPLETE

**Objective**: Implement Tarantula baseline and measurement infrastructure.

**Deliverables**:
- [x] `TarantulaScorer` struct with basic formula
- [x] LCOV parser for `cargo-llvm-cov` integration
- [x] Test pass/fail collector
- [x] Suspiciousness report generator (YAML output)
- [x] Validation framework against manually labeled faults

**Implementation**: `src/tarantula.rs` (~1700 lines, 48 tests)

**Success Metrics**:
| Metric | Target | Measurement |
|--------|--------|-------------|
| Top-1 Accuracy | 25% | % of faults ranked #1 |
| Top-5 Accuracy | 45% | % of faults in top 5 |
| Top-10 Accuracy | 60% | % of faults in top 10 |
| Execution Time | <30s | For 1000 statements |

**Validation Protocol**:
1. Create labeled test suite with 50 known faults
2. Run Tarantula on each
3. Record rank of actual faulty statement
4. Calculate accuracy metrics
5. Document failure cases for Kaizen analysis

### 4.2 Phase 2: Enhancement (Kaizen) ✅ COMPLETE

**Trigger**: Phase 1 metrics achieved AND failure analysis complete.

**Deliverables**:
- [x] Ochiai formula implementation
- [x] DStar formula with tunable exponent
- [x] A/B comparison framework (all formulas computed in parallel)
- [x] Formula selection heuristics
- [x] SZZ algorithm for bug-introducing commit detection
- [x] HybridFaultLocalizer combining SBFL + historical data
- [x] CLI command (`oip localize`)
- [x] Report generation (YAML, JSON, Terminal formats)

**Implementation**: Full integration in `src/tarantula.rs` and `src/cli_handlers.rs`

**Success Metrics**:
| Metric | Target | vs Phase 1 |
|--------|--------|------------|
| Top-1 Accuracy | 35% | +10% |
| Top-5 Accuracy | 55% | +10% |
| Top-10 Accuracy | 70% | +10% |

### 4.3 Phase 3: Hybridization (Jidoka)

**Trigger**: Phase 2 accuracy plateaus AND mutation testing integrated.

**Deliverables**:
- [ ] Metallaxis-FL integration
- [ ] SBFL + MBFL combination scoring
- [ ] Confidence calibration

**Success Metrics**:
| Metric | Target | vs Phase 2 |
|--------|--------|------------|
| Unique Faults Found | +15% | Faults missed by SBFL alone |
| Combined Accuracy | 80% | Top-10 accuracy |

### 4.4 Phase 4: Learning (Respect for People)

**Trigger**: 5000+ labeled examples AND Phase 3 validated.

**Deliverables**:
- [ ] Feature extraction pipeline
- [ ] DeepFL model training
- [ ] Explainability layer (SHAP values)
- [ ] Human-readable fault explanations

**Success Metrics**:
| Metric | Target | vs Phase 3 |
|--------|--------|------------|
| Top-1 Accuracy | 50% | +15% |
| Explanation Quality | 4/5 | Developer survey rating |

### 4.5 Phase 5: RAG Enhancement (trueno-rag Integration) ✅ COMPLETE

**Trigger**: Phase 2 complete AND bug knowledge base established.
**Status**: Implemented 2025-12-01

**Objective**: Enhance fault localization with semantic search over historical bugs, similar code patterns, and contextual fix suggestions using trueno-rag.

**Deliverables**:
- [ ] Bug knowledge base schema and import pipeline
- [ ] trueno-rag integration for bug report indexing
- [ ] Code pattern indexing with semantic chunking
- [ ] RRF/DBSF fusion of SBFL + RAG scores
- [ ] Fix suggestion retrieval and ranking
- [ ] Enhanced explanations with historical context

**Dependencies**:
```toml
trueno-rag = "0.1"
trueno = "0.7"
trueno-db = "0.3"
```

**Success Metrics**:
| Metric | Target | vs Phase 2 |
|--------|--------|------------|
| Bug Retrieval Precision | 70% | New capability |
| Fix Suggestion Accuracy | 50% | New capability |
| Developer Time Saved | 30% | Measured via survey |
| Explanation Helpfulness | 4/5 | Developer rating |

**Review Questions for Team**:
1. Should we prioritize RAG enhancement or MBFL (Phase 3) next?
2. What is the minimum viable bug knowledge base size to be useful?
3. Should fix suggestions be opt-in or default behavior?
4. What embedding model should we use for code similarity?

---

## 5. Report Format

### 5.1 YAML Output Schema

```yaml
fault_localization_report:
  version: "1.0"
  generated_at: "2025-12-01T10:30:00Z"
  repository: "organizational-intelligence-plugin"
  commit: "abc123def"

  failing_tests:
    - name: "test_classifier_memory_safety"
      file: "src/classifier.rs"
      line: 142

  suspiciousness_ranking:
    - rank: 1
      file: "src/classifier.rs"
      line: 87
      statement: "let result = unsafe { ptr.read() };"
      scores:
        tarantula: 0.92
        ochiai: 0.89
        dstar: 0.95
      confidence: 0.87
      explanation: |
        This statement is executed by 95% of failing tests
        but only 12% of passing tests. The unsafe block
        suggests potential memory safety issues.

    - rank: 2
      file: "src/classifier.rs"
      line: 91
      statement: "buffer.push(result);"
      scores:
        tarantula: 0.78
        ochiai: 0.81
        dstar: 0.82
      confidence: 0.72

  quality_signals:
    avg_churn_score: 4.2
    avg_tdg_score: 67.5
    coverage_density: 0.85

  recommendations:
    - "Investigate unsafe block at classifier.rs:87"
    - "Consider adding bounds checking before buffer.push()"
    - "High churn in this file correlates with defect density"
```

### 5.2 Visual Report (Terminal)

```
╔══════════════════════════════════════════════════════════════╗
║           FAULT LOCALIZATION REPORT - Tarantula              ║
╠══════════════════════════════════════════════════════════════╣
║ Repository: organizational-intelligence-plugin               ║
║ Commit: abc123def                                            ║
║ Failing Tests: 3                                             ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  TOP SUSPICIOUS STATEMENTS                                   ║
║  ─────────────────────────────────────────────────────────  ║
║                                                              ║
║  #1  src/classifier.rs:87      ████████████████░░░░  0.92   ║
║      let result = unsafe { ptr.read() };                     ║
║      ⚠️  Executed by 95% failing / 12% passing               ║
║                                                              ║
║  #2  src/classifier.rs:91      ██████████████░░░░░░  0.78   ║
║      buffer.push(result);                                    ║
║      ⚠️  Executed by 88% failing / 23% passing               ║
║                                                              ║
║  #3  src/classifier.rs:85      ████████████░░░░░░░░  0.65   ║
║      let ptr = data.as_ptr();                                ║
║      ⚠️  Executed by 80% failing / 35% passing               ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  CONFIDENCE: HIGH (based on 47 tests, 3 failing)             ║
║  RECOMMENDATION: Focus investigation on unsafe block         ║
╚══════════════════════════════════════════════════════════════╝
```

---

## 6. Peer-Reviewed References

The following ten peer-reviewed publications provide the empirical and theoretical foundation for this specification:

| # | Citation | Key Contribution | Relevance |
|---|----------|------------------|-----------|
| **[1]** | Jones, J.A., Harrold, M.J. (2005). *Empirical evaluation of the Tarantula automatic fault-localization technique.* ASE '05, pp. 273-282. ACM. | Introduced Tarantula formula and color visualization | Foundation technique |
| **[2]** | Abreu, R., Zoeteweij, P., Golsteijn, R., Van Gemund, A.J. (2009). *A practical evaluation of spectrum-based fault localization.* JSS 82(11), pp. 1780-1792. | Comprehensive SBFL comparison; Ochiai superiority | Formula selection |
| **[3]** | Wong, W.E., Debroy, V., Gao, R., Li, Y. (2014). *The DStar method for effective software fault localization.* IEEE TSE 40(1), pp. 1-21. | Introduced DStar with tunable exponent | Advanced SBFL |
| **[4]** | Just, R., Jalali, D., Ernst, M.D. (2014). *Defects4J: A database of existing faults to enable controlled testing studies.* ISSTA '14, pp. 437-440. ACM. | Standard benchmark dataset | Evaluation framework |
| **[5]** | Papadakis, M., Le Traon, Y. (2015). *Metallaxis-FL: Mutation-based fault localization.* STVR 25(5-7), pp. 605-628. Wiley. | MBFL technique using mutation analysis | Hybrid approach |
| **[6]** | Li, X., Li, W., Zhang, Y., Zhang, L. (2019). *DeepFL: Integrating multiple fault diagnosis dimensions for deep fault localization.* ISSTA '19, pp. 169-180. ACM. | Deep learning for FL; multi-feature integration | Learning-based FL |
| **[7]** | Śliwerski, J., Zimmermann, T., Zeller, A. (2005). *When do changes induce fixes?* MSR '05, pp. 1-5. ACM. | Introduced SZZ algorithm | Ground truth labeling |
| **[8]** | Pearson, S., Campos, J., Just, R., et al. (2017). *Evaluating and improving fault localization.* ICSE '17, pp. 609-620. IEEE. | Comprehensive FL evaluation methodology | Evaluation best practices |
| **[9]** | Zou, D., Liang, J., Xiong, Y., et al. (2019). *An empirical study of fault localization families and their combinations.* IEEE TSE 47(2), pp. 332-347. | Family-wise FL comparison; combination strategies | Technique combinations |
| **[10]** | Keller, F., Grunske, L., Heiden, S., et al. (2017). *A critical evaluation of spectrum-based fault localization techniques on a large-scale software system.* ICSME '17, pp. 92-103. IEEE. | Industrial-scale SBFL evaluation | Real-world applicability |

---

## 7. API Design

### 7.1 Core Types

```rust
/// Represents a code location for fault localization
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct StatementId {
    pub file: PathBuf,
    pub line: usize,
    pub column: Option<usize>,
}

/// Coverage information for a single statement
#[derive(Debug, Clone)]
pub struct StatementCoverage {
    pub id: StatementId,
    pub executed_by_passed: usize,
    pub executed_by_failed: usize,
}

/// Result of fault localization analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultLocalizationResult {
    pub rankings: Vec<SuspiciousnessRanking>,
    pub formula_used: FaultLocalizationFormula,
    pub confidence: f32,
    pub total_passed_tests: usize,
    pub total_failed_tests: usize,
}

/// Individual ranking entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuspiciousnessRanking {
    pub rank: usize,
    pub statement: StatementId,
    pub suspiciousness: f32,
    pub scores: HashMap<String, f32>,  // Multiple formulas
    pub explanation: String,
}

/// Available fault localization formulas
#[derive(Debug, Clone, Copy)]
pub enum FaultLocalizationFormula {
    Tarantula,
    Ochiai,
    DStar { exponent: u32 },
    Combined,
}
```

### 7.2 Main Interface

```rust
/// Fault localizer trait for different implementations
pub trait FaultLocalizer {
    fn localize(
        &self,
        coverage: &[StatementCoverage],
        total_passed: usize,
        total_failed: usize,
    ) -> FaultLocalizationResult;

    fn formula(&self) -> FaultLocalizationFormula;
}

/// Builder for configuring fault localization
pub struct FaultLocalizerBuilder {
    formula: FaultLocalizationFormula,
    top_n: usize,
    include_explanations: bool,
}

impl FaultLocalizerBuilder {
    pub fn new() -> Self { /* ... */ }
    pub fn formula(mut self, f: FaultLocalizationFormula) -> Self { /* ... */ }
    pub fn top_n(mut self, n: usize) -> Self { /* ... */ }
    pub fn with_explanations(mut self) -> Self { /* ... */ }
    pub fn build(self) -> Box<dyn FaultLocalizer> { /* ... */ }
}
```

---

## 8. Success Criteria Summary

| Phase | Technique | Top-1 | Top-5 | Top-10 | Gate | Status |
|-------|-----------|-------|-------|--------|------|--------|
| 1 | Tarantula | 25% | 45% | 60% | Baseline | ✅ Complete |
| 2 | Ochiai/DStar/SZZ | 35% | 55% | 70% | +10% improvement | ✅ Complete |
| 3 | MBFL Hybrid | 45% | 65% | 80% | +15% unique faults | Pending |
| 4 | DeepFL | 50% | 75% | 85% | Explainability 4/5 | Pending |
| 5 | RAG (trueno-rag) | N/A | N/A | N/A | 70% retrieval precision | ✅ Complete |
| 6 | Weighted Ensemble | 50% | 75% | N/A | MAP 0.60+ | ✅ Complete |
| 7 | Calibrated Probability | N/A | N/A | N/A | ECE < 0.05 | ✅ Complete |

---

## 9. Conclusion

This specification provides a phased approach to integrating Tarantula-style fault localization into the Organizational Intelligence Plugin. By following Toyota Way principles—starting simple, measuring continuously, and evolving based on data—we avoid premature optimization while building toward state-of-the-art fault localization capabilities.

The combination of git history analysis (SZZ for ground truth) with spectrum-based fault localization creates a powerful synergy: historical data trains and validates our techniques, while real-time fault localization helps developers quickly identify defect locations when tests fail.

**Next Steps**:
1. Implement Phase 1 Tarantula scorer
2. Integrate with `cargo-llvm-cov` coverage output
3. Create labeled validation dataset from this repository's history
4. Measure baseline metrics
5. Proceed to Phase 2 only after validation

---

**Document History:**
- 2025-12-01 v1.3.0: Phase 6 & 7 Implementation Complete
  - Implemented Weighted Ensemble Risk Score (Phase 6)
  - Implemented Calibrated Defect Probability (Phase 7)
  - Added CLI flags: --ensemble, --ensemble-model, --include-churn, --calibrated, --calibration-model, --confidence-threshold
  - Created ensemble_predictor.rs module (~1550 lines, 30 tests)
  - Added WeightedEnsembleModel with EM-based weak supervision
  - Added CalibratedDefectPredictor with Isotonic calibration
  - Added FileFeatures, LabelingFunction trait, and 5 labeling function implementations
  - All 720 tests passing (690 lib + 30 ensemble_predictor)
  - Updated aprender dependency to v0.14.0
- 2025-12-01 v1.2.0: Phase 5 Implementation Complete
  - Implemented RAG-enhanced fault localization with trueno-rag
  - Added CLI flags: --rag, --knowledge-base, --fusion, --similar-bugs
  - Created rag_localization.rs module (~700 lines)
  - Added BugKnowledgeBase with BM25 indexing
  - Implemented LocalizationFusion strategies (RRF, Linear, DBSF, SbflOnly)
  - All 23 RAG tests passing
  - Updated book documentation
- 2025-12-01 v1.1.0: Added Approach 6 (RAG-Enhanced Fault Localization with trueno-rag)
  - Added Phase 5 implementation plan
  - Updated status: Phase 1-2 complete
  - Added review questions for team
- 2025-12-01 v1.0.0: Initial specification created
