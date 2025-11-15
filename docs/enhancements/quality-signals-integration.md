# Quality Signals Integration for Defect Reporting
## Phase 1.5 Enhancement Specification

**Status**: Proposed
**Priority**: High
**Impact**: Transforms defect detection into actionable intelligence
**Estimated Effort**: 2-3 days

---

## Problem Statement

Current Phase 1 defect reporting provides:
- ✅ **What** defects exist (category, frequency)
- ✅ **Where** they are (repository, commit)
- ❌ **Why** they happened (missing quality context)
- ❌ **How urgent** they are (missing prioritization signals)

**Example Current Output**:
```yaml
- category: ConfigurationErrors
  frequency: 10
  confidence: 0.72
  examples:
  - "bd221b0f: adding config"
```

**User Question**: *"Are these config errors in well-tested code or technical debt hell?"*
**Answer**: *Unknown - we need quality signals!*

---

## Proposed Enhancement

Integrate **pmat quality signals** with defect detection to provide context-rich intelligence.

### Quality Signals to Capture

#### 1. Technical Debt Gradient (TDG)
- **Per-file TDG scores** from `pmat analyze tdg`
- **Before/after TDG** for defect fixes
- **Average TDG** of files with defect category

**Value**:
- High TDG + defects = **Refactoring needed**
- Low TDG + defects = **Isolated issues**

#### 2. Code Complexity
- **Cyclomatic complexity** from pmat
- **Function length** metrics
- **Nesting depth**

**Value**:
- High complexity + defects = **Simplification needed**
- Predict defect-prone areas

#### 3. Test Coverage
- **Line coverage** from `pmat analyze coverage`
- **Branch coverage**
- **Mutation test score**

**Value**:
- Low coverage + defects = **Need more tests**
- Validate: "Does testing prevent defects?"

#### 4. SATD (Self-Admitted Technical Debt)
- **TODO/FIXME/HACK** comments from `pmat analyze satd`
- **Debt location** correlation with defects

**Value**:
- SATD markers predict future defects
- Quantify "known issues" vs "surprises"

#### 5. Code Churn
- **Lines added/removed** per commit
- **File change frequency**
- **Author count** (coordination complexity)

**Value**:
- High churn + defects = **Unstable area**
- Low churn + defects = **Regression**

#### 6. Dead Code
- **Unreachable code percentage** from `pmat analyze dead-code`

**Value**:
- Dead code + defects = **Cleanup needed**

---

## Enhanced Schema

### DefectPattern (Enhanced)

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefectPattern {
    pub category: DefectCategory,
    pub frequency: usize,
    pub confidence: f32,

    // NEW: Quality signals aggregated across all instances
    pub quality_signals: QualitySignals,

    // Enhanced examples with per-instance metrics
    pub examples: Vec<DefectInstance>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualitySignals {
    // TDG signals
    pub avg_tdg_score: f32,
    pub max_tdg_score: f32,
    pub tdg_trend: TdgTrend,  // Increasing/Decreasing/Stable

    // Complexity signals
    pub avg_cyclomatic_complexity: f32,
    pub avg_function_length: f32,

    // Coverage signals
    pub avg_test_coverage: f32,
    pub avg_mutation_score: Option<f32>,

    // Debt signals
    pub satd_instances: usize,
    pub dead_code_percentage: f32,

    // Churn signals
    pub avg_lines_changed: usize,
    pub avg_files_per_commit: f32,
    pub high_churn_correlation: bool,  // > 100 lines changed
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefectInstance {
    pub commit_hash: String,
    pub message: String,
    pub files_affected: Vec<String>,

    // Per-instance quality metrics
    pub tdg_before: Option<f32>,
    pub tdg_after: Option<f32>,
    pub complexity: Option<f32>,
    pub test_coverage: Option<f32>,
    pub satd_count: usize,
    pub lines_changed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TdgTrend {
    Improving,    // TDG decreasing with fixes
    Degrading,    // TDG increasing with fixes
    Stable,       // No significant change
}
```

### Example Enhanced Output

```yaml
defect_patterns:
- category: ConfigurationErrors
  frequency: 10
  confidence: 0.72
  quality_signals:
    avg_tdg_score: 52.3          # ⚠️ Above threshold (50)
    max_tdg_score: 73.1           # ⚠️ One very bad file
    tdg_trend: Degrading          # 🚨 Getting worse!
    avg_cyclomatic_complexity: 15.2  # ⚠️ High
    avg_function_length: 125      # ⚠️ Long functions
    avg_test_coverage: 0.58       # ⚠️ Below 85% target
    avg_mutation_score: null      # Not measured
    satd_instances: 8             # 🚨 "TODO: fix config"
    dead_code_percentage: 0.12    # Some cleanup needed
    avg_lines_changed: 67         # Moderate churn
    avg_files_per_commit: 3.2
    high_churn_correlation: true  # ⚠️ Large changes
  examples:
  - commit_hash: "bd221b0f"
    message: "adding config"
    files_affected:
    - "src/config.rs"
    - "tests/config_test.rs"
    - "Cargo.toml"
    tdg_before: 45.2
    tdg_after: 52.8               # 🚨 TDG increased by 7.6!
    complexity: 18                # High
    test_coverage: 0.55           # Low
    satd_count: 3                 # "TODO: validate schema"
    lines_changed: 89
```

### Actionable Insights Report

```yaml
insights:
  high_priority_areas:
  - category: ConfigurationErrors
    reason: "High TDG (52.3), degrading trend, low coverage (58%)"
    recommendation: "Refactor config module before adding features"
    estimated_impact: "Could prevent 60% of future config errors"

  correlations_found:
  - pattern: "Config errors correlate with TDG > 50 (r=0.82)"
    action: "Set TDG warning threshold at 45"

  - pattern: "Security issues have 2x SATD markers vs other categories"
    action: "Prioritize SATD cleanup in security-critical code"

  - pattern: "Integration failures happen in high-churn files (>100 LOC/commit)"
    action: "Add integration tests to frequently changed modules"
```

---

## Implementation Approach

### Phase 1.5.1: Data Collection (1 day)

**Add pmat integration to git analyzer**:

```rust
// src/git.rs - Enhanced CommitInfo
pub struct CommitInfo {
    pub hash: String,
    pub message: String,
    pub author: String,
    pub timestamp: i64,

    // NEW: Quality metrics
    pub files_changed: Vec<FileChange>,
    pub lines_added: usize,
    pub lines_removed: usize,
}

pub struct FileChange {
    pub path: String,
    pub lines_added: usize,
    pub lines_removed: usize,
    pub tdg_score: Option<f32>,      // Run pmat tdg on file
    pub complexity: Option<f32>,      // Extract from pmat
    pub coverage: Option<f32>,        // From coverage report
}
```

**Integration points**:
1. After cloning repo, run `pmat analyze tdg --path {repo}`
2. Parse TDG output, map scores to files
3. Extract complexity from TDG analysis
4. Read coverage data if available (`target/coverage/lcov.info`)

### Phase 1.5.2: Aggregation (1 day)

**Enhance OrgAnalyzer**:

```rust
impl OrgAnalyzer {
    fn aggregate_with_signals(
        &self,
        commits: &[CommitInfo],
        classifications: &[(CommitInfo, Classification)],
    ) -> Vec<EnhancedDefectPattern> {
        let mut category_stats = HashMap::new();

        for (commit, classification) in classifications {
            let stats = category_stats.entry(classification.category)
                .or_insert_with(CategoryStatsEnhanced::new);

            // Aggregate quality signals
            stats.add_instance(commit, classification);
            stats.update_quality_signals(commit);
        }

        category_stats.into_values()
            .map(|s| s.into_pattern())
            .collect()
    }
}
```

### Phase 1.5.3: Reporting (0.5 days)

**Update YAML output schema** to include quality_signals and insights.

### Phase 1.5.4: Insights Generation (0.5 days)

**Add correlation analysis**:

```rust
impl InsightsGenerator {
    pub fn generate_insights(patterns: &[EnhancedDefectPattern]) -> Insights {
        Insights {
            high_priority_areas: self.identify_priority_areas(patterns),
            correlations: self.find_correlations(patterns),
            recommendations: self.generate_recommendations(patterns),
        }
    }

    fn find_correlations(&self, patterns: &[EnhancedDefectPattern]) -> Vec<Correlation> {
        // Statistical analysis
        // - TDG vs defect frequency (Pearson correlation)
        // - Coverage vs defect frequency
        // - Complexity vs defect severity
        // - SATD count vs future defects
    }
}
```

---

## Benefits

### 1. Prioritization Intelligence

**Before**: "You have 10 config errors"
**After**: "You have 10 config errors, 7 in high-TDG files with low coverage - **refactor urgently**"

### 2. Root Cause Analysis

**Correlation Examples**:
- "80% of config errors happen in files with TDG > 50"
- "Security issues correlate with SATD markers (r=0.75)"
- "Integration failures happen in files changed >5 times/month"

### 3. Predictive Insights

**Proactive Warnings**:
- "Files with TDG > 60 have 3x defect rate - refactor before next feature"
- "Modules without tests have 5x config errors - add test coverage"

### 4. Validation of Practices

**Evidence**:
- "EXTREME TDD repos have 50% fewer logic errors"
- "pmat validation reduces integration failures by 70%"
- "High coverage (>85%) correlates with low defect rate"

---

## Example Use Cases

### Use Case 1: Sprint Planning

**Team Question**: "Which defects should we fix first?"

**Enhanced Report Answer**:
```
Priority 1: ConfigurationErrors in src/config.rs
  - TDG: 73.1 (highest in codebase)
  - Coverage: 45% (lowest)
  - SATD: 5 markers
  - Churn: High (changed 12 times last month)
  → Refactor before adding OAuth integration

Priority 2: IntegrationFailures in paiml-mcp-agent-toolkit
  - TDG: 48.2 (acceptable)
  - Coverage: 78% (good)
  - High churn: 8 files changed per commit
  → Add integration tests, reduce coupling
```

### Use Case 2: Technical Debt Paydown

**Management Question**: "Where should we invest refactoring effort?"

**Enhanced Report Answer**:
```
Refactoring ROI Analysis:
1. Config module: High TDG (52.3) + 10 defects = 82% reduction potential
2. Auth module: Medium TDG (38.1) + 3 defects = 45% reduction potential
3. Utils module: Low TDG (15.2) + 1 defect = 12% reduction potential

Recommendation: Refactor config module first (2.8x ROI vs utils)
```

### Use Case 3: Hiring/Training

**Question**: "Do junior devs produce more defects in complex code?"

**Enhanced Report Answer**:
```
Analysis by Author Experience + Complexity:
- Junior devs (<6 months): 2.1% defect rate in high-complexity code (>15)
- Junior devs in low-complexity: 0.8% defect rate
- Senior devs in high-complexity: 1.2% defect rate

Insight: Complexity matters more than experience
Action: Simplify codebase, pair juniors on complex modules
```

---

## Implementation Checklist

### Phase 1.5.1: Data Collection
- [ ] Add `pmat tdg` integration to git analyzer
- [ ] Parse TDG output and map to files
- [ ] Extract cyclomatic complexity
- [ ] Read coverage data from lcov.info
- [ ] Count SATD markers (grep for TODO/FIXME/HACK)
- [ ] Calculate churn metrics

### Phase 1.5.2: Aggregation
- [ ] Update DefectPattern schema with QualitySignals
- [ ] Update DefectInstance schema with per-commit metrics
- [ ] Implement aggregation logic in OrgAnalyzer
- [ ] Calculate trends (TDG improving/degrading)

### Phase 1.5.3: Reporting
- [ ] Update YAML serialization
- [ ] Add insights section to report
- [ ] Generate correlation analysis
- [ ] Create priority rankings

### Phase 1.5.4: Visualization
- [ ] Add markdown summary with charts
- [ ] Generate priority heatmap
- [ ] Create correlation graphs
- [ ] Export to CSV for further analysis

---

## Success Metrics

**Tool Enhancement Metrics**:
- ✅ All defects have TDG scores
- ✅ Coverage data captured for >80% of repos
- ✅ At least 3 correlations found per analysis
- ✅ Priority ranking changes team's fix order

**Business Impact Metrics** (after 1 month):
- 📊 30% reduction in high-TDG defects
- 📊 50% of fixes address root causes (vs symptoms)
- 📊 Team velocity increases (fixing right things first)

---

## Example Enhanced Report

See: `examples/enhanced-defect-report.yaml` (to be created)

---

**Conclusion**: Adding pmat quality signals transforms defect detection from "what's broken" to "why it breaks and what to fix first" - making it dramatically more actionable for teams.
