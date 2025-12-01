# Example: Fault Localization

Demonstrates Tarantula-style Spectrum-Based Fault Localization (SBFL) to identify suspicious code when tests fail.

## Running the Example

```bash
cargo run --example fault_localization
```

## What This Example Does

1. Creates synthetic coverage data simulating passing and failing tests
2. Runs three SBFL formulas (Tarantula, Ochiai, DStar)
3. Demonstrates SZZ algorithm for bug-introducing commit detection
4. Combines SBFL with historical SZZ data using HybridFaultLocalizer
5. Shows RAG-enhanced localization with trueno-rag (Phase 5)
6. Demonstrates Weighted Ensemble Risk Scores (Phase 6)
7. Shows Calibrated Defect Probability predictions (Phase 7)
8. Generates reports in multiple formats (YAML, JSON, terminal)

## Source Code

```rust
// examples/fault_localization.rs
use organizational_intelligence_plugin::tarantula::{
    FaultLocalizationResult, HybridFaultLocalizer, LocalizationConfig,
    ReportFormat, SbflFormula, StatementCoverage, StatementId,
    SzzAnalyzer, SzzResult, TarantulaIntegration,
};
use std::collections::HashMap;

fn main() {
    println!("🔍 Tarantula Fault Localization Demo\n");

    // Simulate coverage data from a bug scenario
    let coverage = create_test_coverage();

    // Run SBFL with different formulas
    demo_sbfl_formulas(&coverage);

    // Demonstrate SZZ algorithm
    demo_szz_algorithm();

    // Show hybrid approach
    demo_hybrid_localization(&coverage);
}
```

## Sample Output

```
🔍 Tarantula Fault Localization Demo

📊 SBFL Formula Comparison:

Tarantula Results:
  #1 src/parser.rs:50 - 0.909 (bug location!)
  #2 src/parser.rs:55 - 0.500
  #3 src/parser.rs:60 - 0.333

Ochiai Results:
  #1 src/parser.rs:50 - 0.949
  #2 src/parser.rs:55 - 0.577
  #3 src/parser.rs:60 - 0.408

DStar (exponent=2) Results:
  #1 src/parser.rs:50 - 100.000
  #2 src/parser.rs:55 - 1.000
  #3 src/parser.rs:60 - 0.333

🔬 SZZ Bug-Introducing Commit Detection:

Bug-fixing commits identified: 2
  - abc123: "fix: null pointer in parser"
  - def456: "bugfix: race condition handler"

Bug-introducing commits traced:
  - bad001 introduced bug at src/parser.rs:50
  - bad002 introduced bug at src/handler.rs:100

🎯 Hybrid Fault Localization:

Combined SBFL + Historical scores (α=0.7):
  #1 src/parser.rs:50 - 0.927 (SBFL: 0.909, Historical: 0.800)
  #2 src/handler.rs:100 - 0.650 (SBFL: 0.500, Historical: 0.850)
```

## Key Concepts Demonstrated

### 1. SBFL Formulas

```rust
// Tarantula - balanced approach
let tarantula_score = tarantula(failed, passed, total_failed, total_passed);

// Ochiai - often better precision
let ochiai_score = ochiai(failed, passed, total_failed);

// DStar - aggressive ranking
let dstar_score = dstar(failed, passed, total_failed, 2);
```

### 2. SZZ Algorithm

```rust
// Identify bug-fixing commits
let fixes = SzzAnalyzer::identify_bug_fixes(&commits);

// Trace back to introducing commits
let szz_result = SzzAnalyzer::trace_introducing_commits(
    "fix_commit",
    "fix: null pointer",
    &changed_lines,
    &blame_data,
);
```

### 3. Hybrid Localization

```rust
// Combine SBFL with historical data
let combined = HybridFaultLocalizer::combine_scores(
    &sbfl_result,
    &historical_suspiciousness,
    0.7, // alpha: 70% SBFL, 30% historical
);
```

### 4. RAG-Enhanced Localization (Phase 5)

```rust
use organizational_intelligence_plugin::rag_localization::{
    BugDocument, BugKnowledgeBase, RagFaultLocalizer,
    RagLocalizationConfig, LocalizationFusion,
};

// Create bug knowledge base
let mut kb = BugKnowledgeBase::new();
kb.add_bug(
    BugDocument::new("BUG-001", "Null pointer in parser", DefectCategory::MemorySafety)
        .with_description("Parser crashes on empty input")
        .with_fix_commit("abc123")
        .with_affected_files(vec!["src/parser.rs"])
        .with_fix_pattern("Use unwrap_or_default()")
);

// Configure RAG-enhanced localization
let config = RagLocalizationConfig::new()
    .with_formula(SbflFormula::Tarantula)
    .with_fusion(LocalizationFusion::RRF { k: 60.0 })
    .with_similar_bugs(5)
    .with_explanations(true);

// Run RAG-enhanced localization
let rag_localizer = RagFaultLocalizer::new(kb, config);
let result = rag_localizer.localize(&coverage, 100, 10);

// Results include similar bugs and suggested fixes
for ranking in &result.rankings {
    println!("#{} {}:{} - {:.3}",
        ranking.sbfl_ranking.rank,
        ranking.sbfl_ranking.statement.file.display(),
        ranking.sbfl_ranking.statement.line,
        ranking.combined_score
    );
    for bug in &ranking.similar_bugs {
        println!("  → Similar: {} ({:.0}%)", bug.summary, bug.similarity * 100.0);
    }
}
```

### 5. Weighted Ensemble Model (Phase 6)

```rust
use organizational_intelligence_plugin::ensemble_predictor::{
    FileFeatures, WeightedEnsembleModel,
    SbflLabelingFunction, TdgLabelingFunction, ChurnLabelingFunction,
    ComplexityLabelingFunction, RagSimilarityLabelingFunction,
};

// Create feature vectors for files
let features = vec![
    FileFeatures {
        file_path: "src/parser.rs".to_string(),
        sbfl_score: 0.92,
        tdg_score: Some(0.65),
        churn_count: Some(25),
        complexity: Some(15.0),
        rag_similarity: Some(0.85),
    },
    FileFeatures {
        file_path: "src/handler.rs".to_string(),
        sbfl_score: 0.78,
        tdg_score: Some(0.45),
        churn_count: Some(8),
        complexity: Some(8.0),
        rag_similarity: Some(0.72),
    },
];

// Create labeling functions (weak supervision signals)
let labeling_functions: Vec<Box<dyn LabelingFunction>> = vec![
    Box::new(SbflLabelingFunction::new(0.7)),      // SBFL threshold
    Box::new(TdgLabelingFunction::new(0.6)),       // TDG threshold
    Box::new(ChurnLabelingFunction::new(20)),      // Churn threshold
    Box::new(ComplexityLabelingFunction::new(12.0)), // Complexity threshold
    Box::new(RagSimilarityLabelingFunction::new(0.75)), // RAG threshold
];

// Fit ensemble model using EM algorithm
let mut ensemble = WeightedEnsembleModel::new(labeling_functions);
ensemble.fit(&features);

// Get learned signal weights
println!("Learned Signal Weights:");
for (name, weight) in ensemble.get_weights() {
    println!("  {}: {:.1}%", name, weight * 100.0);
}

// Predict risk scores
let predictions = ensemble.predict(&features);
for (file, risk) in features.iter().zip(predictions.iter()) {
    println!("{} - Risk: {:.1}%", file.file_path, risk * 100.0);
}
```

### 6. Calibrated Defect Prediction (Phase 7)

```rust
use organizational_intelligence_plugin::ensemble_predictor::{
    CalibratedDefectPredictor, CalibratedPrediction, ConfidenceLevel,
};

// Create calibrated predictor (wraps ensemble model)
let mut predictor = CalibratedDefectPredictor::new(labeling_functions);

// Fit with historical data (features + known defect labels)
let historical_labels = vec![true, false, true, false, false];
predictor.fit(&historical_features, &historical_labels);

// Get calibrated predictions with confidence intervals
let predictions: Vec<CalibratedPrediction> = predictor.predict(&features);

for (file, pred) in features.iter().zip(predictions.iter()) {
    let (lo, hi) = pred.confidence_interval;
    let level = match pred.confidence_level {
        ConfidenceLevel::High => "HIGH",
        ConfidenceLevel::Medium => "MEDIUM",
        ConfidenceLevel::Low => "LOW",
    };

    println!("{} - P(defect) = {:.0}% ± {:.0}% [{}]",
        file.file_path,
        pred.probability * 100.0,
        (hi - lo) / 2.0 * 100.0,
        level
    );

    // Show contributing factors
    for (factor, contribution) in &pred.contributing_factors {
        println!("  ├─ {}: {:.1}%", factor, contribution * 100.0);
    }
}
```

## Toyota Way Principles

| Principle | Application |
|-----------|-------------|
| **Genchi Genbutsu** | Uses actual test coverage and historical bugs, not estimates |
| **Muda** | Lightweight SBFL before expensive mutation testing; RAG only for top-N |
| **Muri** | Configurable top-N and similar-bugs limits prevent information overload |
| **Jidoka** | Human-readable explanations with context from knowledge base |
| **Kaizen** | Bug knowledge base grows with each fix, improving future localization |
| **Hansei** | Calibrated predictions include uncertainty quantification for honest assessment |

## Integration with pmat

```rust
// Enrich with TDG scores
if let Ok(tdg) = PmatIntegration::analyze_tdg(&repo_path) {
    TarantulaIntegration::enrich_with_tdg(&mut result, &tdg.file_scores);
}
```

## CLI Usage with RAG

```bash
# Create a knowledge base from historical bugs
cat > bugs.yaml << 'EOF'
- id: "BUG-001"
  title: "Null pointer in parser"
  description: "Parser crashes when input is empty"
  category: MemorySafety
  fix_commit: "abc123"
  fix_diff: "- data.unwrap()\n+ data.unwrap_or_default()"
  affected_files: ["src/parser.rs"]
  severity: 4
  symptoms: ["panic", "unwrap failed"]
  root_cause: "Missing None check"
  fix_pattern: "Use unwrap_or_default()"
EOF

# Run RAG-enhanced fault localization
oip localize \
  --passed-coverage passed.lcov \
  --failed-coverage failed.lcov \
  --passed-count 100 \
  --failed-count 10 \
  --formula ochiai \
  --rag \
  --knowledge-base bugs.yaml \
  --fusion rrf \
  --similar-bugs 5 \
  --format terminal
```

## CLI Usage with Ensemble and Calibration

### Weighted Ensemble Model (Phase 6)

```bash
# Run with ensemble model combining multiple signals
oip localize \
  --passed-coverage passed.lcov \
  --failed-coverage failed.lcov \
  --passed-count 100 \
  --failed-count 10 \
  --formula tarantula \
  --ensemble \
  --enrich-tdg \
  --repo . \
  --format terminal
```

Output:
```
🔮 Running Weighted Ensemble Model (Phase 6)...
   ✅ Ensemble model fitted on 10 files

   Ensemble Risk Predictions:
   #1 src/parser.rs - Risk: 78.5%
   #2 src/handler.rs - Risk: 65.2%

   Learned Signal Weights:
      SBFL: 35.2%
      TDG: 22.1%
      Churn: 18.5%
      Complexity: 14.8%
```

### Calibrated Predictions (Phase 7)

```bash
# Run with calibrated probability output
oip localize \
  --passed-coverage passed.lcov \
  --failed-coverage failed.lcov \
  --calibrated \
  --confidence-threshold 0.5 \
  --format terminal
```

Output:
```
📊 Running Calibrated Defect Prediction (Phase 7)...
   Confidence threshold: 50%

   Calibrated Predictions (above 50% threshold):
   #1 src/parser.rs:87 - P(defect) = 73% ± 12% [HIGH]
      ├─ SBFL: 35.0%
      ├─ TDG: 22.0%
   #2 src/handler.rs:100 - P(defect) = 58% ± 22% [MEDIUM]
```

### Full Pipeline (All Phases)

```bash
# Combined: SBFL + RAG + Ensemble + Calibration
oip localize \
  --passed-coverage passed.lcov \
  --failed-coverage failed.lcov \
  --formula ochiai \
  --rag \
  --knowledge-base bugs.yaml \
  --ensemble \
  --calibrated \
  --confidence-threshold 0.6 \
  --enrich-tdg \
  --repo . \
  --format terminal
```

## See Also

- [Localize CLI Command](../cli/localize.md)
- [Tarantula Specification](../../docs/specifications/tarantula-defect-specification.md)
- [SZZ Algorithm Paper](https://dl.acm.org/doi/10.1145/1083142.1083147)
- [trueno-rag crate](https://crates.io/crates/trueno-rag)
