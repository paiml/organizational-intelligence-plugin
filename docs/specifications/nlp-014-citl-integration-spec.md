---
title: Depyler CITL Integration for Ground-Truth Training Labels
issue: NLP-014
status: Draft
created: 2025-11-28
updated: 2025-11-28
priority: P1
depends_on: [NLP-011, NLP-013]
integrates: [depyler, alimentar]
---

# NLP-014: Depyler CITL Integration for Ground-Truth Training Labels

**Ticket ID**: NLP-014
**Status**: Draft
**Priority**: P1
**Estimated Effort**: 4-6 days
**Integrations**: depyler (CITL producer), alimentar (data pipeline)

## Summary

Integrate Depyler's Compiler-in-the-Loop (CITL) diagnostic output as a ground-truth training signal for OIP's defect classifier. This addresses the core limitation identified in NLP-011: the ML classifier overfits due to limited training data (508 examples) with noisy labels derived from commit message pattern matching.

Depyler's CITL produces compiler-verified error codes (E0308, E0277, etc.) that map deterministically to OIP DefectCategory labels, providing high-confidence training data without human labeling.

## Motivation

### Problem Statement

From NLP-011 Validation Report:
- **Test Accuracy**: 54.55% (below 80% target)
- **Overfitting**: 100% train → 46% validation
- **Root Cause**: Insufficient training data with noisy labels

Current OIP training pipeline:
```
Git Commits → Pattern Match ("config" → ConfigurationErrors) → Training
                    ↑
              Low confidence (0.70)
              High noise
              No ground truth
```

### Proposed Solution

Leverage Depyler's CITL as a label oracle:
```
Python Source → Transpile → cargo clippy → Error Code → DefectCategory
                                 ↑
                          Machine-verified
                          High confidence (0.95)
                          Deterministic mapping
```

### Expected Outcomes

| Metric | Current (NLP-011) | Target (NLP-014) |
|--------|-------------------|------------------|
| Training examples | 508 | 5,000+ |
| Label confidence | 0.70 avg | 0.90+ avg |
| Test accuracy | 54.55% | 75%+ |
| Overfitting gap | 54% | <15% |

## Requirements

### Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1 | Import Depyler CITL JSONL export format | P0 |
| FR-2 | Map rustc error codes to DefectCategory | P0 |
| FR-3 | Map Clippy lint codes to DefectCategory | P0 |
| FR-4 | Extend TrainingExample with compiler fields | P1 |
| FR-5 | Extend CommitFeatures vector (8→14 dims) | P1 |
| FR-6 | CLI command: `oip train import-depyler` | P1 |
| FR-7 | Support merge strategies (append/replace/weighted) | P2 |
| FR-8 | Validate imported examples against schema | P1 |
| FR-9 | Integrate alimentar DataLoader for streaming | P1 |
| FR-10 | Support Parquet format for large corpora | P1 |
| FR-11 | Configurable batch size and shuffling | P2 |

### Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-1 | Import throughput | ≥50,000 examples/sec (alimentar) |
| NFR-2 | Memory overhead | <100MB streaming (any corpus size) |
| NFR-3 | Schema validation latency | <1ms per example |
| NFR-4 | Backwards compatibility | Existing models must load |
| NFR-5 | Parquet columnar scan | ≥100,000 rows/sec |
| NFR-6 | Batch prefetch latency | <10ms (async I/O) |

## Technical Design

### 3.1 Error Code Taxonomy Mapping

```rust
/// Map rustc error code to OIP DefectCategory
pub fn rustc_to_defect_category(code: &str) -> Option<DefectCategory> {
    match code {
        // Type system
        "E0308" => Some(DefectCategory::TypeErrors),
        "E0412" => Some(DefectCategory::TypeAnnotationGaps),

        // Ownership/borrowing
        "E0502" | "E0503" | "E0505" => Some(DefectCategory::OwnershipBorrow),
        "E0382" | "E0507" => Some(DefectCategory::MemorySafety),

        // Traits
        "E0277" => Some(DefectCategory::TraitBounds),

        // Name resolution
        "E0425" | "E0433" => Some(DefectCategory::StdlibMapping),

        // AST/structure
        "E0599" | "E0615" => Some(DefectCategory::ASTTransform),
        "E0614" => Some(DefectCategory::OperatorPrecedence),

        // Configuration
        "E0658" => Some(DefectCategory::ConfigurationErrors),

        _ => None, // Unknown codes fallback to rule-based
    }
}

/// Map Clippy lint to OIP DefectCategory
pub fn clippy_to_defect_category(lint: &str) -> Option<DefectCategory> {
    match lint {
        "clippy::unwrap_used" | "clippy::expect_used" | "clippy::panic"
            => Some(DefectCategory::ApiMisuse),
        "clippy::todo" | "clippy::unreachable"
            => Some(DefectCategory::LogicErrors),
        "clippy::cognitive_complexity"
            => Some(DefectCategory::PerformanceIssues),
        "clippy::too_many_arguments" | "clippy::match_single_binding"
            => Some(DefectCategory::ASTTransform),
        "clippy::needless_collect"
            => Some(DefectCategory::IteratorChain),
        "clippy::manual_map"
            => Some(DefectCategory::ComprehensionBugs),
        _ => None,
    }
}
```

### 3.2 Import Schema

Depyler exports JSONL format:
```json
{
  "source_file": "example_dict_ops.py",
  "error_code": "E0308",
  "clippy_lint": null,
  "level": "error",
  "message": "mismatched types: expected `i32`, found `&str`",
  "oip_category": "TypeErrors",
  "confidence": 0.95,
  "span": {
    "line_start": 42,
    "column_start": 12
  },
  "suggestion": {
    "replacement": ".parse::<i32>()",
    "applicability": "MaybeIncorrect"
  },
  "timestamp": 1732752000,
  "depyler_version": "3.21.0"
}
```

### 3.3 Data Pipeline with Alimentar

For scalable ingestion of large CITL corpora (50K+ examples), integrate [alimentar](https://github.com/paiml/alimentar) as the data loading layer.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Data Pipeline Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Depyler CITL Export                                            │
│       │                                                          │
│       ▼                                                          │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │    JSONL     │────▶│  Alimentar   │────▶│     OIP      │    │
│  │   /Parquet   │     │  DataLoader  │     │   Trainer    │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│                              │                                   │
│                              ├── Batching (configurable)        │
│                              ├── Shuffling (epoch-level)        │
│                              ├── Streaming (memory-bounded)     │
│                              └── Parallel prefetch              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Alimentar Integration

```rust
use alimentar::{ArrowDataset, DataLoader, JsonLinesSource};

/// Load CITL corpus with alimentar for scalable training
pub fn create_citl_dataloader(
    corpus_path: &Path,
    batch_size: usize,
    shuffle: bool,
) -> Result<DataLoader<impl Dataset>> {
    let dataset = if corpus_path.extension() == Some("parquet".as_ref()) {
        ArrowDataset::from_parquet(corpus_path)?
    } else {
        // JSONL streaming
        ArrowDataset::from_json_lines(corpus_path)?
    };

    let mut loader = DataLoader::new(dataset)
        .batch_size(batch_size)
        .prefetch(2);  // Parallel prefetch

    if shuffle {
        loader = loader.shuffle(true);
    }

    Ok(loader)
}

/// Training loop with alimentar
pub fn train_with_citl(
    loader: DataLoader<impl Dataset>,
    model: &mut HybridClassifier,
) -> Result<TrainingStats> {
    let mut stats = TrainingStats::default();

    for batch in loader {
        let examples: Vec<TrainingExample> = batch
            .column("data")?
            .iter()
            .filter_map(|row| serde_json::from_value(row).ok())
            .collect();

        model.train_batch(&examples)?;
        stats.batches_processed += 1;
        stats.examples_processed += examples.len();
    }

    Ok(stats)
}
```

#### Parquet Export (Recommended for Large Corpora)

```bash
# Depyler: export as Parquet for efficient columnar access
depyler oracle export-oip \
    --input-dir ./examples \
    --output ./training_corpus/citl.parquet \
    --format parquet

# OIP: import with alimentar streaming
oip train import-depyler \
    --corpus ./training_corpus/citl.parquet \
    --batch-size 256 \
    --shuffle
```

### 3.4 Extended TrainingExample

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    // Existing fields
    pub message: String,
    pub label: DefectCategory,
    pub confidence: f32,
    pub commit_hash: String,
    pub author: String,
    pub timestamp: i64,
    pub lines_added: u32,
    pub lines_removed: u32,
    pub files_changed: u32,

    // New fields (NLP-014)
    #[serde(default)]
    pub error_code: Option<String>,
    #[serde(default)]
    pub clippy_lint: Option<String>,
    #[serde(default)]
    pub has_suggestion: bool,
    #[serde(default)]
    pub suggestion_applicability: Option<String>,
    #[serde(default)]
    pub source: TrainingSource,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum TrainingSource {
    #[default]
    CommitMessage,
    DepylerCitl,
    Manual,
}
```

### 3.4 Extended Feature Vector

```rust
pub struct CommitFeatures {
    // Existing (8 dims)
    pub defect_category: u8,
    pub files_changed: f32,
    pub lines_added: f32,
    pub lines_deleted: f32,
    pub complexity_delta: f32,
    pub timestamp: f64,
    pub hour_of_day: u8,
    pub day_of_week: u8,

    // New (6 dims) - NLP-014
    pub error_code_class: u8,        // 0=type,1=borrow,2=name,3=trait,4=other
    pub has_suggestion: u8,           // 0 or 1
    pub suggestion_applicability: u8, // 0=none,1=machine,2=maybe,3=placeholder
    pub clippy_lint_count: u8,        // 0-255
    pub span_line_delta: f32,         // Distance from function start
    pub diagnostic_confidence: f32,   // From taxonomy mapping
}

impl CommitFeatures {
    pub const DIMENSION: usize = 14;  // Extended from 8
}
```

## CLI Interface

### Import Command

```bash
# Import Depyler CITL corpus (streaming with alimentar)
oip train import-depyler \
    --corpus /path/to/depyler/training_corpus/citl.parquet \
    --merge-strategy append \
    --min-confidence 0.80 \
    --reweight 1.5 \
    --batch-size 256 \
    --shuffle

# Options:
#   --corpus          Path to Depyler JSONL/Parquet export (required)
#   --merge-strategy  append|replace|weighted (default: append)
#   --min-confidence  Minimum confidence threshold (default: 0.75)
#   --reweight        Weight multiplier for CITL examples (default: 1.0)
#   --batch-size      Alimentar batch size (default: 128)
#   --shuffle         Shuffle examples per epoch (default: true)
#   --validate-only   Validate schema without importing
```

### Export Statistics Command

```bash
# Show training data statistics by source
oip train stats --by-source

# Output:
# Source          Examples  Avg Conf  Categories
# CommitMessage       508     0.70    12/18
# DepylerCitl       4,521     0.92    14/18
# Total             5,029     0.88    15/18
```

## Implementation Plan

### Phase 1: Core Import (2 days)
- [ ] Define `DepylerExport` struct matching Depyler schema
- [ ] Implement `rustc_to_defect_category()` mapping
- [ ] Implement `clippy_to_defect_category()` mapping
- [ ] Add `import_depyler_corpus()` to TrainingDataExtractor
- [ ] Unit tests for all mappings

### Phase 1.5: Alimentar Integration (1 day)
- [ ] Add `alimentar` dependency to Cargo.toml
- [ ] Implement `create_citl_dataloader()` for JSONL/Parquet
- [ ] Implement `train_with_citl()` batched training loop
- [ ] Add Parquet export support to Depyler (coordinate)
- [ ] Benchmark: verify ≥50K examples/sec throughput

### Phase 2: Feature Extension (1 day)
- [ ] Extend `TrainingExample` with new fields
- [ ] Extend `CommitFeatures` to 14 dimensions
- [ ] Ensure backwards compatibility (serde defaults)
- [ ] Update ML pipeline for extended features

### Phase 3: CLI & Integration (1 day)
- [ ] Add `oip train import-depyler` command
- [ ] Add `oip train stats --by-source`
- [ ] Integration tests with sample Depyler export
- [ ] Documentation updates

### Phase 4: Validation (1 day)
- [ ] Generate large corpus from Depyler (5K+ examples)
- [ ] Train hybrid model (commits + CITL)
- [ ] Compare accuracy vs NLP-011 baseline
- [ ] Document results in validation report

## Test Plan

### Unit Tests

```rust
#[test]
fn test_rustc_type_error_mapping() {
    assert_eq!(
        rustc_to_defect_category("E0308"),
        Some(DefectCategory::TypeErrors)
    );
}

#[test]
fn test_clippy_api_misuse_mapping() {
    assert_eq!(
        clippy_to_defect_category("clippy::unwrap_used"),
        Some(DefectCategory::ApiMisuse)
    );
}

#[test]
fn test_import_preserves_confidence() {
    let json = r#"{"error_code":"E0308","confidence":0.95,...}"#;
    let example = import_depyler_example(json).unwrap();
    assert!((example.confidence - 0.95).abs() < 0.001);
}
```

### Integration Tests

```rust
#[test]
fn test_end_to_end_import() {
    let corpus = create_test_corpus(100);
    let mut extractor = TrainingDataExtractor::new(0.75);
    let stats = extractor.import_depyler_corpus(&corpus).unwrap();
    assert_eq!(stats.imported, 100);
}

#[test]
fn test_hybrid_training_improves_accuracy() {
    // Train on commit messages only
    let baseline = train_and_evaluate(commit_examples);

    // Train on commits + CITL
    let hybrid = train_and_evaluate(commit_examples + citl_examples);

    assert!(hybrid.accuracy > baseline.accuracy);
}
```

## Success Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Import works | Pass | Integration test |
| Schema validated | 100% | All fields parsed |
| Mapping coverage | ≥80% | Error codes mapped |
| Accuracy improvement | +10%+ | vs NLP-011 baseline |
| No regressions | Pass | Existing tests green |

## References

1. **Depyler CITL Spec**: `../depyler/docs/specifications/verbose-compiler-diagnostics-citl-spec.md` Section 11
2. **NLP-011 Validation**: `docs/validation/NLP-011-depyler-validation-report.md`
3. **Alimentar**: `../alimentar` - Data loading, batching, and streaming for ML pipelines
4. **Zimmermann et al. (2009)**: Cross-project defect prediction benefits from multi-signal sources
5. **Pan & Yang (2010)**: Transfer learning improves model performance with limited target data
6. **Apache Arrow/Parquet**: Columnar format for efficient ML data loading

## Appendix A: Full Error Code Mapping Table

| Error Code | Rust Meaning | DefectCategory | Confidence |
|------------|--------------|----------------|------------|
| E0308 | Mismatched types | TypeErrors | 0.95 |
| E0277 | Trait not satisfied | TraitBounds | 0.95 |
| E0502 | Cannot borrow as mutable | OwnershipBorrow | 0.95 |
| E0503 | Cannot use after move | OwnershipBorrow | 0.95 |
| E0505 | Cannot move out of borrowed | OwnershipBorrow | 0.95 |
| E0382 | Use of moved value | MemorySafety | 0.90 |
| E0507 | Cannot move out of ref | MemorySafety | 0.90 |
| E0425 | Cannot find value | StdlibMapping | 0.85 |
| E0433 | Cannot find crate/module | StdlibMapping | 0.85 |
| E0412 | Cannot find type | TypeAnnotationGaps | 0.85 |
| E0599 | No method found | ASTTransform | 0.80 |
| E0614 | Cannot dereference | OperatorPrecedence | 0.80 |
| E0615 | Tuple index on non-tuple | ASTTransform | 0.80 |
| E0658 | Unstable feature | ConfigurationErrors | 0.75 |

## Appendix B: Clippy Lint Mapping Table

| Clippy Lint | DefectCategory | Rationale |
|-------------|----------------|-----------|
| `clippy::unwrap_used` | ApiMisuse | Improper error handling |
| `clippy::expect_used` | ApiMisuse | Improper error handling |
| `clippy::panic` | ApiMisuse | Uncontrolled termination |
| `clippy::todo` | LogicErrors | Incomplete implementation |
| `clippy::unreachable` | LogicErrors | Dead code paths |
| `clippy::cognitive_complexity` | PerformanceIssues | Maintainability |
| `clippy::too_many_arguments` | ASTTransform | Function signature |
| `clippy::needless_collect` | IteratorChain | Iterator misuse |
| `clippy::manual_map` | ComprehensionBugs | Pattern translation |
| `clippy::match_single_binding` | ASTTransform | Match misuse |
