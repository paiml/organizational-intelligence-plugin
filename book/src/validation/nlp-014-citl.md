# NLP-014: CITL Integration

This document describes the Compiler-in-the-Loop (CITL) integration implemented in NLP-014 to address the training data limitations identified in NLP-011.

## Problem Statement

From NLP-011 validation:
- Current corpus: 508 training examples
- Classifier accuracy: 54%
- Class imbalance: 8 categories with <10 examples

## Solution: Ground-Truth Labels from Compiler

CITL integration provides deterministic, high-confidence training labels by mapping:

1. **rustc error codes** → DefectCategory (14 mappings)
2. **clippy lints** → DefectCategory (10 mappings)

### Why CITL?

| Source | Confidence | Volume | Cost |
|--------|------------|--------|------|
| Manual labeling | Variable | Low | High |
| Commit message heuristics | 60-80% | Medium | Low |
| **CITL (rustc/clippy)** | **95%+** | **High** | **Zero** |

## Implementation

### Phase 1: Error Code Taxonomy

```rust
pub fn rustc_to_defect_category(code: &str) -> Option<DefectCategory> {
    match code {
        "E0308" => Some(DefectCategory::TypeErrors),
        "E0277" => Some(DefectCategory::TraitBounds),
        "E0502" | "E0503" | "E0505" => Some(DefectCategory::OwnershipBorrow),
        "E0382" | "E0507" => Some(DefectCategory::MemorySafety),
        // ... 14 total mappings
    }
}
```

### Phase 2: Extended TrainingExample

New CITL-specific fields:

```rust
pub struct TrainingExample {
    // Existing fields...
    pub error_code: Option<String>,        // e.g., "E0308"
    pub clippy_lint: Option<String>,       // e.g., "clippy::unwrap_used"
    pub has_suggestion: bool,              // Compiler fix available
    pub suggestion_applicability: Option<SuggestionApplicability>,
    pub source: TrainingSource,            // CommitMessage | DepylerCitl
}
```

### Phase 3: Extended CommitFeatures

Expanded from 8 to 14 dimensions:

```rust
pub struct CommitFeatures {
    // Original 8 dimensions...
    pub error_code_class: u8,           // ErrorCodeClass enum
    pub has_suggestion: u8,             // 0 or 1
    pub suggestion_applicability: u8,   // 0-4 scale
    pub clippy_lint_count: u8,          // Count of clippy warnings
    pub span_line_delta: f32,           // Normalized span size
    pub diagnostic_confidence: f32,     // Compiler confidence
}
```

### Phase 4: alimentar Integration

Efficient data loading with Arrow/Parquet:

```rust
let loader = CitlDataLoader::with_config(CitlLoaderConfig {
    batch_size: 1024,
    shuffle: true,
    min_confidence: 0.75,
    merge_strategy: MergeStrategy::Weighted(2),
    weight: 1.5,
});

let (examples, stats) = loader.load_parquet("depyler-corpus.parquet")?;
```

## Validation Targets

| Metric | Before (NLP-011) | Target | Method |
|--------|------------------|--------|--------|
| Training examples | 508 | 5,000+ | CITL corpus |
| Classifier accuracy | 54% | 75%+ | Ground-truth labels |
| Category coverage | 10/18 | 18/18 | Error code mapping |
| Confidence mean | 72% | 90%+ | Compiler diagnostics |

## Category Mappings

### rustc Error Codes

| Category | Error Codes | Confidence |
|----------|-------------|------------|
| Type Errors | E0308 | 95% |
| Type Annotation Gaps | E0412 | 85% |
| Ownership/Borrow | E0502, E0503, E0505 | 95% |
| Memory Safety | E0382, E0507 | 90% |
| Trait Bounds | E0277 | 95% |
| Stdlib Mapping | E0425, E0433 | 85% |
| AST Transform | E0599, E0615 | 80% |
| Operator Precedence | E0614 | 80% |
| Configuration Errors | E0658 | 75% |

### Clippy Lints

| Category | Lints |
|----------|-------|
| API Misuse | unwrap_used, expect_used, panic |
| Logic Errors | todo, unreachable |
| Performance Issues | cognitive_complexity |
| AST Transform | too_many_arguments, match_single_binding |
| Comprehension Bugs | manual_map |
| Iterator Chain | needless_collect |

## Usage

### CLI Import

```bash
# Import JSONL corpus
oip import-depyler depyler-corpus.jsonl --min-confidence 0.8

# Import Parquet with weighted merge
oip import-depyler depyler-corpus.parquet --merge weighted
```

### Programmatic API

```rust
use organizational_intelligence_plugin::citl::*;

// Load and convert
let (exports, stats) = import_depyler_corpus("corpus.jsonl", 0.75)?;
let examples = convert_to_training_examples(&exports);

// Validate schema
let validation = validate_citl_schema("corpus.parquet")?;
assert!(validation.is_valid);
```

### Example

```bash
cargo run --example citl_import
```

## Test Coverage

- 73 tests for CITL module
- Property-based tests for encoding/decoding
- Integration tests for JSONL and Parquet loading
- Schema validation tests

## Files Modified

- `src/citl.rs` - Core CITL module (new)
- `src/training.rs` - Extended TrainingExample
- `src/features.rs` - Extended CommitFeatures (8 → 14 dims)
- `src/cli.rs` - ImportDepyler command
- `src/cli_handlers.rs` - Handler implementation

## References

- [NLP-011 depyler Validation](./nlp-011-depyler.md)
- [Import Depyler CLI](../cli/import-depyler.md)
- [NLP-014 Specification](../../docs/specifications/nlp-014-citl-integration-spec.md)
