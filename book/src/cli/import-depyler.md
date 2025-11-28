# Import Depyler (CITL Integration)

The `import-depyler` command imports ground-truth training labels from Depyler's Compiler-in-the-Loop (CITL) exports. This enables high-confidence training data extraction from rustc error codes and clippy lints.

## Synopsis

```bash
oip import-depyler <PATH> [OPTIONS]
```

## Description

Depyler generates JSONL exports containing compiler diagnostics with precise error codes and suggested fixes. The `import-depyler` command:

1. Parses CITL export files (JSONL or Parquet format)
2. Maps rustc error codes to OIP's 18-category taxonomy
3. Maps clippy lints to defect categories
4. Generates high-confidence training examples
5. Validates schema compatibility

## Arguments

| Argument | Description |
|----------|-------------|
| `PATH` | Path to Depyler export file (.jsonl or .parquet) |

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--min-confidence` | 0.75 | Minimum confidence threshold for imports |
| `--merge` | append | Merge strategy: `append`, `replace`, or `weighted` |
| `--validate` | true | Validate schema before import |
| `--output` | - | Output path for converted training data |

## Examples

### Basic Import

```bash
# Import from JSONL export
oip import-depyler depyler-export.jsonl

# Import from Parquet file
oip import-depyler depyler-corpus.parquet
```

### With Options

```bash
# High-confidence only with weighted merge
oip import-depyler depyler-export.jsonl \
  --min-confidence 0.9 \
  --merge weighted

# Replace existing training data
oip import-depyler depyler-export.jsonl --merge replace
```

### Validate Schema

```bash
# Check schema compatibility without importing
oip import-depyler depyler-export.jsonl --validate-only
```

## CITL Export Format

Depyler exports diagnostics in JSONL format:

```json
{
  "source_file": "lib.rs",
  "error_code": "E0308",
  "clippy_lint": null,
  "level": "error",
  "message": "mismatched types: expected `i32`, found `&str`",
  "oip_category": null,
  "confidence": 0.95,
  "span": {"line_start": 42, "column_start": 12},
  "suggestion": {"replacement": ".parse::<i32>()", "applicability": "MaybeIncorrect"},
  "timestamp": 1732752000,
  "depyler_version": "3.21.0"
}
```

## Error Code Mappings

### rustc Error Codes

| Error Code | Category | Confidence |
|------------|----------|------------|
| E0308 | Type Errors | 95% |
| E0277 | Trait Bounds | 95% |
| E0502, E0503, E0505 | Ownership/Borrow | 95% |
| E0382, E0507 | Memory Safety | 90% |
| E0425, E0433 | Stdlib Mapping | 85% |
| E0412 | Type Annotation Gaps | 85% |
| E0599, E0615 | AST Transform | 80% |
| E0614 | Operator Precedence | 80% |
| E0658 | Configuration Errors | 75% |

### Clippy Lints

| Lint | Category |
|------|----------|
| `clippy::unwrap_used`, `clippy::expect_used`, `clippy::panic` | API Misuse |
| `clippy::todo`, `clippy::unreachable` | Logic Errors |
| `clippy::cognitive_complexity` | Performance Issues |
| `clippy::too_many_arguments`, `clippy::match_single_binding` | AST Transform |
| `clippy::manual_map` | Comprehension Bugs |
| `clippy::needless_collect` | Iterator Chain |

## Merge Strategies

### Append (Default)

Adds CITL examples to existing training data:

```bash
oip import-depyler depyler-export.jsonl --merge append
```

### Replace

Replaces all training data with CITL examples:

```bash
oip import-depyler depyler-export.jsonl --merge replace
```

### Weighted

Applies weight multiplier to CITL examples for higher influence:

```bash
oip import-depyler depyler-export.jsonl --merge weighted
```

## Output

```
📦 Importing Depyler CITL corpus...
   File: depyler-export.jsonl
   Format: JSONL
   Total records: 5,432

✅ Import complete:
   Imported: 4,891 examples
   Skipped (low confidence): 312
   Skipped (unknown category): 229

📊 Category distribution:
   Type Errors: 1,234 (25.2%)
   Ownership/Borrow: 892 (18.2%)
   API Misuse: 645 (13.2%)
   ...

🎯 Average confidence: 89.3%
```

## See Also

- [Extract Training Data](./extract-training-data.md)
- [Train Classifier](./train-classifier.md)
- [NLP-014 Specification](../validation/nlp-014-citl.md)
