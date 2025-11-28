# Example: CITL Import

This example demonstrates how to import ground-truth training labels from Depyler's Compiler-in-the-Loop (CITL) exports.

## Running the Example

```bash
cargo run --example citl_import
```

## What It Demonstrates

1. **rustc Error Code Mappings** - How error codes map to defect categories
2. **Clippy Lint Mappings** - How clippy lints map to categories
3. **DepylerExport Parsing** - Parsing JSONL export format
4. **TrainingExample Conversion** - Converting exports to training data
5. **CitlDataLoader Configuration** - Configuring the data loader

## Sample Output

```
🔧 CITL Integration - Depyler Ground-Truth Labels

NLP-014: Compiler-in-the-Loop training data extraction

📋 rustc Error Code → DefectCategory Mappings:
   E0308 (Type mismatch) → Type Errors
   E0382 (Use of moved value) → Memory Safety
   E0502 (Borrow while mutably borrowed) → Ownership/Borrow
   E0277 (Trait bound not satisfied) → Trait Bounds
   E0599 (Method not found) → AST Transform
   E0425 (Unresolved name) → Stdlib Mapping

📋 Clippy Lint → DefectCategory Mappings:
   clippy::unwrap_used (Panicking unwrap) → API Misuse
   clippy::expect_used (Panicking expect) → API Misuse
   clippy::todo (Unfinished code) → Logic Errors
   clippy::cognitive_complexity (Complex function) → Performance Issues
   clippy::needless_collect (Unnecessary collect) → Iterator Chain

📦 Parsing Depyler JSONL Export:
   lib.rs [E0308]: mismatched types: expected `i32`, found `&str`
   main.rs [clippy::unwrap_used]: used `unwrap()` on `Option` value
   parser.rs [E0502]: cannot borrow `self.data` as mutable

🔄 Converting to TrainingExamples:
   Converted 3 exports → 3 training examples
   - [Type Errors] mismatched types: expected `i32`, found  (conf: 95%)
   - [API Misuse] used `unwrap()` on `Option` value (conf: 88%)
   - [Ownership/Borrow] cannot borrow `self.data` as mutable (conf: 92%)

⚙️  CitlDataLoader Configuration:
   Batch size: 1024
   Min confidence: 75%
   Shuffle: true
   Weight: 1.5x

📚 CITL-Mapped Defect Categories:
   Type Errors ← E0308, E0412
   Ownership/Borrow ← E0502, E0503, E0505
   Memory Safety ← E0382, E0507
   ...

🎯 CITL Integration Complete!
   Target: 5K+ ground-truth examples → 75%+ classifier accuracy
```

## Code Walkthrough

### Error Code Mapping

```rust
use organizational_intelligence_plugin::citl::rustc_to_defect_category;

// Map rustc error codes to defect categories
let category = rustc_to_defect_category("E0308");
assert_eq!(category, Some(DefectCategory::TypeErrors));
```

### Clippy Lint Mapping

```rust
use organizational_intelligence_plugin::citl::clippy_to_defect_category;

// Map clippy lints to defect categories
let category = clippy_to_defect_category("clippy::unwrap_used");
assert_eq!(category, Some(DefectCategory::ApiMisuse));
```

### Loading CITL Corpus

```rust
use organizational_intelligence_plugin::citl::{CitlDataLoader, CitlLoaderConfig};

let loader = CitlDataLoader::with_config(CitlLoaderConfig {
    batch_size: 1024,
    shuffle: true,
    min_confidence: 0.75,
    merge_strategy: MergeStrategy::Append,
    weight: 1.0,
});

// Load from JSONL
let (examples, stats) = loader.load_jsonl("depyler-export.jsonl")?;

// Load from Parquet (streaming)
let iter = loader.load_parquet("depyler-corpus.parquet")?;
for batch in iter {
    process_batch(batch);
}
```

### Schema Validation

```rust
use organizational_intelligence_plugin::citl::validate_citl_schema;

let validation = validate_citl_schema("export.parquet")?;
if !validation.is_valid {
    println!("Missing fields: {:?}", validation.missing_fields);
}
```

## Related Examples

- [Classify Defects](./classify-defects.md) - Rule-based classification
- [Train Custom Model](./train-custom-model.md) - Training with CITL data
- [depyler Validation](./depyler-validation.md) - Full validation pipeline

## See Also

- [Import Depyler CLI](../cli/import-depyler.md)
- [NLP-014: CITL Integration](../validation/nlp-014-citl.md)
