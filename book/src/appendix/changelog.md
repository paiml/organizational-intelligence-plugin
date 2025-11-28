# Changelog

All notable changes to OIP are documented here.

## [0.2.0] - 2024-11-28

### Added

#### NLP-014: CITL Integration
- **rustc Error Code Mappings**: 14 error codes mapped to DefectCategory
  - E0308 → Type Errors
  - E0277 → Trait Bounds
  - E0502/E0503/E0505 → Ownership/Borrow
  - E0382/E0507 → Memory Safety
  - E0425/E0433 → Stdlib Mapping
  - And more...

- **Clippy Lint Mappings**: 10 lints mapped to DefectCategory
  - unwrap_used/expect_used/panic → API Misuse
  - todo/unreachable → Logic Errors
  - cognitive_complexity → Performance Issues

- **Extended TrainingExample**: New CITL-specific fields
  - `error_code`: rustc error code (e.g., "E0308")
  - `clippy_lint`: clippy lint name
  - `has_suggestion`: Whether compiler fix is available
  - `suggestion_applicability`: Fix confidence level
  - `source`: Training data source (CommitMessage/DepylerCitl)

- **Extended CommitFeatures**: 8 → 14 dimensions
  - error_code_class, has_suggestion, suggestion_applicability
  - clippy_lint_count, span_line_delta, diagnostic_confidence

- **alimentar DataLoader Integration**
  - `CitlDataLoader`: Efficient batch loading
  - Parquet format support with Arrow
  - JSONL streaming support
  - Schema validation

- **Merge Strategies**: Append, Replace, Weighted

- **CLI Command**: `oip import-depyler`

- **Example**: `cargo run --example citl_import`

- **Documentation**:
  - CLI: import-depyler.md
  - Validation: nlp-014-citl.md
  - Example: citl-import.md

### Changed

- Extended feature vector dimension from 8 to 14
- Updated coverage to exclude alimentar dependency (92.70%)

### Fixed

- Coverage calculation now excludes path dependencies

## [0.1.0] - 2024-11-01

### Added

- Initial release
- 18-category defect taxonomy
- Rule-based classifier (Tier 1)
- TF-IDF + Random Forest classifier (Tier 2)
- GPU-accelerated correlation matrix
- CLI: analyze, extract-training-data, train-classifier, summarize, review-pr
- WASM browser demo
- 90% test coverage enforcement
