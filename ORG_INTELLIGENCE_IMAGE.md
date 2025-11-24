# Organizational Intelligence Plugin - Project Image

**Generated**: 2025-11-24
**Repository**: https://github.com/paiml/organizational-intelligence-plugin
**Purpose**: GPU-accelerated defect pattern analysis for GitHub organizations

---

## Executive Summary

The Organizational Intelligence Plugin (OIP-GPU) is a production-ready tool for analyzing defect patterns in software repositories using GPU acceleration and SIMD optimizations.

### Project Health
- **Overall Score**: 68.3%
- **Maintainability Index**: 70.0
- **Test Coverage**: 65.0% (124 passing tests)
- **Code Quality**: Production-ready with zero clippy warnings

### Architecture
- **Language**: Rust (Edition 2021)
- **Lines of Code**: ~10,000+ (across 35 primary files)
- **Functions**: 83 analyzed
- **GPU Support**: WebGPU (wgpu 0.19) + SIMD fallback

---

## System Architecture

### Core Modules

```
organizational-intelligence-plugin/
├── src/
│   ├── analyzer.rs         # Repository analysis orchestration
│   ├── classifier.rs       # Defect classification (10 categories)
│   ├── features.rs         # Feature extraction (8 dimensions)
│   ├── correlation.rs      # SIMD/GPU correlation computation
│   ├── ml.rs              # K-means clustering
│   ├── sliding_window.rs  # Concept drift detection
│   ├── storage.rs         # JSON persistence layer
│   ├── gpu_main.rs        # CLI interface
│   ├── perf.rs            # Performance utilities
│   ├── config.rs          # Configuration management
│   ├── error.rs           # Error handling with recovery
│   └── observability.rs   # Tracing/metrics
├── benches/              # Criterion benchmarks
├── tests/                # Integration tests
└── docs/                 # Specifications & guides
```

### Data Flow

```
Local Git Repo → Classifier → Feature Extractor → Storage (JSON)
                                      ↓
                              GPU Correlation ← SIMD Fallback
                                      ↓
                              Query Engine → Results (Table/JSON/CSV)
```

---

## Capabilities

### 1. Defect Classification
**10 Categories** with rule-based pattern matching:

| Category | Confidence | Pattern Examples |
|----------|------------|------------------|
| Memory Safety | 85% | "use after free", "null pointer", "buffer overflow" |
| Concurrency | 80% | "race condition", "deadlock", "thread safety" |
| Security | 90% | "sql injection", "xss", "authentication" |
| Logic Errors | 70% | "off by one", "incorrect logic", "wrong condition" |
| API Misuse | 75% | "wrong parameter", "missing error handling" |
| Resource Leaks | 80% | "file handle leak", "not closed" |
| Type Errors | 75% | "type mismatch", "casting error" |
| Configuration | 70% | "config", "environment variable" |
| Performance | 65% | "slow", "inefficient", "n+1 query" |
| Integration | 70% | "compatibility", "version mismatch" |

### 2. Feature Extraction
**8-dimensional feature vectors** for GPU processing:
- Defect category (0-9)
- Files changed
- Lines added
- Lines deleted
- Complexity delta
- Timestamp
- Hour of day
- Day of week

### 3. Performance Optimization

**Batch Processing**:
- `BatchFeatureExtractor`: Configurable batch sizes (default 1000)
- Performance statistics tracking
- Streaming support

**Caching**:
- `CachedCorrelation`: LRU cache with 5-minute TTL
- Symmetric key optimization for 2x cache hits
- Cache hit rate tracking

**Compute Backends**:
- GPU: WebGPU via wgpu (Vulkan/Metal/DX12)
- SIMD: AVX-512 > AVX2 > scalar fallback
- Auto-selection based on availability

### 4. Query Interface

**Natural Language Queries**:
- `"show all defects"` - Full distribution
- `"most common defect"` - Frequency-sorted
- `"count by category"` - Category breakdown

**Output Formats**:
- Table (CLI-friendly)
- JSON (programmatic)
- CSV (spreadsheet)
- YAML (config-friendly)

---

## Quality Metrics

### Test Coverage
- **Total Tests**: 124 passing, 6 ignored (GPU hardware tests)
- **Integration Tests**: 20 E2E CLI tests
- **Unit Tests**: 104 module tests
- **Coverage**: 65% (target: 95% per certeza standards)

### Performance Characteristics

| Operation | Complexity | Optimized |
|-----------|-----------|-----------|
| Feature Extraction | O(n) | ✅ Batch |
| Pearson Correlation | O(n²) | ✅ Cached |
| K-means Clustering | O(n³) | ⚠️ High |
| Sliding Window | O(n³) | ⚠️ High |

**Note**: O(n³) operations identified in `src/ml.rs` and `src/sliding_window.rs` for optimization in future sprints.

### Code Quality

**Strengths**:
- ✅ Zero clippy warnings
- ✅ Comprehensive error handling with recovery hints
- ✅ Structured logging with tracing
- ✅ Pre-commit hooks (fmt, clippy, tests)

**Technical Debt Hotspots**:
1. `src/ml.rs` - 7 defects (TDG: 2.5)
2. `src/sliding_window.rs` - 9 defects (TDG: 2.5)
3. `src/summarizer.rs` - 6 defects (TDG: 2.5)

**Median Metrics**:
- Cyclomatic Complexity: 1.00 (excellent)
- Cognitive Complexity: 0.00 (excellent)
- Function Count: 83
- Provability: 43%

---

## Development Process

### CI/CD Pipeline
- **Security Audit**: cargo-audit vulnerability scanning
- **MSRV**: Rust 1.75.0 minimum
- **Multi-platform**: Linux, macOS, Windows
- **Release Automation**: GitHub Actions with checksums

### Quality Gates (Pre-commit)
1. Code formatting (`cargo fmt`)
2. Linting (`cargo clippy -- -D warnings`)
3. Fast tests (`cargo test`)
4. Pre-commit hooks enforced

### Configuration
- **File**: `.oip.yaml` (YAML config)
- **Environment**: `OIP_*` variable overrides
- **Logging**: Configurable levels (trace/debug/info/warn/error)

---

## Sprint History

### Sprint v0.3.0: GPU Acceleration (6 commits)
- GPU correlation computation
- GPU/CPU equivalence tests
- Sliding window for concept drift
- SMOTE for class imbalance
- ML models (K-means)
- Performance benchmarks

### Sprint v0.4.0: Production Hardening (6 commits)
- E2E CLI integration tests (20 tests)
- Error handling with recovery hints
- Observability (tracing/metrics)
- Configuration management
- Performance utilities
- User guide documentation

### Sprint v0.5.0: CI/CD (2 commits)
- Security audit workflow
- MSRV verification (1.75.0)
- Release automation
- Multi-platform builds

### Sprint v0.6.0: Performance Optimization (1 commit)
- Batch feature extractor
- Cached correlation
- 11 new tests

### UAT: Production Validation (1 commit)
- Local repository analysis
- JSON persistence
- Tested on depyler (500-1000 commits)

**Total Deliverables**: 16 commits, 124 tests

---

## Usage Examples

### Analyze a Local Repository
```bash
./target/release/oip-gpu analyze \
    --local ../depyler \
    --output depyler.db \
    --max-commits 1000
```

### Query Results
```bash
# Show all defects
./target/release/oip-gpu query --input depyler.db "show all defects"

# Most common defects
./target/release/oip-gpu query --input depyler.db "most common defect"

# Export to JSON
./target/release/oip-gpu query \
    --input depyler.db \
    --format json \
    --export results.json \
    "count by category"
```

### Demo Script
```bash
./demo.sh  # Interactive 5-step demonstration
```

---

## Real-World Results: depyler Analysis

**Dataset**: 1000 commits from depyler repository

**Distribution**:
- Category 0 (General): 626 (62.6%)
- Category 9 (Documentation): 120 (12.0%)
- Category 5 (Performance): 84 (8.4%)
- Category 8 (Refactoring): 72 (7.2%)
- Category 6 (Security): 55 (5.5%)
- Other: 43 (4.3%)

**Storage**: 160 KB for 1000 feature vectors (8 dimensions each)

**Performance**: ~100 commits/second extraction rate

---

## Future Roadmap

### Short-term (Q1 2025)
- [ ] Reduce O(n³) complexity in ML/sliding window modules
- [ ] Increase test coverage to 95%
- [ ] Parquet storage backend (replace JSON)
- [ ] ARM64 (Apple Silicon) release builds

### Mid-term (Q2 2025)
- [ ] Real-time correlation computation
- [ ] Shell completions (bash/zsh/fish)
- [ ] WebAssembly build for browser demos
- [ ] Comprehensive benchmark suite

### Long-term (2025+)
- [ ] Distributed analysis for large organizations
- [ ] Machine learning model training pipeline
- [ ] Real-time drift detection alerts
- [ ] Integration with GitHub Actions

---

## Dependencies

**Core**:
- `trueno` (0.7.1) - SIMD tensor operations
- `trueno-db` (0.3.2) - Columnar storage
- `wgpu` (0.19) - GPU compute
- `tokio` (1.40) - Async runtime
- `clap` (4.5) - CLI framework

**Analysis**:
- `git2` (0.20) - Git repository access
- `octocrab` (0.39) - GitHub API
- `chrono` (0.4) - Date/time handling

**Quality**:
- `tracing` (0.1) - Structured logging
- `criterion` (0.5) - Benchmarking
- `anyhow` (1.0) - Error handling

---

## Team & Contacts

**Authors**: OIP Team
**License**: MIT
**Repository**: https://github.com/paiml/organizational-intelligence-plugin
**Documentation**: See `docs/USER_GUIDE.md` and `QUICKSTART.md`

**Contributing**: Follow pre-commit quality gates, maintain 95% test coverage, adhere to TDD practices.

---

*This organizational image was generated using pmat analysis tools and represents the current state of the project as of 2025-11-24.*
