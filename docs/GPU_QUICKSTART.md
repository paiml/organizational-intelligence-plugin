# GPU-Accelerated Analysis Quick Start

Quick guide to using oip-gpu for defect pattern analysis.

## Installation

```bash
cargo install --path . --bin oip-gpu
```

## GPU Hardware Requirements

### Phase 1 (Current - SIMD Only)
- **CPU**: Any x86_64 with AVX2 or ARM with NEON
- **No GPU required**: Phase 1 uses CPU SIMD via trueno
- **Backends**: Automatically selects best available (AVX-512 > AVX2 > scalar)

### Phase 2 (GPU Acceleration - In Progress)
- **GPU**: Any GPU with Vulkan 1.2+, Metal, or DirectX 12 support
- **Compilation**: Requires `--features gpu` flag
- **Platforms**: Linux (Vulkan), macOS (Metal), Windows (DX12/Vulkan)
- **Memory**: 2GB+ VRAM recommended for large correlation matrices
- **Fallback**: Gracefully degrades to SIMD if GPU unavailable

### Compile with GPU Support

```bash
# Build with GPU features (requires GPU hardware)
cargo build --release --features gpu

# Run GPU-accelerated correlation
oip-gpu correlate --backend gpu --input features.db
```

### Check GPU Availability

```bash
# Will report GPU adapter info or fallback to SIMD
oip-gpu benchmark --suite correlation --backend gpu
```

### GPU Implementation Status

**✅ Complete (Phase 1):**
- SIMD-accelerated correlation via trueno
- Feature extraction and storage
- Columnar in-memory database
- Natural language queries
- Benchmark framework

**🚧 In Progress (Phase 2 - PHASE2-001):**
- GPU correlation matrix computation
- wgpu backend initialization ✅
- WGSL compute shader ✅
- GPU buffer management ✅
- Bind groups and dispatch (TODO)
- Result readback (TODO)

**📋 Planned (Phase 2):**
- GPU/CPU equivalence tests (tolerance 1e-4)
- Sliding window correlation (concept drift)
- 20-50x speedup validation

## Quick Start

### 1. Analyze a Repository

```bash
# Analyze rust-lang/rust (max 1000 commits)
oip-gpu analyze --repo rust-lang/rust --output features.db

# Output:
# 🔍 Analyzing repository: rust-lang/rust
# 📊 Analyzing commits (max 1000)...
# ✅ Found 8 defect categories
# 🔧 Extracting features for GPU processing...
# ✅ Extracted 245 feature vectors
# 💾 Saving to features.db...
# ✨ Analysis complete!
```

### 2. Run Benchmarks

```bash
# Run all benchmarks
oip-gpu benchmark --suite all

# Run specific suite
oip-gpu benchmark --suite correlation
```

### 3. Force Backend

```bash
# Force SIMD backend
oip-gpu analyze --repo rust-lang/rust --backend simd --output out.db

# Force GPU backend (Phase 2)
oip-gpu analyze --repo rust-lang/rust --backend gpu --output out.db
```

## API Reference

### FeatureExtractor

Converts defect metadata into GPU-friendly numerical features.

```rust
use organizational_intelligence_plugin::features::FeatureExtractor;

let extractor = FeatureExtractor::new();
let features = extractor.extract(
    1,              // category (0-9)
    3,              // files changed
    100,            // lines added
    50,             // lines deleted
    1700000000,     // timestamp
)?;

// Convert to vector for GPU
let vec = features.to_vector();
assert_eq!(vec.len(), 8);  // 8 dimensions
```

### FeatureStore

GPU-friendly columnar storage for features.

```rust
use organizational_intelligence_plugin::storage::FeatureStore;

let mut store = FeatureStore::new()?;

// Insert single feature
store.insert(features)?;

// Bulk insert (optimized)
store.bulk_insert(vec![f1, f2, f3])?;

// Query by category
let cat1 = store.query_by_category(1)?;

// Convert to GPU vectors
let vectors = store.to_vectors();
```

### Correlation Analysis

SIMD-accelerated Pearson correlation.

```rust
use organizational_intelligence_plugin::correlation::pearson_correlation;
use trueno::Vector;

let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);

let r = pearson_correlation(&x, &y)?;
// r = 1.0 (perfect correlation)
```

## Performance

Phase 1 SIMD performance (trueno backend):

| Operation | Size | Time | Notes |
|-----------|------|------|-------|
| Pearson correlation | 10K | ~50µs | AVX-512/AVX2 |
| Feature extraction | 1K | ~200µs | Temporal parsing |
| Bulk insert | 10K | ~1ms | In-memory |
| Query by category | 10K | ~10µs | Linear scan |

Run benchmarks: `cargo bench`

## Architecture

```
GitHub → OrgAnalyzer → FeatureExtractor → FeatureStore → GPU Vectors
         (OIP)         (TASK-001)         (TASK-004)     (Phase 2)
```

## Next Steps

- **Phase 2**: GPU acceleration (correlation, clustering)
- **Phase 3**: ML models (prediction, anomaly detection)
- See: `docs/specifications/GPU-correlation-predictions-spec.md`

## Troubleshooting

### "pmat not installed"

TDG analysis is optional. Install pmat:
```bash
cargo install pmat
```

### "Repository not found"

Ensure repo is public or set GITHUB_TOKEN:
```bash
export GITHUB_TOKEN=ghp_...
```

### Slow analysis

Limit commits:
```bash
oip-gpu analyze --repo owner/repo --output out.db
# Default: 1000 commits (Phase 1)
```

## Examples

See `examples/` directory for full usage examples.
