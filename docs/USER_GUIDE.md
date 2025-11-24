# OIP-GPU User Guide

GPU-accelerated defect pattern analysis for GitHub repositories.

## Quick Start

```bash
# Install
cargo install --path . --bin oip-gpu

# Analyze a repository
oip-gpu analyze --repo rust-lang/rust --output features.db

# Query patterns
oip-gpu query --input features.db "most common defect"

# Train predictor
oip-gpu predict --input features.db --train

# Cluster patterns
oip-gpu cluster --input features.db --k 5
```

## Commands

### analyze

Analyze GitHub repositories for defect patterns.

```bash
# Single repository
oip-gpu analyze --repo owner/repo --output features.db

# Multiple repositories
oip-gpu analyze --repos "owner/repo1,owner/repo2" --output features.db

# Entire organization
oip-gpu analyze --org rust-lang --output features.db

# With options
oip-gpu analyze --repo owner/repo \
  --output features.db \
  --since 2024-01-01 \
  --workers 4
```

**Options:**
- `--repo` - Single repository (owner/repo)
- `--repos` - Comma-separated repositories
- `--org` - GitHub organization name
- `--output` - Output database file (default: oip-gpu.db)
- `--since` - Only analyze commits after date (YYYY-MM-DD)
- `--workers` - Parallel worker count (default: auto)

### correlate

Compute correlation matrices between defect features.

```bash
# Basic correlation
oip-gpu correlate --input features.db --output correlations.json

# With sliding windows (concept drift detection)
oip-gpu correlate --input features.db --window 6months --output drift.json

# Force SIMD backend
oip-gpu correlate --input features.db --backend simd
```

**Options:**
- `--input` - Input database from analyze
- `--output` - Output file for correlation matrix
- `--window` - Sliding window size for drift detection
- `--backend` - Compute backend (auto/gpu/simd/cpu)

### predict

Train and use defect prediction models.

```bash
# Train model
oip-gpu predict --input features.db --train --model model.bin

# Make predictions
oip-gpu predict --input features.db --model model.bin --predict

# With SMOTE balancing
oip-gpu predict --input features.db --train --smote --model model.bin
```

**Options:**
- `--input` - Input database
- `--model` - Model file path
- `--train` - Train new model
- `--predict` - Make predictions
- `--smote` - Apply SMOTE oversampling

### cluster

Discover defect patterns using K-means clustering.

```bash
# Basic clustering
oip-gpu cluster --input features.db --k 5

# With output
oip-gpu cluster --input features.db --k 10 --output clusters.json

# Elbow method (find optimal k)
oip-gpu cluster --input features.db --elbow
```

**Options:**
- `--input` - Input database
- `--k` - Number of clusters (default: 5)
- `--output` - Output file for cluster assignments
- `--elbow` - Run elbow method to find optimal k

### query

Natural language queries on defect data.

```bash
# Common queries
oip-gpu query --input features.db "most common defect"
oip-gpu query --input features.db "count by category"
oip-gpu query --input features.db "show all defects"
```

**Supported Queries:**
- "most common defect" - Show most frequent defect category
- "count by category" - Count defects per category
- "show all" - List all defects

### benchmark

Run performance benchmarks.

```bash
# All benchmarks
oip-gpu benchmark --suite all

# Specific suite
oip-gpu benchmark --suite correlation
oip-gpu benchmark --suite feature_extraction
oip-gpu benchmark --suite storage
```

## Configuration

### Configuration File

Create `.oip.yaml` in your project root:

```yaml
analysis:
  max_commits: 1000
  workers: 4
  cache_dir: ".oip-cache"
  include_merges: false

ml:
  n_trees: 100
  max_depth: 10
  k_clusters: 5
  smote_k: 5
  smote_ratio: 0.5

storage:
  default_output: "oip-gpu.db"
  compress: true
  batch_size: 1000

compute:
  backend: "auto"
  workgroup_size: 256
  gpu_enabled: true

logging:
  level: "info"
  json: false
```

### Environment Variables

Override configuration with environment variables:

```bash
# Analysis
export OIP_MAX_COMMITS=2000
export OIP_WORKERS=8
export OIP_CACHE_DIR="/tmp/oip-cache"

# ML
export OIP_K_CLUSTERS=10

# Compute
export OIP_BACKEND=simd
export OIP_GPU_ENABLED=false

# Logging
export OIP_LOG_LEVEL=debug
export OIP_LOG_JSON=true

# GitHub (for private repos)
export GITHUB_TOKEN=ghp_...
```

## Global Options

Available for all commands:

```bash
oip-gpu --verbose <command>     # Enable debug logging
oip-gpu --backend gpu <command>  # Force GPU backend
oip-gpu --backend simd <command> # Force SIMD backend
oip-gpu --config path.yaml <command> # Custom config file
```

## Compute Backends

### Auto (Default)

Automatically selects best available backend:
1. GPU (if available and enabled)
2. SIMD (AVX-512 > AVX2 > scalar)

### GPU

Requires:
- Vulkan 1.2+ (Linux), Metal (macOS), or DirectX 12 (Windows)
- 2GB+ VRAM recommended
- Compile with `--features gpu`

```bash
cargo build --release --features gpu
oip-gpu --backend gpu analyze --repo owner/repo
```

### SIMD

CPU-based SIMD acceleration:
- AVX-512 (Intel Skylake+, AMD Zen4+)
- AVX2 (Intel Haswell+, AMD Excavator+)
- Scalar fallback

```bash
oip-gpu --backend simd analyze --repo owner/repo
```

## Error Handling

### Common Errors

**Repository not found:**
```
Error: Repository not found: owner/repo
Hint: Check the repository name format (owner/repo) and ensure it exists
```

**Authentication required:**
```
Error: Authentication required: GitHub API rate limit
Hint: Set GITHUB_TOKEN environment variable
```

**GPU unavailable:**
```
Error: GPU not available: No suitable adapter found
Hint: Use --backend simd for CPU fallback, or install GPU drivers
```

### Recovery

Most errors are recoverable. Check:
1. Repository name format (owner/repo)
2. GITHUB_TOKEN for private repos
3. GPU drivers for GPU backend
4. Input file exists for analysis commands

## Examples

### Analyze Open Source Project

```bash
# Analyze rust-lang/rust
oip-gpu analyze --repo rust-lang/rust --output rust.db

# Find patterns
oip-gpu cluster --input rust.db --k 5 --output patterns.json

# Query results
oip-gpu query --input rust.db "most common defect"
```

### Compare Multiple Repos

```bash
# Analyze multiple repos
oip-gpu analyze --repos "tokio-rs/tokio,async-rs/async-std" --output async.db

# Compute correlations
oip-gpu correlate --input async.db --output correlations.json
```

### Detect Concept Drift

```bash
# Analyze with time windows
oip-gpu analyze --repo owner/repo --output features.db

# Detect drift over 6-month windows
oip-gpu correlate --input features.db --window 6months --output drift.json
```

### Train Prediction Model

```bash
# Analyze training data
oip-gpu analyze --org my-org --output training.db

# Train with SMOTE balancing
oip-gpu predict --input training.db --train --smote --model defect-model.bin

# Predict on new data
oip-gpu predict --input new-data.db --model defect-model.bin --predict
```

## Performance Tips

1. **Use SIMD for small datasets** (<10K features)
2. **Use GPU for large datasets** (>10K features)
3. **Limit commits** with `--since` for faster analysis
4. **Increase workers** for parallel repository analysis
5. **Enable caching** to avoid re-cloning repos

## Troubleshooting

### Slow Analysis

```bash
# Limit commits
oip-gpu analyze --repo owner/repo --since 2024-01-01

# Increase workers
oip-gpu analyze --repo owner/repo --workers 8
```

### Out of Memory

```bash
# Reduce batch size in config
storage:
  batch_size: 500

# Or use smaller commit limit
oip-gpu analyze --repo owner/repo --since 2024-06-01
```

### GPU Not Detected

```bash
# Check GPU support
vulkaninfo  # Linux
# or
system_profiler SPDisplaysDataType  # macOS

# Fall back to SIMD
oip-gpu --backend simd analyze --repo owner/repo
```

## API Reference

See `docs/GPU_QUICKSTART.md` for library API documentation.

## Support

- Issues: https://github.com/anthropics/claude-code/issues
- Docs: `docs/specifications/GPU-correlation-predictions-spec.md`
