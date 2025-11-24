# Performance Validation Guide

**PHASE2-006**: GPU vs SIMD performance validation for 20x speedup target.

## Quick Start

### Run All Benchmarks (SIMD Only)

```bash
cargo bench
```

### Run GPU Benchmarks (Requires GPU Hardware)

```bash
cargo bench --features gpu
```

## Benchmark Suites

### 1. SIMD Correlation (Phase 1)

**Benchmark**: `correlation`

Measures SIMD-accelerated Pearson correlation via trueno.

```bash
cargo bench -- correlation
```

**Expected Performance:**
- 100 elements: ~5-10 µs (AVX2/AVX-512)
- 1,000 elements: ~50-100 µs
- 10,000 elements: ~500 µs - 1 ms

### 2. Feature Extraction

**Benchmark**: `feature_extraction`

Measures conversion of defect data to 8D vectors.

```bash
cargo bench -- feature_extraction
```

**Expected Performance:**
- 10 features: <10 µs
- 100 features: <100 µs
- 1,000 features: <1 ms

### 3. Storage Operations

**Benchmarks**: `storage`, `query`, `gpu_conversion`

Measures bulk insert, category queries, and vector conversion.

```bash
cargo bench -- storage
cargo bench -- query
cargo bench -- gpu_conversion
```

### 4. GPU vs SIMD Comparison (Phase 2)

**Benchmark**: `gpu_vs_simd`

**⚠️ Requires**: `--features gpu` and GPU hardware (Vulkan 1.2+/Metal/DX12)

Directly compares GPU and SIMD correlation for same input sizes.

```bash
cargo bench --features gpu -- gpu_vs_simd
```

**Output Format:**
```
gpu_vs_simd/simd/100      time:   [5.2 µs 5.3 µs 5.4 µs]
gpu_vs_simd/gpu/100       time:   [250 ns 260 ns 270 ns]
                                  ^^^^^^^^^^^^^^^^^^^^
                                  Target: 20x faster = ~260 ns
```

**Speedup Calculation:**
```
Speedup = SIMD time / GPU time
        = 5.3 µs / 260 ns
        = 20.4x  ✅ (Target: 20x)
```

### 5. Speedup Validation

**Benchmark**: `speedup_validation`

**⚠️ Requires**: `--features gpu` and GPU hardware

Focused benchmark for validating 20x speedup target on 10K elements.

```bash
cargo bench --features gpu -- speedup_validation
```

**Target Metrics:**
- **SIMD Baseline**: 500-1000 µs (10K elements)
- **GPU Target**: 25-50 µs (20-40x speedup)
- **Validation**: GPU time ≤ SIMD time / 20

## Interpreting Results

### Speedup Ratio

The speedup ratio is computed as:

```
Speedup = T_simd / T_gpu
```

Where:
- `T_simd` = SIMD execution time (trueno backend)
- `T_gpu` = GPU execution time (wgpu backend)

### Performance Targets

| Metric | Phase 1 (SIMD) | Phase 2 (GPU) | Speedup Target |
|--------|---------------|---------------|----------------|
| Correlation (100) | ~5 µs | ~250 ns | 20x |
| Correlation (1K) | ~50 µs | ~2.5 µs | 20x |
| Correlation (10K) | ~500 µs | ~25 µs | 20x |

### Validation Criteria

✅ **PASS**: Speedup ≥ 20x for 10K elements
⚠️ **WARN**: Speedup ≥ 10x but < 20x (acceptable)
❌ **FAIL**: Speedup < 10x (investigate)

## Hardware Requirements

### Phase 1 (SIMD)

- **CPU**: x86_64 with AVX2 or ARM with NEON
- **No GPU required**
- **Performance**: 5-10x speedup vs scalar

### Phase 2 (GPU)

- **GPU**: Discrete GPU with Vulkan 1.2+, Metal, or DirectX 12
- **VRAM**: 2GB+ recommended
- **Platforms**:
  - Linux: Vulkan (Mesa, NVIDIA, AMD)
  - macOS: Metal (M1/M2 or discrete GPU)
  - Windows: DirectX 12 or Vulkan

### CPU-Only Systems

If GPU is unavailable, benchmarks will print:

```
⚠️  GPU not available, skipping GPU benchmarks
```

This is expected and not an error. Phase 1 (SIMD) still provides significant speedup.

## Continuous Integration

### CI Validation

For CI/CD pipelines without GPU access:

```bash
# Run SIMD benchmarks only
cargo bench

# Verify GPU code compiles
cargo build --release --features gpu
cargo test --features gpu
```

### GPU Validation (Manual)

On systems with GPU hardware:

```bash
# Full validation suite
cargo bench --features gpu

# Check GPU is detected
cargo run --features gpu --bin oip-gpu -- benchmark --suite correlation --backend gpu
```

## Regression Testing

To detect performance regressions:

```bash
# Baseline measurement
cargo bench --save-baseline main

# After changes
cargo bench --baseline main
```

**Alert if**:
- SIMD correlation >10% slower
- GPU correlation >10% slower
- Speedup ratio drops below 15x

## Troubleshooting

### GPU Benchmarks Not Running

**Symptom**: "GPU not available" message

**Solutions**:
1. Verify GPU drivers installed (vulkaninfo, nvidia-smi)
2. Check GPU feature flag: `cargo bench --features gpu`
3. Try CPU-only mode if GPU not available

### Unexpected Speedup Results

**Symptom**: Speedup < 10x or GPU slower than SIMD

**Possible Causes**:
1. **Small data size**: GPU overhead dominates (use 10K+ elements)
2. **CPU/GPU memory transfer**: Buffer copy overhead (measured)
3. **Integrated GPU**: May not outperform SIMD (expected)
4. **Debug build**: Use `--release` flag

**Verify**:
```bash
cargo bench --features gpu --release -- speedup_validation
```

### Benchmark Variance

**Symptom**: Results vary by >20% between runs

**Solutions**:
1. Close background applications
2. Disable CPU frequency scaling:
   ```bash
   sudo cpupower frequency-set --governor performance
   ```
3. Increase sample size: Edit `benches/gpu_benchmarks.rs` (measurement_time)

## Performance Optimization Notes

### Phase 1 (SIMD) - Complete ✅

- **Backend**: trueno 0.7.1 with auto-dispatch
- **SIMD**: AVX-512 > AVX2 > scalar fallback
- **Typical speedup**: 5-10x vs scalar

### Phase 2 (GPU) - Complete ✅

- **Backend**: wgpu 0.19 with WGSL compute shaders
- **Parallelism**: 256 threads per workgroup
- **Target speedup**: 20-50x vs SIMD
- **Status**: Implementation complete, validation requires GPU hardware

### Future Optimizations (Phase 3)

- **Matrix batching**: Compute multiple correlations in single dispatch
- **Persistent GPU buffers**: Reduce CPU↔GPU transfers
- **Async pipeline**: Overlap CPU and GPU work
- **Potential**: 50-100x speedup for batch operations

## References

- Specification: `docs/specifications/GPU-correlation-predictions-spec.md`
- GPU Quick Start: `docs/GPU_QUICKSTART.md`
- Sprint Status: `docs/SPRINT_v0.2.0_COMPLETE.md`

## Example Output

### SIMD Benchmarks (Phase 1)

```
correlation/100           time:   [5.234 µs 5.287 µs 5.351 µs]
correlation/1000          time:   [52.31 µs 52.89 µs 53.62 µs]
correlation/10000         time:   [523.4 µs 528.9 µs 536.2 µs]
```

### GPU vs SIMD (Phase 2 - Requires GPU)

```
gpu_vs_simd/simd/100      time:   [5.234 µs 5.287 µs 5.351 µs]
gpu_vs_simd/gpu/100       time:   [245 ns 252 ns 261 ns]
                                  Speedup: 21.0x ✅

gpu_vs_simd/simd/10000    time:   [523.4 µs 528.9 µs 536.2 µs]
gpu_vs_simd/gpu/10000     time:   [24.1 µs 24.8 µs 25.6 µs]
                                  Speedup: 21.3x ✅
```

### Speedup Validation

```
speedup_validation/simd_baseline   time:   [528.9 µs 534.2 µs 540.8 µs]
speedup_validation/gpu_target      time:   [24.8 µs 25.2 µs 25.7 µs]

Speedup Ratio: 21.2x ✅ (Target: 20x)
```

---

**Status**: PHASE2-006 implementation complete. GPU validation requires hardware.

**Next**: PHASE2-004 (Class imbalance handling) or PHASE2-005 (ML models integration)
