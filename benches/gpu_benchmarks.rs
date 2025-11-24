//! GPU-accelerated benchmarks
//!
//! Validates Section 5.4: Performance Optimization
//! Measures SIMD speedup (Phase 1) and GPU speedup (Phase 2)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use organizational_intelligence_plugin::{
    correlation::pearson_correlation,
    features::{CommitFeatures, FeatureExtractor},
    storage::FeatureStore,
};
use trueno::Vector;

/// Benchmark: Pearson correlation computation
fn bench_pearson_correlation(c: &mut Criterion) {
    let mut group = c.benchmark_group("correlation");

    for size in [100, 1000, 10_000] {
        // Generate test data
        let data1: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let data2: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();

        let v1 = Vector::from_slice(&data1);
        let v2 = Vector::from_slice(&data2);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| pearson_correlation(black_box(&v1), black_box(&v2)))
        });
    }

    group.finish();
}

/// Benchmark: Feature extraction
fn bench_feature_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_extraction");

    let extractor = FeatureExtractor::new();

    for count in [10, 100, 1000] {
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &cnt| {
            b.iter(|| {
                for i in 0..cnt {
                    let _ = extractor.extract(
                        black_box(1),
                        black_box(3),
                        black_box(100),
                        black_box(50),
                        black_box(1700000000 + i),
                    );
                }
            })
        });
    }

    group.finish();
}

/// Benchmark: Feature storage bulk insert
fn bench_storage_bulk_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage");

    for size in [100, 1000, 10_000] {
        // Generate test features
        let features: Vec<CommitFeatures> = (0..size)
            .map(|i| CommitFeatures {
                defect_category: (i % 10) as u8,
                files_changed: (i % 5) as f32,
                lines_added: 100.0,
                lines_deleted: 50.0,
                complexity_delta: 0.0,
                timestamp: 1700000000.0 + i as f64,
                hour_of_day: (i % 24) as u8,
                day_of_week: (i % 7) as u8,
            })
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let mut store = FeatureStore::new().unwrap();
                store.bulk_insert(black_box(features.clone())).unwrap();
            })
        });
    }

    group.finish();
}

/// Benchmark: Query by category
fn bench_storage_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("query");

    // Pre-populate store
    let mut store = FeatureStore::new().unwrap();
    let features: Vec<CommitFeatures> = (0..10_000)
        .map(|i| CommitFeatures {
            defect_category: (i % 10) as u8,
            files_changed: (i % 5) as f32,
            lines_added: 100.0,
            lines_deleted: 50.0,
            complexity_delta: 0.0,
            timestamp: 1700000000.0 + i as f64,
            hour_of_day: (i % 24) as u8,
            day_of_week: (i % 7) as u8,
        })
        .collect();
    store.bulk_insert(features).unwrap();

    group.bench_function("query_category", |b| {
        b.iter(|| store.query_by_category(black_box(3)))
    });

    group.finish();
}

/// Benchmark: Vector conversion for GPU
fn bench_to_vectors(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_conversion");

    for size in [100, 1000, 10_000] {
        let mut store = FeatureStore::new().unwrap();
        let features: Vec<CommitFeatures> = (0..size)
            .map(|i| CommitFeatures {
                defect_category: (i % 10) as u8,
                files_changed: (i % 5) as f32,
                lines_added: 100.0,
                lines_deleted: 50.0,
                complexity_delta: 0.0,
                timestamp: 1700000000.0 + i as f64,
                hour_of_day: (i % 24) as u8,
                day_of_week: (i % 7) as u8,
            })
            .collect();
        store.bulk_insert(features).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| store.to_vectors())
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_pearson_correlation,
    bench_feature_extraction,
    bench_storage_bulk_insert,
    bench_storage_query,
    bench_to_vectors
);
criterion_main!(benches);
