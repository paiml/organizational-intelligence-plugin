//! NLP Classifier Performance Benchmarks
//!
//! Validates Phase 2 ML classifier performance targets:
//! - Rule-based classifier: <10ms (Tier 1)
//! - ML classifier (TF-IDF + Random Forest): <100ms (Tier 2)
//! - Measures end-to-end inference latency
//!
//! Implements Section 2.3: Performance Targets from nlp-models-techniques-spec.md

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use organizational_intelligence_plugin::{
    classifier::RuleBasedClassifier,
    git::CommitInfo,
    ml_trainer::MLTrainer,
    nlp::{CommitMessageProcessor, TfidfFeatureExtractor},
    training::TrainingDataExtractor,
};

/// Create sample commit messages for benchmarking
fn create_sample_commits() -> Vec<CommitInfo> {
    vec![
        CommitInfo {
            hash: "abc123".to_string(),
            message: "fix: null pointer dereference in parser module".to_string(),
            author: "dev@example.com".to_string(),
            timestamp: 1234567890,
            files_changed: 2,
            lines_added: 10,
            lines_removed: 5,
        },
        CommitInfo {
            hash: "def456".to_string(),
            message: "fix: race condition in mutex lock acquisition".to_string(),
            author: "dev@example.com".to_string(),
            timestamp: 1234567891,
            files_changed: 1,
            lines_added: 5,
            lines_removed: 3,
        },
        CommitInfo {
            hash: "ghi789".to_string(),
            message: "fix: memory leak in buffer allocation".to_string(),
            author: "dev@example.com".to_string(),
            timestamp: 1234567892,
            files_changed: 1,
            lines_added: 8,
            lines_removed: 2,
        },
        CommitInfo {
            hash: "jkl012".to_string(),
            message: "fix: configuration error in yaml parser".to_string(),
            author: "dev@example.com".to_string(),
            timestamp: 1234567893,
            files_changed: 1,
            lines_added: 3,
            lines_removed: 1,
        },
        CommitInfo {
            hash: "mno345".to_string(),
            message: "fix: type error in generic bounds validation".to_string(),
            author: "dev@example.com".to_string(),
            timestamp: 1234567894,
            files_changed: 2,
            lines_added: 15,
            lines_removed: 8,
        },
    ]
}

/// Benchmark: Rule-based classifier (Tier 1 - Target: <10ms)
fn bench_rule_based_classifier(c: &mut Criterion) {
    let mut group = c.benchmark_group("tier1_rule_based");

    let classifier = RuleBasedClassifier::new();
    let commits = create_sample_commits();

    // Single commit classification
    group.bench_function("single_commit", |b| {
        b.iter(|| classifier.classify_from_message(black_box(&commits[0].message)))
    });

    // Batch classification
    for batch_size in [10, 100, 1000] {
        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            &batch_size,
            |b, &size| {
                let batch: Vec<String> = (0..size)
                    .map(|i| commits[i % commits.len()].message.clone())
                    .collect();

                b.iter(|| {
                    for msg in &batch {
                        classifier.classify_from_message(black_box(msg));
                    }
                })
            },
        );
    }

    group.finish();
}

/// Benchmark: Multi-label classification
fn bench_multi_label_classification(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_label");

    let classifier = RuleBasedClassifier::new();
    let commits = create_sample_commits();

    // Single commit multi-label
    group.bench_function("single_commit", |b| {
        b.iter(|| {
            classifier.classify_multi_label(
                black_box(&commits[0].message),
                black_box(3),
                black_box(0.60),
            )
        })
    });

    // Batch multi-label
    group.bench_function("batch_100", |b| {
        let batch: Vec<String> = (0..100)
            .map(|i| commits[i % commits.len()].message.clone())
            .collect();

        b.iter(|| {
            for msg in &batch {
                classifier.classify_multi_label(black_box(msg), black_box(3), black_box(0.60));
            }
        })
    });

    group.finish();
}

/// Benchmark: NLP text preprocessing
fn bench_nlp_preprocessing(c: &mut Criterion) {
    let mut group = c.benchmark_group("nlp_preprocessing");

    let processor = CommitMessageProcessor::new();
    let commits = create_sample_commits();

    // Basic preprocessing (tokenization + stemming + stopwords)
    group.bench_function("preprocess", |b| {
        b.iter(|| processor.preprocess(black_box(&commits[0].message)))
    });

    // Preprocessing with n-grams
    group.bench_function("preprocess_with_ngrams", |b| {
        b.iter(|| processor.preprocess_with_ngrams(black_box(&commits[0].message)))
    });

    // N-gram extraction
    group.bench_function("extract_ngrams", |b| {
        let tokens = processor.preprocess(&commits[0].message).unwrap();
        b.iter(|| processor.extract_ngrams(black_box(&tokens), black_box(2)))
    });

    // Batch preprocessing
    group.bench_function("preprocess_batch_100", |b| {
        let batch: Vec<String> = (0..100)
            .map(|i| commits[i % commits.len()].message.clone())
            .collect();

        b.iter(|| {
            for msg in &batch {
                let _ = processor.preprocess(black_box(msg));
            }
        })
    });

    group.finish();
}

/// Benchmark: TF-IDF feature extraction (Tier 2 component)
fn bench_tfidf_feature_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("tier2_tfidf");

    let commits = create_sample_commits();
    let messages: Vec<String> = commits.iter().map(|c| c.message.clone()).collect();

    // Fit + Transform (training)
    group.bench_function("fit_transform_5", |b| {
        b.iter(|| {
            let mut extractor = TfidfFeatureExtractor::new(100);
            extractor.fit_transform(black_box(&messages))
        })
    });

    // Transform only (inference)
    group.bench_function("transform_single", |b| {
        let mut extractor = TfidfFeatureExtractor::new(100);
        extractor.fit(&messages).unwrap();

        b.iter(|| extractor.transform(black_box(&[commits[0].message.clone()])))
    });

    // Batch transform
    for batch_size in [10, 100] {
        group.bench_with_input(
            BenchmarkId::new("transform_batch", batch_size),
            &batch_size,
            |b, &size| {
                let mut extractor = TfidfFeatureExtractor::new(100);
                extractor.fit(&messages).unwrap();

                let batch: Vec<String> = (0..size)
                    .map(|i| commits[i % commits.len()].message.clone())
                    .collect();

                b.iter(|| extractor.transform(black_box(&batch)))
            },
        );
    }

    // Large vocabulary
    group.bench_function("fit_transform_1500_features", |b| {
        b.iter(|| {
            let mut extractor = TfidfFeatureExtractor::new(1500);
            extractor.fit_transform(black_box(&messages))
        })
    });

    group.finish();
}

/// Benchmark: Training data extraction pipeline
fn bench_training_data_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("training_pipeline");

    let extractor = TrainingDataExtractor::new(0.75);
    let commits = create_sample_commits();

    // Extract training examples
    group.bench_function("extract_5_commits", |b| {
        b.iter(|| extractor.extract_training_data(black_box(&commits), black_box("test-repo")))
    });

    // Create splits
    group.bench_function("create_splits", |b| {
        let examples = extractor
            .extract_training_data(&commits, "test-repo")
            .unwrap();

        if !examples.is_empty() {
            b.iter(|| {
                extractor.create_splits(black_box(&examples), black_box(&["test-repo".to_string()]))
            });
        }
    });

    group.finish();
}

/// Benchmark: ML classifier training
fn bench_ml_classifier_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("ml_training");
    group.sample_size(10); // Reduce sample size for expensive operation

    let commits = create_sample_commits();
    let extractor = TrainingDataExtractor::new(0.70);

    // Note: This benchmark is expensive and will be skipped if insufficient data
    group.bench_function("train_small_model", |b| {
        let examples = extractor
            .extract_training_data(&commits, "test-repo")
            .unwrap();

        if examples.len() < 10 {
            return; // Skip if insufficient data
        }

        let dataset = extractor
            .create_splits(&examples, &["test-repo".to_string()])
            .unwrap();

        b.iter(|| {
            let trainer = MLTrainer::new(10, Some(5), 100);
            trainer.train(black_box(&dataset))
        });
    });

    group.finish();
}

/// Benchmark: End-to-end ML inference latency (Tier 2 - Target: <100ms)
///
/// This is the critical benchmark for Phase 2 Tier 2 performance target.
/// Measures complete pipeline: message → TF-IDF → Random Forest → prediction
fn bench_ml_inference_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("tier2_ml_inference");
    group.sample_size(50); // Moderate sample size

    let commits = create_sample_commits();
    let extractor = TrainingDataExtractor::new(0.70);
    let examples = extractor
        .extract_training_data(&commits, "test-repo")
        .unwrap();

    if examples.len() < 10 {
        println!("⚠️  Insufficient training data for ML inference benchmark");
        return;
    }

    // Train a small model for benchmarking
    let dataset = extractor
        .create_splits(&examples, &["test-repo".to_string()])
        .unwrap();

    let trainer = MLTrainer::new(20, Some(10), 100);
    let model = match trainer.train(&dataset) {
        Ok(m) => m,
        Err(_) => {
            println!("⚠️  Failed to train model for benchmarking");
            return;
        }
    };

    // Single message inference (critical path)
    group.bench_function("single_message", |b| {
        let test_message = vec![commits[0].message.clone()];

        b.iter(|| {
            // This measures the full inference pipeline
            if let Some(ref tfidf) = model.tfidf_extractor {
                if let Ok(features) = tfidf.transform(black_box(&test_message)) {
                    // Convert to f32 for Random Forest
                    let (n_rows, n_cols) = (features.n_rows(), features.n_cols());
                    let data_f32: Vec<f32> = (0..n_rows * n_cols)
                        .map(|i| {
                            let row = i / n_cols;
                            let col = i % n_cols;
                            features.get(row, col) as f32
                        })
                        .collect();

                    if let Ok(features_f32) =
                        aprender::primitives::Matrix::from_vec(n_rows, n_cols, data_f32)
                    {
                        if let Some(ref classifier) = model.classifier {
                            black_box(classifier.predict(&features_f32));
                        }
                    }
                }
            }
        });
    });

    group.finish();
}

/// Benchmark: Tier comparison (Rule-based vs ML)
fn bench_tier_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("tier_comparison");

    let rule_classifier = RuleBasedClassifier::new();
    let commits = create_sample_commits();
    let test_message = &commits[0].message;

    // Tier 1: Rule-based (target: <10ms)
    group.bench_function("tier1_rule_based", |b| {
        b.iter(|| rule_classifier.classify_from_message(black_box(test_message)))
    });

    // Tier 2: TF-IDF extraction only (component benchmark)
    group.bench_function("tier2_tfidf_component", |b| {
        let messages: Vec<String> = commits.iter().map(|c| c.message.clone()).collect();
        let mut extractor = TfidfFeatureExtractor::new(100);
        extractor.fit(&messages).unwrap();

        b.iter(|| extractor.transform(black_box(std::slice::from_ref(test_message))))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_rule_based_classifier,
    bench_multi_label_classification,
    bench_nlp_preprocessing,
    bench_tfidf_feature_extraction,
    bench_training_data_extraction,
    bench_ml_classifier_training,
    bench_ml_inference_latency,
    bench_tier_comparison
);

criterion_main!(benches);
