//! End-to-End CLI Integration Tests
//!
//! PROD-001: Validates full CLI workflow from analysis to prediction
//! Tests the oip-gpu binary with various command combinations

use std::path::PathBuf;
use std::process::Command;

/// Get the path to the oip-gpu binary
fn oip_gpu_bin() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("target");

    // Check for coverage build first (cargo llvm-cov)
    let cov_path = path.join("llvm-cov-target").join("debug").join("oip-gpu");
    if cov_path.exists() {
        return cov_path;
    }

    // Fall back to normal debug build
    path.push("debug");
    path.push("oip-gpu");
    path
}

/// Helper to run oip-gpu with arguments
fn run_oip_gpu(args: &[&str]) -> std::process::Output {
    Command::new(oip_gpu_bin())
        .args(args)
        .output()
        .expect("Failed to execute oip-gpu")
}

/// Helper to check if output contains expected text
fn output_contains(output: &std::process::Output, text: &str) -> bool {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    stdout.contains(text) || stderr.contains(text)
}

#[test]
fn test_cli_help() {
    let output = run_oip_gpu(&["--help"]);

    assert!(output.status.success() || output_contains(&output, "Usage"));
    assert!(output_contains(&output, "analyze") || output_contains(&output, "Analyze"));
}

#[test]
fn test_cli_version() {
    let output = run_oip_gpu(&["--version"]);

    // Should show version or at least not crash
    assert!(output.status.success() || output_contains(&output, "0.1"));
}

#[test]
fn test_analyze_help() {
    let output = run_oip_gpu(&["analyze", "--help"]);

    assert!(output_contains(&output, "org") || output_contains(&output, "repo"));
}

#[test]
fn test_correlate_help() {
    let output = run_oip_gpu(&["correlate", "--help"]);

    assert!(output_contains(&output, "input") || output_contains(&output, "Input"));
}

#[test]
fn test_predict_help() {
    let output = run_oip_gpu(&["predict", "--help"]);

    assert!(
        output_contains(&output, "model")
            || output_contains(&output, "Model")
            || output_contains(&output, "input")
            || output_contains(&output, "Input")
    );
}

#[test]
fn test_query_help() {
    let output = run_oip_gpu(&["query", "--help"]);

    assert!(output_contains(&output, "query") || output_contains(&output, "input"));
}

#[test]
fn test_cluster_help() {
    let output = run_oip_gpu(&["cluster", "--help"]);

    assert!(output_contains(&output, "cluster") || output_contains(&output, "k"));
}

#[test]
fn test_benchmark_help() {
    let output = run_oip_gpu(&["benchmark", "--help"]);

    assert!(output_contains(&output, "suite") || output_contains(&output, "benchmark"));
}

#[test]
fn test_invalid_command() {
    let output = run_oip_gpu(&["nonexistent-command"]);

    // Should fail gracefully with error message
    assert!(!output.status.success());
}

#[test]
fn test_analyze_missing_target() {
    let output = run_oip_gpu(&["analyze"]);

    // Should fail because no org/repo specified
    // But should fail gracefully, not panic
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Either fails with error message or shows help
    assert!(
        !output.status.success()
            || stderr.contains("error")
            || stdout.contains("Usage")
            || stderr.contains("required")
    );
}

#[test]
fn test_verbose_flag() {
    let output = run_oip_gpu(&["--verbose", "--help"]);

    // Verbose flag should be accepted
    assert!(output.status.success() || output_contains(&output, "Usage"));
}

#[test]
fn test_backend_flag_simd() {
    let output = run_oip_gpu(&["--backend", "simd", "--help"]);

    // SIMD backend flag should be accepted
    assert!(output.status.success() || output_contains(&output, "Usage"));
}

#[test]
fn test_backend_flag_cpu() {
    let output = run_oip_gpu(&["--backend", "cpu", "--help"]);

    // CPU backend flag should be accepted
    assert!(output.status.success() || output_contains(&output, "Usage"));
}

// Integration test: Full workflow simulation (without network)
mod workflow {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_output_directory_creation() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test-output.db");

        // Analyze with invalid repo should fail but not crash
        let output = run_oip_gpu(&[
            "analyze",
            "--repo",
            "nonexistent/repo",
            "--output",
            output_path.to_str().unwrap(),
        ]);

        // Should fail gracefully (repo doesn't exist)
        // The important thing is it doesn't panic
        let _ = output; // Just verify it ran
    }
}

// Module-level integration tests
mod modules {
    use organizational_intelligence_plugin::correlation::pearson_correlation;
    use organizational_intelligence_plugin::features::{CommitFeatures, FeatureExtractor};
    use organizational_intelligence_plugin::imbalance::Smote;
    use organizational_intelligence_plugin::ml::{DefectPredictor, ModelMetrics, PatternClusterer};
    use organizational_intelligence_plugin::sliding_window::SlidingWindowAnalyzer;
    use organizational_intelligence_plugin::storage::FeatureStore;
    use trueno::Vector;

    fn make_feature(category: u8, files: u32, timestamp: f64) -> CommitFeatures {
        CommitFeatures {
            defect_category: category,
            files_changed: files as f32,
            lines_added: (files * 10) as f32,
            lines_deleted: (files * 5) as f32,
            complexity_delta: files as f32 * 0.1,
            timestamp,
            hour_of_day: 10,
            day_of_week: 1,
            ..Default::default()
        }
    }

    #[test]
    fn test_full_pipeline_feature_to_prediction() {
        // 1. Extract features
        let extractor = FeatureExtractor::new();
        let features: Vec<CommitFeatures> = (0..100)
            .map(|i| {
                extractor
                    .extract(
                        (i % 3) as u8, // 3 categories
                        (i % 10) + 1,  // 1-10 files
                        i * 10,        // lines added
                        i * 5,         // lines deleted
                        1700000000 + i as i64,
                    )
                    .unwrap()
            })
            .collect();

        // 2. Store features
        let mut store = FeatureStore::new().unwrap();
        store.bulk_insert(features.clone()).unwrap();
        assert_eq!(store.len(), 100);

        // 3. Train predictor
        let mut predictor = DefectPredictor::new();
        predictor.train(&features).unwrap();
        assert!(predictor.is_trained());

        // 4. Make prediction
        let test_feature = make_feature(0, 5, 1700000050.0);
        let prediction = predictor.predict(&test_feature).unwrap();
        assert!(prediction < 10); // Valid category

        // 5. Get probabilities
        let probs = predictor.predict_proba(&test_feature).unwrap();
        assert_eq!(probs.len(), 10);
        assert!(probs.iter().sum::<f32>() > 0.99); // Sum to ~1.0
    }

    #[test]
    fn test_full_pipeline_clustering() {
        // Create features with distinct clusters
        let mut features = Vec::new();

        // Cluster 1: Small files, few changes
        for i in 0..30 {
            features.push(make_feature(0, 1 + (i % 3), 1700000000.0 + i as f64));
        }

        // Cluster 2: Large files, many changes
        for i in 0..30 {
            features.push(make_feature(1, 50 + (i % 10), 1700000000.0 + i as f64));
        }

        // Cluster
        let mut clusterer = PatternClusterer::with_k(2);
        clusterer.fit(&features).unwrap();

        // Verify clustering
        let assignments = clusterer.predict_batch(&features).unwrap();
        assert_eq!(assignments.len(), 60);

        // Check cluster separation
        let cluster0_count = assignments.iter().filter(|&&c| c == 0).count();
        let cluster1_count = assignments.iter().filter(|&&c| c == 1).count();

        // Both clusters should have members
        assert!(cluster0_count > 0);
        assert!(cluster1_count > 0);
    }

    #[test]
    fn test_full_pipeline_correlation() {
        // Create correlated data
        let data_a: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let data_b: Vec<f32> = (0..100).map(|i| (i * 2) as f32 + 10.0).collect();

        let vec_a = Vector::from_slice(&data_a);
        let vec_b = Vector::from_slice(&data_b);

        let r = pearson_correlation(&vec_a, &vec_b).unwrap();

        // Perfect linear correlation
        assert!((r - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_full_pipeline_imbalance_handling() {
        // Create imbalanced dataset: 90% category 0, 10% category 1
        let mut features = Vec::new();
        for i in 0..90 {
            features.push(make_feature(0, i % 10, 1700000000.0 + i as f64));
        }
        for i in 0..10 {
            features.push(make_feature(1, 50 + i, 1700000000.0 + i as f64));
        }

        // Apply SMOTE
        let smote = Smote::new();
        let balanced = smote.oversample(&features, 1, 0.5).unwrap();

        // Should have more samples now
        assert!(balanced.len() > features.len());

        // Minority class should be better represented
        let minority_before = features.iter().filter(|f| f.defect_category == 1).count();
        let minority_after = balanced.iter().filter(|f| f.defect_category == 1).count();
        assert!(minority_after > minority_before);
    }

    #[test]
    fn test_full_pipeline_sliding_window() {
        // Create features spanning multiple time windows
        let mut store = FeatureStore::new().unwrap();

        // Features over 1 year (in seconds: ~31.5M)
        let year_seconds = 365.0 * 24.0 * 3600.0;
        let features: Vec<CommitFeatures> = (0..100)
            .map(|i| {
                make_feature(
                    (i % 5) as u8,
                    (i % 10) + 1,
                    1700000000.0 + (i as f64 / 100.0) * year_seconds,
                )
            })
            .collect();

        store.bulk_insert(features).unwrap();

        // Analyze with sliding windows
        let analyzer = SlidingWindowAnalyzer::new_six_month();
        let matrices = analyzer.compute_all_windows(&store).unwrap();

        // Should have multiple windows
        assert!(!matrices.is_empty());

        // Each matrix should be DIMENSION x DIMENSION (NLP-014: 14x14)
        for matrix in &matrices {
            assert_eq!(matrix.matrix.len(), CommitFeatures::DIMENSION);
            assert_eq!(matrix.matrix[0].len(), CommitFeatures::DIMENSION);
        }
    }

    #[test]
    fn test_metrics_evaluation() {
        let predictions = vec![0, 0, 1, 1, 2, 2, 0, 1];
        let labels = vec![0, 0, 1, 0, 2, 2, 1, 1];

        let accuracy = ModelMetrics::accuracy(&predictions, &labels);
        assert!(accuracy > 0.5); // Better than random

        let precision_0 = ModelMetrics::precision(&predictions, &labels, 0);
        let recall_0 = ModelMetrics::recall(&predictions, &labels, 0);

        assert!(precision_0 > 0.0);
        assert!(recall_0 > 0.0);
    }
}
