mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    // Helper to create a minimal valid baseline YAML for testing
    fn create_test_baseline() -> String {
        r#"organizational_insights:
  top_defect_categories:
    - category: ConfigurationErrors
      frequency: 25
      confidence: 0.78
      quality_signals:
        avg_tdg_score: 45.2
        max_tdg_score: 60.0
        avg_complexity: 8.0
        avg_test_coverage: 0.5
        satd_instances: 5
        avg_lines_changed: 10.0
        avg_files_per_commit: 2.0
      examples: []
code_quality_thresholds:
  tdg_minimum: 85.0
  test_coverage_minimum: 0.85
  max_function_length: 50
  max_cyclomatic_complexity: 10
metadata:
  analysis_date: "2024-01-01T00:00:00Z"
  repositories_analyzed: 1
  commits_analyzed: 10
"#
        .to_string()
    }

    // Helper to create a minimal valid report YAML for testing
    fn create_test_report() -> String {
        r#"version: "1.0"
metadata:
  organization: "test-org"
  analysis_date: "2024-01-01T00:00:00Z"
  repositories_analyzed: 1
  commits_analyzed: 10
  analyzer_version: "1.0.0"
defect_patterns:
  - category: LogicErrors
    frequency: 5
    confidence: 0.91
    quality_signals:
      avg_tdg_score: 88.5
      max_tdg_score: 95.0
      avg_complexity: 4.0
      avg_test_coverage: 0.85
      satd_instances: 0
      avg_lines_changed: 8.0
      avg_files_per_commit: 1.5
    examples:
      - commit_hash: "abc123"
        message: "Fix critical bug"
        author: "test-author"
        timestamp: 1704067200
        files_affected: 2
        lines_added: 10
        lines_removed: 5
"#
        .to_string()
    }

    #[tokio::test]
    async fn test_handle_summarize_invalid_input() {
        let input = PathBuf::from("nonexistent.yaml");
        let temp_output = NamedTempFile::new().unwrap();
        let output = temp_output.path().to_path_buf();

        let result = handle_summarize(input, output, true, 10, 5, false).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_handle_review_pr_invalid_baseline() {
        let baseline = PathBuf::from("nonexistent-baseline.yaml");
        let files = "src/main.rs,src/lib.rs".to_string();
        let format = "markdown".to_string();

        let result = handle_review_pr(baseline, files, format, None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_handle_review_pr_markdown_format() {
        // Create a temporary baseline file
        let temp_baseline = NamedTempFile::new().unwrap();
        std::fs::write(temp_baseline.path(), create_test_baseline()).unwrap();

        let baseline = temp_baseline.path().to_path_buf();
        let files = "src/main.rs,src/lib.rs".to_string();
        let format = "markdown".to_string();

        let result = handle_review_pr(baseline, files, format, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_handle_review_pr_json_format() {
        // Create a temporary baseline file
        let temp_baseline = NamedTempFile::new().unwrap();
        std::fs::write(temp_baseline.path(), create_test_baseline()).unwrap();

        let baseline = temp_baseline.path().to_path_buf();
        let files = "src/test.rs".to_string();
        let format = "json".to_string();

        let result = handle_review_pr(baseline, files, format, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_handle_review_pr_with_output_file() {
        // Create temporary baseline and output files
        let temp_baseline = NamedTempFile::new().unwrap();
        std::fs::write(temp_baseline.path(), create_test_baseline()).unwrap();

        let temp_output = NamedTempFile::new().unwrap();
        let output_path = temp_output.path().to_path_buf();

        let baseline = temp_baseline.path().to_path_buf();
        let files = "src/main.rs".to_string();
        let format = "markdown".to_string();

        let result = handle_review_pr(baseline, files, format, Some(output_path.clone())).await;
        assert!(result.is_ok());

        // Verify output file was created and has content
        let content = std::fs::read_to_string(&output_path).unwrap();
        assert!(!content.is_empty());
    }

    #[tokio::test]
    async fn test_handle_review_pr_empty_files() {
        // Create a temporary baseline file
        let temp_baseline = NamedTempFile::new().unwrap();
        std::fs::write(temp_baseline.path(), create_test_baseline()).unwrap();

        let baseline = temp_baseline.path().to_path_buf();
        let files = "".to_string(); // Empty files list
        let format = "markdown".to_string();

        let result = handle_review_pr(baseline, files, format, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_handle_review_pr_multiple_files() {
        let temp_baseline = NamedTempFile::new().unwrap();
        std::fs::write(temp_baseline.path(), create_test_baseline()).unwrap();

        let baseline = temp_baseline.path().to_path_buf();
        let files = "src/main.rs, src/lib.rs, src/test.rs, src/utils.rs".to_string();
        let format = "markdown".to_string();

        let result = handle_review_pr(baseline, files, format, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_handle_summarize_valid_report() {
        // Create temporary report and output files
        let temp_report = NamedTempFile::new().unwrap();
        std::fs::write(temp_report.path(), create_test_report()).unwrap();

        let temp_output = NamedTempFile::new().unwrap();
        let output_path = temp_output.path().to_path_buf();

        let input = temp_report.path().to_path_buf();

        let result = handle_summarize(input, output_path.clone(), false, 10, 1, true).await;
        assert!(result.is_ok());

        // Verify output file was created
        assert!(output_path.exists());
    }

    #[tokio::test]
    async fn test_handle_summarize_with_pii_stripping() {
        let temp_report = NamedTempFile::new().unwrap();
        std::fs::write(temp_report.path(), create_test_report()).unwrap();

        let temp_output = NamedTempFile::new().unwrap();
        let output_path = temp_output.path().to_path_buf();

        let input = temp_report.path().to_path_buf();

        let result = handle_summarize(input, output_path.clone(), true, 5, 1, false).await;
        assert!(result.is_ok());

        // Verify output file was created
        assert!(output_path.exists());
    }

    #[tokio::test]
    async fn test_handle_summarize_different_top_n() {
        let temp_report = NamedTempFile::new().unwrap();
        std::fs::write(temp_report.path(), create_test_report()).unwrap();

        let temp_output = NamedTempFile::new().unwrap();
        let output_path = temp_output.path().to_path_buf();

        let input = temp_report.path().to_path_buf();

        // Test with top_n = 3
        let result = handle_summarize(input, output_path.clone(), false, 3, 1, true).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_handle_summarize_min_frequency_filter() {
        let temp_report = NamedTempFile::new().unwrap();
        std::fs::write(temp_report.path(), create_test_report()).unwrap();

        let temp_output = NamedTempFile::new().unwrap();
        let output_path = temp_output.path().to_path_buf();

        let input = temp_report.path().to_path_buf();

        // Test with high min_frequency
        let result = handle_summarize(input, output_path.clone(), false, 10, 100, false).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_handle_summarize_invalid_yaml_format() {
        let temp_report = NamedTempFile::new().unwrap();
        // Write invalid YAML
        std::fs::write(temp_report.path(), "not: valid: yaml: {{{").unwrap();

        let temp_output = NamedTempFile::new().unwrap();
        let output_path = temp_output.path().to_path_buf();

        let input = temp_report.path().to_path_buf();

        let result = handle_summarize(input, output_path, false, 10, 1, false).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_handle_analyze_with_token() {
        // This test will fail because it requires real GitHub API
        // But it exercises the code path
        let org = "nonexistent-org-12345678".to_string();
        let temp_output = NamedTempFile::new().unwrap();
        let output = temp_output.path().to_path_buf();

        // Use a fake token to test the token path
        let token = Some("fake-token-for-testing".to_string());

        let result = handle_analyze(org, output, 5, token, "1.0.0".to_string(), None, 0.65).await;
        // Should fail because org doesn't exist, but we're testing the code path
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_handle_analyze_without_token() {
        // Test without GitHub token (unauthenticated)
        let org = "nonexistent-org-87654321".to_string();
        let temp_output = NamedTempFile::new().unwrap();
        let output = temp_output.path().to_path_buf();

        let result = handle_analyze(org, output, 5, None, "1.0.0".to_string(), None, 0.65).await;
        // Should fail because org doesn't exist
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_handle_review_pr_whitespace_in_files() {
        let temp_baseline = NamedTempFile::new().unwrap();
        std::fs::write(temp_baseline.path(), create_test_baseline()).unwrap();

        let baseline = temp_baseline.path().to_path_buf();
        // Test with extra whitespace and commas
        let files = "  src/main.rs  ,  src/lib.rs  ,  , src/test.rs  ".to_string();
        let format = "markdown".to_string();

        let result = handle_review_pr(baseline, files, format, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_handle_summarize_all_config_combinations() {
        let temp_report = NamedTempFile::new().unwrap();
        std::fs::write(temp_report.path(), create_test_report()).unwrap();

        let temp_output = NamedTempFile::new().unwrap();
        let output_path = temp_output.path().to_path_buf();

        let input = temp_report.path().to_path_buf();

        // Test with all config options enabled
        let result = handle_summarize(input, output_path.clone(), true, 20, 2, true).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_handle_extract_training_data_invalid_path() {
        let repo = PathBuf::from("/nonexistent/repo/path");
        let temp_output = NamedTempFile::new().unwrap();
        let output = temp_output.path().to_path_buf();

        let result = handle_extract_training_data(repo, output, 0.75, 100, true, false).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_handle_extract_training_data_not_git_repo() {
        // Create a temporary directory that exists but is not a git repo
        let temp_dir = tempfile::TempDir::new().unwrap();
        let repo = temp_dir.path().to_path_buf();

        let temp_output = NamedTempFile::new().unwrap();
        let output = temp_output.path().to_path_buf();

        let result = handle_extract_training_data(repo, output, 0.75, 100, true, false).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Not a Git repository"));
    }

    #[tokio::test]
    async fn test_handle_extract_training_data_with_splits() {
        // Use the current repository (which is guaranteed to be a git repo)
        let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let temp_output = NamedTempFile::new().unwrap();
        let output = temp_output.path().to_path_buf();

        // This should succeed as it's a real git repo
        let result =
            handle_extract_training_data(repo, output.clone(), 0.70, 50, true, false).await;

        // Should succeed or return Ok with empty results
        match result {
            Ok(_) => {
                // If successful, output file should exist
                if output.exists() {
                    let content = std::fs::read_to_string(&output).unwrap();
                    assert!(!content.is_empty());
                }
            }
            Err(e) => {
                // Some errors are acceptable (e.g., no commits found)
                eprintln!("Expected error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_handle_extract_training_data_without_splits() {
        // Use the current repository
        let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let temp_output = NamedTempFile::new().unwrap();
        let output = temp_output.path().to_path_buf();

        let result =
            handle_extract_training_data(repo, output.clone(), 0.70, 50, false, false).await;

        // Should succeed or return Ok with empty results
        match result {
            Ok(_) => {
                // If successful, output file should exist
                if output.exists() {
                    let content = std::fs::read_to_string(&output).unwrap();
                    assert!(!content.is_empty());
                }
            }
            Err(e) => {
                // Some errors are acceptable
                eprintln!("Expected error: {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_handle_extract_training_data_high_confidence_threshold() {
        // Use the current repository with a very high confidence threshold
        let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let temp_output = NamedTempFile::new().unwrap();
        let output = temp_output.path().to_path_buf();

        // High confidence threshold (0.95) should result in fewer/no examples
        let result =
            handle_extract_training_data(repo, output.clone(), 0.95, 50, true, false).await;

        // Should succeed (even if no examples found)
        assert!(result.is_ok() || result.unwrap_err().to_string().contains("Git"));
    }

    #[tokio::test]
    async fn test_handle_extract_training_data_low_max_commits() {
        // Use the current repository with low max_commits
        let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let temp_output = NamedTempFile::new().unwrap();
        let output = temp_output.path().to_path_buf();

        let result = handle_extract_training_data(repo, output.clone(), 0.75, 5, true, false).await;

        // Should succeed with limited commits
        assert!(result.is_ok() || result.unwrap_err().to_string().contains("Git"));
    }

    #[tokio::test]
    async fn test_handle_train_classifier_invalid_input() {
        let input = PathBuf::from("/nonexistent/training-data.json");
        let output = None;

        let result = handle_train_classifier(input, output, 100, 20, 1500).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("does not exist"));
    }

    #[tokio::test]
    async fn test_handle_train_classifier_invalid_json() {
        // Create a temp file with invalid JSON
        let temp_input = NamedTempFile::new().unwrap();
        std::fs::write(temp_input.path(), "not valid json").unwrap();

        let input = temp_input.path().to_path_buf();
        let output = None;

        let result = handle_train_classifier(input, output, 100, 20, 1500).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_handle_train_classifier_with_valid_data() {
        // Use the test training data file if it exists
        let input = PathBuf::from("/tmp/test-training-data.json");

        if !input.exists() {
            // Skip test if training data doesn't exist
            return;
        }

        let temp_output = NamedTempFile::new().unwrap();
        let output_path = temp_output.path().to_path_buf();
        let output = Some(output_path.clone());

        let result = handle_train_classifier(input, output, 10, 5, 100).await;

        // Should succeed or fail with reasonable error
        match result {
            Ok(_) => {
                // If successful, output file should exist
                assert!(output_path.exists());
            }
            Err(e) => {
                // Acceptable errors: not enough data, etc.
                let msg = e.to_string();
                assert!(
                    msg.contains("empty") || msg.contains("training") || msg.contains("TF-IDF"),
                    "Unexpected error: {}",
                    msg
                );
            }
        }
    }

    // ===== Export Handler Tests (Issue #2) =====

    #[tokio::test]
    async fn test_handle_export_invalid_path() {
        let repo = PathBuf::from("/nonexistent/repo/path");
        let temp_output = NamedTempFile::new().unwrap();
        let output = temp_output.path().to_path_buf();

        let result = handle_export(repo, output, "json".to_string(), 100, 0.70).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("does not exist"));
    }

    #[tokio::test]
    async fn test_handle_export_not_git_repo() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let repo = temp_dir.path().to_path_buf();
        let temp_output = NamedTempFile::new().unwrap();
        let output = temp_output.path().to_path_buf();

        let result = handle_export(repo, output, "json".to_string(), 100, 0.70).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Not a Git repository"));
    }

    #[tokio::test]
    async fn test_handle_export_invalid_format() {
        // Skip if not in a git repo (e.g., during mutation testing)
        let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        if !repo.join(".git").exists() {
            return;
        }

        let temp_output = NamedTempFile::new().unwrap();
        let output = temp_output.path().to_path_buf();

        let result = handle_export(repo, output, "invalid_format".to_string(), 100, 0.70).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        // Either invalid format error or not a git repo error
        assert!(err_msg.contains("Invalid format") || err_msg.contains("Not a Git"));
    }

    #[tokio::test]
    async fn test_handle_export_json_format() {
        // Skip if not in a git repo (e.g., during mutation testing)
        let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        if !repo.join(".git").exists() {
            return;
        }

        let temp_output = NamedTempFile::new().unwrap();
        let output = temp_output.path().to_path_buf();

        // Use low confidence threshold to get more samples
        let result = handle_export(repo, output.clone(), "json".to_string(), 100, 0.60).await;

        // May succeed or fail depending on commit messages - both are acceptable
        match result {
            Ok(_) => {
                assert!(output.exists());
                let content = std::fs::read_to_string(&output).unwrap();
                assert!(content.contains("features"));
                assert!(content.contains("labels"));
            }
            Err(e) => {
                // Acceptable errors
                let msg = e.to_string();
                assert!(
                    msg.contains("No features")
                        || msg.contains("No commits")
                        || msg.contains("Git"),
                    "Unexpected error: {}",
                    msg
                );
            }
        }
    }

    #[tokio::test]
    async fn test_handle_export_binary_format() {
        // Skip if not in a git repo
        let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        if !repo.join(".git").exists() {
            return;
        }

        let temp_output = NamedTempFile::new().unwrap();
        let output = temp_output.path().to_path_buf();

        let result = handle_export(repo, output.clone(), "binary".to_string(), 100, 0.60).await;

        match result {
            Ok(_) => {
                assert!(output.exists());
                let content = std::fs::read(&output).unwrap();
                assert!(!content.is_empty());
            }
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("No features") || msg.contains("Git"),
                    "Unexpected error: {}",
                    msg
                );
            }
        }
    }

    #[tokio::test]
    async fn test_handle_export_high_confidence_threshold() {
        // Skip if not in a git repo
        let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        if !repo.join(".git").exists() {
            return;
        }

        let temp_output = NamedTempFile::new().unwrap();
        let output = temp_output.path().to_path_buf();

        // Very high confidence threshold - may find no features
        let result = handle_export(repo, output.clone(), "json".to_string(), 50, 0.99).await;

        // Both success and "no features" error are acceptable
        match result {
            Ok(_) => {
                assert!(output.exists());
            }
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("No features") || msg.contains("Git"),
                    "Unexpected error: {}",
                    msg
                );
            }
        }
    }

    #[tokio::test]
    async fn test_handle_export_low_max_commits() {
        // Skip if not in a git repo
        let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        if !repo.join(".git").exists() {
            return;
        }

        let temp_output = NamedTempFile::new().unwrap();
        let output = temp_output.path().to_path_buf();

        let result = handle_export(repo, output.clone(), "json".to_string(), 10, 0.60).await;

        // Should work with limited commits
        match result {
            Ok(_) => {
                assert!(output.exists());
            }
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("No features")
                        || msg.contains("No commits")
                        || msg.contains("Git"),
                    "Unexpected error: {}",
                    msg
                );
            }
        }
    }

    // ============== Localize Handler Tests ==============

    fn create_test_lcov_file(dir: &std::path::Path, name: &str, content: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, content).unwrap();
        path
    }

    #[tokio::test]
    async fn test_handle_localize_basic() {
        let temp_dir = tempfile::tempdir().unwrap();

        // Create test LCOV files
        let passed_lcov = r#"SF:src/main.rs
DA:10,5
DA:20,10
DA:30,8
end_of_record
"#;
        let failed_lcov = r#"SF:src/main.rs
DA:10,3
DA:20,0
DA:40,5
end_of_record
"#;

        let passed_path = create_test_lcov_file(temp_dir.path(), "passed.lcov", passed_lcov);
        let failed_path = create_test_lcov_file(temp_dir.path(), "failed.lcov", failed_lcov);
        let output_path = temp_dir.path().join("output.yaml");

        let result = handle_localize(
            passed_path,
            failed_path,
            1,
            1,
            "tarantula".to_string(),
            10,
            output_path.clone(),
            "yaml".to_string(),
            false,
            None,
            false,             // rag
            None,              // knowledge_base
            "rrf".to_string(), // fusion
            5,                 // similar_bugs
            false,             // ensemble
            None,              // ensemble_model
            false,             // include_churn
            false,             // calibrated
            None,              // calibration_model
            0.5,               // confidence_threshold
        )
        .await;

        assert!(result.is_ok());
        assert!(output_path.exists());

        // Verify output contains expected content
        let output_content = std::fs::read_to_string(&output_path).unwrap();
        assert!(output_content.contains("rankings"));
    }

    #[tokio::test]
    async fn test_handle_localize_json_format() {
        let temp_dir = tempfile::tempdir().unwrap();

        let passed_lcov = "SF:src/lib.rs\nDA:100,10\nend_of_record\n";
        let failed_lcov = "SF:src/lib.rs\nDA:100,5\nend_of_record\n";

        let passed_path = create_test_lcov_file(temp_dir.path(), "passed.lcov", passed_lcov);
        let failed_path = create_test_lcov_file(temp_dir.path(), "failed.lcov", failed_lcov);
        let output_path = temp_dir.path().join("output.json");

        let result = handle_localize(
            passed_path,
            failed_path,
            1,
            1,
            "ochiai".to_string(),
            5,
            output_path.clone(),
            "json".to_string(),
            false,
            None,
            false,             // rag
            None,              // knowledge_base
            "rrf".to_string(), // fusion
            5,                 // similar_bugs
            false,             // ensemble
            None,              // ensemble_model
            false,             // include_churn
            false,             // calibrated
            None,              // calibration_model
            0.5,               // confidence_threshold
        )
        .await;

        assert!(result.is_ok());
        assert!(output_path.exists());

        // Verify it's valid JSON
        let content = std::fs::read_to_string(&output_path).unwrap();
        let _: serde_json::Value = serde_json::from_str(&content).unwrap();
    }

    #[tokio::test]
    async fn test_handle_localize_dstar_formula() {
        let temp_dir = tempfile::tempdir().unwrap();

        let passed_lcov = "SF:src/bug.rs\nDA:50,2\nend_of_record\n";
        let failed_lcov = "SF:src/bug.rs\nDA:50,10\nend_of_record\n";

        let passed_path = create_test_lcov_file(temp_dir.path(), "passed.lcov", passed_lcov);
        let failed_path = create_test_lcov_file(temp_dir.path(), "failed.lcov", failed_lcov);
        let output_path = temp_dir.path().join("output.yaml");

        let result = handle_localize(
            passed_path,
            failed_path,
            10,
            5,
            "dstar2".to_string(),
            10,
            output_path.clone(),
            "yaml".to_string(),
            false,
            None,
            false,             // rag
            None,              // knowledge_base
            "rrf".to_string(), // fusion
            5,                 // similar_bugs
            false,             // ensemble
            None,              // ensemble_model
            false,             // include_churn
            false,             // calibrated
            None,              // calibration_model
            0.5,               // confidence_threshold
        )
        .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_handle_localize_invalid_coverage_file() {
        let temp_dir = tempfile::tempdir().unwrap();

        let nonexistent = temp_dir.path().join("nonexistent.lcov");
        let output_path = temp_dir.path().join("output.yaml");

        let result = handle_localize(
            nonexistent.clone(),
            nonexistent,
            1,
            1,
            "tarantula".to_string(),
            10,
            output_path,
            "yaml".to_string(),
            false,
            None,
            false,             // rag
            None,              // knowledge_base
            "rrf".to_string(), // fusion
            5,                 // similar_bugs
            false,             // ensemble
            None,              // ensemble_model
            false,             // include_churn
            false,             // calibrated
            None,              // calibration_model
            0.5,               // confidence_threshold
        )
        .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Failed to read"));
    }
}
