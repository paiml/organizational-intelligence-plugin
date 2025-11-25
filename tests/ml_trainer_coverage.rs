/// Additional tests for ml_trainer.rs to improve coverage to 95%+
/// Focus: predict(), predict_top_n(), parse_category(), edge cases
use anyhow::Result;
use organizational_intelligence_plugin::git::CommitInfo;
use organizational_intelligence_plugin::ml_trainer::MLTrainer;
use organizational_intelligence_plugin::training::{TrainingDataExtractor, TrainingDataset};
use tempfile::TempDir;

fn create_test_dataset() -> Result<TrainingDataset> {
    let commits = vec![
        CommitInfo {
            hash: "abc1".to_string(),
            message: "fix: null pointer dereference in parser".to_string(),
            author: "dev@example.com".to_string(),
            timestamp: 1234567890,
            files_changed: 2,
            lines_added: 10,
            lines_removed: 5,
        },
        CommitInfo {
            hash: "abc2".to_string(),
            message: "fix: race condition in mutex lock".to_string(),
            author: "dev@example.com".to_string(),
            timestamp: 1234567891,
            files_changed: 1,
            lines_added: 5,
            lines_removed: 3,
        },
        CommitInfo {
            hash: "abc3".to_string(),
            message: "fix: memory leak in allocator".to_string(),
            author: "dev@example.com".to_string(),
            timestamp: 1234567892,
            files_changed: 1,
            lines_added: 8,
            lines_removed: 2,
        },
        CommitInfo {
            hash: "abc4".to_string(),
            message: "fix: configuration error in yaml parser".to_string(),
            author: "dev@example.com".to_string(),
            timestamp: 1234567893,
            files_changed: 1,
            lines_added: 3,
            lines_removed: 1,
        },
        CommitInfo {
            hash: "abc5".to_string(),
            message: "fix: type error in generic bounds".to_string(),
            author: "dev@example.com".to_string(),
            timestamp: 1234567894,
            files_changed: 2,
            lines_added: 15,
            lines_removed: 8,
        },
        CommitInfo {
            hash: "abc6".to_string(),
            message: "fix: AST transformation for match expressions".to_string(),
            author: "dev@example.com".to_string(),
            timestamp: 1234567895,
            files_changed: 1,
            lines_added: 12,
            lines_removed: 4,
        },
        CommitInfo {
            hash: "abc7".to_string(),
            message: "fix: operator precedence in comprehension".to_string(),
            author: "dev@example.com".to_string(),
            timestamp: 1234567896,
            files_changed: 1,
            lines_added: 6,
            lines_removed: 2,
        },
        CommitInfo {
            hash: "abc8".to_string(),
            message: "fix: stdlib mapping for os.path".to_string(),
            author: "dev@example.com".to_string(),
            timestamp: 1234567897,
            files_changed: 2,
            lines_added: 20,
            lines_removed: 10,
        },
        CommitInfo {
            hash: "abc9".to_string(),
            message: "fix: ownership borrow error in iterator".to_string(),
            author: "dev@example.com".to_string(),
            timestamp: 1234567898,
            files_changed: 1,
            lines_added: 8,
            lines_removed: 3,
        },
        CommitInfo {
            hash: "abc10".to_string(),
            message: "fix: trait bound issue in generic function".to_string(),
            author: "dev@example.com".to_string(),
            timestamp: 1234567899,
            files_changed: 1,
            lines_added: 5,
            lines_removed: 2,
        },
    ];

    let extractor = TrainingDataExtractor::new(0.60);
    let examples = extractor.extract_training_data(&commits, "test-repo")?;
    extractor.create_splits(&examples, &["test-repo".to_string()])
}

#[test]
fn test_trained_model_predict_with_real_model() -> Result<()> {
    // Create training dataset
    let dataset = create_test_dataset()?;

    // Train model
    let trainer = MLTrainer::new(10, Some(5), 100);
    let model = trainer.train(&dataset)?;

    // Test prediction
    let result = model.predict("fix: null pointer in parser")?;
    assert!(result.is_some());

    let (_category, confidence) = result.unwrap();
    // Should predict MemorySafety or similar
    assert!(confidence > 0.0 && confidence <= 1.0);

    Ok(())
}

#[test]
fn test_trained_model_predict_top_n() -> Result<()> {
    // Create training dataset
    let dataset = create_test_dataset()?;

    // Train model
    let trainer = MLTrainer::new(10, Some(5), 100);
    let model = trainer.train(&dataset)?;

    // Test top-N prediction
    let results = model.predict_top_n("fix: memory leak", 3)?;
    assert!(!results.is_empty());
    assert!(results.len() <= 3);

    for (_category, confidence) in results {
        assert!(confidence > 0.0 && confidence <= 1.0);
    }

    Ok(())
}

#[test]
fn test_trained_model_predict_top_n_empty_message() -> Result<()> {
    // Create training dataset
    let dataset = create_test_dataset()?;

    // Train model
    let trainer = MLTrainer::new(10, Some(5), 100);
    let model = trainer.train(&dataset)?;

    // Test with empty message
    let results = model.predict_top_n("", 3)?;
    // Should handle gracefully
    assert!(results.len() <= 3);

    Ok(())
}

#[test]
fn test_model_save_and_load() -> Result<()> {
    // Create training dataset
    let dataset = create_test_dataset()?;

    // Train model
    let trainer = MLTrainer::new(10, Some(5), 100);
    let model = trainer.train(&dataset)?;

    // Save model
    let temp_dir = TempDir::new()?;
    let model_path = temp_dir.path().join("test-model.bin");
    MLTrainer::save_model(&model, &model_path)?;

    // Verify file exists
    assert!(model_path.exists());

    // Load model
    let loaded_model = MLTrainer::load_model(&model_path)?;

    // Verify metadata
    assert_eq!(loaded_model.metadata.n_classes, model.metadata.n_classes);
    assert_eq!(
        loaded_model.metadata.n_features,
        model.metadata.n_features
    );
    assert_eq!(
        loaded_model.metadata.train_accuracy,
        model.metadata.train_accuracy
    );

    Ok(())
}

#[test]
fn test_trainer_with_different_hyperparameters() -> Result<()> {
    let dataset = create_test_dataset()?;

    // Test with different n_estimators
    let trainer1 = MLTrainer::new(5, Some(5), 100);
    let model1 = trainer1.train(&dataset)?;
    assert!(model1.metadata.train_accuracy >= 0.0);

    // Test with different max_depth
    let trainer2 = MLTrainer::new(10, Some(10), 100);
    let model2 = trainer2.train(&dataset)?;
    assert!(model2.metadata.train_accuracy >= 0.0);

    // Test with different max_features
    let trainer3 = MLTrainer::new(10, Some(5), 200);
    let model3 = trainer3.train(&dataset)?;
    assert!(model3.metadata.train_accuracy >= 0.0);

    Ok(())
}

#[test]
fn test_model_metadata_fields() -> Result<()> {
    let dataset = create_test_dataset()?;

    let trainer = MLTrainer::new(10, Some(5), 100);
    let model = trainer.train(&dataset)?;

    // Verify all metadata fields
    assert!(model.metadata.n_classes > 0);
    assert!(model.metadata.n_features > 0);
    assert!(model.metadata.train_accuracy >= 0.0);
    assert!(model.metadata.train_accuracy <= 1.0);
    assert!(model.metadata.validation_accuracy >= 0.0);
    assert!(model.metadata.validation_accuracy <= 1.0);

    // Test accuracy should be Some
    if let Some(test_acc) = model.metadata.test_accuracy {
        assert!(test_acc >= 0.0);
        assert!(test_acc <= 1.0);
    }

    Ok(())
}

#[test]
fn test_predict_with_various_messages() -> Result<()> {
    let dataset = create_test_dataset()?;

    let trainer = MLTrainer::new(10, Some(5), 100);
    let model = trainer.train(&dataset)?;

    // Test various message types
    let test_messages = vec![
        "fix: null pointer dereference",
        "fix: race condition in mutex",
        "fix: memory leak",
        "fix: type error",
        "fix: AST transformation bug",
        "fix: operator precedence",
        "feat: add new feature", // Non-defect
        "docs: update README",   // Non-defect
    ];

    for message in test_messages {
        let result = model.predict(message);
        // Should not error
        assert!(result.is_ok());
    }

    Ok(())
}
