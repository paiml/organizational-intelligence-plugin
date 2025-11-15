// Unit tests for Rule-Based Classifier
// Following EXTREME TDD: Tests first
// Phase 1: Simple heuristic-based classification with confidence scores

use organizational_intelligence_plugin::classifier::{
    Classification, DefectCategory, RuleBasedClassifier,
};

#[test]
fn test_classifier_can_be_created() {
    // RED: This will fail until we implement RuleBasedClassifier
    let _classifier = RuleBasedClassifier::new();
}

#[test]
fn test_defect_categories_exist() {
    // Test that all 10 defect categories are defined
    let categories = vec![
        DefectCategory::MemorySafety,
        DefectCategory::ConcurrencyBugs,
        DefectCategory::LogicErrors,
        DefectCategory::ApiMisuse,
        DefectCategory::ResourceLeaks,
        DefectCategory::TypeErrors,
        DefectCategory::ConfigurationErrors,
        DefectCategory::SecurityVulnerabilities,
        DefectCategory::PerformanceIssues,
        DefectCategory::IntegrationFailures,
    ];

    assert_eq!(categories.len(), 10);
}

#[test]
fn test_classify_memory_safety_from_commit_message() {
    // RED: Test classifying based on commit message patterns
    let classifier = RuleBasedClassifier::new();
    let commit_message = "fix: use-after-free in buffer handling";

    let classification = classifier
        .classify_from_message(commit_message)
        .expect("Should classify");

    assert_eq!(classification.category, DefectCategory::MemorySafety);
    assert!(classification.confidence > 0.0);
    assert!(classification.confidence <= 1.0);
    assert!(!classification.explanation.is_empty());
}

#[test]
fn test_classify_concurrency_from_commit_message() {
    let classifier = RuleBasedClassifier::new();
    let commit_message = "fix: data race in shared counter";

    let classification = classifier
        .classify_from_message(commit_message)
        .expect("Should classify");

    assert_eq!(classification.category, DefectCategory::ConcurrencyBugs);
    assert!(classification.confidence > 0.5);
}

#[test]
fn test_classify_security_from_commit_message() {
    let classifier = RuleBasedClassifier::new();
    let commit_message = "security: prevent SQL injection in query builder";

    let classification = classifier
        .classify_from_message(commit_message)
        .expect("Should classify");

    assert_eq!(classification.category, DefectCategory::SecurityVulnerabilities);
}

#[test]
fn test_classification_includes_explanation() {
    // CRITICAL: Every classification must include explanation (Respect for People)
    let classifier = RuleBasedClassifier::new();
    let commit_message = "fix: null pointer dereference";

    let classification = classifier
        .classify_from_message(commit_message)
        .expect("Should classify");

    assert!(!classification.explanation.is_empty());
    assert!(classification
        .explanation
        .contains("null pointer dereference"));
}

#[test]
fn test_classification_confidence_range() {
    // Confidence must be between 0.0 and 1.0
    let classifier = RuleBasedClassifier::new();
    let commit_message = "fix: memory leak in connection pool";

    let classification = classifier
        .classify_from_message(commit_message)
        .expect("Should classify");

    assert!(classification.confidence >= 0.0);
    assert!(classification.confidence <= 1.0);
}

#[test]
fn test_unclassifiable_message_returns_none() {
    // Messages without defect keywords should return None
    let classifier = RuleBasedClassifier::new();
    let commit_message = "docs: update README with installation instructions";

    let result = classifier.classify_from_message(commit_message);

    assert!(result.is_none(), "Documentation changes should not classify as defects");
}

#[test]
fn test_case_insensitive_matching() {
    // Pattern matching should be case-insensitive
    let classifier = RuleBasedClassifier::new();
    let commit_message = "Fix: RACE CONDITION in async handler";

    let classification = classifier
        .classify_from_message(commit_message)
        .expect("Should classify");

    assert_eq!(classification.category, DefectCategory::ConcurrencyBugs);
}

#[test]
fn test_multiple_pattern_matches_uses_highest_confidence() {
    // If multiple patterns match, use the one with highest confidence
    let classifier = RuleBasedClassifier::new();
    let commit_message = "fix: deadlock and race condition in mutex handling";

    let classification = classifier
        .classify_from_message(commit_message)
        .expect("Should classify");

    // Should classify as concurrency bug (both deadlock and race are concurrency issues)
    assert_eq!(classification.category, DefectCategory::ConcurrencyBugs);
    assert!(classification.confidence > 0.7, "Multiple matches should increase confidence");
}
