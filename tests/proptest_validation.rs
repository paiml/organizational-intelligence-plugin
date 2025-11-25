/// Property-based tests using proptest for validation
/// Tests invariants and properties across random inputs
use organizational_intelligence_plugin::classifier::{DefectCategory, RuleBasedClassifier};
use organizational_intelligence_plugin::git::CommitInfo;
use organizational_intelligence_plugin::training::TrainingDataExtractor;
use proptest::prelude::*;

// Strategy for generating random commit messages
fn commit_message_strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        // Fix messages
        Just("fix: null pointer dereference".to_string()),
        Just("fix: memory leak in allocator".to_string()),
        Just("fix: race condition in mutex".to_string()),
        Just("fix: type error in generics".to_string()),
        Just("fix: configuration parsing bug".to_string()),
        Just("fix: API misuse in client".to_string()),
        Just("fix: resource leak in file handler".to_string()),
        Just("fix: security vulnerability in auth".to_string()),
        Just("fix: performance issue in loop".to_string()),
        Just("fix: integration test failure".to_string()),
        Just("fix: operator precedence bug".to_string()),
        Just("fix: type annotation missing".to_string()),
        Just("fix: stdlib mapping error".to_string()),
        Just("fix: AST transformation issue".to_string()),
        Just("fix: comprehension bug".to_string()),
        Just("fix: iterator chain error".to_string()),
        Just("fix: ownership borrow issue".to_string()),
        Just("fix: trait bound violation".to_string()),
        // Non-fix messages (should not be classified as defects)
        Just("feat: add new feature".to_string()),
        Just("docs: update README".to_string()),
        Just("chore: update dependencies".to_string()),
        Just("refactor: clean up code".to_string()),
        Just("test: add unit tests".to_string()),
        // Random strings
        "[a-zA-Z0-9 :_-]{10,100}".prop_map(|s| s),
    ]
}

// Strategy for generating random confidence thresholds
fn confidence_strategy() -> impl Strategy<Value = f32> {
    0.0f32..=1.0f32
}

// Strategy for generating commit info
fn commit_info_strategy() -> impl Strategy<Value = CommitInfo> {
    (
        "[a-f0-9]{40}",                   // hash
        commit_message_strategy(),        // message
        "[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+", // author
        0i64..2000000000i64,              // timestamp
        0usize..100usize,                 // files_changed
        0usize..1000usize,                // lines_added
        0usize..1000usize,                // lines_removed
    )
        .prop_map(
            |(hash, message, author, timestamp, files_changed, lines_added, lines_removed)| {
                CommitInfo {
                    hash,
                    message,
                    author,
                    timestamp,
                    files_changed,
                    lines_added,
                    lines_removed,
                }
            },
        )
}

proptest! {
    /// Property: Classifier always returns a valid category or None
    #[test]
    fn classifier_returns_valid_category(message in commit_message_strategy()) {
        let classifier = RuleBasedClassifier::new();
        if let Some(classification) = classifier.classify_from_message(&message) {
            // Category must be one of the 18 valid categories
            prop_assert!(matches!(
                classification.category,
                DefectCategory::MemorySafety
                    | DefectCategory::ConcurrencyBugs
                    | DefectCategory::LogicErrors
                    | DefectCategory::ApiMisuse
                    | DefectCategory::ResourceLeaks
                    | DefectCategory::TypeErrors
                    | DefectCategory::ConfigurationErrors
                    | DefectCategory::SecurityVulnerabilities
                    | DefectCategory::PerformanceIssues
                    | DefectCategory::IntegrationFailures
                    | DefectCategory::OperatorPrecedence
                    | DefectCategory::TypeAnnotationGaps
                    | DefectCategory::StdlibMapping
                    | DefectCategory::ASTTransform
                    | DefectCategory::ComprehensionBugs
                    | DefectCategory::IteratorChain
                    | DefectCategory::OwnershipBorrow
                    | DefectCategory::TraitBounds
            ));
        }
    }

    /// Property: Classification confidence is always between 0 and 1
    #[test]
    fn classifier_confidence_in_valid_range(message in commit_message_strategy()) {
        let classifier = RuleBasedClassifier::new();
        if let Some(classification) = classifier.classify_from_message(&message) {
            prop_assert!(classification.confidence >= 0.0);
            prop_assert!(classification.confidence <= 1.0);
        }
    }

    /// Property: Same message always produces same classification (deterministic)
    #[test]
    fn classifier_is_deterministic(message in commit_message_strategy()) {
        let classifier = RuleBasedClassifier::new();
        let result1 = classifier.classify_from_message(&message);
        let result2 = classifier.classify_from_message(&message);

        match (result1, result2) {
            (Some(c1), Some(c2)) => {
                prop_assert_eq!(c1.category, c2.category);
                prop_assert!((c1.confidence - c2.confidence).abs() < f32::EPSILON);
            }
            (None, None) => {}
            _ => prop_assert!(false, "Classification should be deterministic"),
        }
    }

    /// Property: Training extractor with high confidence threshold produces
    /// fewer or equal examples than with low threshold
    #[test]
    fn training_extractor_confidence_monotonic(
        commits in prop::collection::vec(commit_info_strategy(), 1..20),
        low_conf in 0.3f32..0.5f32,
        high_conf in 0.7f32..0.9f32
    ) {
        let low_extractor = TrainingDataExtractor::new(low_conf);
        let high_extractor = TrainingDataExtractor::new(high_conf);

        let low_examples = low_extractor.extract_training_data(&commits, "test-repo");
        let high_examples = high_extractor.extract_training_data(&commits, "test-repo");

        if let (Ok(low), Ok(high)) = (low_examples, high_examples) {
            prop_assert!(high.len() <= low.len(),
                "High confidence threshold ({}) should produce <= examples than low threshold ({})",
                high_conf, low_conf);
        }
    }

    /// Property: Training examples always have valid confidence scores
    #[test]
    fn training_examples_have_valid_confidence(
        commits in prop::collection::vec(commit_info_strategy(), 1..10),
        min_conf in confidence_strategy()
    ) {
        let extractor = TrainingDataExtractor::new(min_conf);
        if let Ok(examples) = extractor.extract_training_data(&commits, "test-repo") {
            for example in examples {
                prop_assert!(example.confidence >= 0.0);
                prop_assert!(example.confidence <= 1.0);
                // All examples should meet minimum confidence threshold
                prop_assert!(example.confidence >= min_conf,
                    "Example confidence {} should be >= min_conf {}",
                    example.confidence, min_conf);
            }
        }
    }

    /// Property: DefectCategory Display and Debug are non-empty
    #[test]
    fn defect_category_display_non_empty(category in prop_oneof![
        Just(DefectCategory::MemorySafety),
        Just(DefectCategory::ConcurrencyBugs),
        Just(DefectCategory::LogicErrors),
        Just(DefectCategory::ApiMisuse),
        Just(DefectCategory::ResourceLeaks),
        Just(DefectCategory::TypeErrors),
        Just(DefectCategory::ConfigurationErrors),
        Just(DefectCategory::SecurityVulnerabilities),
        Just(DefectCategory::PerformanceIssues),
        Just(DefectCategory::IntegrationFailures),
        Just(DefectCategory::OperatorPrecedence),
        Just(DefectCategory::TypeAnnotationGaps),
        Just(DefectCategory::StdlibMapping),
        Just(DefectCategory::ASTTransform),
        Just(DefectCategory::ComprehensionBugs),
        Just(DefectCategory::IteratorChain),
        Just(DefectCategory::OwnershipBorrow),
        Just(DefectCategory::TraitBounds),
    ]) {
        let display = format!("{}", category);
        let debug = format!("{:?}", category);

        prop_assert!(!display.is_empty(), "Display should not be empty");
        prop_assert!(!debug.is_empty(), "Debug should not be empty");
    }

    /// Property: CommitInfo preserves all fields through clone
    #[test]
    fn commit_info_clone_preserves_fields(commit in commit_info_strategy()) {
        let cloned = commit.clone();

        prop_assert_eq!(commit.hash, cloned.hash);
        prop_assert_eq!(commit.message, cloned.message);
        prop_assert_eq!(commit.author, cloned.author);
        prop_assert_eq!(commit.timestamp, cloned.timestamp);
        prop_assert_eq!(commit.files_changed, cloned.files_changed);
        prop_assert_eq!(commit.lines_added, cloned.lines_added);
        prop_assert_eq!(commit.lines_removed, cloned.lines_removed);
    }
}

#[cfg(test)]
mod additional_property_tests {
    use super::*;
    use organizational_intelligence_plugin::nlp::TfidfFeatureExtractor;

    proptest! {
        /// Property: TF-IDF vocabulary size is bounded by max_features
        #[test]
        fn tfidf_vocabulary_bounded(
            messages in prop::collection::vec("[a-zA-Z ]{5,50}", 2..10),
            max_features in 10usize..100usize
        ) {
            let mut extractor = TfidfFeatureExtractor::new(max_features);
            if extractor.fit_transform(&messages).is_ok() {
                prop_assert!(extractor.vocabulary_size() <= max_features);
            }
        }

        /// Property: TF-IDF produces consistent output dimensions
        #[test]
        fn tfidf_output_dimensions_consistent(
            messages in prop::collection::vec("[a-zA-Z ]{5,50}", 2..10)
        ) {
            let mut extractor = TfidfFeatureExtractor::new(100);
            if let Ok(matrix) = extractor.fit_transform(&messages) {
                prop_assert_eq!(matrix.n_rows(), messages.len());
                prop_assert!(matrix.n_cols() <= 100);
            }
        }
    }
}
