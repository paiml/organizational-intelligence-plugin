// Example: Classify defects from commit messages
// Demonstrates the rule-based classifier in action
//
// Usage:
//   cargo run --example classify_defects
//
// This example shows Phase 1 classifier capabilities

use organizational_intelligence_plugin::classifier::{DefectCategory, RuleBasedClassifier};

fn main() {
    // Initialize logging
    tracing_subscriber::fmt::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("🤖 Rule-Based Defect Classifier - Example\n");
    println!("Phase 1: Heuristic pattern matching with confidence scores\n");

    // Create classifier
    let classifier = RuleBasedClassifier::new();

    // Test commit messages (representing typical bug fixes)
    let test_messages = vec![
        "fix: use-after-free in buffer handling",
        "fix: race condition in async handler",
        "security: prevent SQL injection in query builder",
        "fix: memory leak in connection pool",
        "fix: deadlock when acquiring multiple locks",
        "fix: null pointer dereference in parser",
        "fix: off-by-one error in array indexing",
        "perf: optimize slow database query",
        "fix: unchecked error in file operations",
        "docs: update installation guide", // Should not classify
        "feat: add user authentication",   // Should not classify
    ];

    println!("📋 Classifying {} commit messages:\n", test_messages.len());

    let mut classified = 0;
    let mut unclassified = 0;

    for message in &test_messages {
        println!("Message: \"{}\"", message);

        match classifier.classify_from_message(message) {
            Some(classification) => {
                println!("  ✅ Category: {}", classification.category.as_str());
                println!("  📊 Confidence: {:.0}%", classification.confidence * 100.0);
                println!("  💡 Explanation: {}", classification.explanation);
                println!(
                    "  🔍 Matched patterns: {}",
                    classification.matched_patterns.join(", ")
                );
                classified += 1;
            }
            None => {
                println!("  ⏭️  Not classified as a defect (documentation/feature/etc)");
                unclassified += 1;
            }
        }
        println!();
    }

    // Summary
    println!("📊 Summary:");
    println!(
        "   Classified: {} ({:.0}%)",
        classified,
        (classified as f32 / test_messages.len() as f32) * 100.0
    );
    println!(
        "   Unclassified: {} ({:.0}%)",
        unclassified,
        (unclassified as f32 / test_messages.len() as f32) * 100.0
    );

    // Demonstrate category coverage
    println!("\n📚 All 10 Defect Categories:");
    let categories = [
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

    for (i, category) in categories.iter().enumerate() {
        println!("   {}. {}", i + 1, category.as_str());
    }

    println!("\n🎯 Phase 1 Classifier Complete!");
    println!("   Next: Collect user feedback for Phase 2 ML training");
}
